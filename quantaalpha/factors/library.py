"""
Factor library manager: save experiment output to unified JSON factor library.
Called from quantaalpha/pipeline/loop.py feedback step.
"""

import json
import hashlib
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

DEFAULT_FACTOR_CACHE_DIR = os.environ.get(
    "FACTOR_CACHE_DIR",
    "data/results/factor_cache",
)


def _per_factor_metrics(backtest_results: Any, factor_expr: str) -> dict:
    """This factor's own combiner credit, pulled out of the batch's results (C4).

    ``factor_attribution`` is ``{expression: {...}}`` built by the ICIR
    combiner's ``_icir_attribution`` and carried through by the runner. Each
    record holds the factor's share of the composite's weight (``weight``), the
    signed mean ICIR weight (``weight_raw``), cross-seed stability
    (``weight_stability``), its own full-sample ``rank_ic``/``ic_mean``/
    ``ic_std``, and its share of the book's rebalance cost
    (``turnover_share``).

    Returns ``{}`` when attribution is unavailable -- the LightGBM path produces
    no per-factor weights, and a cache miss yields none either -- so callers see
    an explicitly empty dict rather than the batch numbers wearing a per-factor
    label, which would be worse than nothing.
    """
    if not isinstance(backtest_results, dict) or not factor_expr:
        return {}
    attribution = backtest_results.get("factor_attribution")
    if not isinstance(attribution, dict):
        return {}
    rec = attribution.get(factor_expr)
    return dict(rec) if isinstance(rec, dict) else {}


def _admitted_flag(backtest_results: Any):
    """Whether F_Theta admitted this factor, or None if the arm has no verdict.

    The net-cost runner emits `feasible`; a Qlib-only metrics Series carries
    no such key, so Qlib-only libraries record None rather than a misleading
    False.
    """
    if not isinstance(backtest_results, dict):
        return None
    # `in_zoo` is the current signal (set by NetCostFactorRunner for every
    # factor that entered the repository). `feasible` is the older gated form,
    # still read so pre-existing libraries keep working.
    value = backtest_results.get("in_zoo")
    if value is None:
        value = backtest_results.get("feasible")
    if value is None:
        return None
    if isinstance(value, str):
        return value.strip().lower() in ("true", "1", "yes")
    return bool(value)


class FactorLibraryManager:
    """Manage unified factor library (CRUD)."""

    def __init__(self, library_path: str):
        self.library_path = Path(library_path)
        self.data = self._load()

    def _load(self) -> dict:
        if self.library_path.exists():
            try:
                with open(self.library_path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except (json.JSONDecodeError, Exception) as e:
                logger.warning(f"Factor library file corrupted, recreating: {e}")
        return self._fresh()

    @staticmethod
    def _fresh() -> dict:
        """A new library, seeded from Alpha158(20) when asked.

        Seeding has to happen HERE rather than in a driver script. The library
        filename carries the run's stamp (``all_factors_library_<arm>_<STAMP>``)
        and the stamp does not exist until the run starts, so a script that
        pre-writes a seeded file writes one nothing will ever open -- which is
        exactly what an earlier attempt did.

        This seeds the pool the *generator* sees, not the E_Θ repository: the
        runner rehydrates its repository from the ledger, so a seeded
        factor still has to earn admission on its own contribution. That is the
        intended split -- the pool is what the search starts from, not what it
        is credited with.

        Which is why seeds are written ``admitted: False``. They used to be
        written ``True``, and that quietly broke the split the paragraph above
        describes: ``_zoo.json`` is ``write_admitted_subset()``, which selects
        on exactly this flag rather than on the ledger, so all 20 public
        Alpha158 factors were exported as part of the mined repository and
        every figure computed from that file credited the search with them.
        The repository was never affected -- it comes from the ledger -- so the
        two disagreed about what had been admitted, and the file that gets
        backtested was the one that was wrong.

        ``False`` rather than ``None`` because the verdict is not missing: a
        seed has not passed E_Θ. If the search later proposes the same
        expression and it earns admission, the normal update path sets the flag
        and the factor is credited then, on its own contribution.

        Off unless ``QA_SEED_POOL=true``, and a failure to seed degrades to an
        empty library rather than taking the run down with it.
        """
        base = {
            "metadata": {
                "created_at": datetime.now().isoformat(),
                "last_updated": datetime.now().isoformat(),
                "total_factors": 0,
                "version": "1.0",
            },
            "factors": {},
        }
        if os.environ.get("QA_SEED_POOL", "").lower() != "true":
            return base
        try:
            import hashlib

            from quantaalpha.factors.seed_pool import SEED_POOL

            for name, expr in SEED_POOL.items():
                fid = hashlib.md5(f"seed::{name}::{expr}".encode()).hexdigest()[:16]
                base["factors"][fid] = {
                    "factor_id": fid, "factor_name": name,
                    "factor_expression": expr, "factor_implementation_code": "",
                    "factor_description": f"Alpha158(20) seed factor {name}.",
                    "factor_formulation": expr, "cache_location": {},
                    "metadata": {"source": "alpha158_20_seed_pool",
                                 "round_number": -1, "evolution_phase": "seed"},
                    "backtest_results": {}, "feedback": "", "admitted": False,
                }
            base["metadata"]["total_factors"] = len(base["factors"])
            base["metadata"]["seeded_from"] = "alpha158_20"
            logger.info(f"Factor library seeded with {len(base['factors'])} "
                        "Alpha158(20) factors")
        except Exception as e:
            logger.warning(f"Could not seed the factor library, starting empty: {e}")
        return base

    def admitted_ids(self) -> list:
        """Factor ids that passed F_Theta, in insertion order."""
        return [
            fid for fid, f in self.data.get("factors", {}).items()
            if isinstance(f, dict) and f.get("admitted") is True
        ]

    def write_admitted_subset(self, path) -> int:
        """Write a library JSON containing only the admitted factors.

        The effective-alpha repository, in the same schema, so it can be fed
        straight to `run_backtest --factor-json`. Returns the count written.
        """
        ids = self.admitted_ids()
        subset = {
            "metadata": {
                **self.data.get("metadata", {}),
                "derived_from": str(self.library_path),
                "subset": "admitted_only",
                "total_factors": len(ids),
            },
            "factors": {fid: self.data["factors"][fid] for fid in ids},
        }
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", encoding="utf-8") as f:
            json.dump(subset, f, ensure_ascii=False, indent=2, default=str)
        logger.info(f"Wrote {len(ids)} admitted factor(s) to {out}")
        return len(ids)

    def write_zoo_subset(self, path, exprs) -> int:
        """Write a library JSON containing only the factors currently in the zoo.

        ``exprs`` is the active repository -- admissions minus evictions,
        replayed in order -- which is what the combiner/book actually run on.
        Same schema as ``write_admitted_subset``, so it drops straight into
        ``run_backtest --factor-json``.

        Why this exists separately from ``write_admitted_subset``: the
        ``admitted`` flag is set on admission and never cleared when a factor is
        later evicted by the cap, prune, or replace paths, so selecting on it
        re-exports factors the gate explicitly removed -- the deliverable
        overcounts the live zoo by every eviction. Selecting on the ledger-
        reconstructed zoo (the same source the runner rehydrates from on resume)
        keeps the ``_zoo.json`` honest. Returns the count written.
        """
        want = set(exprs)
        factors = self.data.get("factors", {})
        # One entry per zoo expression. The library can hold several entries
        # for the same expression (a factor re-proposed across rounds); keep
        # only the one that earned admission, falling back to the first seen,
        # so the deliverable never carries redundant copies of one signal.
        picked: dict[str, str] = {}
        for fid, f in factors.items():
            if not isinstance(f, dict):
                continue
            expr = f.get("factor_expression")
            if expr not in want:
                continue
            cur = picked.get(expr)
            if cur is None or (f.get("admitted") and not factors[cur].get("admitted")):
                picked[expr] = fid
        subset_factors = {fid: factors[fid] for fid in picked.values()}
        # Drop the main library's `admitted_count`/`admitted_factor_ids`: they
        # count the cumulative-ever-admitted set (stale on eviction) and would
        # sit next to `total_factors` showing two different numbers. The active
        # zoo is described by `total_factors` + `subset`, not by admission flags.
        meta = {k: v for k, v in self.data.get("metadata", {}).items()
                if k not in ("admitted_count", "admitted_factor_ids")}
        meta.update({
            "derived_from": str(self.library_path),
            "subset": "active_zoo",
            "total_factors": len(subset_factors),
        })
        subset = {"metadata": meta, "factors": subset_factors}
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", encoding="utf-8") as f:
            json.dump(subset, f, ensure_ascii=False, indent=2, default=str)
        logger.info(f"Wrote {len(subset_factors)} active-zoo factor(s) to {out}")
        return len(subset_factors)

    def _save(self):
        self.data["metadata"]["last_updated"] = datetime.now().isoformat()
        self.data["metadata"]["total_factors"] = len(self.data["factors"])
        admitted = self.admitted_ids()
        self.data["metadata"]["admitted_factor_ids"] = admitted
        self.data["metadata"]["admitted_count"] = len(admitted)
        self.library_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.library_path, "w", encoding="utf-8") as f:
            json.dump(self.data, f, ensure_ascii=False, indent=2, default=str)

    def add_factors_from_experiment(
        self,
        experiment,
        experiment_id: str = "unknown",
        round_number: int = 0,
        hypothesis: Optional[str] = None,
        feedback: Any = None,
        initial_direction: Optional[str] = None,
        user_initial_direction: Optional[str] = None,
        planning_direction: Optional[str] = None,
        evolution_phase: str = "original",
        trajectory_id: str = "",
        parent_trajectory_ids: Optional[list] = None,
    ):
        """Extract factors from a QlibFactorExperiment and write to library."""
        if experiment is None:
            logger.warning("experiment is None, skip saving factors")
            return
        backtest_results = self._extract_backtest_results(experiment)
        feedback_dict = self._extract_feedback(feedback)
        sub_tasks = getattr(experiment, "sub_tasks", []) or []
        sub_workspaces = getattr(experiment, "sub_workspace_list", []) or []

        for idx, task in enumerate(sub_tasks):
            factor_name = getattr(task, "factor_name", getattr(task, "name", f"factor_{idx}"))
            factor_expr = getattr(task, "factor_expression", "")
            factor_desc = getattr(task, "factor_description", getattr(task, "description", ""))
            factor_form = getattr(task, "factor_formulation", "")

            factor_id = hashlib.md5(
                f"{factor_name}_{factor_expr}".encode()
            ).hexdigest()[:16]

            code = ""
            cache_location = {}
            if idx < len(sub_workspaces):
                ws = sub_workspaces[idx]
                code_dict = getattr(ws, "code_dict", {})
                code = "\n".join(
                    f"File: {fname}\n\n{content}"
                    for fname, content in code_dict.items()
                )
                ws_path = getattr(ws, "workspace_path", None)
                if ws_path:
                    ws_path = Path(ws_path)
                    workspace_suffix = ""
                    for part in ws_path.parts:
                        if part.startswith("workspace_"):
                            workspace_suffix = part.replace("workspace_", "")
                            break
                    h5_file = ws_path / "result.h5"
                    cache_location = {
                        "workspace_suffix": workspace_suffix,
                        "workspace_path": str(ws_path.parent),
                        "factor_dir": ws_path.name,
                    }
                    if h5_file.exists():
                        cache_location["result_h5_path"] = str(h5_file)
                    else:
                        logger.warning(
                            f"result.h5 missing for {factor_name} ({h5_file}), will recompute from expression in backtest"
                        )

            factor_entry = {
                "factor_id": factor_id,
                "factor_name": factor_name,
                "factor_expression": factor_expr,
                "factor_implementation_code": code,
                "factor_description": factor_desc,
                "factor_formulation": factor_form,
                "cache_location": cache_location,
                "metadata": {
                    "experiment_id": experiment_id,
                    "round_number": round_number,
                    "evolution_phase": evolution_phase,
                    "trajectory_id": trajectory_id,
                    "parent_trajectory_ids": parent_trajectory_ids or [],
                    "hypothesis": str(hypothesis) if hypothesis else "",
                    "initial_direction": initial_direction or "",
                    "planning_direction": planning_direction or "",
                    "created_at": datetime.now().isoformat(),
                },
                "backtest_results": backtest_results,
                # C4: this factor's OWN metrics, not the batch's.
                #
                # ``backtest_results`` above is one dict per EXPERIMENT, written
                # identically to all N factors of the batch -- so every factor of
                # a hypothesis carried the same RankIC and the library was not
                # rankable (measured: 78 factors sharing 26 distinct IC values).
                # The combiner already computes per-factor credit in
                # ``_icir_attribution`` and the operator surfaces it as
                # ``factor_attribution`` keyed by expression; it just was never
                # written down. Empty on the LightGBM path or a cache miss.
                "factor_metrics": _per_factor_metrics(backtest_results, factor_expr),
                "feedback": feedback_dict,
                # Did this factor pass F_Theta and enter the effective-alpha
                # repository? The library records every trial (which the ledger
                # needs), but `zoo` is only the admitted subset -- keep the two
                # distinguishable without having to dig into backtest_results.
                # None when the arm produced no feasibility verdict (control).
                "admitted": _admitted_flag(backtest_results),
            }

            self.data["factors"][factor_id] = factor_entry

            if factor_expr and cache_location.get("result_h5_path"):
                self._sync_h5_to_md5_cache(factor_expr, cache_location["result_h5_path"])

        self._save()
        logger.info(
            f"Saved {len(sub_tasks)} factors to {self.library_path} (backtest_results: {len(backtest_results)} metrics)"
        )

    @staticmethod
    def _sync_h5_to_md5_cache(factor_expression: str, h5_path: str,
                                cache_dir: Optional[str] = None) -> bool:
        """Sync factor values from result.h5 to MD5 cache dir (.pkl). Returns True on success."""
        cache_dir = Path(cache_dir or DEFAULT_FACTOR_CACHE_DIR)
        h5_file = Path(h5_path)

        if not h5_file.exists():
            return False

        md5_key = hashlib.md5(factor_expression.encode()).hexdigest()
        pkl_file = cache_dir / f"{md5_key}.pkl"

        if pkl_file.exists():
            return True

        try:
            cache_dir.mkdir(parents=True, exist_ok=True)
            result = pd.read_hdf(str(h5_file))
            result.to_pickle(pkl_file)
            logger.debug(f"Synced factor cache -> {pkl_file.name}")
            return True
        except Exception as e:
            logger.debug(f"Sync factor cache failed [{h5_path}]: {e}")
            return False

    @staticmethod
    def check_cache_status(library_path: str,
                           cache_dir: Optional[str] = None) -> dict:
        """Check cache status for each factor in library. Returns:
            {
                "total": int,
                "h5_cached": int,
                "md5_cached": int,
                "need_compute": int,
                "factors": [ { "factor_id", "factor_name", "status" }, ... ]
            }
        """
        cache_dir = Path(cache_dir or DEFAULT_FACTOR_CACHE_DIR)

        with open(library_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        factors = data.get("factors", {})
        total = len(factors)
        h5_cached = 0
        md5_cached = 0
        need_compute = 0
        details = []

        for fid, finfo in factors.items():
            expr = finfo.get("factor_expression", "")
            cloc = finfo.get("cache_location", {})
            h5_path = cloc.get("result_h5_path", "")

            status = "need_compute"
            # Check h5 cache
            if h5_path and Path(h5_path).exists():
                status = "h5_cached"
                h5_cached += 1
            # Check MD5 cache
            elif expr:
                md5_key = hashlib.md5(expr.encode()).hexdigest()
                if (cache_dir / f"{md5_key}.pkl").exists():
                    status = "md5_cached"
                    md5_cached += 1

            if status == "need_compute":
                need_compute += 1

            details.append({
                "factor_id": fid,
                "factor_name": finfo.get("factor_name", fid),
                "status": status,
            })

        return {
            "total": total,
            "h5_cached": h5_cached,
            "md5_cached": md5_cached,
            "need_compute": need_compute,
            "factors": details,
        }

    @staticmethod
    def warm_cache_from_json(library_path: str,
                             cache_dir: Optional[str] = None) -> dict:
        """Walk factor library JSON and sync all available result.h5 to MD5 cache dir. Returns:
            { "total": int, "synced": int, "skipped": int, "failed": int,
              "already_cached": int, "no_source": int }
        """
        cache_dir_path = Path(cache_dir or DEFAULT_FACTOR_CACHE_DIR)

        with open(library_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        factors = data.get("factors", {})
        synced = 0
        skipped = 0
        failed = 0
        already_cached = 0
        no_source = 0

        for fid, finfo in factors.items():
            expr = finfo.get("factor_expression", "")
            cloc = finfo.get("cache_location", {})
            h5_path = cloc.get("result_h5_path", "")

            if not expr or not h5_path:
                no_source += 1
                skipped += 1
                continue

            md5_key = hashlib.md5(expr.encode()).hexdigest()
            pkl_file = cache_dir_path / f"{md5_key}.pkl"

            if pkl_file.exists():
                already_cached += 1
                skipped += 1
                continue

            if not Path(h5_path).exists():
                failed += 1
                continue

            try:
                cache_dir_path.mkdir(parents=True, exist_ok=True)
                result = pd.read_hdf(str(h5_path))
                result.to_pickle(pkl_file)
                synced += 1
            except Exception:
                failed += 1

        return {
            "total": len(factors),
            "synced": synced,
            "skipped": skipped,
            "failed": failed,
            "already_cached": already_cached,
            "no_source": no_source,
        }

    @staticmethod
    def _extract_backtest_results(experiment) -> dict:
        """Extract backtest metrics from experiment.result (pandas Series) as dict."""
        result = getattr(experiment, "result", None)
        if result is None:
            return {}
        if isinstance(result, pd.Series):
            out = {}
            for key, val in result.items():
                # NaN/Inf -> None for JSON
                if isinstance(val, (float, np.floating)):
                    if np.isnan(val) or np.isinf(val):
                        out[str(key)] = None
                    else:
                        out[str(key)] = round(float(val), 8)
                else:
                    out[str(key)] = val
            return out

        if isinstance(result, pd.DataFrame):
            try:
                return {
                    str(k): round(float(v), 8) if isinstance(v, (float, np.floating)) and not np.isnan(v) else None
                    for k, v in result.iloc[:, 0].items()
                }
            except Exception:
                pass

        if isinstance(result, dict):
            return result

        return {}

    @staticmethod
    def _extract_feedback(feedback) -> dict:
        """Convert feedback object to serializable dict."""
        if feedback is None:
            return {}
        if isinstance(feedback, dict):
            return feedback

        out = {}
        for attr in ["observations", "hypothesis_evaluation", "decision", "reason",
                      "new_hypothesis", "feedback_str"]:
            val = getattr(feedback, attr, None)
            if val is not None:
                out[attr] = str(val) if not isinstance(val, (bool, int, float)) else val
        if not out:
            out["raw"] = str(feedback)
        return out
