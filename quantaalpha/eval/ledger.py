"""Append-only trial ledger.

One row per evaluation, carrying enough to reconstruct the trial: the factor,
the protocol hash, **and the repository hash**. The last one is not optional —
``m(f) = E_Θ(f; zoo)`` is deterministic with respect to the *pair*, so a row
without ``zoo_hash`` is not reproducible.

This is also the substrate for the explicitly out-of-scope work (deflated
Sharpe, multiple-testing correction): those need the full record of every
trial attempted, not just the ones that survived, which is why every
evaluation is written here regardless of feasibility.
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

DEFAULT_LEDGER_PATH = os.environ.get("QA_LEDGER", "data/results/eval_ledger.jsonl")


class Ledger:
    """JSONL sink. Opened in append mode and flushed per write."""

    def __init__(self, path: str | os.PathLike[str] | None = None) -> None:
        self.path = Path(path or DEFAULT_LEDGER_PATH)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, record: dict[str, Any]) -> None:
        row = {"ts": datetime.now(timezone.utc).isoformat(), **record}
        try:
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(row, default=str, sort_keys=True) + "\n")
                handle.flush()
        except OSError as exc:
            # A ledger write must never take down a mining run; losing a row is
            # bad, losing the run is worse.
            logger.warning("ledger append failed (%s): %s", self.path, exc)

    def read(self) -> list[dict[str, Any]]:
        """Every well-formed record, skipping any that are not.

        The ledger is read while other processes are appending to it (the
        repository is rehydrated from here), and a record can exceed the size
        POSIX guarantees atomic for an append. A torn final line must cost one
        row, not raise -- the same trade-off ``append`` already makes.
        """
        if not self.path.exists():
            return []
        rows: list[dict[str, Any]] = []
        with self.path.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    logger.warning("skipping malformed ledger line in %s", self.path)
        return rows

    def __len__(self) -> int:
        return len(self.read())


def replay_repository(path: str | os.PathLike[str] | None = None) -> dict[str, dict]:
    """The repository as the ledger says it currently stands.

    Replays records IN ORDER: an admitted batch adds its expressions, an
    eviction record removes them. Order matters -- a factor admitted, evicted,
    then re-admitted must end up present, and one admitted then evicted must
    not. Rejected batches are recorded under ``rejected_exprs`` and contribute
    nothing.

    Shared by the runner (which rehydrates from it across process boundaries)
    and the evolution controller (which sizes the repository to decide whether
    to keep mining), so the two cannot disagree about what was admitted.
    """
    repo: dict[str, dict] = {}
    # QA_RESTORE_CAP_EVICTED=1 skips library_cap evictions on replay: those
    # factors were removed ONLY by the count cap (admission.max_library /
    # QA_MAX_LIBRARY), not by any quality gate, so raising or removing the cap
    # should let them back into the zoo. Read here -- not at each caller -- so
    # the runner rehydrate, the evolution controller, and the _zoo.json writer
    # all agree on the same zoo (the "cannot disagree" invariant this function
    # exists to enforce). Default off (env unset) is byte-identical to the old
    # behavior. Pair with QA_MAX_LIBRARY on restart.
    _skip_cap = os.environ.get("QA_RESTORE_CAP_EVICTED") == "1"
    for record in Ledger(path).read():
        _ev = record.get("evicted_exprs") or []
        if _skip_cap and record.get("eviction_rule") == "library_cap":
            _ev = []  # keep cap-evicted factors; the cap that removed them is overridden
        for expr in _ev:
            repo.pop(expr, None)
        metrics = record.get("metrics") or {}
        # The per-factor tear sheet (t_nw, rank_ic_neutral, ic_breakeven_book,
        # sign, mechanism, ...) is written to its OWN ledger field, not merged
        # into the batch ``metrics`` at persist time. Admission merges it into
        # the in-memory repository (net_cost_runner.py:540) so the replace duel
        # and eviction can rank on |t_nw|; rehydration MUST do the same, or a
        # fresh runner (one per evolution task) rebuilds the repository from the
        # ledger with batch metrics only -- every incumbent then lacks t_nw,
        # ``_research_score`` returns -inf, and the replace duel lets a WEAKER
        # near-duplicate replace a stronger incumbent (anti-learning, measured
        # 2026-08-24: a |t| 8.92 WMA-smoothed factor replaced a |t| 11.15 one).
        sheets = record.get("factor_tearsheets") or {}
        for expr in record.get("factor_exprs") or []:
            if expr:
                m = dict(metrics)
                sheet = sheets.get(expr) if isinstance(sheets, dict) else None
                if isinstance(sheet, dict):
                    m.update(sheet)
                repo[expr] = m
    return repo


def replay_t_history(path: str | os.PathLike[str] | None = None) -> list[float]:
    """``|t_nw|`` of EVERY factor scored so far this run, for the FDR gate.

    The Benjamini-Hochberg correction runs over the full family of tests, not
    the survivors -- a correction computed only over admitted factors is
    truncated to its own winners and can never bind (see test_winsor_fdr F4).
    So this reads every record's ``factor_tearsheets`` (admitted or rejected)
    and returns ``abs(t_nw)`` for each, in evaluation order.

    Eviction/transition records carry no ``factor_tearsheets`` and contribute
    nothing, so a factor is counted once, at the batch it was scored -- never
    re-added when it is later evicted. Records without tearsheets (pre-fix
    runs) are skipped too, making rehydration safe against a ledger that
    predates this change.
    """
    out: list[float] = []
    try:
        for record in Ledger(path).read():
            sheets = record.get("factor_tearsheets") or {}
            if isinstance(sheets, dict):
                sheets = sheets.values()
            for sheet in sheets:
                try:
                    t = float(sheet.get("t_nw"))
                except (TypeError, ValueError):
                    continue
                if t == t:  # not NaN
                    out.append(abs(t))
    except OSError:
        pass
    return out


__all__ = ["DEFAULT_LEDGER_PATH", "Ledger", "replay_repository", "replay_t_history"]
