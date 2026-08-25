from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import yaml

from quantaalpha.log import logger
from quantaalpha.llm.client import APIBackend


def _load_prompts(prompt_file: Path) -> dict[str, str]:
    if not prompt_file.exists():
        return {}
    try:
        return yaml.safe_load(prompt_file.read_text(encoding="utf-8")) or {}
    except Exception as exc:
        logger.warning(f"Failed to load planning prompts: {exc}")
        return {}


def _extract_json(text: str) -> str:
    if not text:
        return ""
    t = text.strip()
    fence = re.search(r"```json\s*(.*?)```", t, re.DOTALL | re.IGNORECASE)
    if fence:
        t = fence.group(1).strip()
    start = t.find("{")
    end = t.rfind("}")
    if start != -1 and end != -1 and end > start:
        return t[start : end + 1]
    return t


def _parse_directions(message: str, n: int) -> list[str] | None:
    frag = _extract_json(message)
    try:
        data = json.loads(frag)
    except Exception as exc:
        # Say WHICH failure it was. The caller logs "parse failed" and retries,
        # but bad JSON, a missing key and too-few-directions need different
        # fixes, and one message for all three hides that.
        logger.warning(f"directions: JSON decode failed ({exc}); "
                       f"first 120 chars: {(frag or '')[:120]!r}")
        return None
    arr = data.get("directions") if isinstance(data, dict) else None
    if not isinstance(arr, list):
        _shape = list(data)[:8] if isinstance(data, dict) else type(data).__name__
        logger.warning(f"directions: response had no 'directions' list (keys={_shape})")
        return None
    vals = [str(x).strip() for x in arr if isinstance(x, str) and x.strip()]
    if len(vals) < n:
        logger.warning(f"directions: got {len(vals)} usable, need {n}")
        return None
    return vals


def _no_directions(where: str, n: int) -> list[str]:
    """A failed planning call yields NOTHING. Loudly.

    This replaces ``_fallback_directions``, which returned ``n`` canned strings
    of the form ``"{base} + short-term momentum signal with volume
    confirmation"`` / ``"+ volatility regime switch"`` / ``"+ sector
    neutralization"`` / ``"+ fundamental proxy alignment"`` and seven more.

    Every one of those is a market prior the author picked, not one the system
    measured. Injected on an LLM failure they are indistinguishable downstream
    from a direction the planner reasoned its way to, so a factor descended from
    one would credit the search with an answer it was handed -- and the mined
    result would silently be partly ours rather than the system's.

    An empty list is honest and visible. The caller decides whether to stop (no
    initial directions => nothing to mine, and that should be a loud failure
    rather than eight canned ones) or to continue on what it already has (a
    reseed that returns nothing simply does not reseed this round).
    """
    # NOTE: quantaalpha's RDAgentLog takes a single message argument -- no
    # printf-style formatting -- so build the string here.
    logger.error(
        f"{where} produced no directions after all attempts; returning NONE "
        f"rather than substituting canned ones ({n} slot(s) unfilled)"
    )
    return []


def _market_context() -> str:
    """Universe + TRAIN WINDOW ONLY, read from Θ at runtime.

    The generator reasons better when it knows which market and which era it is
    mining -- "US large caps 1995-2005" and "an emerging market 2016-2020" have
    genuinely different microstructure, and a model that knows neither has to
    guess. But telling it is only safe if the horizon is CLAMPED.

    Two rules make this leak-free:

    1. **Only the TRAIN window is disclosed.** ``Θ.splits.valid`` (the in-loop
       selection window) and ``Θ.splits.final_test`` (the holdout) are NEVER
       named. The model cannot aim a factor at a period it does not know the
       boundaries of.
    2. **An explicit no-hindsight instruction.** Knowing "2016-2020" invites a
       model to recall what happened in 2021-2025 and design for it. The block
       below forbids reasoning from any event after the train end date.

    Everything here is derived from the loaded protocol, so pointing the system
    at a different index or a different era changes this text automatically --
    no market-specific string is hardcoded (see the market-agnostic constraint:
    priors must be LEARNED from the disclosed window, never asserted).

    Returns "" if Θ cannot be loaded, so planning never breaks on this.
    """
    try:
        from quantaalpha.eval.protocol import default_protocol_path, load_protocol
        import os

        path = os.environ.get("QA_PROTOCOL") or default_protocol_path()
        theta = load_protocol(path)
        train_start, train_end = theta.splits.train
        universe = getattr(theta, "universe", None) or getattr(theta, "benchmark", "")

        # Clamp the disclosure to BEFORE THE EARLIEST FOLD'S VALIDATION WINDOW.
        #
        # ``Θ.splits.train`` is only the *last* fold's training span. With
        # walk_forward.folds > 1 the earlier folds validate on periods that sit
        # INSIDE that span, so disclosing train_end hands the model hindsight
        # over the very windows it is about to be selected on. Measured on
        # protocol_csi300_meanvar_soft_linear at folds=3:
        #
        #   fold 1  train 2005-01-01..2012-12-09   valid 2012-12-31..2015-12-31
        #   fold 2  train 2005-01-01..2015-12-10   valid 2016-01-01..2018-12-31
        #   fold 3  train 2005-01-01..2018-12-10   valid 2019-01-01..2021-12-31
        #
        # The prompt said "reason about 2005-01-01 to 2018-12-31" while folds 1
        # and 2 validate across 2012-2018 -- six years of the selection window
        # disclosed as free research material. The cutoff must be the day before
        # the EARLIEST validation day across all folds, not the configured
        # train end. folds<=1 reproduces the old value exactly (fold 1's
        # validation IS Θ.splits.valid, which starts after train_end).
        try:
            import pandas as pd
            n_folds = int(getattr(theta.walk_forward, "folds", 1) or 1)
            if getattr(theta.walk_forward, "enabled", False) and n_folds > 1:
                folds = theta.splits.walk_forward_folds(n_folds)
                if folds:
                    earliest_valid = min(pd.Timestamp(v[0]) for _, v in folds)
                    clamp = (earliest_valid - pd.Timedelta(days=1)).strftime("%Y-%m-%d")
                    if clamp < train_end:
                        logger.info(
                            f"market context: clamping the disclosed window to "
                            f"{clamp} (was {train_end}) -- {len(folds)} "
                            f"walk-forward folds validate from "
                            f"{earliest_valid.strftime('%Y-%m-%d')}")
                        train_end = clamp
        except Exception as exc:
            # A clamp failure must FAIL CLOSED, not disclose the wider window.
            logger.warning(
                f"could not clamp the disclosed window to the earliest fold "
                f"({type(exc).__name__}: {exc}); planning proceeds with NO "
                f"window disclosure rather than risk leaking the selection period")
            return ""
    except Exception as exc:
        # Loud. A silent "" here means the model plans WITHOUT the disclosed
        # train window or universe and nothing says so -- the prompt quietly
        # loses the only context grounding it in this dataset.
        logger.warning(f"market context unavailable ({type(exc).__name__}: {exc}); "
                       f"directions will be planned with NO window/universe "
                       f"disclosure")
        return ""

    return (
        "Market context (use it to reason; do NOT go beyond it):\n"
        f"  - Universe / benchmark: {universe}\n"
        f"  - Research window you may reason about: {train_start} to {train_end}\n"
        "  - Infer this market's microstructure for yourself -- typical holding\n"
        "    horizons, whether short-horizon returns tend to continue or revert,\n"
        "    liquidity and trading-cost conditions, session structure. Do not\n"
        "    assume behaviour observed in other markets or other eras carries\n"
        "    over; state the direction you expect and let measurement correct it.\n"
        "  - STRICT KNOWLEDGE CUTOFF -- this is the most important rule here.\n"
        f"    Act as a researcher whose knowledge ENDS on {train_end}. You may\n"
        f"    use ONLY what was known, published or observable on or before\n"
        f"    {train_end}. This is a cutoff on KNOWLEDGE, not merely on data:\n"
        f"      * No events after {train_end} -- crises, rallies, drawdowns,\n"
        "        policy or regulatory changes, index reconstitutions, changes in\n"
        "        tick size, trading hours, settlement or short-selling rules.\n"
        f"      * No factor ideas, anomalies or signal constructions that were\n"
        f"        first published, popularized or widely adopted after {train_end}.\n"
        f"      * No microstructure or market-behaviour knowledge learned after\n"
        f"        {train_end} -- including how this market's participant mix,\n"
        "        liquidity, volatility regime or reversal/momentum behaviour\n"
        "        EVOLVED later, and including which factors later decayed,\n"
        "        crowded out, or stopped working.\n"
        "      * No hindsight phrasing of any kind ('this later proved',\n"
        "      'it is now known that', 'in subsequent years').\n"
        f"    If a justification only makes sense to someone who lived past\n"
        f"    {train_end}, it is disallowed -- rewrite it from what a researcher\n"
        f"    standing on {train_end} could actually have argued.\n"
        "    A factor motivated by hindsight looks excellent in validation and\n"
        "    fails in live trading, which is the exact failure this system\n"
        "    exists to detect.\n"
    )


def _seed_context() -> str:
    """The Alpha158(20) seed library as direction-planning context.

    Delegates to ``seed_pool.seed_context`` rather than rendering a second copy
    here. Until 2026-08-16 this function had its own inline renderer, and the
    two headers had already drifted -- enough that a paper-baseline comparison
    would differ by the seed wording as well as by the thing under test.

    The seeds never enter the zoo or ledger (``admitted: False`` in library.py);
    this only steers what the generator proposes. Returns "" if the pool cannot
    be loaded, so a missing seed module never breaks planning.
    """
    try:
        from quantaalpha.factors.seed_pool import seed_context
        return seed_context()
    except Exception as exc:
        logger.warning(f"seed context unavailable ({type(exc).__name__}: {exc}); "
                       f"directions will be planned with NO seed library")
        return ""


def generate_parallel_directions(
    initial_direction: str,
    n: int,
    prompt_file: Path,
    max_attempts: int = 5,
    use_llm: bool = True,
    seed_in_generation: bool = True,
) -> list[str]:
    n = max(1, int(n))
    prompts = _load_prompts(prompt_file)
    sys_tpl = prompts.get("system", "")
    user_tpl = prompts.get("user", "")
    output_format = prompts.get("output_format", "")

    # Alpha158(20) seed library as orthogonal-signal context (common-mode, both
    # arms; user decision 2026-08-12). Seeds stay admitted:False and never enter
    # the zoo -- this only steers what the LLM proposes. "" when disabled, so a
    # prompt without {seed_context} is unaffected (extra .format kwargs ignored).
    # Market + train-window context always leads: it is the frame the seeds are
    # judged against, and it carries the no-hindsight clamp.
    seed_context = _seed_context() if seed_in_generation else ""
    seed_context = "\n\n".join(x for x in (_market_context(), seed_context) if x)
    fmt = dict(initial_direction=initial_direction, n=n, seed_context=seed_context)
    system_prompt = sys_tpl.format(**fmt)
    user_prompt = user_tpl.format(**fmt)
    if output_format:
        if "{n}" in output_format:
            output_format = output_format.replace("{n}", str(n))
        user_prompt = f"{user_prompt}\n\n{output_format}"

    if not use_llm:
        return _no_directions("generate_parallel_directions (use_llm=False)", n)

    for attempt in range(1, max_attempts + 1):
        try:
            resp = APIBackend().build_messages_and_create_chat_completion(
                user_prompt=user_prompt, system_prompt=system_prompt, json_mode=False
            )
            directions = _parse_directions(resp, n)
            if directions:
                return directions[:n]
            system_prompt += "\n\nStrictly output valid JSON. No extra text."
            logger.warning(f"Planning parse failed (attempt {attempt}), retrying...")
        except Exception as exc:
            logger.warning(f"Planning LLM call failed (attempt {attempt}): {exc}")

    return _no_directions("generate_parallel_directions", n)


def generate_informed_directions(
    initial_direction: str,
    n: int,
    prompt_file: Path,
    history_summary: str,
    max_attempts: int = 5,
    use_llm: bool = True,
    seed_in_generation: bool = True,
) -> list[str]:
    """Generate NEW directions informed by the run's trial history.

    Called by the controller when the repository has stopped growing: the
    existing directions have saturated the signal space they can reach, and
    re-breeding within them (mutation/crossover) cannot open new space. The
    ``history_summary`` carries, per prior direction, which factors were
    admitted (with metrics) and which were rejected (binned by reason), so the
    LLM proposes directions ORTHOGONAL to the admitted signal space and away
    from exhausted directions.

    Mirrors ``generate_parallel_directions`` but defaults
    A reseed never injects canned ``base + ...``
    directions, because those are not informed by the history and would just
    re-saturate. On LLM failure the controller retries at the next stale
    window.
    """
    n = max(1, int(n))
    prompts = _load_prompts(prompt_file)
    sys_tpl = prompts.get("system", "")
    user_tpl = prompts.get("user", "")
    output_format = prompts.get("output_format", "")

    # Same Alpha158(20) orthogonal-signal context as round-0 planning (common-
    # mode). Defaults to True so the controller's reseed call (which does not
    # pass this kwarg) still gets the seeds.
    seed_context = _seed_context() if seed_in_generation else ""
    seed_context = "\n\n".join(x for x in (_market_context(), seed_context) if x)
    fmt = dict(initial_direction=initial_direction or "", n=n,
               history_summary=history_summary or "", seed_context=seed_context)
    system_prompt = sys_tpl.format(**fmt)
    user_prompt = user_tpl.format(**fmt)
    if output_format:
        if "{n}" in output_format:
            output_format = output_format.replace("{n}", str(n))
        user_prompt = f"{user_prompt}\n\n{output_format}"

    if not use_llm:
        return _no_directions("generate_informed_directions (use_llm=False)", n)

    for attempt in range(1, max_attempts + 1):
        try:
            resp = APIBackend().build_messages_and_create_chat_completion(
                user_prompt=user_prompt, system_prompt=system_prompt, json_mode=False
            )
            directions = _parse_directions(resp, n)
            if directions:
                return directions[:n]
            system_prompt += "\n\nStrictly output valid JSON. No extra text."
            logger.warning(f"Informed-planning parse failed (attempt {attempt}), retrying...")
        except Exception as exc:
            logger.warning(f"Informed-planning LLM call failed (attempt {attempt}): {exc}")

    return _no_directions("generate_informed_directions", n)


def load_run_config(config_path: Path) -> dict[str, Any]:
    if not config_path.exists():
        return {}
    try:
        return yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    except Exception as exc:
        logger.warning(f"Failed to load run config: {exc}")
        return {}

