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
    except Exception:
        return None
    arr = data.get("directions") if isinstance(data, dict) else None
    if not isinstance(arr, list):
        return None
    vals = [str(x).strip() for x in arr if isinstance(x, str) and x.strip()]
    return vals if len(vals) >= n else None


def _fallback_directions(initial_direction: str, n: int) -> list[str]:
    base = initial_direction.strip() if initial_direction else "market microstructure"
    patterns = [
        f"{base} + short-term momentum signal with volume confirmation",
        f"{base} + volatility regime switch using rolling variance",
        f"{base} + liquidity/turnover adjustment for noise reduction",
        f"{base} + cross-sectional rank with sector-neutralization",
        f"{base} + intraday reversal vs overnight drift decomposition",
        f"{base} + fundamental proxy alignment (price-to-book, earnings momentum)",
        f"{base} + calendar effects and seasonality-aware normalization",
        f"{base} + risk-adjusted return features (downside volatility focus)",
    ]
    out = []
    for i in range(n):
        out.append(patterns[i % len(patterns)])
    return out


def _seed_context() -> str:
    """The Alpha158(20) seed library, framed as orthogonal-signal context.

    Lists the 20 canonical signals (name + mining-DSL expression) with an
    instruction to propose NEW directions/factors ORTHOGONAL to the signal
    space they already cover -- NOT to copy them. The seeds never enter the
    zoo or ledger (``admitted:False`` in library.py); this only steers what the
    generator proposes, common-mode across both arms (user decision 2026-08-12:
    a generation change, not an eval-objective one, so the arms still differ
    only in the objective). Returns "" if the pool cannot be loaded, so a
    missing seed module never breaks planning.
    """
    try:
        from quantaalpha.factors.seed_pool import SEED_POOL
    except Exception:
        return ""
    if not SEED_POOL:
        return ""
    lines = [
        "Canonical Alpha158(20) reference signals -- do NOT copy these. Propose "
        "NEW directions/factors ORTHOGONAL to the signal space they already cover:"
    ]
    for name, expr in SEED_POOL.items():
        lines.append(f"  - {name}: {expr}")
    return "\n".join(lines)


def generate_parallel_directions(
    initial_direction: str,
    n: int,
    prompt_file: Path,
    max_attempts: int = 5,
    use_llm: bool = True,
    allow_fallback: bool = True,
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
    seed_context = _seed_context() if seed_in_generation else ""
    fmt = dict(initial_direction=initial_direction, n=n, seed_context=seed_context)
    system_prompt = sys_tpl.format(**fmt)
    user_prompt = user_tpl.format(**fmt)
    if output_format:
        if "{n}" in output_format:
            output_format = output_format.replace("{n}", str(n))
        user_prompt = f"{user_prompt}\n\n{output_format}"

    if not use_llm:
        return _fallback_directions(initial_direction, n) if allow_fallback else []

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

    return _fallback_directions(initial_direction, n) if allow_fallback else []


def generate_informed_directions(
    initial_direction: str,
    n: int,
    prompt_file: Path,
    history_summary: str,
    max_attempts: int = 5,
    use_llm: bool = True,
    allow_fallback: bool = False,
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
    ``allow_fallback=False`` -- a reseed must not inject canned ``base + ...``
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
    fmt = dict(initial_direction=initial_direction or "", n=n,
               history_summary=history_summary or "", seed_context=seed_context)
    system_prompt = sys_tpl.format(**fmt)
    user_prompt = user_tpl.format(**fmt)
    if output_format:
        if "{n}" in output_format:
            output_format = output_format.replace("{n}", str(n))
        user_prompt = f"{user_prompt}\n\n{output_format}"

    if not use_llm:
        return _fallback_directions(initial_direction, n) if allow_fallback else []

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

    return _fallback_directions(initial_direction, n) if allow_fallback else []


def load_run_config(config_path: Path) -> dict[str, Any]:
    if not config_path.exists():
        return {}
    try:
        return yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    except Exception as exc:
        logger.warning(f"Failed to load run config: {exc}")
        return {}

