"""The Alpha158(20) seed pool, translated into the mining DSL.

The paper states that *"the initial seed factor pool is derived from the public
Alpha158(20) subset"*. The code did not do that: ``factor_zoo_path`` was
``null``, so ``FactorRegulator.alphazoo`` was an empty frame, ``match_alphazoo``
had nothing to compare against, and the novelty/duplication check the regulator
is supposed to perform never ran. This supplies the pool.

**Why a translation and not the Qlib strings.** ``FactorLoader.ALPHA158_20_FACTORS``
is Qlib syntax -- ``Ref``, ``Mean``, ``Std``, ``Min``, ``Max``. Two consumers
need these expressions and both want the mining DSL:

* the regulator compares *subtrees* against candidate expressions, and mined
  candidates are written with ``DELAY``/``TS_MEAN``/``TS_STD``. ``Ref($close,1)``
  and ``DELAY($close,1)`` are the same factor and structurally distinct
  identifiers, so leaving the Qlib names in place would make every comparison
  miss and the novelty check would stay inert in a subtler way.
* seeded factors are *executed*, and ``function_lib`` has no ``Ref``.

The mapping is exact rather than approximate. Qlib's ``Mean``/``Std``/``Min``/
``Max`` over a window are rolling, so they become ``TS_MEAN``/``TS_STD``/
``TS_MIN``/``TS_MAX``; the DSL's bare ``MIN``/``MAX`` are *cross-sectional* and
would silently change the factor's meaning. ``Ref`` is ``DELAY``.
"""

from __future__ import annotations

import csv
from pathlib import Path

# Qlib operator -> mining DSL operator. Window-taking Qlib reducers are rolling.
TRANSLATION = {
    "Ref": "DELAY",
    "Mean": "TS_MEAN",
    "Std": "TS_STD",
    "Min": "TS_MIN",
    "Max": "TS_MAX",
}

# name -> expression, in the mining DSL. Derived from
# FactorLoader.ALPHA158_20_FACTORS; see translate_alpha158_20 for the check that
# keeps the two in step.
SEED_POOL: dict[str, str] = {
    "ROC0": "($close-$open)/$open",
    "ROC1": "$close/DELAY($close, 1)-1",
    "ROC5": "($close-DELAY($close, 5))/DELAY($close, 5)",
    "ROC10": "($close-DELAY($close, 10))/DELAY($close, 10)",
    "ROC20": "($close-DELAY($close, 20))/DELAY($close, 20)",
    "VRATIO5": "$volume/TS_MEAN($volume, 5)",
    "VRATIO10": "$volume/TS_MEAN($volume, 10)",
    "VSTD5_RATIO": "TS_STD($volume, 5)/TS_MEAN($volume, 5)",
    "RANGE": "($high-$low)/$open",
    "VOLATILITY5": "TS_STD($close, 5)/$close",
    "VOLATILITY10": "TS_STD($close, 10)/$close",
    "RET_VOL5": "TS_STD($close/DELAY($close, 1)-1, 5)",
    "RSV5": "($close-TS_MIN($low, 5))/(TS_MAX($high, 5)-TS_MIN($low, 5)+1e-12)",
    "RSV10": "($close-TS_MIN($low, 10))/(TS_MAX($high, 10)-TS_MIN($low, 10)+1e-12)",
    "HIGH_RATIO5": "$close/TS_MAX($high, 5)-1",
    "LOW_RATIO5": "$close/TS_MIN($low, 5)-1",
    "SHADOW_RATIO": "($high-$close)/($close-$low+1e-12)",
    "BODY_RATIO": "($close-$open)/($high-$low+1e-12)",
    "MA_RATIO5_10": "TS_MEAN($close, 5)/TS_MEAN($close, 10)-1",
    "MA_RATIO10_20": "TS_MEAN($close, 10)/TS_MEAN($close, 20)-1",
}


def translate_alpha158_20() -> dict[str, str]:
    """Translate the canonical Qlib definitions, for checking ``SEED_POOL``.

    Kept as a function rather than used to build ``SEED_POOL`` at import time so
    that the shipped pool is a fixed, reviewable artefact -- a protocol input
    should not change because a token-substitution rule was edited. The test is
    that the two agree.
    """
    import re

    from quantaalpha.backtest.factor_loader import FactorLoader

    out = {}
    for name, expr in FactorLoader.ALPHA158_20_FACTORS.items():
        translated = expr
        for qlib_op, dsl_op in TRANSLATION.items():
            # Word boundary on the left and "(" on the right: without it, "Ref"
            # would also rewrite the "Ref" inside a longer identifier.
            translated = re.sub(rf"\b{qlib_op}\(", f"{dsl_op}(", translated)
        out[name] = translated
    return out


def write_alphazoo_csv(path: str | Path) -> Path:
    """Write the pool as the two-column CSV ``FactorRegulator`` reads.

    ``match_alphazoo`` unpacks each row as ``(name, expression)``, so the file
    must have exactly two columns in that order.
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["name", "expression"])
        for name, expr in SEED_POOL.items():
            w.writerow([name, expr])
    return p


def seed_context(header: str | None = None) -> str:
    """The Alpha158(20) pool rendered as prompt context.

    Lives here, beside ``SEED_POOL``, because more than one prompt path needs it
    and a second copy would drift. ``planning._seed_context`` delegates to this;
    the hypothesis/factor-generation path in ``proposal.py`` uses it too.

    That second consumer is the point. The paper's seed pool exists to tell the
    generator which signal space is ALREADY covered, and until 2026-08-16 it
    reached the LLM exactly once per run -- in the direction-planning system
    prompt. Hypothesis generation and factor construction, the ~50 batches where
    expressions are actually written, never saw it. Steering the topic without
    steering the expression is consistent with the ~40% of batches that came
    back as duplicates of the book.

    Returns "" when the pool cannot be loaded, so a missing/empty pool degrades
    to no context rather than breaking generation.
    """
    if not SEED_POOL:
        return ""
    head = header or (
        "Canonical Alpha158(20) reference signals. This is the PUBLIC signal "
        "space -- it is already covered, and a factor that re-expresses one of "
        "these adds nothing a measurement can credit:"
    )
    return "\n".join([head] + [f"  - {n}: {e}" for n, e in SEED_POOL.items()])


__all__ = ["SEED_POOL", "TRANSLATION", "translate_alpha158_20", "write_alphazoo_csv",
           "seed_context"]
