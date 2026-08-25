"""The admission verdict, shared across the eval and evolution layers.

Lives in ``quantaalpha.core`` so ``eval.admission`` can emit a **structured**
verdict alongside its prose ``reason`` without importing from
``pipeline.evolution`` (layering: eval must not depend on pipeline), and
``pipeline.evolution.diagnosis`` can read it back without re-deriving it by
substring-matching the prose.

Before this module existed, ``diagnosis.classify_verdict`` recovered the verdict
by substring-matching ``admission.decide``'s ``reason``/``pathology`` strings, so
a prose rewording silently degraded every reject to ``MARGINAL``. Now
``admission.decide`` sets ``Decision.verdict`` at each branch (the branch already
knows its own verdict), ``as_record`` serializes it to ``.value``, and it threads
through ``_to_series`` -> ``backtest_metrics``. ``classify_verdict`` reads that
field first and falls back to the prose only when it is absent -- and logs a
warning when it does, so the fallback can no longer be silent.
"""
from __future__ import annotations

from enum import Enum


class Verdict(str, Enum):
    """Why the parent's terminal reward fell where it did.

    Set structurally by ``admission.decide`` (each return branch carries its own
    verdict) and threaded through ``Decision.as_record`` -> ``_to_series`` ->
    ``backtest_metrics``. ``diagnosis.classify_verdict`` reads it directly; the
    legacy prose-substring path remains only as a fallback for records written
    before this field existed.

    Each value maps to a distinct generator instruction: a *resolvably negative*
    contribution is not an unresolved measurement, and the two say opposite
    things about what to do next -- exactly the distinction ``admission.decide``
    was written to preserve in its prose, now preserved structurally.
    """

    NET_HARMFUL = "net_harmful"    # resolvably NEGATIVE: book gets worse (mean<0, |t| large)
    MARGINAL = "marginal"          # not resolved: mean <= k*se (t too small to trust)
    REDUNDANT = "redundant"        # rho_max > rho_bar -- a duplicate of the zoo
    TOO_SPARSE = "too_sparse"      # coverage < min_coverage -- cannot be priced
    CONSTANT = "constant"          # zero-variance signal
    ADMITTED = "admitted"          # passed admission (a winner)
    REPLACED = "replaced"          # admitted by displacing an incumbent
    BOOTSTRAP = "bootstrap"        # admitted during min_size bootstrap (no real verdict)
    NO_MECHANISM = "no_mechanism"  # no stated economic story, or an unfalsifiable one
    NO_DATA = "no_data"            # no usable measurement / gating disabled
    FULL = "full"                  # repository full, did not beat weakest incumbent


__all__ = ["Verdict"]