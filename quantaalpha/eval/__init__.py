"""``E_Θ`` — the frozen, deterministic evaluation engine.

Implements the net-of-cost, capacity-aware objective of
``problem_formulation.tex`` §3 as a pure function of ``(f, zoo, Θ)``.

The package is deliberately self-contained: it depends on Qlib for data and on
``factors.coder.factor_ast`` for the complexity statistic, but on nothing else
in QuantaAlpha. That is what makes Properties 1-3 checkable by hand.

Submodules are imported lazily so that ``import quantaalpha.eval`` stays cheap
and does not drag in Qlib or LightGBM until an evaluation actually runs.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from quantaalpha.eval.protocol import (
    DEFAULT_PROTOCOL_PATH,
    Protocol,
    default_protocol_path,
    load_protocol,
)

if TYPE_CHECKING:  # pragma: no cover
    from quantaalpha.eval.ledger import Ledger
    from quantaalpha.eval.operator import EvaluationOperator

_LAZY = {
    "EvaluationOperator": ("quantaalpha.eval.operator", "EvaluationOperator"),
    "Ledger": ("quantaalpha.eval.ledger", "Ledger"),
}


def __getattr__(name: str) -> Any:
    if name in _LAZY:
        module_path, attr = _LAZY[name]
        from importlib import import_module

        return getattr(import_module(module_path), attr)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "DEFAULT_PROTOCOL_PATH",
    "EvaluationOperator",
    "Ledger",
    "Protocol",
    "default_protocol_path",
    "load_protocol",
]
