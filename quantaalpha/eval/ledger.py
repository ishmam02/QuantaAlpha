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
        if not self.path.exists():
            return []
        with self.path.open(encoding="utf-8") as handle:
            return [json.loads(line) for line in handle if line.strip()]

    def __len__(self) -> int:
        return len(self.read())


__all__ = ["DEFAULT_LEDGER_PATH", "Ledger"]
