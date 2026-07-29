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
    for record in Ledger(path).read():
        for expr in record.get("evicted_exprs") or []:
            repo.pop(expr, None)
        metrics = record.get("metrics") or {}
        for expr in record.get("factor_exprs") or []:
            if expr:
                repo[expr] = metrics
    return repo


__all__ = ["DEFAULT_LEDGER_PATH", "Ledger", "replay_repository"]
