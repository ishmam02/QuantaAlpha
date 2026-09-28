# 06 — Global append-only factor and trajectory stores

**Depends on:** 01. **Blocks:** 07, 08.

## Why

Two things currently die at the end of a run:

* **Factor libraries** are written per run as `data/factorlib/all_factors_library_<id>.json`.
  There is no single store spanning mines.
* **Trajectories** are held in a `TrajectoryPool` whose `fresh_start` defaults to
  `True` — so hypotheses, lineage, verdicts and refine history are discarded
  exactly when a later mine would want them.

The user's requirement is that **everything survives** — admitted and rejected
alike, UUID retained, never deleted — because rejected factors are the record of
what was tried, and a factor that failed in one regime is a candidate in the next.

## Goal

One append-only store for factors and one for trajectories, both keyed by stable
id, both spanning every mine, neither ever deleting.

## Approach

**Factor store** — `data/factorlib/global/factors.jsonl` (append-only) plus an
index. `library.py` already mints `factor_id = md5(...)` and keys
`self.data["factors"][factor_id]`, so the id exists; this promotes it to a global
namespace. Per factor: `factor_id`, expression, hypothesis, mechanism, declared
`expected_ic_sign`, originating mine id, phase, lineage, admission verdict and
metrics, and its decay history. **Rejected factors are first-class rows**, with
the rejection gate and number.

A migration reads the existing per-run libraries into the global store, recording
which file each row came from. Idempotent: re-running does not duplicate.

**Trajectory store** — `data/results/trajectories/global.json`, with
`TrajectoryPool(save_path=…, fresh_start=False)`. The dataclass already round-trips
through `to_dict`/`from_dict` and already carries `parent_ids`, `refine_actions`
and `expected_ic_sign`, so no schema change is needed — only persistence and a
`mine_id` field.

**Collision safety.** `factor_id` is a hash of the expression, so the same
expression mined twice is one row with two provenance entries — not two rows. Make
that explicit, with a test; silent duplicate ids would corrupt the archive in `08`.

## Validation

* Migration of the existing libraries preserves total factor count and every
  admitted flag; verified by a count reconciliation per source file.
* Re-running the migration changes nothing (byte-identical store).
* A pool loaded with `fresh_start=False` recovers lineage: `get_ancestors` resolves
  across two different mines.
* Rejected factors are present and queryable by rejection gate.

## Risks

* `theta.hash` is `sha256(asdict(Protocol))[:16]` and **cannot be reproduced across
  code versions** — a new dataclass field changes it for an unchanged YAML. Store
  the protocol's *content* alongside each row; never try to identify an old
  artifact by re-hashing its YAML.
* The store grows forever. That is intended, but keep rows small — no signal
  payloads, no DataFrames, ids and scalars only.
