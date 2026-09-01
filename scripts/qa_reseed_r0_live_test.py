#!/usr/bin/env python
"""Live LLM test: does the Layer-1 fix change the FACTORS the reseed produces?

The unit tests pin the plumbing (digest carries failure memory + sign lesson
when the flag is on; the operator-novelty gate keeps/rejects). This answers the
real question the user asked -- "test how well the reseed is working WITH THE
LLM CALL ... don't just test the dirs but the FACTORS generated" -- by running
the reseed direction planner and the per-batch factor generator against the
REAL glm-5.2:cloud, with the r0 (round-0, blind) trajectories from the last
production mine as the ONLY context.

Two conditions, same r0 context:
  OFF  -- old digest (no failure memory, no sign lesson, no gate) = the path the
         production mine (pid 42967) is running right now.
  ON   -- QA_RESEED_FAILURE_MEMORY=1 + QA_RESEED_OP_NOVELTY_GATE=1 + top-K=3
         = the Layer-1 fix.

For each condition: build the reseed digest, call the direction planner (LLM),
apply the gate (ON only), then for the first N kept directions generate FACTORS
via the same AlphaAgentLoop hypothesis+constructor the mine uses (LLM), and
report the operator families and the sign framing of the GENERATED FACTORS.

The monoculture this targets is measured on the r0 context: TS_MEAN dominates
the admitted zoo, and the realized neutralized IC is predominantly negative.
So a working fix should push the ON-condition factors OFF TS_MEAN (toward the
unused operator families the coverage block names) and should let the sign
lesson register. The OFF condition is the control.

Reads the production pool read-only; writes nothing to production paths
(throwaway EXPERIMENT_ID + /tmp pool + /tmp workspace; factor generation calls
only factor_propose + factor_construct, never run(), so no library/ledger
writes). Run with the mine stopped so the box is free:

    python scripts/qa_reseed_r0_live_test.py [--factors N]   (default N=3)
"""
import argparse
import json
import os
import sys
import threading
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
os.chdir(REPO)

# Load .env (LLM keys + model) the way cli.py does, so this runs standalone.
from dotenv import load_dotenv
load_dotenv(REPO / ".env")

# Throwaway experiment env so NOTHING touches production paths. The factor
# generation below calls only factor_propose + factor_construct (no run(), no
# library/ledger save), but the scenario/loop construction may read these.
os.environ.setdefault("EXPERIMENT_ID", "qa_reseed_test")
os.environ.setdefault("WORKSPACE_PATH", "/tmp/qa_reseed_test_ws")
os.environ.setdefault("PICKLE_CACHE_FOLDER_PATH_STR", "/tmp/qa_reseed_test_pickle")
os.environ.setdefault("QA_LEDGER", "/tmp/qa_reseed_test_ledger.jsonl")
for _d in ("/tmp/qa_reseed_test_ws", "/tmp/qa_reseed_test_pickle"):
    Path(_d).mkdir(parents=True, exist_ok=True)

_PROMPTS = REPO / "quantaalpha" / "pipeline" / "prompts"
PROD_POOL = REPO / "data" / "results" / "trajectory_pool_meanvar_20260825_123942.json"

from quantaalpha.pipeline.evolution.controller import (
    EvolutionConfig, EvolutionController, RoundPhase,
)
from quantaalpha.pipeline.evolution.trajectory import StrategyTrajectory
from quantaalpha.pipeline.loop import AlphaAgentLoop
from quantaalpha.pipeline.settings import ALPHA_AGENT_FACTOR_PROP_SETTING as PS
from quantaalpha.factors.operator_coverage import (
    extract_operators, top_exercised_operators, coverage_block,
)


# --------------------------------------------------------------------------- #
# FULL-context: load the ENTIRE production pool (all rounds, all phases) as the
# reseed's history -- what the production reseed actually sees when it fires
# after many stale rounds, not just round 0. This is the fix for directive 3:
# "show all the rounds before reseed not just r0".
# --------------------------------------------------------------------------- #
_PHASE = {
    "original": RoundPhase.ORIGINAL,
    "mutation": RoundPhase.MUTATION,
    "crossover": RoundPhase.CROSSOVER,
}


def load_full():
    d = json.load(open(PROD_POOL))
    ts = d["trajectories"]
    items = list(ts.values()) if isinstance(ts, dict) else ts
    # Rebuild StrategyTrajectory objects for EVERY trajectory across every round
    # and phase (original + mutation + crossover), not just round 0. The
    # substantive reseed blocks (what-worked-well, failure modes, sign, coverage)
    # all scan get_all() over every round+phase, so this is what makes the digest
    # carry the real multi-round history.
    trajs = []
    for t in items:
        rnd = t.get("round_idx", 0)
        ph = _PHASE.get(t.get("phase", "original"), RoundPhase.ORIGINAL)
        tid = t.get("trajectory_id") or StrategyTrajectory.generate_id(
            t["direction_id"], rnd, ph)
        trajs.append(StrategyTrajectory(
            trajectory_id=tid, direction_id=t["direction_id"], round_idx=rnd,
            phase=ph, factors=t.get("factors") or [],
            backtest_metrics=dict(t.get("backtest_metrics") or {}),
            hypothesis=t.get("hypothesis", ""),
        ))
    # Direction labels = the LATEST round's original hypotheses. In production a
    # reseed builds the digest with the CURRENT (exhausted) directions and the
    # LLM proposes new ones, so the latest originals are the faithful label set.
    # direction_id is reused across rounds, so all rounds' originals map to those
    # slots and the digest tallies each direction's accumulated outcomes.
    originals = [t for t in trajs if t.phase == RoundPhase.ORIGINAL]
    if not originals:
        return trajs, ["explore cross-sectional equity factors"]
    latest = max(t.round_idx for t in originals)
    by_dir = {}
    for t in originals:
        if t.round_idx == latest:
            by_dir.setdefault(t.direction_id, t.hypothesis)
    directions = [by_dir[i] for i in sorted(by_dir)]
    return trajs, directions


def build_controller(trajs, directions):
    cfg = EvolutionConfig(
        num_directions=len(directions), max_rounds=20,
        mutation_enabled=True, crossover_enabled=True, fresh_start=True,
        reseed_after_stale_rounds=2, directions=directions,
        initial_direction="explore cross-sectional equity factors",
        informed_prompt_path=str(_PROMPTS / "informed_planning_prompts.yaml"),
        mutation_prompt_path=str(_PROMPTS / "evolution_prompts.yaml"),
        crossover_prompt_path=str(_PROMPTS / "evolution_prompts.yaml"),
        pool_save_path="/tmp/qa_reseed_test_pool.json",
    )
    ctl = EvolutionController(cfg)
    for t in trajs:
        ctl.pool.add(t)
    # Real attempt count per direction = # of original-phase trajectories for
    # that direction across ALL rounds; mark saturated once it clears the stale
    # threshold so the digest renders the reseed as a justified, exhausted one.
    from collections import Counter
    orig_counts = Counter(
        t.direction_id for t in trajs if t.phase == RoundPhase.ORIGINAL)
    for did, st in enumerate(ctl._direction_status):
        st["attempts"] = orig_counts.get(did, 0)
        st["saturated"] = st["attempts"] >= cfg.reseed_after_stale_rounds
    ctl._current_round = max((t.round_idx for t in trajs), default=0)
    ctl._current_phase = RoundPhase.ORIGINAL
    return ctl


# --------------------------------------------------------------------------- #
# Operator / sign helpers.
# --------------------------------------------------------------------------- #
def expr_ops(exprs):
    ops = []
    for e in exprs:
        ops.extend(sorted(extract_operators(e)))
    return ops


def sign_of(text):
    """Heuristic sign frame from text: HIGHER/positive vs LOWER/negative."""
    t = (text or "").lower()
    hi = t.count("higher") + t.count("positive") + t.count("momentum") + t.count("increases")
    lo = t.count("lower") + t.count("negative") + t.count("reversion") + t.count("decreases")
    if hi > lo:
        return "HIGHER"
    if lo > hi:
        return "LOWER"
    return "?"


def factor_expressions(factor):
    out = []
    for t in getattr(factor, "sub_tasks", None) or []:
        e = getattr(t, "factor_expression", None) or getattr(t, "expression", None)
        if e:
            out.append(str(e))
        for s in getattr(t, "sub_tasks", None) or []:
            e2 = getattr(s, "factor_expression", None) or getattr(s, "expression", None)
            if e2:
                out.append(str(e2))
    return out


# --------------------------------------------------------------------------- #
# Factor generation via the same AlphaAgentLoop the mine uses.
# --------------------------------------------------------------------------- #
def make_factor_loop(strategy_suffix):
    """Construct ONE AlphaAgentLoop (loads the scenario/trace/constructor once)."""
    return AlphaAgentLoop(
        PS, potential_direction="__seed__", stop_event=threading.Event(),
        use_local=True, strategy_suffix=strategy_suffix, evolution_phase="original",
    )


def gen_factors_for_direction(loop, direction, strategy_suffix):
    """Generate factors for one direction. Returns (hypothesis_text, sign, exprs)."""
    scen = getattr(loop.hypothesis_generator, "scen", None)
    HypGen = type(loop.hypothesis_generator)
    eff = (direction or "") + "\n" + strategy_suffix
    loop.hypothesis_generator = HypGen(scen, eff)
    idea = loop.factor_propose({})
    factor = loop.factor_construct({"factor_propose": idea})
    hyp = getattr(idea, "hypothesis", "") or ""
    mech = getattr(idea, "mechanism", "") or ""
    exp_sign = (getattr(idea, "expected_ic_sign", "") or "").strip().lower()
    exprs = factor_expressions(factor)
    sign = sign_of(mech) if sign_of(mech) != "?" else sign_of(hyp)
    if exp_sign in ("positive", "negative"):
        sign = "HIGHER" if exp_sign == "positive" else "LOWER"
    return hyp, mech, sign, exprs


# --------------------------------------------------------------------------- #
# Main.
# --------------------------------------------------------------------------- #
def run_condition(ctl, label, flags_on, strategy_suffix, n_factors):
    print(f"\n{'='*78}\nCONDITION: {label}  (flags {'ON' if flags_on else 'OFF'})\n{'='*78}")
    if flags_on:
        os.environ["QA_RESEED_FAILURE_MEMORY"] = "1"
        os.environ["QA_RESEED_OP_NOVELTY_GATE"] = "1"
        os.environ["QA_RESEED_OP_TOPK"] = "3"
    else:
        for k in ("QA_RESEED_FAILURE_MEMORY", "QA_RESEED_OP_NOVELTY_GATE"):
            os.environ.pop(k, None)

    digest = ctl._build_reseed_digest()
    has_cov = "Operator coverage across admitted factors" in digest
    has_sign = "Direction, measured across this run" in digest
    has_fm = "What the measurements have shown so far" in digest
    print(f"digest markers: coverage={'Y' if has_cov else 'N'}  "
          f"sign-lesson={'Y' if has_sign else 'N'}  failure-modes={'Y' if has_fm else 'N'}")
    print(f"digest length: {len(digest)} chars")
    if has_sign or has_fm:
        # Show the failure-memory + sign portion only (the new Layer-1 content).
        cut = digest
        if "## Direction, measured across this run" in cut:
            cut = cut[cut.index("## Direction, measured across this run"):]
        print("--- NEW digest content (sign lesson + failure modes) ---")
        print(cut[:1400])

    # Direction planner (real LLM).
    try:
        dirs = ctl._generate_informed_directions(digest)
    except Exception as e:
        print(f"!! direction planner failed: {e!r}")
        return {"label": label, "dirs": [], "factors": []}
    print(f"\ndirection planner returned {len(dirs)} direction(s):")
    for i, d in enumerate(dirs):
        print(f"  [{i}] ops={sorted(set(_hyp_ops(d)))} sign={sign_of(d)} :: {d[:110]}")

    # Gate (ON only).
    pre = len(dirs)
    dirs = ctl._filter_novel_directions(dirs, digest)
    if flags_on:
        print(f"gate: kept {len(dirs)}/{pre} (rejected {pre - len(dirs)})")

    # Factor generation for the first N kept directions.
    factors = []
    if dirs:
        print(f"\ngenerating FACTORS for first {min(n_factors, len(dirs))} direction(s) ...")
        loop = make_factor_loop(strategy_suffix)
        for d in dirs[:n_factors]:
            try:
                hyp, mech, sign, exprs = gen_factors_for_direction(loop, d, strategy_suffix)
            except Exception as e:
                print(f"  !! factor gen failed for a direction: {e!r}")
                factors.append({"dir": d, "exprs": [], "ops": [], "sign": "?", "err": repr(e)})
                continue
            ops = expr_ops(exprs)
            print(f"\n  DIR: {d[:100]}")
            print(f"    hypothesis: {hyp[:140]}")
            print(f"    sign frame : {sign}")
            print(f"    {len(exprs)} factor(s):")
            for e in exprs:
                print(f"      - {e}")
            print(f"    operators : {sorted(set(ops))}")
            factors.append({"dir": d, "exprs": exprs, "ops": sorted(set(ops)),
                            "sign": sign, "err": None})
    return {"label": label, "dirs": dirs, "factors": factors}


def _hyp_ops(text):
    """Operators named in a direction/hypothesis text (word-boundary)."""
    from quantaalpha.factors.operator_coverage import DECLARED_OPERATORS
    import re as _re
    return {op for op in DECLARED_OPERATORS if _re.search(r"\b" + _re.escape(op) + r"\b", text or "")}


def summarize(results):
    print(f"\n{'='*78}\nSUMMARY: does the fix move the reseed off the monoculture?\n{'='*78}")
    # Admitted-zoo monoculture baseline over the FULL pool (all rounds).
    trajs, _ = load_full()
    adm = [t for t in trajs if str((t.backtest_metrics or {}).get("verdict", "")).lower()
           in ("admitted", "replaced", "bootstrap")]
    adm_exprs = []
    for t in adm:
        for f in (t.factors or []):
            e = f.get("expression", "") if isinstance(f, dict) else ""
            if e:
                adm_exprs.append(e)
    top3 = top_exercised_operators(adm_exprs, 3)
    print(f"admitted zoo (full pool): {len(adm)} factors; top-3 operators = {top3}")
    for r in results:
        all_ops = []
        signs = []
        for f in r["factors"]:
            all_ops.extend(f["ops"])
            signs.append(f["sign"])
        from collections import Counter
        oc = Counter(all_ops)
        sc = Counter(signs)
        print(f"\n{r['label']}: {len(r['factors'])} direction(s) factor-generated")
        print(f"  operator tally: {dict(oc.most_common())}")
        print(f"  sign tally    : {dict(sc)}")
        off_top3 = [o for o in oc if o not in set(top3)]
        print(f"  operators OUTSIDE the r0 top-3: {off_top3 or '(none -- still monoculture)'}")


def dump_factors(results, path):
    """Flatten generated factors to [{cond,name,expr,dir,sign}] for the scorer.

    A direction can yield several expressions (sub_tasks); the scorer scores each
    expr SOLO, so flatten to one row per expression. ``cond`` is derived from the
    label's leading token (OFF/ON) so the scorer can group them.
    """
    import json as _json
    out = []
    for r in results:
        cond = r["label"].split()[0].upper()
        i = 0
        for f in r["factors"]:
            for e in (f.get("exprs") or []):
                out.append({"cond": cond, "name": f"{cond}-{i+1}",
                            "expr": e, "dir": (f.get("dir") or "")[:160],
                            "sign": f.get("sign", "?")})
                i += 1
    Path(path).write_text(_json.dumps(out, indent=2))
    print(f"\ndumped {len(out)} factor expressions ({sum(1 for x in out if x['cond']=='OFF')} OFF, "
          f"{sum(1 for x in out if x['cond']=='ON')} ON) -> {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--factors", type=int, default=3,
                    help="directions per condition to factor-generate (default 3)")
    ap.add_argument("--dump", default=None,
                    help="dump generated factor expressions to this JSON for the scorer")
    args = ap.parse_args()

    if not os.environ.get("OPENAI_API_KEY") and not os.environ.get("CHAT_API_KEY"):
        print("WARNING: no OPENAI_API_KEY / CHAT_API_KEY in .env; LLM calls will fail.")

    trajs, directions = load_full()
    rounds = sorted({t.round_idx for t in trajs})
    phases = sorted({t.phase.value for t in trajs})
    adm_n = sum(1 for t in trajs if str((t.backtest_metrics or {}).get("verdict", "")).lower()
                in ("admitted", "replaced", "bootstrap"))
    print(f"FULL context: {len(trajs)} trajectories across rounds {rounds[0]}..{rounds[-1]} "
          f"({len(rounds)} rounds), phases={phases}; {len(directions)} directions; {adm_n} admitted")
    ctl = build_controller(trajs, directions)
    strategy_suffix = ctl._build_zoo_context()
    print(f"strategy_suffix: {len(strategy_suffix)} chars (the per-batch r0 zoo context)")

    saved = {k: os.environ.get(k) for k in
             ("QA_RESEED_FAILURE_MEMORY", "QA_RESEED_OP_NOVELTY_GATE", "QA_RESEED_OP_TOPK")}
    try:
        off = run_condition(ctl, "OFF (old digest, no gate)", False, strategy_suffix, args.factors)
        on = run_condition(ctl, "ON (failure memory + sign lesson + gate)", True, strategy_suffix, args.factors)
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v

    summarize([off, on])
    if args.dump:
        dump_factors([off, on], args.dump)
    print("\nDone.")


if __name__ == "__main__":
    main()