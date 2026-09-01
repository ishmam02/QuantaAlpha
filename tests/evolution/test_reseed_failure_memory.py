"""Failure-memory layering for the reseed digest + direction-novelty gate.

The reseed digest is the DIRECTION PLANNER's input -- the layer that picks the
operator family and the sign frame. The failure-mode counts and the measured
realized-IC sign previously reached only the per-batch generator
(``_build_zoo_context``), AFTER the direction had locked in the family and the
sign, so the planner kept choosing the monoculture operator and the
opposite-framed hypotheses (measured: TS_MEAN 59% of the zoo, 23 of 56
operators never exercised; reseeds stayed TS_MEAN-heavy and HIGHER-framed).

These pin the two fixes, both env-gated default-off so the frozen protocol
path is byte-identical:

  F1  sign-direction block measures the realized-IC sign (tearsheet-sourced,
      min 8 sample, else [])
  F2  failure-mode counts + sign block reach the reseed digest when
      QA_RESEED_FAILURE_MEMORY is on; ABSENT when off (byte-identical)
  F3  operator-novelty gate keeps a direction naming an unused operator and
      rejects one naming only the top-K exercised; inert when the flag is off
  F4  gate no-ops when there is no admitted zoo to measure a monoculture from
  F5  the digest + the gate carry no prescription / no market prior (the
      standing hard rules)
"""
import os
from pathlib import Path
from unittest import mock

from quantaalpha.pipeline.evolution.controller import (
    EvolutionConfig, EvolutionController, RoundPhase,
)
from quantaalpha.pipeline.evolution.trajectory import StrategyTrajectory

_PROMPTS = Path(__file__).resolve().parents[2] / "quantaalpha" / "pipeline" / "prompts"

# Env flags under test. Saved/restored so a test failure cannot leak a flag.
_FLAGS = ("QA_RESEED_FAILURE_MEMORY", "QA_RESEED_OP_NOVELTY_GATE", "QA_RESEED_OP_TOPK")


def _cfg(**over) -> EvolutionConfig:
    base = dict(
        num_directions=3, max_rounds=20,
        mutation_enabled=True, crossover_enabled=True, fresh_start=True,
        reseed_after_stale_rounds=2,
        directions=["d0", "d1", "d2"],
        initial_direction="explore momentum and reversal",
        informed_prompt_path=str(_PROMPTS / "informed_planning_prompts.yaml"),
        mutation_prompt_path=str(_PROMPTS / "evolution_prompts.yaml"),
        crossover_prompt_path=str(_PROMPTS / "evolution_prompts.yaml"),
    )
    base.update(over)
    return EvolutionConfig(**base)


def _traj(did, rnd, metrics, factors=None) -> StrategyTrajectory:
    return StrategyTrajectory(
        trajectory_id=StrategyTrajectory.generate_id(did, rnd, RoundPhase.ORIGINAL),
        direction_id=did, round_idx=rnd, phase=RoundPhase.ORIGINAL,
        factors=factors or [], backtest_metrics=dict(metrics),
    )


def _ctl(trajs, directions=None):
    ctl = EvolutionController(_cfg(directions=directions or ["d0", "d1", "d2"]))
    for t in trajs:
        ctl.pool.add(t)
    return ctl


def _tsheet(rank_ic_neutral=None, rank_ic=None, **extra):
    """A factor tearsheet. ``_failure_patterns`` trips a bucket only when a
    sheet carries the field that bucket measures (``t_nw`` for no_signal,
    ``rho_max`` for duplicate, ``ic_pos_frac`` for unstable, ``exposure_size``
    + ``rank_ic`` for size_exposure) -- so the F2 mock must populate them."""
    s = {}
    if rank_ic_neutral is not None:
        s["rank_ic_neutral"] = rank_ic_neutral
    if rank_ic is not None:
        s["rank_ic"] = rank_ic
    s.update(extra)
    return s


class _Env:
    """Save/clear the flags for the duration of a block, restore on exit."""
    def __enter__(self):
        self._saved = {f: os.environ.pop(f, None) for f in _FLAGS}
        return self

    def __exit__(self, *exc):
        for f, v in self._saved.items():
            if v is None:
                os.environ.pop(f, None)
            else:
                os.environ[f] = v


def test_f1_sign_direction_block_measures():
    """F1: the block states the negative share from tearsheets, min 8 sample."""
    with _Env():
        neg = [_traj(i, 0, {"factor_tearsheets": {"f": _tsheet(-0.02)}}) for i in range(9)]
        pos = [_traj(i, 0, {"factor_tearsheets": {"f": _tsheet(0.03)}}) for i in range(2)]
        ctl = _ctl(neg + pos)
        blk = ctl._sign_direction_block(ctl.pool.get_all())
        assert blk and "Direction, measured across this run" in blk[0]
        assert "9 of 11" in blk[1] and "NEGATIVE" in blk[1] and "(82%)" in blk[1]
        # min-sample guard: 7 factors is below 8 -> nothing asserted
        few = [_traj(i, 0, {"factor_tearsheets": {"f": _tsheet(-0.02)}}) for i in range(7)]
        assert _ctl(few)._sign_direction_block(_ctl(few).pool.get_all()) == []
        # rank_ic fallback when rank_ic_neutral absent
        only_raw = [_traj(i, 0, {"factor_tearsheets": {"f": _tsheet(rank_ic=-0.05)}}) for i in range(9)]
        blk2 = _ctl(only_raw)._sign_direction_block(_ctl(only_raw).pool.get_all())
        assert blk2 and "9 of 9" in blk2[1]
    print("OK  F1: sign-direction block measures realized-IC sign, min 8 sample")


def test_f2_digest_carries_failure_memory_when_on_absent_when_off():
    """F2: flag on -> digest has the failure modes + sign block; off -> neither."""
    # Tearsheets carry t_nw<3 (no_signal), rho_max>=0.60 (duplicate), and
    # ic_pos_frac<0.52 (unstable) so _failure_patterns has buckets to count.
    sheets = [_traj(i, 0, {
        "verdict": "admitted" if i % 2 == 0 else "replaced",
        "delta_mean": 0.1, "U": 0.5, "rho_max": 0.2,
        "factor_tearsheets": {"f": _tsheet(
            rank_ic_neutral=-0.03 + 0.001 * i, rank_ic=-0.10,
            t_nw=2.0, rho_max=0.70, ic_pos_frac=0.50,
            exposure_size=0.5, monotonicity=0.20)},
    }, factors=[{"expression": "TS_MEAN($close,20)"}]) for i in range(12)]
    with _Env():
        # OFF: the new blocks must be ABSENT (byte-identical to before).
        ctl = _ctl(sheets)
        off = ctl._build_reseed_digest()
        assert "What the measurements have shown so far" not in off, "F2 off: failure modes leaked in"
        assert "Direction, measured across this run" not in off, "F2 off: sign block leaked in"
        # ON: both blocks present.
        os.environ["QA_RESEED_FAILURE_MEMORY"] = "1"
        on = ctl._build_reseed_digest()
        assert "What the measurements have shown so far" in on, "F2 on: failure modes missing"
        assert "Direction, measured across this run" in on, "F2 on: sign block missing"
        assert "NEGATIVE" in on  # 12 negatives -> the measured direction
    print("OK  F2: failure memory in digest when flag on, absent when off")


def test_f3_gate_keeps_novel_rejects_monoculture_inert_when_off():
    """F3: gate keeps an unused-operator direction, rejects a top-K-only one."""
    # Admitted zoo is TS_MEAN-only, so top-3 exercised = {TS_MEAN}.
    adm = [_traj(i, 0, {"verdict": "admitted", "U": 0.5, "delta_mean": 0.1,
                       "factor_tearsheets": {"f": _tsheet(-0.02)}},
                 factors=[{"expression": "TS_MEAN($close,20)"}])
           for i in range(6)]
    with _Env():
        ctl = _ctl(adm)
        novel_dir = "A high 14-day RSI predicts LOWER forward returns."
        mono_dir = "A high 20-day average (TS_MEAN) of volume predicts HIGHER."
        # OFF: inert, both kept.
        kept_off = ctl._filter_novel_directions([novel_dir, mono_dir], "digest")
        assert kept_off == [novel_dir, mono_dir], "F3 off: gate is not inert"
        # ON: RSI kept (not in top-3), TS_MEAN-only rejected.
        os.environ["QA_RESEED_OP_NOVELTY_GATE"] = "1"
        os.environ["QA_RESEED_OP_TOPK"] = "3"
        kept_on = ctl._filter_novel_directions([novel_dir, mono_dir], "digest")
        assert kept_on == [novel_dir], f"F3 on: expected [novel], got {kept_on}"
    print("OK  F3: gate keeps novel, rejects monoculture; inert when off")


def test_f4_gate_noops_without_admitted_zoo():
    """F4: with no admitted factors there is no monoculture to measure -> no-op."""
    # All rejected: nothing admitted, so _admitted_expressions() is empty.
    rej = [_traj(i, 0, {"verdict": "marginal", "delta_mean": -0.01,
                       "factor_tearsheets": {"f": _tsheet(-0.02)}},
                 factors=[{"expression": "TS_MEAN($close,20)"}])
           for i in range(6)]
    with _Env():
        os.environ["QA_RESEED_OP_NOVELTY_GATE"] = "1"
        ctl = _ctl(rej)
        dirs = ["A high TS_MEAN predicts HIGHER.", "A high RSI predicts LOWER."]
        assert ctl._filter_novel_directions(dirs, "d") == dirs, \
            "F4: gate must not fire with no admitted zoo to measure"
    print("OK  F4: gate no-ops when no admitted zoo (early run)")


def test_f5_no_prescription_no_market_prior():
    """F5: the new digest text + gate retry carry no remedy token / market name."""
    BANNED = ["you should", "try ", "instead use", "lengthen", "shorten",
              "prefer ", "we recommend", "make sure to", "always use",
              "CSI300", "csi 300", "reversion", "momentum prior"]
    adm = [_traj(i, 0, {"verdict": "admitted", "U": 0.5, "delta_mean": 0.1,
                       "factor_tearsheets": {"f": _tsheet(
                           rank_ic_neutral=-0.03, rank_ic=-0.10,
                           t_nw=2.0, rho_max=0.70, ic_pos_frac=0.50)}},
                 factors=[{"expression": "TS_MEAN($close,20)"}])
           for i in range(12)]
    with _Env():
        os.environ["QA_RESEED_FAILURE_MEMORY"] = "1"
        os.environ["QA_RESEED_OP_NOVELTY_GATE"] = "1"
        os.environ["QA_RESEED_OP_TOPK"] = "3"
        ctl = _ctl(adm)
        digest = ctl._build_reseed_digest().lower()
        for tok in BANNED:
            assert tok not in digest, f"F5: prescription/prior '{tok}' leaked into digest"
        # Gate retry path: force all-rejected so the retry measurement is built,
        # and capture the measurement string (not the LLM call).
        with mock.patch.object(ctl, "_generate_informed_directions", return_value=[]):
            ctl._filter_novel_directions(["A high TS_MEAN predicts HIGHER."], digest)
        # Reconstruct the retry measurement the gate builds and vet it end to end:
        # the fixed framing prose + the coverage block of the admitted zoo.
        from quantaalpha.factors.operator_coverage import (
            coverage_block, top_exercised_operators)
        admitted = ctl._admitted_expressions()
        top3 = sorted(set(top_exercised_operators(admitted, 3)))
        retry_meas = (
            "\n\n## Operator-novelty outcome\n"
            f"The previous directions exercised only the top-3 most-used "
            f"operators ({', '.join(top3)}) and named no operator outside that "
            f"set.\n" + (coverage_block(admitted) or "")
        ).lower()
        for tok in BANNED:
            assert tok not in retry_meas, f"F5: prescription/prior '{tok}' in retry measurement"
    print("OK  F5: no prescription token, no market prior in digest or gate")


def test_f6_what_worked_well_ranks_strongest_all_phases():
    """F6: the always-on 'what worked well' block ranks admitted factors by |t|,
    scans ALL phases (a mutation child surfaces), excludes rejected and
    no-tearsheet admits, and carries no prescription / market prior."""
    BANNED = ["you should", "try ", "instead use", "lengthen", "shorten",
              "prefer ", "we recommend", "make sure to", "always use",
              "CSI300", "csi 300", "reversion", "momentum prior"]

    def _adm(did, rnd, phase, expr, t_nw, ric=-0.02):
        t = _traj(did, rnd, {"verdict": "admitted", "U": 0.5, "delta_mean": 0.1,
                            "factor_tearsheets": {expr: _tsheet(
                                rank_ic_neutral=ric, t_nw=t_nw, best_horizon=1)}},
                  factors=[{"expression": expr}])
        t.phase = phase
        return t

    with _Env():
        trajs = [
            # original |t|=4 ; mutation child |t|=8 (the bred winner) ;
            # a rejected |t|=9 (must NOT appear) ; a no-tearsheet admit (skipped).
            _adm(0, 0, RoundPhase.ORIGINAL, "RANK(TS_MEAN($close,5))", 4.0),
            _adm(0, 1, RoundPhase.MUTATION,  "RSI($close,14)",          8.0, ric=-0.05),
            _traj(1, 0, {"verdict": "marginal", "delta_mean": -0.01,
                         "factor_tearsheets": {"TS_RANK($close,60)": _tsheet(
                             rank_ic_neutral=-0.02, t_nw=9.0, best_horizon=1)}},
                  factors=[{"expression": "TS_RANK($close,60)"}]),
            _traj(2, 0, {"verdict": "admitted", "U": 0.4, "delta_mean": 0.05},
                  factors=[{"expression": "DELTA($close,5)"}]),
        ]
        ctl = _ctl(trajs)
        block = ctl._what_worked_well_block()
        assert block and "What worked well" in block[0], "F6: block missing"
        joined = "\n".join(block)
        # Ranked by |t| desc: RSI(8) before TS_MEAN(4); rejected |t|=9 absent.
        assert "RSI($close,14)" in joined and "RANK(TS_MEAN($close,5))" in joined
        assert "TS_RANK($close,60)" not in joined, "F6: rejected factor leaked in"
        assert "DELTA($close,5)" not in joined, "F6: no-tearsheet admit leaked in"
        assert joined.index("RSI($close,14)") < joined.index("RANK(TS_MEAN($close,5))"), \
            "F6: not ranked by |t|"
        assert "yours to determine" in joined
        for tok in BANNED:
            assert tok not in joined.lower(), f"F6: prescription/prior '{tok}' leaked"
        # Empty when no admitted factor carries a tearsheet.
        empty = _ctl([_traj(0, 0, {"verdict": "marginal", "delta_mean": -0.01,
                                  "factor_tearsheets": {"f": _tsheet(
                                      rank_ic_neutral=-0.02, t_nw=9.0)}},
                             factors=[{"expression": "f"}])])
        assert empty._what_worked_well_block() == [], "F6: non-empty with no admits"
    print("OK  F6: what-worked-well ranks strongest, all phases, no prescription")


if __name__ == "__main__":
    test_f1_sign_direction_block_measures()
    test_f2_digest_carries_failure_memory_when_on_absent_when_off()
    test_f3_gate_keeps_novel_rejects_monoculture_inert_when_off()
    test_f4_gate_noops_without_admitted_zoo()
    test_f5_no_prescription_no_market_prior()
    test_f6_what_worked_well_ranks_strongest_all_phases()
    print("\nAll reseed-failure-memory tests PASS")