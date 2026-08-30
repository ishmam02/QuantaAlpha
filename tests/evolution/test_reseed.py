"""Learning-aware reseed: the search regenerates informed directions when the
repository stops growing, and the control arm is untouched.

These five scenarios pin the behaviour the plan specified:

  V1  control-arm no-op      -- no ``U`` on any trajectory => the whole reseed
                                 path is a no-op even with a stuck zoo size,
                                 and ``_build_zoo_context`` is empty.
  V2  grow + mark saturated  -- U-bearing history, monkeypatched LLM output =>
                                 ``_directions`` grows, saturated old ids stay
                                 completed, recently-admitting ids re-open,
                                 phase resets to ORIGINAL.
  V3  sequential transitions -- ``_get_mutation_task`` / ``_get_crossover_task``
                                 divert to ``_get_original_task`` exactly when
                                 ``_reseed_if_stale`` fires, and take their
                                 normal breeding transition otherwise.
  V4  metric extraction      -- ``admitted=False`` surfaces onto the trajectory
                                 extras; a Qlib-only Series gets no ``admitted``
                                 key (control arm byte-identical).
  V5  informed parse         -- ``generate_informed_directions`` parses the LLM
                                 JSON and returns the list; with
                                 ``allow_fallback=False`` a parse failure yields
                                 ``[]`` (no canned directions injected).

The control arm must differ from the treatment arm ONLY in the objective, so
V1 and V4 are the load-bearing A/B-safety checks. Run directly
(``python tests/evolution/test_reseed.py``) or via ``pytest``.
"""
from pathlib import Path

import pandas as pd
from unittest import mock

from quantaalpha.pipeline.evolution.controller import (
    EvolutionConfig,
    EvolutionController,
    RoundPhase,
)
from quantaalpha.pipeline.evolution.trajectory import StrategyTrajectory
from quantaalpha.pipeline.planning import generate_informed_directions

_PROMPTS = Path(__file__).resolve().parents[2] / "quantaalpha" / "pipeline" / "prompts"


def _cfg(**over) -> EvolutionConfig:
    base = dict(
        num_directions=3,
        max_rounds=20,
        mutation_enabled=True,
        crossover_enabled=True,
        fresh_start=True,
        reseed_after_stale_rounds=2,
        directions=["d0", "d1", "d2"],
        initial_direction="explore momentum and reversal",
        informed_prompt_path=str(_PROMPTS / "informed_planning_prompts.yaml"),
        mutation_prompt_path=str(_PROMPTS / "evolution_prompts.yaml"),
        crossover_prompt_path=str(_PROMPTS / "evolution_prompts.yaml"),
    )
    base.update(over)
    return EvolutionConfig(**base)


def _traj(did: int, rnd: int, metrics: dict, factors=None) -> StrategyTrajectory:
    return StrategyTrajectory(
        trajectory_id=StrategyTrajectory.generate_id(did, rnd, RoundPhase.ORIGINAL),
        direction_id=did,
        round_idx=rnd,
        phase=RoundPhase.ORIGINAL,
        factors=factors or [],
        backtest_metrics=dict(metrics),
    )


def test_v1_no_repository_data_is_noop():
    """No repository size available => reseed never fires, zoo context empty.

    This case used to be phrased as "the control arm is a no-op", gated on a
    `_has_treatment_data()` check. The A/B framework was removed, so there is
    no control arm and that method is gone -- but the invariant it protected
    still matters and still holds: before any admission has been recorded there
    is nothing to reseed FROM, and firing anyway would replace the starting
    directions with directions informed by no outcomes at all.
    """
    cfg = _cfg(num_directions=2, directions=["c0", "c1"])
    ctl = EvolutionController(cfg)
    # Trajectories carrying no U and no `admitted` -- the early-run state.
    ctl.pool.add(_traj(0, 0, {"RankIC": 0.31}))
    ctl.pool.add(_traj(1, 0, {"RankIC": 0.27}))

    with mock.patch.object(ctl, "_zoo_size", return_value=None):
        for _ in range(4):
            assert ctl._reseed_if_stale() is False
    assert ctl._stale_rounds == 0
    assert ctl._directions == ["c0", "c1"]
    assert ctl._current_phase == RoundPhase.ORIGINAL
    assert ctl._build_zoo_context() == ""
    print("OK  V1: no repository data is a no-op (gate on data presence)")


def test_v2_grow_and_mark_saturated():
    """U-bearing history + mocked LLM => directions grow, saturated skipped."""
    ctl = EvolutionController(_cfg())
    # ORIGINAL outcomes: dir0 admitted early, dir1 rejected (redundant),
    # dir2 admitted recently.
    ctl.report_task_complete(
        {"phase": RoundPhase.ORIGINAL, "direction_id": 0},
        _traj(0, 0, {"U": 0.5, "admitted": True, "delta_mean": 0.10, "rho_max": 0.3},
              factors=[{"expression": "TS_MEAN($close,5)"}]),
    )
    ctl.report_task_complete(
        {"phase": RoundPhase.ORIGINAL, "direction_id": 1},
        _traj(1, 0, {"U": 0.4, "admitted": False, "delta_mean": -0.05,
                     "rho_max": 0.9, "failed_gates": "rho_max"}),
    )
    ctl.report_task_complete(
        {"phase": RoundPhase.ORIGINAL, "direction_id": 2},
        _traj(2, 2, {"U": 0.6, "admitted": True, "delta_mean": 0.20, "rho_max": 0.2}),
    )
    assert ctl._directions_completed == {0, 1, 2}

    ctl._current_round = 3  # n=2 => dir0 (admit@0) and dir1 (never) saturate, dir2 (admit@2) does not
    ctl._best_zoo_size = 5
    with mock.patch.object(ctl, "_zoo_size", return_value=5), \
         mock.patch.object(ctl, "_generate_informed_directions",
                           return_value=["newA", "newB"]) as gen:
        assert ctl._reseed_if_stale() is False  # stale 0->1, below the n=2 window
        assert gen.call_count == 0
        fired = ctl._reseed_if_stale()           # stale 1->2, fires
    assert fired is True

    # Direction list GREW (never replaced); two new reseed_1 status entries.
    assert ctl._directions == ["d0", "d1", "d2", "newA", "newB"]
    assert ctl._direction_status[3]["source"] == "reseed_1"
    assert ctl._direction_status[4]["source"] == "reseed_1"
    assert ctl._reseed_count == 1
    # Saturated ids (0,1) stay completed; the recently-admitting id 2 re-opens.
    assert ctl._direction_status[0]["saturated"] is True
    assert ctl._direction_status[1]["saturated"] is True
    assert ctl._direction_status[2]["saturated"] is False
    assert ctl._directions_completed == {0, 1}
    assert ctl._current_phase == RoundPhase.ORIGINAL
    print("OK  V2: grew by 2, marked 0/1 saturated, re-opened 2, phase=ORIGINAL")


def test_v3_sequential_transitions_divert_on_reseed():
    """Mutation/crossover transitions return ORIGINAL iff the reseed fires.

    The REAL ``_get_mutation_task`` / ``_get_crossover_task`` must run so they
    reach their transition point and call the mocked ``_get_original_task``;
    mocking the transition method itself would short-circuit the very branch
    under test. So mocks are scoped per scenario, patching only the neighbours
    each one calls.
    """
    ctl = EvolutionController(_cfg())
    sentinel_orig = {"phase": RoundPhase.ORIGINAL, "direction_id": 999, "marker": "orig"}
    sentinel_cx = {"phase": RoundPhase.CROSSOVER, "direction_id": 7, "marker": "cx"}

    # --- mutation transition: reseed fires -> divert to ORIGINAL ---
    ctl._mutation_targets, ctl._mutation_idx = [], 0
    ctl._current_phase = RoundPhase.MUTATION
    with mock.patch.object(ctl, "_prepare_mutation_targets"), \
         mock.patch.object(ctl, "_get_original_task", return_value=sentinel_orig), \
         mock.patch.object(ctl, "_reseed_if_stale", return_value=True):
        assert ctl._get_mutation_task()["marker"] == "orig"

    # --- mutation transition: reseed does not fire -> normal breeding ---
    ctl._mutation_targets, ctl._mutation_idx = [], 0
    ctl._current_phase = RoundPhase.MUTATION
    with mock.patch.object(ctl, "_prepare_mutation_targets"), \
         mock.patch.object(ctl, "_prepare_crossover_groups"), \
         mock.patch.object(ctl, "_get_crossover_task", return_value=sentinel_cx), \
         mock.patch.object(ctl, "_reseed_if_stale", return_value=False):
        got = ctl._get_mutation_task()
    assert got["marker"] == "cx"
    assert ctl._current_phase == RoundPhase.CROSSOVER

    # --- crossover transition: reseed fires -> divert to ORIGINAL ---
    # Real _get_crossover_task runs (idx 0 >= len([]) => transition); only its
    # neighbour _get_original_task is mocked.
    ctl._crossover_groups, ctl._crossover_idx = [], 0
    ctl._current_phase = RoundPhase.CROSSOVER
    with mock.patch.object(ctl, "_get_original_task", return_value=sentinel_orig), \
         mock.patch.object(ctl, "_reseed_if_stale", return_value=True):
        assert ctl._get_crossover_task()["marker"] == "orig"
    print("OK  V3: sequential mutation & crossover divert to ORIGINAL on reseed")


def test_v4_extraction_surfaces_admitted_verdict():
    """``admitted`` rides on treatment extras; control Series is untouched."""
    ctl = EvolutionController(_cfg())

    treatment = pd.Series({
        "U": 0.5, "feasible": True, "admitted": False,
        "delta_mean": -0.05, "rho_max": 0.9,
        "failed_gates": "rho_max", "weakest_dimensions": "size",
    })
    ex = ctl._extract_net_cost_metrics(treatment)
    assert ex["admitted"] is False          # the real verdict, not the feasible fallback
    assert ex["feasible"] is True
    assert ex["U"] == 0.5
    assert ex["failed_gates"] == "rho_max"
    assert ex["weakest_dimensions"] == "size"

    control = pd.Series({"RankIC": 0.31, "ICIR": 1.2})  # no net-cost keys at all
    ex_c = ctl._extract_net_cost_metrics(control)
    assert "admitted" not in ex_c           # control arm gets no injected verdict
    assert "U" not in ex_c
    assert ex_c == {}
    print("OK  V4: admitted verdict surfaced for treatment; control untouched")


def test_v5_informed_directions_parse_and_no_fallback():
    """LLM JSON parses to the list; a parse failure with no fallback returns []."""
    prompt = _PROMPTS / "informed_planning_prompts.yaml"
    assert prompt.exists()

    fake_ok = mock.MagicMock()
    fake_ok.return_value.build_messages_and_create_chat_completion.return_value = (
        '```json\n{"directions": ["alphaA", "alphaB", "alphaC"]}\n```'
    )
    with mock.patch("quantaalpha.pipeline.planning.APIBackend", fake_ok):
        out = generate_informed_directions(
            initial_direction="explore momentum", n=3, prompt_file=prompt,
            history_summary="dir0: redundant", use_llm=True)
    assert out == ["alphaA", "alphaB", "alphaC"]

    fake_bad = mock.MagicMock()
    fake_bad.return_value.build_messages_and_create_chat_completion.return_value = "not json"
    with mock.patch("quantaalpha.pipeline.planning.APIBackend", fake_bad):
        out = generate_informed_directions(
            initial_direction="explore momentum", n=3, prompt_file=prompt,
            history_summary="dir0: redundant", max_attempts=1, use_llm=True)
    # The `allow_fallback` flag is gone because the canned fallback directions it
    # guarded were deleted outright: a hardcoded hypothesis injected on parse
    # failure is a market prior the system did not mine. [] is now unconditional.
    assert out == []
    print("OK  V5: informed directions parse; a parse failure yields [] with no canned directions")


def test_v6_reseed_fires_through_real_get_next_task():
    """End-to-end sequential dispatch: a stalled repository makes the REAL
    ``get_next_task`` return an ORIGINAL task carrying a GROWN direction.

    Only the two external dependencies are mocked -- ``_zoo_size`` (the ledger
    read) and ``_generate_informed_directions`` (the LLM call). The dispatch
    (``get_next_task`` -> ``_get_mutation_task``), the stale detection, the
    digest, the saturation marking, the direction growth and the ORIGINAL task
    emission all run unmodified. ``_prepare_mutation_targets`` is stubbed only
    to reach the mutation transition without exercising parent selection, which
    is orthogonal to the reseed wiring.
    """
    ctl = EvolutionController(_cfg(num_directions=2, directions=["d0", "d1"]))
    # Two ORIGINAL outcomes with U -- both admitted early, so both saturate by
    # round 3 (n=2) and stay completed; the post-reseed ORIGINAL task must then
    # be one of the NEW directions rather than a re-opened old one.
    ctl.report_task_complete(
        {"phase": RoundPhase.ORIGINAL, "direction_id": 0},
        _traj(0, 0, {"U": 0.5, "admitted": True, "delta_mean": 0.10, "rho_max": 0.3},
              factors=[{"expression": "TS_MEAN($close,5)"}]),
    )
    ctl.report_task_complete(
        {"phase": RoundPhase.ORIGINAL, "direction_id": 1},
        _traj(1, 0, {"U": 0.4, "admitted": True, "delta_mean": 0.08, "rho_max": 0.4},
              factors=[{"expression": "TS_STD($close,10)"}]),
    )
    assert ctl._directions_completed == {0, 1}

    # Park the controller at the mutation transition, one stale round short of
    # firing (n=2). Mock only the ledger read + the LLM.
    ctl._current_phase = RoundPhase.MUTATION
    ctl._mutation_targets, ctl._mutation_idx = [], 0
    ctl._current_round = 3
    ctl._best_zoo_size = 5
    # A controller that has already taken its growth baseline, which is what a
    # real run looks like by round 3. Leaving this at its -1 init makes the
    # first check spend itself establishing the baseline (deliberate: it stops a
    # resumed run reading a spurious +1 of growth and masking a stall already in
    # progress), so the reseed would fire one round later than this case asserts.
    ctl._zoo_size_at_last_check = 5
    ctl._stale_rounds = 1
    with mock.patch.object(ctl, "_zoo_size", return_value=5), \
         mock.patch.object(ctl, "_generate_informed_directions",
                           return_value=["newA", "newB"]), \
         mock.patch.object(ctl, "_prepare_mutation_targets"):
        task = ctl.get_next_task()

    # The reseed fired through the real dispatch and grew the direction list.
    assert ctl._reseed_count == 1
    assert ctl._directions == ["d0", "d1", "newA", "newB"]
    assert task is not None
    assert task["phase"] == RoundPhase.ORIGINAL
    assert task["direction_id"] >= 2          # a grown direction, not a re-opened old one
    assert task["direction"] in ("newA", "newB")
    assert ctl._current_phase == RoundPhase.ORIGINAL
    print("OK  V6: real get_next_task dispatch -> reseed fires -> ORIGINAL task"
          " carries a grown direction")


def _cx(did: int, rnd: int, metrics: dict, factors=None) -> StrategyTrajectory:
    """A crossover-phase trajectory (the breeding stock mutation normally uses)."""
    return StrategyTrajectory(
        trajectory_id=StrategyTrajectory.generate_id(did, rnd, RoundPhase.CROSSOVER),
        direction_id=did, round_idx=rnd, phase=RoundPhase.CROSSOVER,
        factors=factors or [], backtest_metrics=dict(metrics),
    )


def _mut(did: int, rnd: int, metrics: dict, factors=None) -> StrategyTrajectory:
    """A mutation-phase trajectory (post-reseed mutation carries reseed genes)."""
    return StrategyTrajectory(
        trajectory_id=StrategyTrajectory.generate_id(did, rnd, RoundPhase.MUTATION),
        direction_id=did, round_idx=rnd, phase=RoundPhase.MUTATION,
        factors=factors or [], backtest_metrics=dict(metrics),
    )


def test_v7_post_reseed_mutation_breeds_fresh_originals():
    """The fix for the reseed dead-end.

    A mutation whose IMMEDIATELY PRECEDING round is a fresh ORIGINAL round (a
    reseed -- the reseed returns the phase to ORIGINAL) breeds from THOSE new
    directions, not from the stale latest crossover. Measured on the production
    pool before this fix: 0/53 reseeded originals were ever bred; every
    post-reseed mutation bred from a crossover 2-4 rounds stale (round 5 bred
    round 2's crossover, round 9 bred round 6's, ... round 21 bred round 18's).

    This runs the REAL ``_prepare_mutation_targets`` -- the only mock is
    ``_cached_diagnosis`` (returned None -> every parent routes to ORTHOGONAL,
    so ``_mutation_targets`` = every admissible parent of the selected round),
    which isolates parent SELECTION from the diagnosis/bucketing logic.
    """
    import os
    ctl = EvolutionController(_cfg())
    # round 0: initial originals -- must NOT be selected at round 5.
    r0 = [_traj(i, 0, {"U": 0.4, "delta_mean": 0.05, "admitted": i == 0,
                       "verdict": "admitted" if i == 0 else "marginal"},
                factors=[{"expression": "TS_MEAN($close,20)"}]) for i in range(2)]
    # round 2: crossover children -- the STALE crossover the old code bred.
    cx2 = [_cx(i, 2, {"U": 0.3, "delta_mean": 0.02, "admitted": False,
                      "verdict": "marginal"},
                 factors=[{"expression": f"RANK($close,{20+i})"}]) for i in range(2)]
    # round 4: ORIGINAL -- the reseed's FRESH directions (the fix must breed THESE).
    r4 = [_traj(10 + i, 4, {"U": 0.5, "delta_mean": 0.08, "admitted": False,
                            "verdict": "marginal"},
                factors=[{"expression": f"RSI($close,{14+i})"}]) for i in range(2)]
    for t in r0 + cx2 + r4:
        ctl.pool.add(t)

    ctl._current_round = 5  # mutation; prev_round=4 is the reseed ORIGINAL round
    saved = os.environ.pop("QA_REQUIRE_FEASIBLE", None)
    try:
        with mock.patch.object(ctl, "_cached_diagnosis", return_value=None):
            ctl._prepare_mutation_targets()
    finally:
        if saved is not None:
            os.environ["QA_REQUIRE_FEASIBLE"] = saved

    selected = {t.trajectory_id for t in ctl._mutation_targets}
    assert selected == {t.trajectory_id for t in r4}, (
        f"V7: post-reseed mutation must breed the round-4 fresh originals, "
        f"got rounds {sorted({t.round_idx for t in ctl._mutation_targets})}")
    assert not (selected & {t.trajectory_id for t in cx2}), "V7: stale crossover leaked in"
    assert not (selected & {t.trajectory_id for t in r0}), "V7: round-0 originals leaked in"
    print("OK  V7: post-reseed mutation breeds the fresh ORIGINAL round, not the stale crossover")


def test_v8_normal_mutation_breeds_latest_crossover_unchanged():
    """The fix changes ONLY the reseed case.

    In the normal cycle the round before a mutation is a CROSSOVER round, so
    ``prev_is_original`` is False and mutation still breeds the latest
    crossover outputs -- byte-identical to the pre-fix behavior. This is the
    guardrail: the surgical fix must not perturb the normal breeding chain.
    """
    import os
    ctl = EvolutionController(_cfg())
    r0 = [_traj(i, 0, {"U": 0.4, "delta_mean": 0.05, "admitted": i == 0,
                       "verdict": "admitted" if i == 0 else "marginal"},
                factors=[{"expression": "TS_MEAN($close,20)"}]) for i in range(2)]
    cx2 = [_cx(i, 2, {"U": 0.3, "delta_mean": 0.02, "admitted": False,
                      "verdict": "marginal"},
                 factors=[{"expression": f"RANK($close,{20+i})"}]) for i in range(2)]
    # round 4 is a CROSSOVER round here (normal cycle) -- the latest crossover.
    cx4 = [_cx(i, 4, {"U": 0.3, "delta_mean": 0.02, "admitted": False,
                      "verdict": "marginal"},
                 factors=[{"expression": f"DELTA($close,{i+1})"}]) for i in range(2)]
    for t in r0 + cx2 + cx4:
        ctl.pool.add(t)

    ctl._current_round = 5  # mutation; prev_round=4 is a CROSSOVER round (normal)
    saved = os.environ.pop("QA_REQUIRE_FEASIBLE", None)
    try:
        with mock.patch.object(ctl, "_cached_diagnosis", return_value=None):
            ctl._prepare_mutation_targets()
    finally:
        if saved is not None:
            os.environ["QA_REQUIRE_FEASIBLE"] = saved

    selected = {t.trajectory_id for t in ctl._mutation_targets}
    assert selected == {t.trajectory_id for t in cx4}, (
        f"V8: normal mutation must breed the LATEST crossover (round 4), "
        f"got rounds {sorted({t.round_idx for t in ctl._mutation_targets})}")
    assert not (selected & {t.trajectory_id for t in cx2}), "V8: older crossover leaked in"
    assert not (selected & {t.trajectory_id for t in r0}), "V8: round-0 originals leaked in"
    print("OK  V8: normal mutation breeds the latest crossover (unchanged by the fix)")


def test_v9_post_reseed_crossover_breeds_fresh_originals():
    """The crossover half of the reseed fix (Option A).

    A crossover whose most recent ORIGINAL round is NEWER than any crossover
    (only a reseed creates an original round once crossovers exist) breeds the
    FRESH reseed originals + the post-reseed mutation, not the stale pre-reseed
    crossover. Measured pre-fix on production pool ``meanvar_20260827_224322``:
    post-reseed crossovers bred the reseed round 0/8 (r6), 0/8 (r10), 0/7 (r14)
    -- ~half their children picked two stale-crossover parents and inherited
    zero reseed genes, so the reseed's new directions only entered the lineage
    via the mutation half.

    Runs the REAL ``_get_crossover_candidates`` (the round-selection logic); it
    reads only ``round_idx``/``phase``, so metric values are placeholders and
    admissibility filtering (applied later in ``_prepare_crossover_groups``) is
    out of scope here.
    """
    ctl = EvolutionController(_cfg())
    r0 = [_traj(i, 0, {"U": 0.4, "delta_mean": 0.05, "admitted": i == 0,
                       "verdict": "admitted" if i == 0 else "marginal"},
                factors=[{"expression": "TS_MEAN($close,20)"}]) for i in range(2)]
    cx2 = [_cx(i, 2, {"U": 0.3, "delta_mean": 0.02, "admitted": False,
                      "verdict": "marginal"},
               factors=[{"expression": f"RANK($close,{20+i})"}]) for i in range(2)]
    # round 4: ORIGINAL -- the reseed's FRESH directions (the fix must breed THESE).
    r4 = [_traj(10 + i, 4, {"U": 0.5, "delta_mean": 0.08, "admitted": False,
                            "verdict": "marginal"},
                factors=[{"expression": f"RSI($close,{14+i})"}]) for i in range(2)]
    # round 5: mutation -- post-reseed, already carries reseed-4 genes via V7.
    m5 = [_mut(20 + i, 5, {"U": 0.45, "delta_mean": 0.06, "admitted": False,
                           "verdict": "marginal"},
               factors=[{"expression": f"RSI($close,{14+i})_m"}]) for i in range(2)]
    for t in r0 + cx2 + r4 + m5:
        ctl.pool.add(t)

    candidates = ctl._get_crossover_candidates()
    selected_rounds = sorted({t.round_idx for t in candidates})
    assert 4 in selected_rounds, "V9: reseed originals (round 4) must be candidates"
    assert 5 in selected_rounds, "V9: post-reseed mutation (round 5) must be candidates"
    assert 2 not in selected_rounds, "V9: stale pre-reseed crossover (round 2) leaked in"
    assert 0 not in selected_rounds, "V9: initial originals (round 0) leaked in"
    print("OK  V9: post-reseed crossover breeds fresh originals + mutation, not stale crossover")


def test_v10_normal_crossover_breeds_latest_mutation_and_crossover_unchanged():
    """The fix changes ONLY the reseed case.

    In the normal cycle no original round is newer than the latest crossover,
    so the post-reseed branch does not trigger and crossover still breeds the
    latest mutation + latest crossover -- byte-identical to the pre-fix
    behavior. This is the guardrail: the surgical fix must not perturb the
    normal breeding chain.
    """
    ctl = EvolutionController(_cfg())
    r0 = [_traj(i, 0, {"U": 0.4, "delta_mean": 0.05, "admitted": i == 0,
                       "verdict": "admitted" if i == 0 else "marginal"},
                factors=[{"expression": "TS_MEAN($close,20)"}]) for i in range(2)]
    cx2 = [_cx(i, 2, {"U": 0.3, "delta_mean": 0.02, "admitted": False,
                      "verdict": "marginal"},
               factors=[{"expression": f"RANK($close,{20+i})"}]) for i in range(2)]
    m3 = [_mut(20 + i, 3, {"U": 0.35, "delta_mean": 0.03, "admitted": False,
                           "verdict": "marginal"},
               factors=[{"expression": f"DELTA($close,{i+1})_m"}]) for i in range(2)]
    # round 4: CROSSOVER (normal cycle) -- the latest crossover.
    cx4 = [_cx(10 + i, 4, {"U": 0.3, "delta_mean": 0.02, "admitted": False,
                           "verdict": "marginal"},
                factors=[{"expression": f"DELTA($close,{i+1})"}]) for i in range(2)]
    for t in r0 + cx2 + m3 + cx4:
        ctl.pool.add(t)

    candidates = ctl._get_crossover_candidates()
    selected_rounds = sorted({t.round_idx for t in candidates})
    # Normal Case 2: latest mutation (round 3) + latest crossover (round 4).
    assert 3 in selected_rounds and 4 in selected_rounds, (
        f"V10: normal crossover must breed latest mutation (3) + latest crossover (4), "
        f"got rounds {selected_rounds}")
    assert 0 not in selected_rounds, "V10: originals (round 0) must not leak into normal crossover"
    assert 2 not in selected_rounds, "V10: older crossover (round 2) must not leak in"
    print("OK  V10: normal crossover breeds latest mutation + latest crossover (unchanged by the fix)")


if __name__ == "__main__":
    test_v1_no_repository_data_is_noop()
    test_v2_grow_and_mark_saturated()
    test_v3_sequential_transitions_divert_on_reseed()
    test_v4_extraction_surfaces_admitted_verdict()
    test_v5_informed_directions_parse_and_no_fallback()
    test_v6_reseed_fires_through_real_get_next_task()
    test_v7_post_reseed_mutation_breeds_fresh_originals()
    test_v8_normal_mutation_breeds_latest_crossover_unchanged()
    test_v9_post_reseed_crossover_breeds_fresh_originals()
    test_v10_normal_crossover_breeds_latest_mutation_and_crossover_unchanged()
    print("\nAll reseed tests passed.")