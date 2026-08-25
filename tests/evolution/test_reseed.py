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


if __name__ == "__main__":
    test_v1_no_repository_data_is_noop()
    test_v2_grow_and_mark_saturated()
    test_v3_sequential_transitions_divert_on_reseed()
    test_v4_extraction_surfaces_admitted_verdict()
    test_v5_informed_directions_parse_and_no_fallback()
    test_v6_reseed_fires_through_real_get_next_task()
    print("\nAll reseed tests passed.")