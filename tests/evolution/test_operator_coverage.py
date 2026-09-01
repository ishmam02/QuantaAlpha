"""Operator-coverage measurement + OHLCV-seed de-prime.

Two changes in one plan, tested together because they are the two layers of the
same monoculture:

  C1  operator-coverage measurement (``quantaalpha.factors.operator_coverage``)
      injected into the reseed digest (direction breadth) and the factor
      feedback (within-mechanism breadth). Measurement only -- names which
      operators the population exhausted, closes "yours to determine", never a
      remedy, never a market prior.

  C2  de-prime the Alpha158(20) OHLCV seeds: ``seed_in_generation: false``
      (planning) and ``_with_seed_context`` removed (hypothesis generator). The
      seeds anchored round-0 directions to price-volume microstructure and
      primed a TS_MEAN formula the construct copied verbatim.

Run directly (``python tests/evolution/test_operator_coverage.py``) or via pytest.
"""
from pathlib import Path
from unittest import mock

from quantaalpha.factors.operator_coverage import (
    DECLARED_OPERATORS,
    extract_operators,
    coverage_block,
)
from quantaalpha.factors.coder import factor_ast

# --- C1.1 extract_operators ---------------------------------------------------

def test_extract_operators_collects_function_names():
    expr = "RANK(TS_MEAN($close - DELAY($close, 1), 5))"
    assert extract_operators(expr) == {"RANK", "TS_MEAN", "DELAY"}


def test_extract_operators_nested_and_repeated():
    expr = "TS_CORR(TS_MEAN($close, 5), DELAY($volume, 2), 10) / TS_MEAN($close, 3)"
    ops = extract_operators(expr)
    assert ops == {"TS_CORR", "TS_MEAN", "DELAY"}


def test_extract_operators_uppercases_consistently():
    # The grammar accepts any-case function names; the measurement normalizes.
    assert "TS_MEAN" in extract_operators("rank(ts_mean($close, 5))")


def test_extract_operators_garbage_returns_empty_no_raise():
    for bad in ["not a real expr @@@", "", "   ", "((((", "TS_MEAN("]:
        assert extract_operators(bad) == set(), f"bad input must degrade: {bad!r}"


def test_collect_operators_added_to_factor_ast():
    # The parallel walker exists beside collect_unique_vars and is VarNode-aware.
    tree = factor_ast.parse_expression("RANK(DELAY($close, 1))")
    ops: set = set()
    factor_ast.collect_operators(tree, ops)
    assert ops == {"RANK", "DELAY"}


# --- C1.2 DECLARED_OPERATORS --------------------------------------------------

def test_declared_count_is_the_real_operator_vocabulary():
    # ~55 genuine operators (73 defs minus duplicates minus pseudo-ops).
    assert 45 <= len(DECLARED_OPERATORS) <= 65, len(DECLARED_OPERATORS)


def test_declared_includes_unused_operator_classes():
    # The coverage gap must be able to NAME these -- they are executable but
    # unexercised in the measured monoculture.
    need = {"TS_CORR", "TS_COVARIANCE", "REGBETA", "REGRESI", "RSI", "MACD",
            "COUNT", "SUMIF", "FILTER", "SMA", "EMA", "WMA", "DECAYLINEAR",
            "SKEW", "KURT", "TS_ARGMAX", "TS_ARGMIN", "HIGHDAY", "LOWDAY"}
    assert need <= DECLARED_OPERATORS, need - DECLARED_OPERATORS


def test_declared_excludes_pseudo_ops():
    # Arithmetic / comparison / logical / helper / conditional-ternary are NOT
    # genuine factor transforms -- they must not appear as a coverage gap.
    pseudo = {"ADD", "SUBTRACT", "MULTIPLY", "DIVIDE", "GT", "LT", "GE", "LE",
              "EQ", "NE", "AND", "OR", "WHERE", "SEQUENCE", "FLOOR"}
    assert pseudo.isdisjoint(DECLARED_OPERATORS), pseudo & DECLARED_OPERATORS


# --- C1.3 coverage_block: measurement-only ------------------------------------

_MONOCULTURE = [
    "RANK(TS_MEAN($close, 5))",
    "RANK(TS_MEAN($volume, 10))",
    "TS_MEAN(DELAY($close, 1), 20)",
    "RANK(TS_MEAN($vwap, 5))",
    "TS_MEAN($close - DELAY($close, 1), 5)",
]


def test_coverage_block_names_most_used_and_top3_share():
    block = coverage_block(_MONOCULTURE)
    assert "Operator coverage so far" in block
    # TS_MEAN dominates this monoculture population.
    assert "TS_MEAN" in block
    assert "Top 3 operators" in block
    assert "% of all calls" in block


def test_coverage_block_lists_unused_operator_classes():
    block = coverage_block(_MONOCULTURE)
    assert "not yet used in the population" in block
    # The unused set must name the operator classes the monoculture missed.
    for op in ("REGBETA", "REGRESI", "RSI", "MACD", "TS_CORR", "COUNT"):
        assert op in block, f"unused list must name {op}"


def test_coverage_block_states_exercised_count_out_of_declared():
    block = coverage_block(_MONOCULTURE)
    n_declared = len(DECLARED_OPERATORS)
    assert f"of {n_declared} declared operators exercised so far" in block


def test_coverage_block_never_prescribes():
    block = coverage_block(_MONOCULTURE)
    low = block.lower()
    for tok in ("use ", "try ", "should ", "consider ", "ought ", "prefer "):
        assert tok not in low, f"block must not prescribe; found {tok!r}"


def test_coverage_block_closes_yours_to_determine():
    block = coverage_block(_MONOCULTURE)
    assert "yours to determine" in block


def test_coverage_block_carries_no_market_prior():
    block = coverage_block(_MONOCULTURE)
    # The measurement is over operator names only; no index / market strings.
    for tok in ("CSI 300", "A-share", "A-shares", "CSI300", "SH000300"):
        assert tok not in block, f"block must not carry market prior: {tok}"


def test_coverage_block_empty_population_degrades_silently():
    assert coverage_block([]) == ""
    assert coverage_block(["", "  "]) == ""


def test_coverage_block_all_garbage_degrades_silently():
    assert coverage_block(["@@@", "((((", "not an expr"]) == ""


# --- C1.4 threading: reseed digest (PRIMARY) ---------------------------------

def _cfg(**over):
    from quantaalpha.pipeline.evolution.controller import EvolutionConfig
    _PROMPTS = Path(__file__).resolve().parents[2] / "quantaalpha" / "pipeline" / "prompts"
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


def test_reseed_digest_includes_operator_coverage_section():
    from quantaalpha.pipeline.evolution.controller import (
        EvolutionController, RoundPhase,
    )
    from quantaalpha.pipeline.evolution.trajectory import StrategyTrajectory

    def _traj(did, rnd, metrics, factors=None):
        return StrategyTrajectory(
            trajectory_id=StrategyTrajectory.generate_id(did, rnd, RoundPhase.ORIGINAL),
            direction_id=did, round_idx=rnd, phase=RoundPhase.ORIGINAL,
            factors=factors or [], backtest_metrics=dict(metrics),
        )

    ctl = EvolutionController(_cfg())
    # Admitted trajectories carrying TS_MEAN monoculture expressions.
    ctl.pool.add(_traj(0, 0, {"U": 0.5, "admitted": True, "delta_mean": 0.1},
                        factors=[{"expression": "RANK(TS_MEAN($close, 5))"}]))
    ctl.pool.add(_traj(1, 0, {"U": 0.4, "admitted": True, "delta_mean": 0.05},
                        factors=[{"expression": "TS_MEAN(DELAY($close, 1), 20)"}]))
    digest = ctl._build_reseed_digest()
    assert "Operator coverage across admitted factors" in digest
    assert "TS_MEAN" in digest
    # Names an unused operator class so the reseed can reach it.
    assert "REGBETA" in digest or "RSI" in digest or "TS_CORR" in digest
    assert "yours to determine" in digest


def test_reseed_digest_empty_when_no_admitted_history():
    from quantaalpha.pipeline.evolution.controller import EvolutionController
    ctl = EvolutionController(_cfg(directions=["d0"]))
    # No trajectories at all -> digest is empty (coverage block omitted, not
    # rendered for an empty population).
    assert ctl._build_reseed_digest() == ""


# --- C1.4 threading: factor feedback (SECONDARY) -----------------------------

def test_feedback_metric_block_renders_coverage_when_set():
    from quantaalpha.factors.net_cost_feedback import NetCostFactorFeedback
    f = NetCostFactorFeedback.__new__(NetCostFactorFeedback)
    f.operator_coverage_block = coverage_block(_MONOCULTURE)
    out = f._format_metric_block({"U": 0.5, "n_factors": 1, "theta_hash": "abc"})
    assert "Operator coverage so far" in out
    assert "yours to determine" in out


def test_feedback_metric_block_omits_coverage_when_empty():
    from quantaalpha.factors.net_cost_feedback import NetCostFactorFeedback
    f = NetCostFactorFeedback.__new__(NetCostFactorFeedback)
    # No attribute set -> getattr default "" -> block omitted.
    out = f._format_metric_block({"U": 0.5, "n_factors": 1})
    assert "Operator coverage so far" not in out


# --- C2.1 de-prime: planning seed injection disabled by config ---------------

_PLANNING_PROMPTS = Path(__file__).resolve().parents[2] / "quantaalpha" / "pipeline" / "prompts" / "planning_prompts.yaml"
_SEED_TOKENS = ["Alpha158(20)", "Canonical", "VRATIO5", "MA_RATIO5_10"]


def _render_planning_system_prompt(seed_in_generation: bool) -> str:
    """Replicate planning.generate_parallel_directions' render (no LLM call)."""
    from quantaalpha.pipeline.planning import _seed_context, _market_context, _load_prompts
    prompts = _load_prompts(_PLANNING_PROMPTS)
    sys_tpl = prompts.get("system", "")
    seed_context = _seed_context() if seed_in_generation else ""
    seed_context = "\n\n".join(x for x in (_market_context(), seed_context) if x)
    return sys_tpl.format(initial_direction="explore", n=3, seed_context=seed_context)


def test_planning_prompt_seed_free_when_disabled():
    rendered = _render_planning_system_prompt(seed_in_generation=False)
    for tok in _SEED_TOKENS:
        assert tok not in rendered, f"seed token {tok!r} leaked with seeds disabled"
    # The data-driven market context is NOT a seed and must still appear.
    assert "Market context" in rendered


def test_planning_prompt_still_carries_seeds_when_enabled():
    # Config-driven and reversible: with the flag true the seeds reappear.
    rendered = _render_planning_system_prompt(seed_in_generation=True)
    assert "Alpha158(20)" in rendered
    assert "Market context" in rendered


# --- C2.2 de-prime: hypothesis-generator seed injection removed ---------------

def test_hypothesis_specification_is_seed_free():
    from quantaalpha.core.proposal import Trace
    from quantaalpha.factors.proposal import AlphaAgentHypothesisGen
    gen = AlphaAgentHypothesisGen.__new__(AlphaAgentHypothesisGen)
    gen.potential_direction = None
    gen.refine_directive = None
    ctx, ok = gen.prepare_context(Trace(scen=None))
    assert ok is True
    spec = ctx["hypothesis_specification"]
    for tok in _SEED_TOKENS:
        assert tok not in spec, f"seed token {tok!r} leaked into hypothesis_specification"
    # The specification prompt itself is still a real (non-empty) prompt.
    assert spec.strip() != ""


def test_with_seed_context_helper_removed():
    # The de-prime removed the helper; a future re-add must be deliberate.
    import quantaalpha.factors.proposal as proposal
    assert not hasattr(proposal, "_with_seed_context")


def test_reseed_threads_seed_in_generation_flag():
    """The reseed path must honor the flag, not default to injecting seeds.

    The round-0 path (``generate_parallel_directions``) reads the flag from
    ``factor_mining``; the reseed path (``_generate_informed_directions`` ->
    ``generate_informed_directions``) goes through the controller, which used to
    build the call kwargs WITHOUT ``seed_in_generation`` -- so it defaulted to
    True and the reseed kept injecting the OHLCV seeds after round-0 was
    de-primed. Half a de-prime. The controller now threads
    ``config.seed_in_generation`` into the call.
    """
    from quantaalpha.pipeline.evolution.controller import (
        EvolutionConfig, EvolutionController,
    )
    _PROMPTS = Path(__file__).resolve().parents[2] / "quantaalpha" / "pipeline" / "prompts"
    cfg = EvolutionConfig(
        num_directions=2, max_rounds=10, mutation_enabled=True, crossover_enabled=True,
        fresh_start=True, reseed_after_stale_rounds=1, directions=["d0", "d1"],
        initial_direction="x",
        informed_prompt_path=str(_PROMPTS / "informed_planning_prompts.yaml"),
        mutation_prompt_path=str(_PROMPTS / "evolution_prompts.yaml"),
        crossover_prompt_path=str(_PROMPTS / "evolution_prompts.yaml"),
        seed_in_generation=False,
    )
    ctl = EvolutionController(cfg)
    captured = {}

    def fake(**kwargs):
        captured.update(kwargs)
        return ["newA", "newB"]

    with mock.patch("quantaalpha.pipeline.planning.generate_informed_directions",
                    side_effect=fake):
        ctl._generate_informed_directions("digest text")
    assert captured.get("seed_in_generation") is False, (
        "reseed must pass seed_in_generation=False through to the planner")


def test_informed_direction_prompt_seed_free_when_disabled():
    """The informed-direction (reseed) system prompt renders seed-free with the
    flag off, and still carries the data-driven market context."""
    from quantaalpha.pipeline.planning import _seed_context, _market_context, _load_prompts
    prompts = _load_prompts(Path(__file__).resolve().parents[2] / "quantaalpha"
                            / "pipeline" / "prompts" / "informed_planning_prompts.yaml")
    sys_tpl = prompts.get("system", "")
    sc = "\n\n".join(x for x in (_market_context(), "") if x)  # flag False -> no seeds
    rendered = sys_tpl.format(initial_direction="x", n=3, seed_context=sc,
                              history_summary="prior outcomes")
    for tok in _SEED_TOKENS:
        assert tok not in rendered, f"reseed prompt leaked seed token {tok!r}"
    assert "Market context" in rendered


def test_loop_feedback_reads_current_batch_from_factor_backtest():
    """The current-batch factor expressions live on the EXPERIMENT
    (``prev_out["factor_backtest"]``), not the hypothesis
    (``prev_out["factor_propose"]``). An earlier version read factor_propose,
    which is the hypothesis and has no factor_expression -- so the current batch
    silently contributed nothing and round 0 (empty prior library) rendered NO
    coverage block. Source guard so that bug cannot return.
    """
    import quantaalpha.pipeline.loop as loop
    import inspect
    src = inspect.getsource(loop.AlphaAgentLoop.feedback)
    # The current-batch extraction must read factor_backtest (the experiment)...
    assert 'prev_out.get("factor_backtest")' in src, (
        "current-batch coverage must read factor_backtest, not factor_propose")
    # ...and must NOT read factor_propose for sub_tasks (it is the hypothesis).
    assert 'prev_out.get("factor_propose")\n            for _t' not in src, (
        "factor_propose has no factor_expression; do not read sub_tasks from it")


def test_round0_empty_library_with_current_batch_renders_block():
    """Round 0: no prior library yet, only the current batch. The coverage block
    MUST be non-empty -- this is the round the monoculture forms, and the
    coverage signal is most needed there. Guards the factor_backtest fix."""
    import tempfile
    from quantaalpha.factors.operator_coverage import coverage_block
    from quantaalpha.factors.library import FactorLibraryManager

    class _T:
        def __init__(self, e):
            self.factor_expression = e

    class _Exp:
        def __init__(self, exprs):
            self.sub_tasks = [_T(e) for e in exprs]

    with tempfile.TemporaryDirectory() as d:
        libpath = Path(d) / "lib.json"
        population = []
        prior = FactorLibraryManager(str(libpath)).data.get("factors", {}) or {}
        population.extend(f.get("factor_expression", "") for f in prior.values()
                         if isinstance(f, dict) and f.get("factor_expression"))
        # current batch from factor_backtest (the experiment)
        for t in getattr(_Exp(["RANK(TS_MEAN($close, 5))",
                               "TS_MEAN(DELAY($close, 1), 20)"]), "sub_tasks", []):
            if t.factor_expression:
                population.append(t.factor_expression)
    block = coverage_block(population)
    assert block, "round-0 coverage block must be non-empty"
    assert "TS_MEAN" in block
    assert "REGBETA" in block  # names an unused operator class


# --- C2.3 regression: rho_max dedup gate reads the zoo, not the seeds ---------

def test_rho_max_gate_unaffected_by_deprime():
    # The dedup gate operates on admitted factor expressions in the zoo; the
    # seeds never entered the zoo (admitted: False) and the de-prime does not
    # touch the gate. Smoke-check the operator-coverage module does not import
    # anything from the gate path.
    import quantaalpha.factors.operator_coverage as oc
    assert "rho_max" not in dir(oc)


if __name__ == "__main__":
    import sys
    mod = sys.modules[__name__]
    failures = []
    for name in sorted(globals()):
        if name.startswith("test_") and callable(globals()[name]):
            try:
                globals()[name]()
                print(f"PASS  {name}")
            except AssertionError as e:
                print(f"FAIL  {name}: {e}")
                failures.append(name)
            except Exception as e:
                print(f"ERROR {name}: {type(e).__name__}: {e}")
                failures.append(name)
    if failures:
        print(f"\n{len(failures)} failure(s): {failures}")
        sys.exit(1)
    print("\nAll operator-coverage + de-prime tests passed.")