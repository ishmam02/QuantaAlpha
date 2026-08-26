"""
Model workflow with session control.
"""

import time
import pandas as pd
from typing import Any

from quantaalpha.pipeline.settings import BaseFacSetting
from quantaalpha.core.developer import Developer
from quantaalpha.core.proposal import (
    Hypothesis2Experiment,
    HypothesisExperiment2Feedback,
    HypothesisGen,  
    Trace,
)
from quantaalpha.core.scenario import Scenario
from quantaalpha.core.utils import import_class
from quantaalpha.log import logger
from quantaalpha.log.time import measure_time
from quantaalpha.utils.workflow import LoopBase, LoopMeta
from quantaalpha.core.exception import FactorEmptyError
import threading


import datetime
import pickle
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from tqdm.auto import tqdm

from quantaalpha.core.exception import CoderError
from quantaalpha.log import logger
from functools import wraps

# Decorator: check stop_event before invoking the function

def stop_event_check(func):
    @wraps(func)
    def wrapper(self, *args, **kwargs):
        if STOP_EVENT is not None and STOP_EVENT.is_set():
            raise Exception("Operation stopped due to stop_event flag.")
        return func(self, *args, **kwargs)
    return wrapper


class AlphaAgentLoop(LoopBase, metaclass=LoopMeta):
    skip_loop_error = (FactorEmptyError,)
    
    @measure_time
    def __init__(
        self,
        PROP_SETTING: BaseFacSetting,
        potential_direction,
        stop_event: threading.Event,
        use_local: bool = True,
        strategy_suffix: str = "",
        evolution_phase: str = "original",
        trajectory_id: str = "",
        parent_trajectory_ids: list = None,
        direction_id: int = 0,
        round_idx: int = 0,
        quality_gate_config: dict = None,
        refine_mode: bool = False,
        refine_directive: dict = None,
        parent_prefix: dict = None,
        crossover_parents: dict = None,
        refine_factors_block: str = "",
        revise_hypothesis_block: str = "",
        crossover_strength_block: str = "",
    ):
        with logger.tag("init"):
            self.use_local = use_local
            # Store initial direction for factor provenance
            self.potential_direction = potential_direction

            # Evolution-related attributes
            self.strategy_suffix = strategy_suffix
            self.evolution_phase = evolution_phase  # original / mutation / crossover
            self.trajectory_id = trajectory_id
            self.parent_trajectory_ids = parent_trajectory_ids or []
            self.direction_id = direction_id
            self.round_idx = round_idx  # 0=original, 1=mutation, 2=crossover, ...

            # Diagnose-and-refine mutation (Eq. 6). Only set when a
            # RefinementOperator diagnosed the parent; absent on orthogonal
            # mutation tasks, so those stay byte-identical to before.
            # ``refine_directive`` is the structured
            # verdict (what to fix); ``parent_prefix`` is the frozen prefix
            # (hypothesis + factor expressions to build on); the two block
            # strings are pre-rendered prompt sections handed to the steps.
            self.refine_mode = refine_mode
            self.refine_directive = refine_directive or None
            self.parent_prefix = parent_prefix or None
            self.crossover_parents = crossover_parents or None
            self.refine_factors_block = refine_factors_block or ""
            self.revise_hypothesis_block = revise_hypothesis_block or ""
            # Crossover (Eq. 7): the two parents' validated strengths, rendered
            # as constructor inspiration ("draw ideas from, do not copy"). Empty
            # on refine/orthogonal tasks; threaded onto the factor constructor
            # below, parallel to ``refine_factors_block``.
            self.crossover_strength_block = crossover_strength_block or ""

            # Quality gate config
            self.quality_gate_config = quality_gate_config or {}

            # For trajectory collection
            self._last_hypothesis = None
            self._last_experiment = None
            self._last_feedback = None
            
            logger.info(f"Initialized AlphaAgentLoop, backtest in {'local' if use_local else 'Docker'}")
            if potential_direction:
                logger.info(f"Initial direction: {potential_direction}")
            if evolution_phase != "original":
                logger.info(f"Evolution phase: {evolution_phase}, round: {round_idx}, trajectory_id: {trajectory_id}")

            consistency_enabled = self.quality_gate_config.get("consistency_enabled", False)
            complexity_enabled = self.quality_gate_config.get("complexity_enabled", True)
            redundancy_enabled = self.quality_gate_config.get("redundancy_enabled", True)
            logger.info(f"Quality gate: consistency={'on' if consistency_enabled else 'off'}, "
                       f"complexity={'on' if complexity_enabled else 'off'}, "
                       f"redundancy={'on' if redundancy_enabled else 'off'}")
                
            scen: Scenario = import_class(PROP_SETTING.scen)(use_local=use_local)
            logger.log_object(scen, tag="scenario")

            # If strategy suffix is set, append it to the direction
            effective_direction = potential_direction
            if strategy_suffix:
                effective_direction = (potential_direction or "") + "\n" + strategy_suffix
            
            self.hypothesis_generator: HypothesisGen = import_class(PROP_SETTING.hypothesis_gen)(scen, effective_direction)
            logger.log_object(self.hypothesis_generator, tag="hypothesis generator")

            # Pass consistency check config into factor constructor
            self.factor_constructor: Hypothesis2Experiment = import_class(PROP_SETTING.hypothesis2experiment)(
                consistency_enabled=consistency_enabled
            )
            logger.log_object(self.factor_constructor, tag="experiment generation")

            # Thread the refine directive into the steps that act on it. The
            # hypothesis generator decides freeze-vs-revise (refine_target);
            # the factor constructor injects the frozen-prefix block when the
            # hypothesis is frozen and only the expressions are refined. Both
            # steps read these attributes via getattr with a default, so an
            # orthogonal loop (no attrs set) is unchanged.
            if self.refine_mode and self.refine_directive:
                self.hypothesis_generator.refine_directive = self.refine_directive
                self.hypothesis_generator.parent_prefix = self.parent_prefix or {}
                self.hypothesis_generator.revise_hypothesis_block = self.revise_hypothesis_block
                self.factor_constructor.refine_directive = self.refine_directive
                self.factor_constructor.parent_prefix = self.parent_prefix or {}
                self.factor_constructor.refine_factors_block = self.refine_factors_block
                # Crossover inspiration block (recombine path). Empty on refine
                # tasks, so the constructor's existing expression-refine logic
                # is unchanged.
                self.factor_constructor.crossover_strength_block = (
                    self.crossover_strength_block
                )
                logger.info(
                    f"Refine mode: target={self.refine_directive.get('refine_target')}, "
                    f"verdict={self.refine_directive.get('verdict')}, "
                    f"weakness={self.refine_directive.get('weakness_dimension')}"
                )

            self.coder: Developer = import_class(PROP_SETTING.coder)(scen)
            logger.log_object(self.coder, tag="coder")
            
            self.runner: Developer = import_class(PROP_SETTING.runner)(scen)
            logger.log_object(self.runner, tag="runner")

            self.summarizer: HypothesisExperiment2Feedback = import_class(PROP_SETTING.summarizer)(scen)
            logger.log_object(self.summarizer, tag="summarizer")
            self.trace = Trace(scen=scen)
            
            global STOP_EVENT
            STOP_EVENT = stop_event
            super().__init__()

    @classmethod
    def load(cls, path, use_local: bool = True):
        """Load existing session."""
        instance = super().load(path)
        instance.use_local = use_local
        logger.info(f"Loaded AlphaAgentLoop, backtest in {'local' if use_local else 'Docker'}")
        return instance

    @measure_time
    @stop_event_check
    def factor_propose(self, prev_out: dict[str, Any]):
        """Propose hypothesis as the basis for factor construction."""
        with logger.tag("r"):  
            idea = self.hypothesis_generator.gen(self.trace)
            logger.log_object(idea, tag="hypothesis generation")
            self._last_hypothesis = idea
        return idea

    @measure_time
    @stop_event_check
    def factor_construct(self, prev_out: dict[str, Any]):
        """Construct multiple factors from the hypothesis."""
        with logger.tag("r"): 
            factor = self.factor_constructor.convert(prev_out["factor_propose"], self.trace)
            self._apply_operator_contract(factor)
            logger.log_object(factor.sub_tasks, tag="experiment generation")
        return factor

    def _apply_operator_contract(self, factor) -> None:
        """Check that the operator actually did structural work.

        Measured behaviour without this: EVERY refine child changes a lookback
        window, whatever weakness the diagnosis named -- one rescue and four
        regressions across two runs. A better diagnosis fed to an operator that
        answers every instruction by editing one constant produces better-
        labelled constant edits, so the diagnosis and this check have to ship
        together.

        Drops a literal-only child only when a sibling survives, so the contract
        can never empty a batch; otherwise it records the violation and lets the
        child through. The point is first to MEASURE how often this happens --
        the fraction falling over a run is the evidence that the diagnosis is
        being acted on rather than merely read.
        """
        try:
            from quantaalpha.pipeline.evolution.operator_contract import (
                ContractReport, check_crossover, check_refine, rejection_note,
            )
        except Exception:
            return
        directive = getattr(self, "refine_directive", None) or {}
        tasks = list(getattr(factor, "sub_tasks", None) or [])
        if not tasks:
            return

        # A crossover carries its two parents' expressions; a refine carries a
        # single parent prefix. Checking a crossover with the refine rule (or
        # vice versa) tests the wrong property.
        cx = getattr(self, "crossover_parents", None) or {}
        a_exprs, b_exprs = list(cx.get("a") or []), list(cx.get("b") or [])
        is_crossover = bool(a_exprs and b_exprs)
        prefix = getattr(self, "parent_prefix", None) or {}
        parents = [f.get("expression", "") for f in (prefix.get("factors") or [])
                   if f.get("expression")]
        # T5: the crossover gate (check_crossover) fires whenever the task
        # carries BOTH parents' expressions, regardless of refine_target
        # (crossover is "recombine", not "expression"). The refine gate
        # (check_refine, literal-only edit) fires only on the expression-refine
        # path. A task with neither payload has nothing to check.
        if not is_crossover and not (directive.get("refine_target") == "expression" and parents):
            return

        report = ContractReport()
        offenders = []
        for t in tasks:
            expr = getattr(t, "factor_expression", None)
            if not expr:
                continue
            if is_crossover:
                r = check_crossover(expr, a_exprs, b_exprs)
                passed = r.ok
                report.note(r)
            else:
                results = [check_refine(expr, p) for p in parents]
                passed = any(r.ok for r in results)
                report.note(results[0] if not passed
                            else results[0].__class__(True, "structural"))
            if not passed:
                offenders.append(t)

        if report.checked:
            kind = "crossover" if is_crossover else "refine"
            what = ("inherited from only ONE parent" if is_crossover
                    else "changed ONLY numeric constants")
            logger.info(
                f"operator contract [{kind}]: {report.rejected}/{report.checked} "
                f"child(ren) {what}"
            )
        if offenders and len(offenders) < len(tasks):
            keep = [t for t in tasks if t not in offenders]
            factor.sub_tasks = keep
            logger.info(
                f"operator contract: dropped {len(offenders)} literal-only "
                f"child(ren); {len(keep)} structural sibling(s) remain"
            )
        elif offenders:
            logger.info(
                "operator contract: ALL children were literal-only; keeping them "
                "rather than emptying the batch. " + rejection_note("literal_only")
            )

    @measure_time
    @stop_event_check
    def factor_calculate(self, prev_out: dict[str, Any]):
        """Compute factor values from factor expressions."""
        with logger.tag("d"):  # develop
            factor = self.coder.develop(prev_out["factor_construct"])
            logger.log_object(factor.sub_workspace_list, tag="coder result")
        return factor
    

    @measure_time
    @stop_event_check
    def factor_backtest(self, prev_out: dict[str, Any]):
        """Run backtest for factors."""
        with logger.tag("ef"):  # evaluate and feedback
            logger.info(f"Start factor backtest (Local: {self.use_local})")
            exp = self.runner.develop(prev_out["factor_calculate"], use_local=self.use_local)
            if exp is None:
                logger.error(f"Factor extraction failed.")
                raise FactorEmptyError("Factor extraction failed.")
            logger.log_object(exp, tag="runner result")
            self._last_experiment = exp
        return exp

    @measure_time
    @stop_event_check
    def feedback(self, prev_out: dict[str, Any]):
        # Operator-coverage measurement for the within-mechanism generator. The
        # construct LLM reads ``hypothesis_and_feedback`` (the channel
        # ``generate_feedback`` fills) in the SAME user prompt that already lists
        # every declared operator via ``function_lib_description`` -- so it can SEE
        # REGBETA/RSI/MACD/COUNT but has no signal about which the population has
        # exhausted. The block below puts that measurement beside the menu: the
        # library-so-far (prior rounds, read before this batch is saved) plus the
        # current batch's factors. Measurement only -- the block names the unused
        # operators as the LOCATION of the convergence and closes "yours to
        # determine", prescribing no remedy and carrying no market prior.
        #
        # ``library_path`` is computed here (hoisted from the save block below) so
        # the population can be read and so the save reuses the same path.
        import os as _os
        from pathlib import Path as _Path
        from quantaalpha.factors.operator_coverage import coverage_block as _coverage_block
        _project_root = _Path(__file__).resolve().parent.parent.parent
        _library_suffix = _os.environ.get("FACTOR_LIBRARY_SUFFIX", "")
        _library_filename = (
            f"all_factors_library_{_library_suffix}.json" if _library_suffix
            else "all_factors_library.json"
        )
        _factorlib_dir = _project_root / "data" / "factorlib"
        library_path = _factorlib_dir / _library_filename
        population_exprs: list[str] = []
        try:
            from quantaalpha.factors.library import FactorLibraryManager as _FLM
            _prior = _FLM(str(library_path)).data.get("factors", {}) or {}
            population_exprs.extend(
                f.get("factor_expression", "") for f in _prior.values()
                if isinstance(f, dict) and f.get("factor_expression")
            )
        except Exception:
            logger.exception("could not read prior factor library for coverage; using current batch only")
        try:
            # The CURRENT batch's factor expressions live on the EXPERIMENT, not
            # the hypothesis: ``prev_out["factor_propose"]`` is the hypothesis
            # (no factor_expression), while ``prev_out["factor_backtest"]`` is
            # the runner-developed experiment whose ``sub_tasks[*].factor_expression``
            # ``add_factors_from_experiment`` reads just below. Without this the
            # current batch would be missing from the population and round 0
            # (empty prior library) would render NO coverage block -- the very
            # round the monoculture forms.
            _exp = prev_out.get("factor_backtest")
            for _t in getattr(_exp, "sub_tasks", []) or []:
                _e = getattr(_t, "factor_expression", "")
                if _e:
                    population_exprs.append(_e)
        except Exception:
            logger.exception("could not read current-batch factor expressions for coverage")
        try:
            _cov = _coverage_block(population_exprs)
        except Exception:
            logger.exception("operator-coverage block failed for feedback; omitting")
            _cov = ""
        # Per-process summarizer instance under parallel execution, so the
        # attribute is local to this worker and never crosses arms.
        try:
            self.summarizer.operator_coverage_block = _cov
        except Exception:
            pass

        feedback = self.summarizer.generate_feedback(prev_out["factor_backtest"], prev_out["factor_propose"], self.trace)
        with logger.tag("ef"):  # evaluate and feedback
            logger.log_object(feedback, tag="feedback")
        self.trace.hist.append((prev_out["factor_propose"], prev_out["factor_backtest"], feedback))
        
        self._last_feedback = feedback

        # Auto-save factors to unified factor library
        try:
            from pathlib import Path
            from quantaalpha.factors.library import FactorLibraryManager

            experiment_id = "unknown"
            if hasattr(self, 'session_folder') and self.session_folder:
                parts = Path(self.session_folder).parts
                for part in parts:
                    if part.startswith("202") and len(part) > 10:
                        experiment_id = part
                        break

            round_number = self.round_idx

            hypothesis_text = None
            if prev_out.get("factor_propose"):
                hypothesis_text = str(prev_out["factor_propose"])

            planning_direction = getattr(self, 'potential_direction', None)
            user_initial_direction = getattr(self, 'user_initial_direction', None)

            evolution_phase = getattr(self, 'evolution_phase', 'original')
            trajectory_id = getattr(self, 'trajectory_id', '')
            parent_trajectory_ids = getattr(self, 'parent_trajectory_ids', [])

            # Factor library path is hoisted above (computed before the feedback
            # call so the operator-coverage population could read the prior
            # library); reuse it here and ensure the directory exists.
            _factorlib_dir.mkdir(parents=True, exist_ok=True)
            manager = FactorLibraryManager(str(library_path))
            manager.add_factors_from_experiment(
                experiment=prev_out["factor_backtest"],
                experiment_id=experiment_id,
                round_number=round_number,
                hypothesis=hypothesis_text,
                feedback=feedback,
                initial_direction=planning_direction,
                user_initial_direction=user_initial_direction,
                planning_direction=planning_direction,
                evolution_phase=evolution_phase,
                trajectory_id=trajectory_id,
                parent_trajectory_ids=parent_trajectory_ids,
            )
            # Two artefacts, deliberately distinct: the full library is the
            # TRIAL RECORD (every factor mined, which the ledger and any later
            # multiple-testing correction need), and the _zoo file is the
            # effective-alpha repository F_zoo actually used for combination and
            # relative ranking. With the absolute floors removed these coincide
            # except for factors that failed to produce a signal, but they are
            # different objects and the head-to-head consumes the zoo.
            zoo_path = library_path.with_name(library_path.stem + "_zoo.json")
            # The zoo is the ledger's active repository -- admissions minus
            # evictions, replayed in order -- which is what the combiner/book
            # actually run on. NOT the library's `admitted` flag: that is set on
            # admission and never cleared on eviction, so selecting on it
            # re-exports factors the cap/prune/replace paths removed and the
            # deliverable overcounts the live zoo (see qa-zoo-json-contaminated).
            # replay_repository is the same source the runner rehydrates from on
            # resume, so the _zoo.json matches the live zoo exactly.
            from quantaalpha.eval.ledger import replay_repository
            n_zoo = manager.write_zoo_subset(zoo_path, replay_repository(self.runner.ledger.path))
            logger.info(
                f"Saved factors to library: {library_path} (phase={evolution_phase}); "
                f"zoo subset: {zoo_path} ({n_zoo} factor(s))"
            )
        except Exception as e:
            logger.warning(f"Failed to save factors to library: {e}")
    
    def _get_trajectory_data(self) -> dict[str, Any]:
        """
        Get trajectory data for the current round (used by evolution controller).
        Method name is prefixed with underscore so the workflow system does not treat it as a step.
        Returns:
            Dict with hypothesis, experiment, feedback, etc.
        """
        return {
            "hypothesis": self._last_hypothesis,
            "experiment": self._last_experiment,
            "feedback": self._last_feedback,
            "direction_id": self.direction_id,
            "evolution_phase": self.evolution_phase,
            "trajectory_id": self.trajectory_id,
            "parent_trajectory_ids": self.parent_trajectory_ids,
            "loop_idx": self.loop_idx,
            "round_idx": self.round_idx,
        }




class BacktestLoop(LoopBase, metaclass=LoopMeta):
    skip_loop_error = (FactorEmptyError,)
    @measure_time
    def __init__(self, PROP_SETTING: BaseFacSetting, factor_path=None):
        with logger.tag("init"):

            self.factor_path = factor_path

            scen: Scenario = import_class(PROP_SETTING.scen)()
            logger.log_object(scen, tag="scenario")

            self.hypothesis_generator: HypothesisGen = import_class(PROP_SETTING.hypothesis_gen)(scen)
            logger.log_object(self.hypothesis_generator, tag="hypothesis generator")

            self.factor_constructor: Hypothesis2Experiment = import_class(PROP_SETTING.hypothesis2experiment)(factor_path=factor_path)
            logger.log_object(self.factor_constructor, tag="experiment generation")

            self.coder: Developer = import_class(PROP_SETTING.coder)(scen, with_feedback=False, with_knowledge=False, knowledge_self_gen=False)
            logger.log_object(self.coder, tag="coder")
            
            self.runner: Developer = import_class(PROP_SETTING.runner)(scen)
            logger.log_object(self.runner, tag="runner")

            self.summarizer: HypothesisExperiment2Feedback = import_class(PROP_SETTING.summarizer)(scen)
            logger.log_object(self.summarizer, tag="summarizer")
            self.trace = Trace(scen=scen)
            super().__init__()

    def factor_propose(self, prev_out: dict[str, Any]):
        """
        Market hypothesis on which factors are built
        """
        with logger.tag("r"):  
            idea = self.hypothesis_generator.gen(self.trace)
            logger.log_object(idea, tag="hypothesis generation")
        return idea
        

    @measure_time
    def factor_construct(self, prev_out: dict[str, Any]):
        """
        Construct a variety of factors that depend on the hypothesis
        """
        with logger.tag("r"): 
            factor = self.factor_constructor.convert(prev_out["factor_propose"], self.trace)
            logger.log_object(factor.sub_tasks, tag="experiment generation")
        return factor

    @measure_time
    def factor_calculate(self, prev_out: dict[str, Any]):
        """
        Debug factors and calculate their values
        """
        with logger.tag("d"):  # develop
            factor = self.coder.develop(prev_out["factor_construct"])
            logger.log_object(factor.sub_workspace_list, tag="coder result")
        return factor
    

    @measure_time
    def factor_backtest(self, prev_out: dict[str, Any]):
        """
        Conduct Backtesting
        """
        with logger.tag("ef"):  # evaluate and feedback
            exp = self.runner.develop(prev_out["factor_calculate"])
            if exp is None:
                logger.error(f"Factor extraction failed.")
                raise FactorEmptyError("Factor extraction failed.")
            logger.log_object(exp, tag="runner result")
        return exp

    @measure_time
    def stop(self, prev_out: dict[str, Any]):
        exit(0)
