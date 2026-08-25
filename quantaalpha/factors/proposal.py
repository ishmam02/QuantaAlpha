import json
from pathlib import Path
from typing import List, Tuple

from jinja2 import Environment, StrictUndefined

from quantaalpha.factors.coder.factor import FactorExperiment, FactorTask
from quantaalpha.components.proposal import FactorHypothesis2Experiment, FactorHypothesisGen
from quantaalpha.core.prompts import Prompts
from quantaalpha.core.proposal import Hypothesis, Scenario, Trace
from quantaalpha.core.experiment import Experiment
from quantaalpha.factors.experiment import QlibFactorExperiment
from quantaalpha.llm.client import APIBackend, robust_json_parse
import os
import pandas as pd
from quantaalpha.log import logger
from quantaalpha.factors.regulator.factor_regulator import FactorRegulator

DEFAULT_HISTORY_LIMIT = 6
MIN_HISTORY_LIMIT = 1
# How many times to re-prompt the hypothesis LLM when its output is unparseable
# (a code block, an empty string, malformed JSON). ``robust_json_parse`` already
# repairs most malformed JSON; this bound covers a response that contains no
# JSON object at all -- a stochastic lapse (one 2026-08-23 hypothesis-revise call
# returned a full Python function). Re-prompting recovers almost all of these;
# the bad response is a random lapse, not a deterministic one.
MAX_PARSE_RETRIES = 3


def render_hypothesis_and_feedback(prompt_dict, trace: Trace, history_limit: int = DEFAULT_HISTORY_LIMIT) -> str:
    """Render hypothesis_and_feedback with configurable history limit."""
    if len(trace.hist) > 0:
        limited_trace = Trace(scen=trace.scen)
        limited_trace.hist = trace.hist[-history_limit:] if history_limit > 0 else trace.hist
        return (
            Environment(undefined=StrictUndefined)
            .from_string(prompt_dict["hypothesis_and_feedback"])
            .render(trace=limited_trace)
        )
    else:
        return "No previous hypothesis and feedback available since it's the first round."


def is_input_length_error(error_msg: str) -> bool:
    """Check if error is due to input length limit."""
    error_indicators = [
        "input length",
        "context length", 
        "maximum context",
        "token limit",
        "InvalidParameter",
        "Range of input length",
        "max_tokens",
        "too long"
    ]
    error_str = str(error_msg).lower()
    return any(indicator.lower() in error_str for indicator in error_indicators)


QlibFactorHypothesis = Hypothesis
qa_prompt_dict = Prompts(file_path=Path(__file__).parent / "prompts" / "proposal.yaml")


# The Alpha158(20) seed context used to be appended to the hypothesis
# specification here (``_with_seed_context``). Removed 2026-08-23: those OHLCV
# seeds primed a TS_MEAN/TS_SUM formula the construct copied verbatim and
# anchored round-0 directions to price-volume microstructure (operator
# monoculture). Direction-planning seed injection is also disabled via
# ``seed_in_generation: false`` in the configs; the data-driven
# ``_market_context()`` still gives the generator its market context.


class AlphaAgentHypothesis(Hypothesis):
    """
    AlphaAgentHypothesis extends the Hypothesis class to include a potential_direction,
    which represents the initial idea or starting point for the hypothesis.
    """

    def __init__(
        self,
        hypothesis: str,
        concise_observation: str,
        concise_justification: str,
        concise_knowledge: str,
        concise_specification: str,
        expected_ic_sign: str = "",
    ) -> None:
        super().__init__(
            hypothesis,
            "",
            "",
            concise_observation,
            concise_justification,
            concise_knowledge,
        )
        self.concise_specification = concise_specification
        # The PRE-REGISTERED direction. The hypothesis prompt demands "EXACTLY
        # ONE of positive or negative" and tells the model it "will be checked
        # against the realized RankIC" -- but until now nothing parsed it, so
        # the commitment was requested, promised a test, and silently dropped.
        # A mechanism that predicts no direction is not a mechanism, and one
        # whose direction is contradicted by measurement has been falsified.
        self.expected_ic_sign = (expected_ic_sign or "").strip().lower()
        
    def __str__(self) -> str:
        return f"""Hypothesis: {self.hypothesis}
                Concise Observation: {self.concise_observation}
                Concise Justification: {self.concise_justification}
                Concise Knowledge: {self.concise_knowledge}
                concise Specification: {self.concise_specification}
                Expected IC Sign: {self.expected_ic_sign}
                """

base_prompt_dict = Prompts(file_path=Path(__file__).parent / "prompts" / "prompts.yaml")

class QlibFactorHypothesisGen(FactorHypothesisGen):
    def __init__(self, scen: Scenario) -> Tuple[dict, bool]:
        super().__init__(scen)

    def prepare_context(self, trace: Trace) -> Tuple[dict, bool]:
        hypothesis_and_feedback = (
            (
                Environment(undefined=StrictUndefined)
                .from_string(base_prompt_dict["hypothesis_and_feedback"])
                .render(trace=trace)
            )
            if len(trace.hist) > 0
            else "No previous hypothesis and feedback available since it's the first round."
        )
        context_dict = {
            "hypothesis_and_feedback": hypothesis_and_feedback,
            "RAG": None,
            "hypothesis_output_format": base_prompt_dict["hypothesis_output_format"],
            "hypothesis_specification": base_prompt_dict["factor_hypothesis_specification"],
        }
        return context_dict, True

    def convert_response(self, response: str) -> Hypothesis:
        response_dict = robust_json_parse(response)
        hypothesis = QlibFactorHypothesis(
            hypothesis=response_dict.get("hypothesis", ""),
            reason=response_dict.get("reason", ""),
            concise_reason=response_dict.get("concise_reason", ""),
            concise_observation=response_dict.get("concise_observation", ""),
            concise_justification=response_dict.get("concise_justification", ""),
            concise_knowledge=response_dict.get("concise_knowledge", ""),
        )
        return hypothesis


class QlibFactorHypothesis2Experiment(FactorHypothesis2Experiment):
    def prepare_context(self, hypothesis: Hypothesis, trace: Trace) -> Tuple[dict | bool]:
        scenario = trace.scen.get_scenario_all_desc()
        experiment_output_format = base_prompt_dict["factor_experiment_output_format"]

        hypothesis_and_feedback = (
            (
                Environment(undefined=StrictUndefined)
                .from_string(base_prompt_dict["hypothesis_and_feedback"])
                .render(trace=trace)
            )
            if len(trace.hist) > 0
            else "No previous hypothesis and feedback available since it's the first round."
        )

        experiment_list: List[FactorExperiment] = [t[1] for t in trace.hist]

        factor_list = []
        for experiment in experiment_list:
            factor_list.extend(experiment.sub_tasks)

        return {
            "target_hypothesis": str(hypothesis),
            "scenario": scenario,
            "hypothesis_and_feedback": hypothesis_and_feedback,
            "experiment_output_format": experiment_output_format,
            "target_list": factor_list,
            "RAG": None,
        }, True

    def convert_response(self, response: str, trace: Trace) -> FactorExperiment:
        response_dict = robust_json_parse(response)
        tasks = []

        for factor_name in response_dict:
            factor_data = response_dict.get(factor_name, {})
            if not isinstance(factor_data, dict):
                continue
            description = factor_data.get("description", "")
            formulation = factor_data.get("formulation", "")
            # expression = factor_data.get("expression", "")
            variables = factor_data.get("variables", {})
            tasks.append(
                FactorTask(
                    factor_name=factor_name,
                    factor_description=description,
                    factor_formulation=formulation,
                    # factor_expression=expression,
                    variables=variables,
                )
            )

        exp = QlibFactorExperiment(tasks)
        exp.based_experiments = [QlibFactorExperiment(sub_tasks=[])] + [t[1] for t in trace.hist if t[2]]

        unique_tasks = []

        for task in tasks:
            duplicate = False
            for based_exp in exp.based_experiments:
                for sub_task in based_exp.sub_tasks:
                    if task.factor_name == sub_task.factor_name:
                        duplicate = True
                        break
                if duplicate:
                    break
            if not duplicate:
                unique_tasks.append(task)

        exp.tasks = unique_tasks
        return exp



qa_prompt_dict = Prompts(file_path=Path(__file__).parent / "prompts" / "prompts.yaml")


def _factors_per_hypothesis() -> int:
    """How many factors one hypothesis should yield, from the run config.

    Read through the environment because these classes are instantiated deep in
    the RD-Agent loop with no handle on the run config, and are pickled (see the
    note above about prompt_dict). `factor_mining.main` exports it.

    This knob was DEAD until now: `factors_per_hypothesis` was read only by
    `expected_factor_count`, a budget estimator that nothing calls, while the
    prompt hardcoded "2-3 Factors per Generation". Setting it to 1 in the config
    therefore changed nothing -- a live run was measured emitting 3 factors per
    hypothesis with the config asking for 1.
    """
    import os
    try:
        return max(int(os.environ.get("QA_FACTORS_PER_HYPOTHESIS", "1")), 1)
    except (TypeError, ValueError):
        return 1

# prompt_dict not as attribute: class instance is pickled later, prompt_dict cannot be pickled
class AlphaAgentHypothesisGen(FactorHypothesisGen):
    def __init__(self, scen: Scenario, potential_direction: str=None) -> Tuple[dict, bool]:
        super().__init__(scen)
        self.potential_direction = potential_direction

    def prepare_context(self, trace: Trace, history_limit: int = DEFAULT_HISTORY_LIMIT) -> Tuple[dict, bool]:

        # Refine-mode, hypothesis layer (refine_target=hypothesis): the premise
        # itself is the fault, so the LLM must REVISE it rather than propose a
        # new one. Inject the pre-rendered revise block (parent hypothesis +
        # diagnosis) as the context, overriding the usual direction/feedback
        # rendering. (The frozen-hypothesis / expression-refine case never
        # reaches here -- gen() short-circuits before the LLM call.)
        _rd = getattr(self, "refine_directive", None)
        if _rd and _rd.get("refine_target") == "hypothesis":
            block = getattr(self, "revise_hypothesis_block", "") or ""
            directive = _rd.get("directive_text", "")
            if block:
                hypothesis_and_feedback = f"{block}\n\n### Refinement directive\n{directive}"
            else:
                hypothesis_and_feedback = (
                    f"Revise the parent hypothesis per this directive:\n{directive}"
                )
        elif len(trace.hist) > 0:
            hypothesis_and_feedback = render_hypothesis_and_feedback(
                qa_prompt_dict, trace, history_limit
            )

        elif self.potential_direction is not None:
            hypothesis_and_feedback = (
                Environment(undefined=StrictUndefined)
                .from_string(qa_prompt_dict["potential_direction_transformation"])
                .render(potential_direction=self.potential_direction)
            ) #
        else:
            hypothesis_and_feedback = "No previous hypothesis and feedback available since it's the first round. You are encouraged to propose an innovative hypothesis that diverges significantly from existing perspectives."
            
        context_dict = {
            "hypothesis_and_feedback": hypothesis_and_feedback,
            "RAG": None,
            "hypothesis_output_format": qa_prompt_dict["hypothesis_output_format"],
            "hypothesis_specification": qa_prompt_dict["factor_hypothesis_specification"],
        }
        return context_dict, True

    def convert_response(self, response: str) -> AlphaAgentHypothesis:
        """
        Convert LLM JSON to AlphaAgentHypothesis; use default empty string for missing fields to avoid KeyError.
        """
        response_dict = robust_json_parse(response)
        # Use get to avoid KeyError on missing fields
        hypothesis = AlphaAgentHypothesis(
            hypothesis=response_dict.get("hypothesis", ""),
            concise_observation=response_dict.get("concise_observation", ""),
            concise_knowledge=response_dict.get("concise_knowledge", ""),
            concise_justification=response_dict.get("concise_justification", ""),
            concise_specification=response_dict.get("concise_specification", ""),
            expected_ic_sign=response_dict.get("expected_ic_sign", ""),
        )
        return hypothesis

    def _warn_if_refine_drifted(self, hypothesis: "AlphaAgentHypothesis") -> None:
        """Soft guard for the hypothesis-revise path (refine_target=hypothesis).

        The revise contract is prompt text only -- the JSON schema is the
        standard hypothesis one, so nothing structurally forces the LLM to
        revise rather than restart orthogonally. If the emitted hypothesis
        shares essentially no wording with its parent, it is an orthogonal
        restart dressed as a refinement. Surface that in the log so the
        failure mode is observable (a refine loop that silently restarts would
        otherwise look like it is exploiting). Observational only: it does not
        block or rewrite the output.
        """
        _rd = getattr(self, "refine_directive", None)
        if not _rd or _rd.get("refine_target") != "hypothesis":
            return
        pp = getattr(self, "parent_prefix", {}) or {}
        parent = pp.get("hypothesis") or _rd.get("parent_hypothesis") or ""
        child = getattr(hypothesis, "hypothesis", "") or ""
        if not parent or not child:
            return
        pw, cw = set(parent.lower().split()), set(child.lower().split())
        if not pw:
            return
        overlap = len(pw & cw) / len(pw)
        if overlap < 0.08:
            logger.warning(
                f"refine[hypothesis] produced an orthogonal restart (word "
                f"overlap {overlap:.2f} with the parent hypothesis) -- the "
                f"directive asked for a revision; the LLM may have ignored it."
            )

    def _gen_with_parse_retry(self, system_prompt: str, user_prompt: str,
                              json_flag: bool) -> "AlphaAgentHypothesis":
        """Call the hypothesis LLM and parse; re-prompt on malformed output.

        The model occasionally returns a code block or an empty string instead
        of a JSON object (one hypothesis-revise call returned a full Python
        function, see the 2026-08-23 push smoke). ``robust_json_parse`` repairs
        most malformed JSON, but a response with no JSON object at all raises.
        Re-prompting a bounded number of times recovers almost all of these --
        the bad response is a stochastic lapse, not a deterministic one. Raises
        the last parse error after ``MAX_PARSE_RETRIES`` so the caller can
        degrade safely.
        """
        last_exc: Exception | None = None
        for attempt in range(MAX_PARSE_RETRIES):
            try:
                resp = APIBackend().build_messages_and_create_chat_completion(
                    user_prompt, system_prompt, json_mode=json_flag)
                return self.convert_response(resp)
            except (json.JSONDecodeError, ValueError) as e:
                last_exc = e
                logger.warning(
                    f"hypothesis-gen returned unparseable output "
                    f"(attempt {attempt + 1}/{MAX_PARSE_RETRIES}: {e}); "
                    f"re-prompting...")
        assert last_exc is not None
        raise last_exc

    def _refine_hypothesis_fallback(self) -> "AlphaAgentHypothesis | None":
        """Degrade-safely fallback for a failed hypothesis-revise.

        When the hypothesis-revise LLM output is unparseable after retries, the
        safe degradation is NOT to kill the task (which loses the child and the
        round's work) but to fall back to the FROZEN parent premise -- the same
        thing the expression-refine short-circuit does. The factor constructor
        then refines the parent expression, so the child is scored as a
        degraded-but-valid expression-refine instead of dying. Returns None for
        non-refine paths (a fresh-gen failure is a real error worth surfacing,
        not something to paper over with a parent that does not exist).
        """
        _rd = getattr(self, "refine_directive", None)
        if not _rd or _rd.get("refine_target") != "hypothesis":
            return None
        pp = getattr(self, "parent_prefix", {}) or {}
        parent_hypothesis = pp.get("hypothesis") or _rd.get("parent_hypothesis") or ""
        parent_sign = str(pp.get("expected_ic_sign")
                          or _rd.get("parent_expected_ic_sign") or "").strip().lower()
        if parent_sign not in ("positive", "negative"):
            parent_sign = ""
        return AlphaAgentHypothesis(
            hypothesis=parent_hypothesis,
            concise_observation="",
            concise_justification="",
            concise_knowledge="",
            concise_specification="",
            expected_ic_sign=parent_sign,
        )

    def gen(self, trace: Trace) -> AlphaAgentHypothesis:
        """Generate hypothesis; supports dynamic history limit for input length."""
        # Refine-mode short-circuit (Eq. 6): when the diagnosis froze the
        # hypothesis (refine_target=EXPRESSION -- the cost-stall case: the
        # premise is sound, only the construction is too expensive), the
        # hypothesis generator returns the FROZEN parent hypothesis verbatim
        # and skips the LLM call entirely. The factor constructor then refines
        # the expressions while keeping this premise; the coder renders code
        # from those expressions, so the whole construction layer is refined
        # without re-proposing the premise. (The hypothesis-revise case,
        # refine_target=HYPOTHESIS, falls through to the LLM call below with the
        # revise block already injected by prepare_context.)
        _rd = getattr(self, "refine_directive", None)
        if _rd and _rd.get("refine_target") == "expression":
            pp = getattr(self, "parent_prefix", {}) or {}
            parent_hypothesis = pp.get("hypothesis") or _rd.get("parent_hypothesis") or ""
            # The concise_* fields are empty: this is a FROZEN premise, not a
            # freshly generated one, so the LLM observation/justification do not
            # apply. The factor constructor only reads ``str(hypothesis)``.
            # The premise is frozen, so its DIRECTIONAL prediction is frozen too
            # and has to travel with it. Dropping it here made every refine child
            # arrive with no stated direction, and the falsifiability gate then
            # rejected all of them -- 36 rejections in one run, silencing the one
            # operator measured to actually improve factors.
            parent_sign = str(pp.get("expected_ic_sign")
                              or _rd.get("parent_expected_ic_sign") or "").strip().lower()
            if parent_sign not in ("positive", "negative"):
                parent_sign = ""
            return AlphaAgentHypothesis(
                hypothesis=parent_hypothesis,
                concise_observation="",
                concise_justification="",
                concise_knowledge="",
                concise_specification="",
                expected_ic_sign=parent_sign,
            )

        # First-class deterministic sign-flip refine (the measured-direction
        # fix): the parent's pre-registered IC direction was OPPOSITE its
        # measured one while the edge cleared the bar (|t| >= k_sigma) -- the
        # construction has a real edge, the hypothesis just labeled the
        # direction backwards. The construction is FROZEN verbatim (the factor
        # constructor re-uses the parent factors unchanged, see the SIGN
        # short-circuit in ``FactorHypothesis2Experiment.convert``); the
        # direction is CORRECTED to the measured one; and the hypothesis records
        # the measured correction so the child's premise states the corrected
        # prediction. No LLM call -- this is the search learning from its
        # measured mistake and correcting it for free. See
        # ``diagnosis.sign_flip_directive``.
        if _rd and _rd.get("refine_target") == "sign":
            pp = getattr(self, "parent_prefix", {}) or {}
            parent_hypothesis = pp.get("hypothesis") or _rd.get("parent_hypothesis") or ""
            corrected_sign = str(pp.get("expected_ic_sign")
                                 or _rd.get("parent_expected_ic_sign") or "").strip().lower()
            if corrected_sign not in ("positive", "negative"):
                corrected_sign = ""
            correction = _rd.get("directive_text") or ""
            # Append the measured correction to the frozen premise so the
            # child's hypothesis states the corrected direction (the premise is
            # otherwise identical to the parent's). ``correction`` is
            # measurement-only prose; it never prescribes a remedy.
            hyp = (parent_hypothesis + "\n\n" + correction).strip() if correction else parent_hypothesis
            return AlphaAgentHypothesis(
                hypothesis=hyp,
                concise_observation="",
                concise_justification="",
                concise_knowledge="",
                concise_specification="",
                expected_ic_sign=corrected_sign,
            )

        history_limit = DEFAULT_HISTORY_LIMIT

        try:
            while history_limit >= MIN_HISTORY_LIMIT:
                try:
                    context_dict, json_flag = self.prepare_context(trace, history_limit)
                    system_prompt = (
                        Environment(undefined=StrictUndefined)
                        .from_string(qa_prompt_dict["hypothesis_gen"]["system_prompt"])
                        .render(
                            targets=self.targets,
                            scenario=self.scen.get_scenario_all_desc(filtered_tag="hypothesis_and_experiment"),
                            hypothesis_output_format=context_dict["hypothesis_output_format"],
                            hypothesis_specification=context_dict["hypothesis_specification"],
                        )
                    )
                    user_prompt = (
                        Environment(undefined=StrictUndefined)
                        .from_string(qa_prompt_dict["hypothesis_gen"]["user_prompt"])
                        .render(
                            targets=self.targets,
                            hypothesis_and_feedback=context_dict["hypothesis_and_feedback"],
                            RAG=context_dict["RAG"],
                            round=len(trace.hist)
                        )
                    )

                    hypothesis = self._gen_with_parse_retry(system_prompt, user_prompt, json_flag)
                    self._warn_if_refine_drifted(hypothesis)
                    return hypothesis

                except Exception as e:
                    if is_input_length_error(str(e)) and history_limit > MIN_HISTORY_LIMIT:
                        history_limit -= 1
                        logger.warning(f"Input length exceeded, retrying with history_limit={history_limit}...")
                    else:
                        raise

            # Last attempt with minimum history limit
            context_dict, json_flag = self.prepare_context(trace, MIN_HISTORY_LIMIT)
            system_prompt = (
                Environment(undefined=StrictUndefined)
                .from_string(qa_prompt_dict["hypothesis_gen"]["system_prompt"])
                .render(
                    targets=self.targets,
                    scenario=self.scen.get_scenario_all_desc(filtered_tag="hypothesis_and_experiment"),
                    hypothesis_output_format=context_dict["hypothesis_output_format"],
                    hypothesis_specification=context_dict["hypothesis_specification"],
                )
            )
            user_prompt = (
                Environment(undefined=StrictUndefined)
                .from_string(qa_prompt_dict["hypothesis_gen"]["user_prompt"])
                .render(
                    targets=self.targets,
                    hypothesis_and_feedback=context_dict["hypothesis_and_feedback"],
                    RAG=context_dict["RAG"],
                    round=len(trace.hist)
                )
            )
            hypothesis = self._gen_with_parse_retry(system_prompt, user_prompt, json_flag)
            self._warn_if_refine_drifted(hypothesis)
            return hypothesis

        except (json.JSONDecodeError, ValueError) as e:
            # The LLM returned unparseable output (a code block / empty string)
            # even after MAX_PARSE_RETRIES re-prompts. For a hypothesis-revise,
            # degrade safely to the frozen parent premise so the task produces a
            # (degraded) scored child instead of dying and losing the round --
            # the same expression-refine semantics the short-circuit uses. For a
            # fresh generation there is no parent to fall back to; a failure
            # there is a real error, so re-raise it.
            fallback = self._refine_hypothesis_fallback()
            if fallback is not None:
                logger.warning(
                    f"hypothesis-revise LLM output unparseable after "
                    f"{MAX_PARSE_RETRIES} retries ({e}); degrading to the frozen "
                    f"parent premise (expression-refine semantics) so the child "
                    f"is scored instead of lost.")
                return fallback
            raise
    
    

class EmptyHypothesisGen(FactorHypothesisGen):
    def __init__(self, scen: Scenario) -> Tuple[dict, bool]:
        super().__init__(scen)
        
    def convert_response(self, *args, **kwargs) -> AlphaAgentHypothesis: 
        return super().convert_response(*args, **kwargs)  
    
    def prepare_context(self, *args, **kwargs) -> Tuple[dict | bool]:
        return super().prepare_context(*args, **kwargs)

    def gen(self, trace: Trace) -> AlphaAgentHypothesis:

        hypothesis = AlphaAgentHypothesis(
            hypothesis="",
            concise_observation="",
            concise_justification="",
            concise_knowledge="",
            concise_specification=""
        )

        return hypothesis




class AlphaAgentHypothesis2FactorExpression(FactorHypothesis2Experiment):
    def __init__(self, *args, consistency_enabled: bool = False, **kwargs):
        super().__init__(*args, **kwargs)
        # Initialize FactorRegulator with config settings
        from quantaalpha.factors.coder.config import FACTOR_COSTEER_SETTINGS
        self.factor_regulator = FactorRegulator(
            factor_zoo_path=FACTOR_COSTEER_SETTINGS.factor_zoo_path,
            duplication_threshold=FACTOR_COSTEER_SETTINGS.duplication_threshold
        )
        
        # Initialize consistency checker if enabled
        self.consistency_enabled = consistency_enabled
        self._quality_gate = None
        
    @property
    def quality_gate(self):
        """Lazy-load FactorQualityGate."""
        if self._quality_gate is None and self.consistency_enabled:
            try:
                from quantaalpha.factors.regulator.consistency_checker import FactorQualityGate
                self._quality_gate = FactorQualityGate(
                    consistency_enabled=self.consistency_enabled,
                    complexity_enabled=True,
                    redundancy_enabled=True
                )
            except ImportError as e:
                logger.warning(f"Could not load consistency checker: {e}")
                self._quality_gate = None
        return self._quality_gate
        
    def prepare_context(self, hypothesis: Hypothesis, trace: Trace, history_limit: int = DEFAULT_HISTORY_LIMIT) -> Tuple[dict | bool]:
        scenario = trace.scen.get_scenario_all_desc()
        # Render the output schema for the requested count. The EXAMPLE teaches
        # harder than the prose: a two-slot schema beside "produce exactly 1"
        # is why a run configured for one factor per hypothesis was measured
        # emitting three.
        experiment_output_format = (
            Environment(undefined=StrictUndefined)
            .from_string(qa_prompt_dict["factor_experiment_output_format"])
            .render(factors_per_hypothesis=_factors_per_hypothesis())
        )
        function_lib_description = qa_prompt_dict['function_lib_description']
        hypothesis_and_feedback = render_hypothesis_and_feedback(
            qa_prompt_dict, trace, history_limit
        )

        experiment_list: List[FactorExperiment] = [t[1] for t in trace.hist]

        factor_list = []
        for experiment in experiment_list:
            factor_list.extend(experiment.sub_tasks)

        # Refine-mode (Eq. 6), expression layer: the hypothesis is FROZEN (the
        # generator returned the parent premise verbatim) and only the
        # construction is to be refined. Inject the pre-rendered frozen-prefix
        # block (parent factor expressions + the directive) into the feedback
        # channel so the LLM refines the parent's expressions rather than
        # inventing unrelated ones. The directive was produced deterministically
        # by diagnosis.diagnose; this block is the AlphaEvolve artifact
        # side-channel that tells the LLM *what* to fix. Only applies when
        # refine_target=expression (the cost-stall family); hypothesis-refine
        # re-derives expressions from the revised premise and has no frozen
        # expression prefix, so the block is empty and this is a no-op.
        _rd = getattr(self, "refine_directive", None)
        if _rd and _rd.get("refine_target") == "expression":
            block = getattr(self, "refine_factors_block", "") or ""
            if block:
                hypothesis_and_feedback = (
                    (hypothesis_and_feedback + "\n\n" if hypothesis_and_feedback else "")
                    + block
                )
        elif _rd and _rd.get("refine_target") == "recombine":
            # Crossover (Eq. 7): the two parents' validated strengths as
            # CONSTRUCTOR inspiration -- "draw ideas from, do not copy". The
            # hypothesis is authored fresh by the generator (this is not a
            # frozen-prefix build); the block only tells the constructor which
            # validated construction patterns to inherit ideas from.
            block = getattr(self, "crossover_strength_block", "") or ""
            if block:
                hypothesis_and_feedback = (
                    (hypothesis_and_feedback + "\n\n" if hypothesis_and_feedback else "")
                    + block
                )

        return {
            "target_hypothesis": str(hypothesis),
            "scenario": scenario,
            "hypothesis_and_feedback": hypothesis_and_feedback,
            "function_lib_description": function_lib_description,
            "experiment_output_format": experiment_output_format,
            "target_list": factor_list,
            "RAG": None,
        }, True
        
    def convert(self, hypothesis: Hypothesis, trace: Trace) -> Experiment:
        """Convert hypothesis to factor expressions; supports dynamic history limit."""
        # First-class deterministic sign-flip refine (the measured-direction
        # fix): the construction is FROZEN verbatim -- the parent's expressions
        # are re-used UNCHANGED, only the pre-registered direction was backwards
        # (corrected on the hypothesis in ``AlphaAgentHypothesisGen.gen``). So
        # build the experiment straight from the parent factor dicts with NO LLM
        # call: the search measured that its prediction was backwards and
        # corrects it, re-testing the identical construction under the corrected
        # direction. Mirrors ``BacktestHypothesis2FactorExpression.convert`` (a
        # no-LLM build from a factor list). ``trace.hist`` is empty at convert
        # time (only the feedback step appends, AFTER construct), so the dedup
        # against ``based_experiments`` is a no-op -- the frozen factors pass
        # through unchanged. See ``diagnosis.sign_flip_directive``.
        _rd = getattr(self, "refine_directive", None)
        if _rd and _rd.get("refine_target") == "sign":
            pp = getattr(self, "parent_prefix", {}) or {}
            parent_factors = pp.get("factors") or _rd.get("parent_factors") or []
            tasks = []
            for i, f in enumerate(parent_factors):
                if not isinstance(f, dict):
                    continue
                expr = f.get("expression") or f.get("factor_expression") or ""
                if not expr:
                    continue
                tasks.append(
                    FactorTask(
                        factor_name=f.get("name") or f.get("factor_name") or f"signflip_{i}",
                        factor_description=f.get("description", ""),
                        factor_formulation=f.get("formulation", ""),
                        factor_expression=expr,
                        variables=f.get("variables", ""),
                    )
                )
            exp = QlibFactorExperiment(tasks)
            exp.based_experiments = [QlibFactorExperiment(sub_tasks=[])] + [
                t[1] for t in trace.hist if t[2]]
            unique_tasks = []
            for task in tasks:
                duplicate = False
                for based_exp in exp.based_experiments:
                    for sub_task in based_exp.sub_tasks:
                        if task.factor_name == sub_task.factor_name:
                            duplicate = True
                            break
                    if duplicate:
                        break
                if not duplicate:
                    unique_tasks.append(task)
            exp.tasks = unique_tasks
            try:
                exp.hypothesis = hypothesis
            except Exception:
                logger.warning("could not attach hypothesis to sign-flip experiment; "
                               "the mechanism gate will have nothing to read")
            return exp

        history_limit = DEFAULT_HISTORY_LIMIT
        
        while history_limit >= MIN_HISTORY_LIMIT:
            try:
                return self._convert_with_history_limit(hypothesis, trace, history_limit)
            except Exception as e:
                if is_input_length_error(str(e)) and history_limit > MIN_HISTORY_LIMIT:
                    history_limit -= 1
                    logger.warning(f"Input length exceeded, retrying with history_limit={history_limit}...")
                else:
                    raise
        
        # Last attempt with minimum history limit
        return self._convert_with_history_limit(hypothesis, trace, MIN_HISTORY_LIMIT)
    
    def _convert_with_history_limit(self, hypothesis: Hypothesis, trace: Trace, history_limit: int) -> Experiment:
        """Convert with given history limit."""
        context, json_flag = self.prepare_context(hypothesis, trace, history_limit)
        system_prompt = (
            Environment(undefined=StrictUndefined)
            .from_string(qa_prompt_dict["hypothesis2experiment"]["system_prompt"])
            .render(
                targets=self.targets,
                scenario=trace.scen.background, # get_scenario_all_desc(filtered_tag="hypothesis_and_experiment"),
                experiment_output_format=context["experiment_output_format"],
                factors_per_hypothesis=_factors_per_hypothesis(),
            )
        )
        user_prompt = (
            Environment(undefined=StrictUndefined)
            .from_string(qa_prompt_dict["hypothesis2experiment"]["user_prompt"])
            .render(
                targets=self.targets,
                target_hypothesis=context["target_hypothesis"],
                hypothesis_and_feedback=context["hypothesis_and_feedback"],
                function_lib_description=context["function_lib_description"],
                target_list=context["target_list"],
                RAG=context["RAG"], 
                expression_duplication=None
            )
        )
        
        # Detect duplicated sub-expressions
        flag = False
        expression_duplication_prompt = None
        while True:
            if flag:
                break
                
            resp = APIBackend().build_messages_and_create_chat_completion(user_prompt, system_prompt, json_mode=json_flag)
            try:
                response_dict = robust_json_parse(resp)
            except json.JSONDecodeError as e:
                logger.warning(f"JSON parse failed: {e}, retrying...")
                continue
            proposed_names = []
            proposed_exprs = []
            
            for i, factor_name in enumerate(response_dict):
                factor_data = response_dict.get(factor_name, {})
                if not isinstance(factor_data, dict):
                    continue
                expr = factor_data.get("expression", "")
                description = factor_data.get("description", "")
                formulation = factor_data.get("formulation", "")
                variables = factor_data.get("variables", {})
                
                # Check if expression is parsable
                if not self.factor_regulator.is_parsable(expr):
                    logger.info(f"Failed to parse expr: {expr}, retrying...")
                    break
                
                success, eval_dict = self.factor_regulator.evaluate(expr)
                if not success:
                    break
                
                # Consistency check (if enabled)
                if self.consistency_enabled and self.quality_gate is not None:
                    try:
                        passed, feedback, results = self.quality_gate.evaluate(
                            hypothesis=str(hypothesis),
                            factor_name=factor_name,
                            factor_description=description,
                            factor_formulation=formulation,
                            factor_expression=expr,
                            variables=variables
                        )
                        
                        # Use corrected expression from consistency check if provided
                        if results.get("corrected_expression") and results["corrected_expression"] != expr:
                            logger.info(f"Consistency check corrected expression: {expr} -> {results['corrected_expression']}")
                            expr = results["corrected_expression"]
                            factor_data["expression"] = expr
                            response_dict[factor_name] = factor_data
                            
                            # Re-check corrected expression
                            if not self.factor_regulator.is_parsable(expr):
                                logger.warning(f"Corrected expression could not be parsed: {expr}")
                                break
                            success, eval_dict = self.factor_regulator.evaluate(expr)
                            if not success:
                                break
                        
                        if not passed:
                            logger.warning(f"Consistency check failed: {factor_name}, feedback: {feedback}")
                    except Exception as e:
                        logger.warning(f"Consistency check error: {e}")
                
                # If expression has problems, regenerate with feedback
                if not self.factor_regulator.is_expression_acceptable(eval_dict):
                    # Calculate ratios for feedback
                    num_all_nodes = eval_dict['num_all_nodes']
                    free_args_ratio = float(eval_dict['num_free_args']) / float(num_all_nodes) if num_all_nodes > 0 else 0.0
                    unique_vars_ratio = float(eval_dict['num_unique_vars']) / float(num_all_nodes) if num_all_nodes > 0 else 0.0
                    
                    # Get symbol length and base features count for complexity feedback
                    symbol_length = eval_dict.get('symbol_length', 0)
                    num_base_features = eval_dict.get('num_base_features', 0)
                    symbol_length_threshold = self.factor_regulator.symbol_length_threshold
                    base_features_threshold = self.factor_regulator.base_features_threshold
                    
                    feedback_item = (
                            Environment(undefined=StrictUndefined)
                            .from_string(qa_prompt_dict["expression_duplication"])
                            .render(
                                prev_expression=expr,
                                duplicated_subtree_size=eval_dict['duplicated_subtree_size'],
                            duplication_threshold=self.factor_regulator.duplication_threshold,
                            duplicated_subtree=eval_dict.get('duplicated_subtree', ''),
                            matched_alpha=eval_dict.get('matched_alpha', ''),
                            free_args_ratio=free_args_ratio,
                            num_free_args=eval_dict['num_free_args'],
                            unique_vars_ratio=unique_vars_ratio,
                            num_unique_vars=eval_dict['num_unique_vars'],
                            num_all_nodes=num_all_nodes,
                            symbol_length=symbol_length,
                            symbol_length_threshold=symbol_length_threshold,
                            num_base_features=num_base_features,
                            base_features_threshold=base_features_threshold
                            )
                        )
                    
                    if expression_duplication_prompt is not None:
                        expression_duplication_prompt = '\n\n'.join([expression_duplication_prompt, feedback_item])
                    else:
                        expression_duplication_prompt = feedback_item
                    
                    user_prompt = (
                        Environment(undefined=StrictUndefined)
                        .from_string(qa_prompt_dict["hypothesis2experiment"]["user_prompt"])
                        .render(
                            targets=self.targets,
                            target_hypothesis=context["target_hypothesis"],
                            hypothesis_and_feedback=context["hypothesis_and_feedback"],
                            function_lib_description=context["function_lib_description"],
                            target_list=context["target_list"],
                            RAG=context["RAG"], 
                            expression_duplication=expression_duplication_prompt
                        )
                    )
                    break
                else:
                    proposed_names.append(factor_name)
                    proposed_exprs.append(expr)
                    if i == len(response_dict) - 1:
                        flag = True
                    else:
                        continue
        

        # Add valid factors to the factor regulator
        self.factor_regulator.add_factor(proposed_names, proposed_exprs)
                
                
        exp = self.convert_response(resp, trace)
        # ATTACH THE HYPOTHESIS TO THE EXPERIMENT.
        #
        # `convert_response` builds the experiment from the response text alone
        # and never sees the hypothesis, so `exp.hypothesis` was unset for every
        # factor this system has ever mined. The runner reads it to record the
        # economic mechanism -- measured across every ledger and factor library
        # on disk, `mechanism` is non-empty ZERO times out of all of them.
        #
        # That was invisible while a missing mechanism only logged a warning.
        # Once it became an admission gate it rejected everything, which is how
        # it was finally noticed.
        try:
            exp.hypothesis = hypothesis
        except Exception:                      # some Experiment types are slotted
            logger.warning("could not attach hypothesis to experiment; the "
                           "mechanism gate will have nothing to read")
        return exp
    

    def convert_response(self, response: str, trace: Trace) -> FactorExperiment:
        response_dict = robust_json_parse(response)
        tasks = []

        for factor_name in response_dict:
            factor_data = response_dict.get(factor_name, {})
            if not isinstance(factor_data, dict):
                continue
            description = factor_data.get("description", "")
            formulation = factor_data.get("formulation", "")
            expression = factor_data.get("expression", "")
            variables = factor_data.get("variables", {})
            tasks.append(
                FactorTask(
                    factor_name=factor_name,
                    factor_description=description,
                    factor_formulation=formulation,
                    factor_expression=expression,
                    variables=variables,
                )
            )
            
        exp = QlibFactorExperiment(tasks)
        exp.based_experiments = [QlibFactorExperiment(sub_tasks=[])] + [t[1] for t in trace.hist if t[2]]

        unique_tasks = []

        for task in tasks:
            duplicate = False
            for based_exp in exp.based_experiments:
                for sub_task in based_exp.sub_tasks:
                    if task.factor_name == sub_task.factor_name:
                        duplicate = True
                        break
                if duplicate:
                    break
            if not duplicate:
                unique_tasks.append(task)

        exp.tasks = unique_tasks
        return exp



class BacktestHypothesis2FactorExpression(FactorHypothesis2Experiment):
    def __init__(self, factor_path, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.factor_path = factor_path
        
    def convert_response(self, *args, **kwargs) -> FactorExperiment:
        return super().convert_response(*args, **kwargs)
        
    def prepare_context(self, *args, **kwargs) -> Tuple[dict | bool]:
        return super().prepare_context(*args, **kwargs)
        
    def convert(self, hypothesis: Hypothesis, trace: Trace) -> FactorExperiment:
        if os.path.exists(self.factor_path):
            tasks = []
            factor_df = pd.read_csv(self.factor_path, usecols=["factor_name", "factor_expression"], index_col=None)
            for index, row in factor_df.iterrows():
                tasks.append(
                    FactorTask(
                        factor_name=row["factor_name"],
                        factor_description="",
                        factor_formulation="",
                        factor_expression=row["factor_expression"],
                        variables="",
                    )
                )
            
            exp = QlibFactorExperiment(tasks)
            exp.based_experiments = [QlibFactorExperiment(sub_tasks=[])] + [t[1] for t in trace.hist if t[2]]

            unique_tasks = []

            for task in tasks:
                duplicate = False
                for based_exp in exp.based_experiments:
                    for sub_task in based_exp.sub_tasks:
                        if task.factor_name == sub_task.factor_name:
                            duplicate = True
                            break
                    if duplicate:
                        break
                if not duplicate:
                    unique_tasks.append(task)

            exp.tasks = unique_tasks
            return exp
            
        else:
            raise ValueError(f"File {self.factor_csv_path} does not exist. ")
        
    