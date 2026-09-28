# 11 — The decay agent (advisory layer)

**Depends on:** 03, 09. **Blocks:** 14.

## Why

The deterministic tiers can only fire after a **sustained** breach — 30 days for
soft, 60 for hard. That is deliberately slow, because a 63-day rolling IC crosses
its own threshold constantly near the boundary and acting on a touch would retire
healthy factors. On the prior 19-factor set the touch and the trigger were **44
days apart**.

Those 30-60 days are the opportunity. An agent reading the same numbers can see a
breach forming and act earlier — and with a medium-frequency factor half-life now
estimated around 18 months, weeks matter. The agent exists to buy that latency
back, nothing more.

## Goal

An advisory agent that can **accelerate** the loop and grant one narrowly-bounded
exemption, whose failure is a non-event.

## Authority (user decision: accelerate + bounded exemption)

**Always permitted:** trigger a refit early; trigger a re-mine early; demote a
factor earlier than the sustained rule would.

**The one delay permitted — bounded regime exemption.** Exempt a factor from a
**soft** demotion when the underperformance is judged regime-explained. Every
exemption must:

* cite a **validated regime gate** — a named, computed condition from `09` that is
  currently true, with the factor's `ic_crash` / `ic_rally` profile as evidence;
* expire after **one quarter**, renewable only on fresh evidence;
* be recorded with its full justification;
* never apply to **hard** decay;
* never cover more than a configured fraction of the active set (default **25%**).

**Never permitted:** delaying hard decay; postponing a scheduled refit or re-mine;
deleting a factor; editing a deterministic threshold.

## Approach

**`quantaalpha/loop/decay_agent.py`** — `advise(asof, states, market_state, census)
-> Recommendation`.

* Input: the tier census, per-factor rolling IC vs baseline **with distance to each
  threshold and days-in-breach so far**, the `09` market state, and time since the
  last refit/re-mine.
* Output: a strict schema — `{accelerate_refit: bool, accelerate_remine: bool,
  demote: [factor_id], exempt: [{factor_id, regime_gate, evidence, expires}],
  rationale: str}`.
* **Schema-validated.** Anything unparseable, any field out of range, any exemption
  missing a regime gate ⇒ the whole recommendation is discarded and logged. Partial
  application of a malformed response is how an advisory layer becomes
  load-bearing by accident.
* Cadence: **weekly, plus immediately on any tier transition** (`03`'s event
  stream). A daily call adds cost without new information, because the underlying
  rule needs sustained days.
* Every invocation and decision is appended to `data/results/decay/agent.jsonl`,
  including the ones where it recommended nothing.

**Wiring into `03`:** the scheduler asks the agent for a recommendation, applies
only what the authority rules permit, and records both what was recommended and
what was applied. Where they differ, the reason is logged — that difference is the
audit trail for whether the agent is trying to exceed its remit.

## Validation

* **The load-bearing test:** with the agent deleted, disabled, raising, timing out,
  and returning malformed JSON — five cases — a simulated year still produces its
  12 refits and 1 re-mine unchanged.
* An exemption without a regime gate is rejected.
* An exemption against hard decay is rejected.
* Exemptions expire: a factor exempted at T is demoted at T+92 days absent renewal.
* The exemption cap binds: the 26th exemption in a 100-factor active set is refused.
* A recommendation to *delay* a refit is refused and logged as an attempted
  overreach.

## Risks

* **Agent-induced churn.** Accelerating refits repeatedly raises turnover and cost.
  The minimum inter-refit gap from `03` binds the agent too, and the ablation
  includes an agent-disabled arm so its net contribution is measurable rather than
  assumed.
* **Exemption abuse is the real hazard** — it is the only path by which a dead
  factor stays in the book. The cap, the expiry, and the hard-decay exclusion are
  all three necessary; do not relax any of them without measuring.
* LLM cost is trivial here (~52 calls/yr plus events) and is not a constraint.
