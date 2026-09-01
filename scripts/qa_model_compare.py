#!/usr/bin/env python
"""Head-to-head comparison of two generation LLMs on the SAME smoke config.

Reads each model's ledger (per-factor tearsheets: t_nw, rank_ic_neutral,
rank_ic, rank_icir, sign_predicted/realized, mechanism_validated) and its run
log (JSON-fix failures, retries, length-truncation, per-batch wall-clock) and
prints a side-by-side verdict.

The two smokes MUST share everything except the model (same config, same frozen
protocol hash, same seed, same QA_TARGET_MINED). Any difference below is then
attributable to the LLM.

Usage::

    python scripts/qa_model_compare.py \\
        --a glm  data/results/ledger_smoke_glm52.jsonl  data/results/smoke_glm52.log \\
        --b kimi data/results/ledger_smoke_kimi3.jsonl data/results/smoke_kimi3.log
"""
from __future__ import annotations

import argparse
import json
import re
import statistics
from pathlib import Path


def _f(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def load_ledger(path: str) -> list[dict]:
    recs = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                recs.append(json.loads(line))
    return recs


def ledger_summary(recs: list[dict]) -> dict:
    n_batches = len(recs)
    n_admit_batches = sum(1 for r in recs if r.get("admitted"))
    zoo_final = max([r.get("zoo_size") or 0 for r in recs], default=0)

    # Per-factor metrics across ALL evaluated factors (admitted + rejected).
    abs_t, rank_ic_neut, rank_icir_all, rank_ic_all = [], [], [], []
    sign_pred, sign_real, mech_validated = 0, 0, 0
    n_sign_clauses = 0
    exprs, top_ops = set(), {}
    n_factors = 0
    for r in recs:
        ts = r.get("factor_tearsheets") or {}
        n_factors += len(ts)
        for expr, sheet in ts.items():
            exprs.add(expr)
            t = _f(sheet.get("t_nw"))
            if t is not None:
                abs_t.append(abs(t))
            ric = _f(sheet.get("rank_ic_neutral"))
            if ric is not None:
                rank_ic_neut.append(abs(ric))
            ricir = _f(sheet.get("rank_icir"))
            if ricir is not None:
                rank_icir_all.append(ricir)
            rraw = _f(sheet.get("rank_ic"))
            if rraw is not None:
                rank_ic_all.append(abs(rraw))
            sp, sr = sheet.get("sign_predicted"), sheet.get("sign_realized")
            if sp is not None:
                sign_pred += 1
                if sr is not None:
                    n_sign_clauses += 1
                    if sp == sr:
                        sign_real += 1
            if sheet.get("mechanism_validated"):
                mech_validated += 1
            # top-level operator = first identifier before '(' in the expr
            m = re.match(r"\s*([A-Za-z_][A-Za-z0-9_]*)\s*\(", expr)
            if m:
                op = m.group(1)
                top_ops[op] = top_ops.get(op, 0) + 1

    def med(xs):
        return statistics.median(xs) if xs else float("nan")

    def p90(xs):
        if not xs:
            return float("nan")
        xs = sorted(xs)
        return xs[min(len(xs) - 1, int(0.9 * len(xs)))]

    n_ge3 = sum(1 for t in abs_t if t >= 3.0)
    return {
        "n_batches": n_batches,
        "n_admit_batches": n_admit_batches,
        "admit_rate": (n_admit_batches / n_batches) if n_batches else float("nan"),
        "zoo_final": zoo_final,
        "n_factors_eval": n_factors,
        "n_distinct_exprs": len(exprs),
        "abs_t_med": med(abs_t),
        "abs_t_p90": p90(abs_t),
        "abs_t_max": max(abs_t) if abs_t else float("nan"),
        "n_t_ge3": n_ge3,
        "frac_t_ge3": (n_ge3 / len(abs_t)) if abs_t else float("nan"),
        "rank_ic_neut_med": med(rank_ic_neut),
        "rank_ic_med": med(rank_ic_all),
        "rank_icir_med": med(rank_icir_all),
        "rank_icir_p90": p90(rank_icir_all),
        "sign_match_rate": (sign_real / n_sign_clauses) if n_sign_clauses else float("nan"),
        "n_sign_clauses": n_sign_clauses,
        "mech_validated": mech_validated,
        "top_ops": top_ops,
    }


def log_summary(path: str) -> dict:
    """Robustness + speed from the run log."""
    text = Path(path).read_text(encoding="utf-8", errors="replace") if Path(path).exists() else ""
    # Robustness counters
    n_retry = len(re.findall(r"Retrying \d", text))
    n_json_fix = len(re.findall(r"Fixed JSON format issues", text))
    n_json_fail = len(re.findall(r"JSON fix failed", text))
    n_length = len(re.findall(r"finish_reason='length'|finish_reason: length|empty response with finish_reason='length'", text))
    n_failed_completion = len(re.findall(r"Failed to create chat completion", text))
    n_bad_request = len(re.findall(r"BadRequestError", text))
    # Per-batch wall-clock from ledger-style timestamps is computed in main();
    # here we also count phase markers.
    n_rounds = len(re.findall(r"round_idx\s*[:=]\s*\d|Round \d+|phase=\w+", text))
    return {
        "n_retry": n_retry,
        "n_json_fix_ok": n_json_fix,
        "n_json_fix_fail": n_json_fail,
        "n_length_trunc": n_length,
        "n_failed_completion": n_failed_completion,
        "n_bad_request": n_bad_request,
        "n_phase_markers": n_rounds,
    }


def batch_durations(recs: list[dict]) -> list[float]:
    """Per-batch wall-clock (seconds) from consecutive record timestamps."""
    ts = []
    for r in recs:
        t = r.get("ts")
        if t is None:
            continue
        # ts may be epoch float or ISO string
        if isinstance(t, (int, float)):
            ts.append(float(t))
        else:
            from datetime import datetime
            s = str(t).replace("Z", "+00:00")
            try:
                ts.append(datetime.fromisoformat(s).timestamp())
            except ValueError:
                pass
    durs = [ts[i] - ts[i - 1] for i in range(1, len(ts)) if ts[i] - ts[i - 1] > 0]
    return durs


def fmt(d, key, fmt_spec="{:.3f}"):
    v = d.get(key)
    if v is None or (isinstance(v, float) and v != v):
        return "n/a"
    if isinstance(v, float):
        return fmt_spec.format(v)
    return str(v)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", nargs=2, metavar=("NAME", "LEDGER"), required=True)
    ap.add_argument("--b", nargs=2, metavar=("NAME", "LEDGER"), required=True)
    args = ap.parse_args()
    models = []
    for name, ledger_path in (args.a, args.b):
        log_path = str(ledger_path).replace("ledger_", "").replace(".jsonl", ".log")
        recs = load_ledger(ledger_path)
        ls = ledger_summary(recs)
        lg = log_summary(log_path)
        durs = batch_durations(recs)
        lg["batch_sec_med"] = statistics.median(durs) if durs else float("nan")
        lg["batch_sec_total"] = sum(durs) if durs else 0.0
        models.append((name, ls, lg, ledger_path, log_path))

    a_name, a, alog, a_led, _ = models[0]
    b_name, b, blog, b_led, _ = models[1]
    print("=" * 78)
    print(f"  LLM HEAD-TO-HEAD:  {a_name}  vs  {b_name}")
    print("=" * 78)
    print(f"  ledger A: {a_led}")
    print(f"  ledger B: {b_led}")
    print("-" * 78)

    def row(label, ka, kb, spec="{:.3f}"):
        print(f"  {label:<34} {fmt(a, ka, spec):>14}   {fmt(b, kb, spec):>14}")

    print("  --- GENERATION YIELD ---")
    row("batches (records)", "n_batches", "n_batches", "{:.0f}")
    row("factors evaluated", "n_factors_eval", "n_factors_eval", "{:.0f}")
    row("distinct expressions", "n_distinct_exprs", "n_distinct_exprs", "{:.0f}")
    row("admitted batches", "n_admit_batches", "n_admit_batches", "{:.0f}")
    row("admission rate", "admit_rate", "admit_rate", "{:.1%}")
    row("final zoo size", "zoo_final", "zoo_final", "{:.0f}")

    print("  --- PER-FACTOR QUALITY (all eval'd factors) ---")
    row("median |t_nw|", "abs_t_med", "abs_t_med", "{:.2f}")
    row("p90 |t_nw|", "abs_t_p90", "abs_t_p90", "{:.2f}")
    row("max |t_nw|", "abs_t_max", "abs_t_max", "{:.2f}")
    row("# |t| >= 3 (gate-clear)", "n_t_ge3", "n_t_ge3", "{:.0f}")
    row("frac |t| >= 3", "frac_t_ge3", "frac_t_ge3", "{:.1%}")
    row("median |rank_ic_neutral|", "rank_ic_neut_med", "rank_ic_neut_med", "{:.4f}")
    row("median |rank_ic| (raw)", "rank_ic_med", "rank_ic_med", "{:.4f}")
    row("median rank_icir", "rank_icir_med", "rank_icir_med", "{:.3f}")
    row("p90 rank_icir", "rank_icir_p90", "rank_icir_p90", "{:.3f}")

    print("  --- SIGN / MECHANISM FALSIFIABILITY ---")
    row("sign-match rate", "sign_match_rate", "sign_match_rate", "{:.1%}")
    row("# sign clauses scored", "n_sign_clauses", "n_sign_clauses", "{:.0f}")
    row("# mechanism_validated", "mech_validated", "mech_validated", "{:.0f}")

    print("  --- ROBUSTNESS (from log) ---")
    def row2(label, ka, kb, spec="{:.0f}"):
        print(f"  {label:<34} {fmt(alog, ka, spec):>14}   {fmt(blog, kb, spec):>14}")
    row2("retries", "n_retry", "n_retry")
    row2("JSON auto-fixes (ok)", "n_json_fix_ok", "n_json_fix_ok")
    row2("JSON fix FAILURES", "n_json_fix_fail", "n_json_fix_fail")
    row2("length-truncated responses", "n_length_trunc", "n_length_trunc")
    row2("failed completions", "n_failed_completion", "n_failed_completion")
    row2("bad-request errors", "n_bad_request", "n_bad_request")

    print("  --- SPEED (from ledger ts) ---")
    row2("median batch wall-clock (s)", "batch_sec_med", "batch_sec_med", "{:.1f}")
    row2("total batch wall-clock (s)", "batch_sec_total", "batch_sec_total", "{:.0f}")

    print("  --- OPERATOR DIVERSITY (top operators used) ---")
    aops = sorted(a["top_ops"].items(), key=lambda kv: -kv[1])[:8]
    bops = sorted(b["top_ops"].items(), key=lambda kv: -kv[1])[:8]
    print(f"  {a_name}: {aops}")
    print(f"  {b_name}: {bops}")
    print(f"  {a_name} distinct top-ops: {len(a['top_ops'])}   |   {b_name} distinct top-ops: {len(b['top_ops'])}")
    print("=" * 78)


if __name__ == "__main__":
    main()