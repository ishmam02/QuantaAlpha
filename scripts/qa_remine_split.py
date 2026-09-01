#!/usr/bin/env python
"""Cut the REMINE protocol at the hard-decay date.

The rule, exactly as specified:

    "from the backtest find when a hard decay takes place and start a remine
     from that time by adjusting the splits -- the date from the hard decay
     onwards will be the new test range and all the old date range is the
     train and valid."

So:  test  = [hard_decay_date, last_available_day]
     train + valid = everything strictly before hard_decay_date

The train/valid boundary inside that earlier span is the only free parameter.
It is set so VALID is the smallest window on which a typical mined factor can
still clear the admission bar -- otherwise the gate rejects for window length
rather than on merit, which is the failure the protocol's own notes record
(RankIC 0.020 over 261d gives t=2.08, under a k_sigma=3 bar). Everything left
over goes to train, so nothing before the decay date is wasted.

Why this answers the question. "Does re-mining fix alpha decay?" is a claim
about what a search can do knowing only what was knowable BEFORE the alpha
died. Any train or validation day at or after the decay date would let the
remine see the regime it is supposed to be blind to, and a positive result
would then be unfalsifiable.

Usage:
    python scripts/qa_remine_split.py --hard-decay 2019-07-31 [--apply]
    python scripts/qa_remine_split.py --from-report data/results/decay_report.json --apply
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

BASE_PROTOCOL = ROOT / "quantaalpha/eval/protocol_csi300_meanvar_soft_linear.yaml"
REMINE_PROTOCOL = ROOT / "quantaalpha/eval/protocol_csi300_remine.yaml"

# Calibrated from the protocol's own stated figure: RankIC 0.020 over 261
# trading days gives t = 2.08  =>  daily IC sd = 0.020*sqrt(261)/2.08.
IC_SD = 0.1553
TYPICAL_IC = 0.020          # what this system actually mines
TRADING_DAYS_PER_YEAR = 243


def min_valid_days(k_sigma: float, ic: float = TYPICAL_IC, sd: float = IC_SD) -> int:
    """Days needed for a factor of size ``ic`` to reach the admission bar."""
    return int(np.ceil((k_sigma * sd / ic) ** 2))


def trading_days(dates: pd.DatetimeIndex, a: str, b: str) -> int:
    return int(((dates >= pd.Timestamp(a)) & (dates <= pd.Timestamp(b))).sum())


def load_dates() -> pd.DatetimeIndex:
    import os
    os.environ.setdefault("QA_PROTOCOL", str(BASE_PROTOCOL))
    from quantaalpha.eval.protocol import load_protocol
    from quantaalpha.eval.data import load_panel
    th = load_protocol(str(BASE_PROTOCOL))
    return pd.DatetimeIndex(load_panel(th, "2004-01-01", "2026-12-31").dates)


def build_split(hard_decay: str, dates: pd.DatetimeIndex, k_sigma: float,
                valid_years: float | None = None) -> dict:
    hd = pd.Timestamp(hard_decay)
    first, last = dates.min(), dates.max()
    if not (first < hd <= last):
        raise SystemExit(f"hard-decay date {hd.date()} outside data {first.date()}..{last.date()}")

    # TEST: the decay date onward. Non-negotiable -- this is the question.
    test = (str(hd.date()), str(last.date()))

    # Everything strictly before the decay date is train+valid.
    pre_end = hd - pd.Timedelta(days=1)
    pre_days = trading_days(dates, str(first.date()), str(pre_end.date()))

    need = min_valid_days(k_sigma)
    if valid_years is not None:
        need = int(valid_years * TRADING_DAYS_PER_YEAR)

    if pre_days <= need:
        raise SystemExit(
            f"only {pre_days} trading days before {hd.date()}, but VALID alone "
            f"needs {need} for a typical IC {TYPICAL_IC} to clear k_sigma={k_sigma}. "
            "The decay date is too early to remine against.")

    # Walk back `need` trading days from the decay date to open VALID.
    pre = dates[dates <= pre_end]
    valid_start = pre[-need]
    train_end = pre[pre < valid_start][-1]

    train = (str(first.date()), str(train_end.date()))
    valid = (str(valid_start.date()), str(pre_end.date()))

    nt, nv, nx = (trading_days(dates, *train), trading_days(dates, *valid),
                  trading_days(dates, *test))
    t_stat = TYPICAL_IC * np.sqrt(nv) / IC_SD
    return {
        "hard_decay_date": str(hd.date()),
        "train": list(train), "valid": list(valid), "final_test": list(test),
        "train_days": nt, "valid_days": nv, "test_days": nx,
        "train_years": round(nt / TRADING_DAYS_PER_YEAR, 2),
        "valid_years": round(nv / TRADING_DAYS_PER_YEAR, 2),
        "test_years": round(nx / TRADING_DAYS_PER_YEAR, 2),
        "k_sigma": k_sigma,
        "valid_t_at_typical_ic": round(float(t_stat), 2),
        "min_admittable_ic": round(float(k_sigma * IC_SD / np.sqrt(nv)), 4),
        "gate_can_admit": bool(t_stat >= k_sigma),
    }


def choose_folds(split: dict, dates: pd.DatetimeIndex) -> int:
    """Most folds whose earliest fold still has a trainable window.

    Each earlier fold steps validation back by its own length and trains from
    the same start, so fold k trains on (valid_start - k*span). A fold with a
    train window shorter than the valid window is not a regime test -- it is a
    fit on too little data -- so that is the cut-off.
    """
    v0, v1 = pd.Timestamp(split["valid"][0]), pd.Timestamp(split["valid"][1])
    span = v1 - v0 + pd.Timedelta(days=1)
    start = pd.Timestamp(split["train"][0])
    n = 1
    while n < 6:
        cand_v0 = v0 - span * n
        cand_train_days = trading_days(dates, str(start.date()),
                                       str((cand_v0 - pd.Timedelta(days=1)).date()))
        if cand_v0 <= start or cand_train_days < split["valid_days"]:
            break
        n += 1
    return n


def apply_split(split: dict, folds: int) -> Path:
    y = yaml.safe_load(BASE_PROTOCOL.read_text())
    y["splits"]["train"] = split["train"]
    y["splits"]["valid"] = split["valid"]
    y["splits"]["final_test"] = split["final_test"]
    y["walk_forward"]["folds"] = folds
    y["walk_forward"]["enabled"] = True
    header = (
        "# REMINE PROTOCOL -- generated by scripts/qa_remine_split.py\n"
        f"# hard-decay date: {split['hard_decay_date']}\n"
        "#\n"
        "# The question: does re-mining recover an alpha that has hard-decayed?\n"
        "# Everything from the hard-decay date onward is TEST; everything before\n"
        "# it is train+valid. No train or validation day touches the post-decay\n"
        "# regime, so a positive result cannot come from having seen it.\n"
        f"#   train {split['train'][0]}..{split['train'][1]}  "
        f"({split['train_days']}d / {split['train_years']}y)\n"
        f"#   valid {split['valid'][0]}..{split['valid'][1]}  "
        f"({split['valid_days']}d / {split['valid_years']}y)  "
        f"t={split['valid_t_at_typical_ic']} at IC {TYPICAL_IC}\n"
        f"#   test  {split['final_test'][0]}..{split['final_test'][1]}  "
        f"({split['test_days']}d / {split['test_years']}y)\n"
        f"#   folds {folds}\n\n")
    REMINE_PROTOCOL.write_text(header + yaml.safe_dump(y, sort_keys=False))
    return REMINE_PROTOCOL


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hard-decay", help="YYYY-MM-DD from the backtest")
    ap.add_argument("--from-report", help="decay report json with hard_decay_date")
    ap.add_argument("--k-sigma", type=float, default=None,
                    help="admission bar (default: read from the base protocol)")
    ap.add_argument("--valid-years", type=float, default=None,
                    help="override the computed minimum valid window")
    ap.add_argument("--apply", action="store_true", help="write the remine protocol")
    a = ap.parse_args()

    hd = a.hard_decay
    if not hd and a.from_report:
        rep = json.loads(Path(a.from_report).read_text())
        hd = rep.get("hard_decay_date") or rep.get("first_hard_decay")
    if not hd:
        raise SystemExit("need --hard-decay or --from-report with a hard-decay date")

    k = a.k_sigma
    if k is None:
        k = float(yaml.safe_load(BASE_PROTOCOL.read_text())
                  .get("admission", {}).get("k_sigma", 3.0))

    dates = load_dates()
    split = build_split(hd, dates, k, a.valid_years)
    folds = choose_folds(split, dates)
    split["folds"] = folds

    print(json.dumps(split, indent=2))
    if not split["gate_can_admit"]:
        print("\nWARNING: the valid window cannot admit a typical factor at this "
              "k_sigma -- widen it with --valid-years or the remine will reject "
              "on window length, not on merit.", file=sys.stderr)
    if a.apply:
        p = apply_split(split, folds)
        print(f"\nwrote {p}", file=sys.stderr)
        print(f"launch with: QA_PROTOCOL={p} EXPERIMENT_ID=remine_$(date +%Y%m%d_%H%M%S) "
              "./scripts/qa_mine_full.sh", file=sys.stderr)


if __name__ == "__main__":
    main()
