"""A non-cn_data provider root (us_data) must resolve instruments correctly.

quantaalpha.eval.data._membership_mask -- the spell-file reader used by
equal_weight_benchmark -- used to hardcode ``~/.qlib/qlib_data/cn_data/instruments/
{market}.txt``, so a US equal-weight benchmark would have silently loaded CSI300
membership. It now derives the instruments dir from ``QLIB_PROVIDER_URI``. This
test points QLIB_PROVIDER_URI at a tmp us_data dir and checks the mask reads the
US spell file; it also pins the backward-compatible ``market`` default on
estimated_dividend_return (which used to hardcode "csi300").
"""
import inspect

import pandas as pd

from quantaalpha.eval import data as D


def test_membership_mask_resolves_provider_root(tmp_path, monkeypatch):
    us = tmp_path / "us_data"
    (us / "instruments").mkdir(parents=True)
    (us / "instruments" / "sp500.txt").write_text(
        "AAPL\t2010-01-01\t2026-01-09\nMSFT\t2013-01-01\t2026-01-09\n")
    monkeypatch.setenv("QLIB_PROVIDER_URI", str(us))

    dates = pd.date_range("2020-01-02", periods=4, freq="B")
    cols = pd.Index(["AAPL", "MSFT", "GOOGL"])
    mask = D._membership_mask("sp500", dates, cols)

    assert mask.loc[:, "AAPL"].all(), "AAPL is a member across the window"
    assert mask.loc[:, "MSFT"].all(), "MSFT is a member across the window"
    assert not mask.loc[:, "GOOGL"].any(), "GOOGL is not in the US spell file"

    # The spell-file window is respected: a date before AAPL's start is False.
    early = pd.date_range("2009-12-28", periods=2, freq="B")
    m2 = D._membership_mask("sp500", early, cols)
    assert not bool(m2.loc[early[0], "AAPL"]), "AAPL not a member before its add-date"
    print("OK  _membership_mask resolves the instruments dir from QLIB_PROVIDER_URI "
          "(us_data/instruments/sp500.txt), not the hardcoded cn_data path")


def test_estimated_dividend_return_market_param_is_backward_compatible():
    sig = inspect.signature(D.estimated_dividend_return)
    assert "market" in sig.parameters, "estimated_dividend_return must accept a market arg"
    assert sig.parameters["market"].default == "csi300", "default stays csi300 (no regression)"
    print("OK  estimated_dividend_return(market='csi300') default preserved; US passes market=theta.market")