"""Tests for rmt_chaos — Random Matrix Theory herding/chaos index.

Written before the implementation (TDD). The index measures "propagation of
chaos" (Deng-Hani-Ma sense) across a stock universe: when cross-sectional
correlations are pure noise, the sample correlation matrix's eigenvalues sit
inside the Marchenko-Pastur bulk (chaos holds → mean-field predictors valid);
when a market/herding mode appears, the top eigenvalue detaches (chaos broken).
"""
import numpy as np
import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# Marchenko-Pastur edges
# ---------------------------------------------------------------------------

def test_mp_edges_q_small_approaches_one():
    from schwab.rmt_chaos import marchenko_pastur_edges
    lo, hi = marchenko_pastur_edges(q=1e-9)
    assert lo == pytest.approx(1.0, abs=1e-3)
    assert hi == pytest.approx(1.0, abs=1e-3)


def test_mp_edges_q_one():
    from schwab.rmt_chaos import marchenko_pastur_edges
    lo, hi = marchenko_pastur_edges(q=1.0)
    assert lo == pytest.approx(0.0, abs=1e-9)
    assert hi == pytest.approx(4.0, abs=1e-9)


def test_mp_edges_scale_with_sigma2():
    from schwab.rmt_chaos import marchenko_pastur_edges
    lo1, hi1 = marchenko_pastur_edges(q=0.25, sigma2=1.0)
    lo2, hi2 = marchenko_pastur_edges(q=0.25, sigma2=2.0)
    assert hi2 == pytest.approx(2.0 * hi1)
    assert lo2 == pytest.approx(2.0 * lo1)


# ---------------------------------------------------------------------------
# Eigenvalues of a correlation matrix
# ---------------------------------------------------------------------------

def test_independent_returns_stay_in_bulk():
    """Pure-noise universe: no eigenvalue should meaningfully exceed the MP
    upper edge (chaos holds)."""
    from schwab.rmt_chaos import correlation_eigenvalues, marchenko_pastur_edges
    rng = np.random.default_rng(0)
    T, N = 1000, 100
    X = rng.standard_normal((T, N))          # independent columns
    eigs = correlation_eigenvalues(X)
    _, hi = marchenko_pastur_edges(q=N / T)
    # Allow a modest finite-size margin above the asymptotic edge
    assert eigs[0] < hi * 1.5
    assert len(eigs) == N
    assert np.all(np.diff(eigs) <= 1e-9)      # sorted descending


def test_common_factor_detaches_top_eigenvalue():
    """Inject a shared market factor: the top eigenvalue must blow past the
    MP edge (chaos broken)."""
    from schwab.rmt_chaos import correlation_eigenvalues, marchenko_pastur_edges
    rng = np.random.default_rng(1)
    T, N = 1000, 100
    factor = rng.standard_normal((T, 1))
    X = 0.7 * factor + 0.3 * rng.standard_normal((T, N))   # strong common mode
    eigs = correlation_eigenvalues(X)
    _, hi = marchenko_pastur_edges(q=N / T)
    assert eigs[0] > hi * 3           # dominant mode far outside the bulk


# ---------------------------------------------------------------------------
# decompose() — the per-window metrics
# ---------------------------------------------------------------------------

def test_market_mode_frac_independent_is_small():
    """Independent universe: variance in the top mode ~ 1/N (max dispersion)."""
    from schwab.rmt_chaos import correlation_eigenvalues, decompose
    rng = np.random.default_rng(2)
    T, N = 1000, 100
    eigs = correlation_eigenvalues(rng.standard_normal((T, N)))
    d = decompose(eigs, N=N, T=T, top_k=20)
    assert d["market_mode_frac"] < 0.10          # near 1/N = 0.01, noise-inflated


def test_market_mode_frac_single_factor_is_large():
    from schwab.rmt_chaos import correlation_eigenvalues, decompose
    rng = np.random.default_rng(3)
    T, N = 1000, 100
    factor = rng.standard_normal((T, 1))
    X = 0.85 * factor + 0.15 * rng.standard_normal((T, N))
    d = decompose(correlation_eigenvalues(X), N=N, T=T, top_k=20)
    assert d["market_mode_frac"] > 0.5           # herding: most variance in mode 1


def test_absorption_ratio_bounded_and_monotone():
    """Absorption ratio in [0,1]; more herding => higher AR."""
    from schwab.rmt_chaos import correlation_eigenvalues, decompose
    rng = np.random.default_rng(4)
    T, N = 800, 60
    calm = correlation_eigenvalues(rng.standard_normal((T, N)))
    factor = rng.standard_normal((T, 1))
    stress = correlation_eigenvalues(0.7 * factor + 0.3 * rng.standard_normal((T, N)))
    ar_calm   = decompose(calm,   N, T, top_k=12)["absorption_ratio"]
    ar_stress = decompose(stress, N, T, top_k=12)["absorption_ratio"]
    assert 0.0 <= ar_calm <= 1.0
    assert 0.0 <= ar_stress <= 1.0
    assert ar_stress > ar_calm


def test_decompose_has_expected_fields():
    from schwab.rmt_chaos import correlation_eigenvalues, decompose
    rng = np.random.default_rng(5)
    d = decompose(correlation_eigenvalues(rng.standard_normal((500, 40))),
                  N=40, T=500, top_k=8)
    for k in ("lambda_max", "mp_upper", "detachment", "market_mode_frac",
              "absorption_ratio", "n_deviating", "effective_rank"):
        assert k in d


# ---------------------------------------------------------------------------
# log_returns + rolling index
# ---------------------------------------------------------------------------

def _fake_prices(n_days=400, n_assets=30, seed=7, common=0.0):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2024-01-01", periods=n_days)
    factor = rng.standard_normal((n_days, 1))
    shocks = rng.standard_normal((n_days, n_assets))
    rets = common * factor + (1 - common) * 0.01 * shocks
    prices = 100 * np.exp(np.cumsum(rets, axis=0))
    return pd.DataFrame(prices, index=dates,
                        columns=[f"S{i:02d}" for i in range(n_assets)])


def test_log_returns_shape_and_no_inf():
    from schwab.rmt_chaos import log_returns
    px = _fake_prices()
    r = log_returns(px)
    assert r.shape[0] == px.shape[0] - 1
    assert np.isfinite(r.values).all()


def test_rolling_chaos_index_produces_dated_series():
    from schwab.rmt_chaos import log_returns, rolling_chaos_index
    r = log_returns(_fake_prices(n_days=400, n_assets=30))
    idx = rolling_chaos_index(r, window=120)
    assert isinstance(idx, pd.DataFrame)
    assert len(idx) > 0
    assert idx.index.is_monotonic_increasing
    assert (idx["market_mode_frac"].between(0, 1)).all()
    assert (idx["N"] > 0).all()


def test_rolling_index_higher_for_herding_universe():
    """A universe with a strong common factor should read a higher median
    market-mode fraction than an independent one."""
    from schwab.rmt_chaos import log_returns, rolling_chaos_index
    calm = rolling_chaos_index(log_returns(_fake_prices(common=0.0, seed=11)),
                               window=120)
    herd = rolling_chaos_index(log_returns(_fake_prices(common=0.6, seed=11)),
                               window=120)
    assert herd["market_mode_frac"].median() > calm["market_mode_frac"].median()


# ---------------------------------------------------------------------------
# lead_lag vs a reference series (e.g. VIX)
# ---------------------------------------------------------------------------

def test_lead_lag_detects_known_offset():
    """If the reference is the index shifted forward by k days, lead_lag must
    report the index leading by ~k."""
    from schwab.rmt_chaos import lead_lag
    dates = pd.bdate_range("2020-01-01", periods=500)
    rng = np.random.default_rng(9)
    sig = pd.Series(rng.standard_normal(500), index=dates)
    ref = sig.shift(5)                        # ref is a 5-day-delayed echo of sig
    res = lead_lag(sig, ref.dropna(), max_lag=15)
    assert res["best_lag"] == pytest.approx(5, abs=1)   # so sig leads ref by +5
    assert res["best_corr"] > 0.5
