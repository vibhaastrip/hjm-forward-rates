
import numpy as np
import pytest

from src.hullwhite import (
    B_function,
    bond_price,
    bond_option_price,
    make_flat_curve,
    short_rate_variance,
)


SIGMA = 0.01
A = 0.1
F_FLAT = 0.04


@pytest.fixture
def flat_curve():
    return make_flat_curve(F_FLAT)


# ----------------------------------------------------------------------
# B(t,T)
# ----------------------------------------------------------------------


def test_B_is_zero_at_maturity():
    """
    B(t,t) = 0: a bond at its own maturity has no sensitivity to the
    short rate, since its payoff is fixed at 1.
    """
    for t in [0.0, 1.0, 5.0]:
        assert B_function(t, t, A) == pytest.approx(0.0, abs=1e-15)


def test_B_saturates_at_one_over_a():
    """
    B -> 1/a as (T-t) -> infinity. Mean reversion bounds how much a
    long-dated bond can respond to today's short rate -- the rate is
    expected to revert long before maturity, so extra maturity adds
    almost no extra sensitivity.
    """
    assert B_function(0.0, 500.0, A) == pytest.approx(1.0 / A, rel=1e-9)


def test_B_matches_accumulated_volatility():
    """
    The structural link back to HJM: sigma*B(t,T) must equal the
    accumulated forward-rate volatility

        integral_t^T sigma*exp(-a(s-t)) ds

    which is the quantity hjm_drift computes internally. This is what
    makes Hull-White the HJM special case rather than a separate model.
    """
    t, T = 0.5, 4.0
    s = np.linspace(t, T, 20001)
    integral = np.trapezoid(SIGMA * np.exp(-A * (s - t)), s)

    assert SIGMA * B_function(t, T, A) == pytest.approx(integral, rel=1e-8)


def test_B_is_increasing_in_maturity():
    """Longer bonds are more rate-sensitive: B increasing in T."""
    maturities = np.linspace(0.1, 30.0, 100)
    B = B_function(0.0, maturities, A)
    assert np.all(np.diff(B) > 0)


# ----------------------------------------------------------------------
# Bond price
# ----------------------------------------------------------------------


def test_bond_price_reproduces_initial_curve(flat_curve):
    """
    At t=0 with r(0)=f(0,0), the model must return the OBSERVED discount
    factor exactly -- this is the defining property of a calibrated
    Hull-White model, and the reason HJM avoids the "inversion of the
    term structure" problem the paper criticises CIR for (Section 8).

    Exact to machine precision: a sign error or a missing convexity term
    in A(t,T) would show up here immediately.
    """
    for T in [0.5, 1.0, 5.0, 10.0, 30.0]:
        model = bond_price(0.0, T, F_FLAT, flat_curve, F_FLAT, SIGMA, A)
        assert model == pytest.approx(flat_curve(T), rel=1e-14)


def test_bond_price_is_one_at_maturity(flat_curve):
    """P(T,T) = 1 for any short rate -- the paper's Section 2 requirement."""
    for r in [0.0, 0.02, 0.06, 0.15]:
        assert bond_price(5.0, 5.0, r, flat_curve, F_FLAT, SIGMA, A) == pytest.approx(1.0)


def test_bond_price_decreasing_in_short_rate(flat_curve):
    """
    Higher short rate -> lower bond price. Follows from the affine form
    P = A*exp(-B*r) with B > 0, but worth pinning: a sign error on B
    would invert this and still pass several other tests.
    """
    rates = np.array([0.01, 0.03, 0.05, 0.08])
    prices = bond_price(1.0, 5.0, rates, flat_curve, F_FLAT, SIGMA, A)
    assert np.all(np.diff(prices) < 0)


def test_bond_price_decreasing_in_maturity(flat_curve):
    """
    P(t,T) decreasing in T for positive rates -- money further away is
    discounted more. This is the property that makes the forward rate
    f = -d(lnP)/dT positive (equation 1's minus sign).
    """
    maturities = np.array([1.5, 2.0, 5.0, 10.0])
    prices = np.array(
        [bond_price(1.0, T, 0.04, flat_curve, F_FLAT, SIGMA, A) for T in maturities]
    )
    assert np.all(np.diff(prices) < 0)


def test_bond_price_rejects_maturity_before_now(flat_curve):
    with pytest.raises(ValueError):
        bond_price(5.0, 2.0, 0.04, flat_curve, F_FLAT, SIGMA, A)


def test_bond_price_zero_vol_limit(flat_curve):
    """
    As sigma -> 0 the convexity term vanishes and the model collapses to
    deterministic discounting. With a flat curve and r(t) = f_flat, the
    price must be exactly exp(-f_flat*(T-t)).

    A clean analytic check that isolates A(t,T)'s first two terms from
    the convexity adjustment.
    """
    t, T = 2.0, 7.0
    price = bond_price(t, T, F_FLAT, flat_curve, F_FLAT, tiny := 1e-12, A)
    assert price == pytest.approx(np.exp(-F_FLAT * (T - t)), rel=1e-9)


# ----------------------------------------------------------------------
# Short rate variance
# ----------------------------------------------------------------------


def test_short_rate_variance_zero_at_time_zero():
    """r(0) is known, so it has no variance."""
    assert short_rate_variance(0.0, SIGMA, A) == pytest.approx(0.0, abs=1e-15)


def test_short_rate_variance_saturates():
    """
    OU variance -> sigma^2/(2a) as t -> infinity. Mean reversion bounds
    long-run uncertainty, unlike a driftless random walk whose variance
    grows without limit.
    """
    assert short_rate_variance(1000.0, SIGMA, A) == pytest.approx(
        SIGMA**2 / (2 * A), rel=1e-9
    )


def test_short_rate_variance_small_t_limit():
    """
    For small t, mean reversion has had no time to act, so the OU
    variance should approach the driftless Brownian value sigma^2 * t.

    Check: (1-exp(-2at))/(2a) -> t as t -> 0.
    """
    t = 1e-4
    assert short_rate_variance(t, SIGMA, A) == pytest.approx(SIGMA**2 * t, rel=1e-3)


# ----------------------------------------------------------------------
# Bond option
# ----------------------------------------------------------------------


def test_put_call_parity(flat_curve):
    """
    C - P = P(0,T) - K*P(0,t*)

    Holds by a static replication argument, independent of the model.
    Validates both branches at once -- but note it does NOT validate nu,
    which cancels in the difference. See test_option_monotonic_in_vol
    and test_option_converges_to_intrinsic_at_zero_vol for that.
    """
    t_star, T = 1.0, 5.0
    for K in [0.70, 0.80, 0.85, 0.95]:
        c = bond_option_price(t_star, T, K, flat_curve, SIGMA, A, "call")
        p = bond_option_price(t_star, T, K, flat_curve, SIGMA, A, "put")
        expected = flat_curve(T) - K * flat_curve(t_star)
        assert c - p == pytest.approx(expected, abs=1e-12)


def test_option_prices_are_positive(flat_curve):
    """Options can't have negative value -- they're optional to exercise."""
    for K in [0.60, 0.80, 0.99]:
        assert bond_option_price(1.0, 5.0, K, flat_curve, SIGMA, A, "call") > 0
        assert bond_option_price(1.0, 5.0, K, flat_curve, SIGMA, A, "put") > 0


def test_call_decreasing_in_strike(flat_curve):
    """Higher strike -> less valuable call."""
    strikes = np.array([0.70, 0.75, 0.80, 0.85, 0.90])
    calls = np.array(
        [bond_option_price(1.0, 5.0, K, flat_curve, SIGMA, A, "call") for K in strikes]
    )
    assert np.all(np.diff(calls) < 0)


def test_option_monotonic_in_vol(flat_curve):
    """
    Call value increasing in sigma -- more volatility means more upside,
    with downside floored at zero.

    This DOES exercise nu, unlike parity, so it catches errors parity
    would miss.
    """
    sigmas = np.array([0.002, 0.005, 0.01, 0.02])
    calls = np.array(
        [bond_option_price(1.0, 5.0, 0.82, flat_curve, s, A, "call") for s in sigmas]
    )
    assert np.all(np.diff(calls) > 0)


def test_option_converges_to_intrinsic_at_zero_vol(flat_curve):
    """
    As sigma -> 0 the bond price becomes deterministic and the call
    collapses to its discounted intrinsic value:

        max(P(0,T) - K*P(0,t*), 0)

    The sharpest available check on nu's magnitude: if nu were wrong by
    a constant factor, this limit would still hold, but if nu failed to
    vanish with sigma the test would fail.
    """
    t_star, T, K = 1.0, 5.0, 0.80
    call = bond_option_price(t_star, T, K, flat_curve, 1e-10, A, "call")
    intrinsic = max(flat_curve(T) - K * flat_curve(t_star), 0.0)
    assert call == pytest.approx(intrinsic, abs=1e-9)


def test_option_nu_factors_as_variance_times_duration(flat_curve):
    """
    Structural check that nu^2 = Var[r(t*)] * B(t*,T)^2 -- short-rate
    uncertainty at expiry, converted into bond-price uncertainty by the
    duration factor.

    We recover nu by inverting the at-the-money-forward case, where
    K = P(0,T)/P(0,t*) makes the log term vanish, leaving
        C = P(0,T)*[Phi(nu/2) - Phi(-nu/2)].
    """
    from scipy.optimize import brentq
    from scipy.stats import norm

    t_star, T = 2.0, 6.0
    K_atmf = flat_curve(T) / flat_curve(t_star)
    call = bond_option_price(t_star, T, K_atmf, flat_curve, SIGMA, A, "call")

    def residual(nu):
        return flat_curve(T) * (norm.cdf(nu / 2) - norm.cdf(-nu / 2)) - call

    nu_recovered = brentq(residual, 1e-8, 1.0)
    nu_expected = np.sqrt(
        short_rate_variance(t_star, SIGMA, A) * B_function(t_star, T, A) ** 2
    )

    assert nu_recovered == pytest.approx(nu_expected, rel=1e-8)


def test_option_rejects_bad_inputs(flat_curve):
    with pytest.raises(ValueError):
        bond_option_price(5.0, 3.0, 0.8, flat_curve, SIGMA, A)  # T <= t_star
    with pytest.raises(ValueError):
        bond_option_price(1.0, 5.0, -0.1, flat_curve, SIGMA, A)  # negative strike
    with pytest.raises(ValueError):
        bond_option_price(1.0, 5.0, 0.8, flat_curve, SIGMA, A, "straddle")