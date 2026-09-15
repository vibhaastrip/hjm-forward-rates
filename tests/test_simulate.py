"""
test_simulate.py

Tests for the HJM forward curve simulator in src/simulate.py.

The central test here is test_drift_matches_analytic_paired_seed, which
isolates the no-arbitrage drift (equation 18) from Monte Carlo noise by
running paired simulations with identical random shocks. The diffusion
terms cancel exactly in the difference, leaving the drift effect
measurable at a tiny fraction of the sample size an unpaired test needs.
"""

import numpy as np
import pytest

from src.simulate import (
    ConstantVol,
    ExponentialVol,
    TwoFactorVol,
    hjm_drift,
    simulate_forward_curves,
)


def _simulate_with_zero_drift(
    initial_curve, maturities, vol_structure, T_horizon, n_steps, n_paths, seed
):
    """
    Re-run of simulate_forward_curves with the drift term forced to zero,
    keeping everything else (including the RNG stream) identical.

    We monkeypatch rather than add a flag to the production function, so
    the simulator's signature stays clean -- a zero-drift HJM model
    admits arbitrage, so it shouldn't be reachable through the normal
    API. It's a measurement device, not a modelling option.
    """
    import src.simulate as sim

    original = sim.hjm_drift
    sim.hjm_drift = lambda vol, t, mats: np.zeros(np.asarray(mats).size)
    try:
        return sim.simulate_forward_curves(
            initial_curve=initial_curve,
            maturities=maturities,
            vol_structure=vol_structure,
            T_horizon=T_horizon,
            n_steps=n_steps,
            n_paths=n_paths,
            seed=seed,
        )
    finally:
        sim.hjm_drift = original


# ----------------------------------------------------------------------
# Drift function: direct analytic checks (no simulation involved)
# ----------------------------------------------------------------------


def test_drift_constant_vol_analytic():
    """
    For constant volatility sigma, equation (18) with phi=0 gives

        alpha(t,T) = sigma * integral_t^T sigma dv = sigma^2 * (T-t)

    This is the paper's equation (27) with the market-price-of-risk term
    dropped (we simulate under Q~). Trapezoidal integration of a constant
    is EXACT, so this should match to machine precision.
    """
    sigma = 0.01
    t = 0.3
    maturities = np.array([1.0, 2.0, 5.0])

    alpha = hjm_drift(ConstantVol(sigma), t, maturities)
    expected = sigma**2 * (maturities - t)

    assert np.allclose(alpha, expected, rtol=1e-12), (
        f"Expected {expected}, got {alpha}"
    )


def test_drift_exponential_vol_analytic():
    """
    For sigma(t,T) = sigma*exp(-a(T-t)), the inner integral is

        integral_t^T sigma*exp(-a(v-t)) dv = (sigma/a)*(1 - exp(-a(T-t)))

    so equation (18) gives

        alpha(t,T) = (sigma^2/a) * exp(-a(T-t)) * (1 - exp(-a(T-t)))

    This is the Hull-White-reducing structure. Trapezoidal integration is
    approximate here (genuine exponential integrand), so we use a fine
    grid and a correspondingly loose tolerance.
    """
    sigma, a = 0.01, 0.1
    t = 0.0
    maturities = np.linspace(0.0, 10.0, 2001)

    alpha = hjm_drift(ExponentialVol(sigma, a), t, maturities)

    tau = maturities - t
    expected = (sigma**2 / a) * np.exp(-a * tau) * (1 - np.exp(-a * tau))

    assert np.allclose(alpha, expected, rtol=1e-6, atol=1e-12), (
        f"Max abs error: {np.max(np.abs(alpha - expected))}"
    )


def test_drift_is_zero_at_maturity_equal_t():
    """
    At T = t the integration range is empty, so alpha(t,t) = 0 for any
    volatility structure. This is the boundary case where an off-by-one
    in the cumulative-integral alignment would show up first.
    """
    for vol in [
        ConstantVol(0.01),
        ExponentialVol(0.01, 0.1),
        TwoFactorVol(0.008, 0.006, 0.3),
    ]:
        alpha = hjm_drift(vol, t=0.5, maturities=np.array([0.5, 1.0, 2.0]))
        assert alpha[0] == pytest.approx(0.0, abs=1e-15)


def test_drift_two_factor_sums_single_factor_contributions():
    """
    Independent Brownian factors contribute additively to the drift,
    because their cross terms dW_i dW_j vanish for i != j. So the
    two-factor drift must equal the sum of the two corresponding
    single-factor drifts computed separately.

    A structural check on the sum-over-factors in hjm_drift, independent
    of any particular analytic formula.
    """
    sigma_1, sigma_2, lam = 0.008, 0.006, 0.3
    t = 0.2
    maturities = np.linspace(0.5, 8.0, 200)

    combined = hjm_drift(TwoFactorVol(sigma_1, sigma_2, lam), t, maturities)

    # Factor 1 alone is a constant-vol structure.
    part_1 = hjm_drift(ConstantVol(sigma_1), t, maturities)
    # Factor 2 alone is exponential with decay rate lam/2.
    part_2 = hjm_drift(ExponentialVol(sigma_2, lam / 2.0), t, maturities)

    assert np.allclose(combined, part_1 + part_2, rtol=1e-10)


# ----------------------------------------------------------------------
# The central test: paired-seed drift isolation
# ----------------------------------------------------------------------


def test_drift_matches_analytic_paired_seed():
    """
    Isolate the drift's effect by differencing two runs that share a
    seed -- one with the no-arbitrage drift, one with drift forced to
    zero.

    Both runs consume the same RNG stream, so the diffusion increments
    are IDENTICAL path-by-path. Subtracting cancels the stochastic term
    exactly, leaving only the accumulated drift:

        f_drift(t,T) - f_nodrift(t,T) = sum_k alpha(t_k,T)*dt

    a deterministic quantity checkable to six significant figures with
    50 paths, rather than the ~400,000 an unpaired mean test would need
    at this signal-to-noise ratio.
    """
    sigma, a = 0.01, 0.1
    maturities = np.linspace(0.0, 10.0, 201)
    initial_curve = np.full(maturities.size, 0.04)
    vol = ExponentialVol(sigma, a)

    T_horizon, n_steps, n_paths, seed = 1.0, 200, 50, 7

    curves_drift, times, _ = simulate_forward_curves(
        initial_curve=initial_curve,
        maturities=maturities,
        vol_structure=vol,
        T_horizon=T_horizon,
        n_steps=n_steps,
        n_paths=n_paths,
        seed=seed,
    )
    curves_nodrift, _, _ = _simulate_with_zero_drift(
        initial_curve, maturities, vol, T_horizon, n_steps, n_paths, seed
    )

    # A maturity comfortably beyond the horizon, so it stays live for
    # every step (no NaNs).
    T_idx = np.argmin(np.abs(maturities - 5.0))
    T = maturities[T_idx]

    diff = curves_drift[:, -1, T_idx] - curves_nodrift[:, -1, T_idx]

    # Diffusion cancels exactly, so every path should give the SAME
    # difference. If this fails, the runs aren't sharing shocks and the
    # rest of the test is meaningless.
    assert np.std(diff) == pytest.approx(0.0, abs=1e-15), (
        f"Paths disagree (std={np.std(diff)}); shocks are not paired."
    )

    observed = diff[0]

    # Analytic target: sum_k alpha(t_k, T) * dt, using the SAME left-
    # endpoint Euler convention the simulator uses, so we're testing the
    # drift formula rather than the discretization scheme.
    dt = T_horizon / n_steps
    t_k = times[:-1]
    tau = T - t_k
    alpha_k = (sigma**2 / a) * np.exp(-a * tau) * (1 - np.exp(-a * tau))
    expected = np.sum(alpha_k) * dt

    assert observed == pytest.approx(expected, rel=1e-5), (
        f"Expected accumulated drift {expected}, got {observed}"
    )


def test_drift_effect_is_small_but_nonzero():
    """
    Guards against a silently-zero drift. Accumulated drift over a
    1-year horizon at these parameters is ~1e-5 -- small, but it must
    not be zero, and must be POSITIVE (equation 18 with phi=0 gives
    alpha >= 0 for nonnegative volatility, since it's sigma times a
    nonnegative accumulated integral).
    """
    sigma, a = 0.01, 0.1
    maturities = np.linspace(0.0, 10.0, 201)
    initial_curve = np.full(maturities.size, 0.04)
    vol = ExponentialVol(sigma, a)

    curves_drift, _, _ = simulate_forward_curves(
        initial_curve, maturities, vol, 1.0, 100, 20, seed=3
    )
    curves_nodrift, _, _ = _simulate_with_zero_drift(
        initial_curve, maturities, vol, 1.0, 100, 20, seed=3
    )

    T_idx = np.argmin(np.abs(maturities - 5.0))
    diff = curves_drift[0, -1, T_idx] - curves_nodrift[0, -1, T_idx]

    assert diff > 0, "No-arbitrage drift should be strictly positive here"
    assert diff < 1e-3, "Drift is implausibly large -- check units"


# ----------------------------------------------------------------------
# Structural checks on the simulator itself
# ----------------------------------------------------------------------


def test_initial_curve_preserved():
    """The simulation must start exactly at the observed input curve."""
    maturities = np.linspace(0.0, 10.0, 51)
    initial_curve = 0.03 + 0.002 * maturities  # upward-sloping

    curves, _, _ = simulate_forward_curves(
        initial_curve, maturities, ConstantVol(0.01), 1.0, 50, 10, seed=1
    )

    for p in range(curves.shape[0]):
        assert np.allclose(curves[p, 0, :], initial_curve)


def test_matured_maturities_are_nan():
    """
    Maturities behind calendar time should be NaN, not stale values --
    a stale value would silently corrupt any bond price computed from
    the curve at a later step.
    """
    maturities = np.linspace(0.0, 10.0, 101)
    initial_curve = np.full(maturities.size, 0.04)

    curves, times, _ = simulate_forward_curves(
        initial_curve, maturities, ConstantVol(0.01), 2.0, 100, 5, seed=1
    )

    final_t = times[-1]
    expired = maturities < final_t
    assert np.all(np.isnan(curves[:, -1, expired]))
    assert not np.any(np.isnan(curves[:, -1, ~expired]))


def test_same_seed_reproduces_identical_paths():
    """Reproducibility: same seed must give bit-identical output."""
    maturities = np.linspace(0.0, 5.0, 51)
    initial_curve = np.full(maturities.size, 0.04)
    vol = ExponentialVol(0.01, 0.1)

    a, _, _ = simulate_forward_curves(initial_curve, maturities, vol, 1.0, 50, 10, seed=99)
    b, _, _ = simulate_forward_curves(initial_curve, maturities, vol, 1.0, 50, 10, seed=99)

    assert np.array_equal(np.nan_to_num(a), np.nan_to_num(b))


def test_different_seeds_give_different_paths():
    """Sanity check that the seed is actually being used."""
    maturities = np.linspace(0.0, 5.0, 51)
    initial_curve = np.full(maturities.size, 0.04)
    vol = ExponentialVol(0.01, 0.1)

    a, _, _ = simulate_forward_curves(initial_curve, maturities, vol, 1.0, 50, 10, seed=1)
    b, _, _ = simulate_forward_curves(initial_curve, maturities, vol, 1.0, 50, 10, seed=2)

    assert not np.array_equal(np.nan_to_num(a), np.nan_to_num(b))

def test_drift_converges_under_grid_refinement():
    """
    The trapezoidal integration inside hjm_drift has O(h^2) error.
    Halving the grid spacing should therefore cut the error by ~4x.

    This quantifies the integration error rather than just tolerating
    it, and confirms the discrepancy in the paired-seed test is
    convergent numerical error, not a formula bug -- a bug would not
    shrink under refinement.
    """
    sigma, a = 0.01, 0.1
    t, T_target = 0.0, 5.0

    errors = []
    for n_points in [101, 201, 401, 801]:
        maturities = np.linspace(0.0, 10.0, n_points)
        alpha = hjm_drift(ExponentialVol(sigma, a), t, maturities)

        idx = np.argmin(np.abs(maturities - T_target))
        tau = maturities[idx] - t
        exact = (sigma**2 / a) * np.exp(-a * tau) * (1 - np.exp(-a * tau))

        errors.append(abs(alpha[idx] - exact))

    # Each refinement halves h, so O(h^2) error should fall by ~4x.
    # Allow a generous band (3x to 5x) for finite-grid effects.
    for coarse, fine in zip(errors[:-1], errors[1:]):
        ratio = coarse / fine
        assert 3.0 < ratio < 5.0, (
            f"Expected ~4x error reduction per refinement, got {ratio:.2f}. "
            f"Errors: {errors}"
        )