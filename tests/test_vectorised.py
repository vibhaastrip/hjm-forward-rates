import numpy as np
import pytest

from src.curve import bond_price_from_forward_curve, bond_prices_vectorised
from src.simulate import ConstantVol, ExponentialVol, simulate_forward_curves


def _reference(curves_slice, maturities, t, T):
    """The per-path implementation, for comparison."""
    n_paths = curves_slice.shape[0]
    out = np.zeros(n_paths)
    for p in range(n_paths):
        row = curves_slice[p]
        live = ~np.isnan(row)
        out[p] = bond_price_from_forward_curve(row[live], maturities[live], t, T)
    return out


@pytest.fixture
def simulated():
    """A small simulation with genuine NaN expiry structure."""
    maturities = np.linspace(0.0, 10.0, 201)
    initial_curve = 0.03 + 0.002 * maturities      # upward-sloping
    curves, times, _ = simulate_forward_curves(
        initial_curve, maturities, ExponentialVol(0.01, 0.1),
        T_horizon=1.0, n_steps=50, n_paths=40, seed=11,
    )
    return curves, times, maturities


def test_matches_reference_on_grid_points(simulated):
    """
    t and T landing exactly on maturity grid points -- the case where
    no interpolation is needed and the two paths through the code
    should be identical.
    """
    curves, times, maturities = simulated
    t = times[-1]
    T = 5.0   # exactly on the grid (spacing 0.05)

    fast = bond_prices_vectorised(curves[:, -1, :], maturities, t, T)
    slow = _reference(curves[:, -1, :], maturities, t, T)

    assert np.allclose(fast, slow, rtol=1e-13, atol=1e-15)


def test_matches_reference_off_grid_points(simulated):
    """
    t and T falling BETWEEN grid points, exercising the interpolation
    weights. This is where a searchsorted off-by-one would show up.
    """
    curves, times, maturities = simulated
    t = times[-1]

    for T in [3.017, 4.499, 7.223, 9.981]:
        fast = bond_prices_vectorised(curves[:, -1, :], maturities, t, T)
        slow = _reference(curves[:, -1, :], maturities, t, T)
        assert np.allclose(fast, slow, rtol=1e-12, atol=1e-15), (
            f"Mismatch at T={T}: max diff "
            f"{np.max(np.abs(fast - slow)):.3e}"
        )


def test_matches_reference_at_intermediate_times(simulated):
    """
    Evaluate at several points along the time axis, not just the end.
    Earlier times have fewer expired maturities, so this varies the
    live mask.
    """
    curves, times, maturities = simulated

    for k in [1, 10, 25, 40, -1]:
        t = times[k]
        T = 6.0
        fast = bond_prices_vectorised(curves[:, k, :], maturities, t, T)
        slow = _reference(curves[:, k, :], maturities, t, T)
        assert np.allclose(fast, slow, rtol=1e-12, atol=1e-15), (
            f"Mismatch at k={k} (t={t})"
        )


def test_matches_reference_short_maturity(simulated):
    """
    T only just beyond t -- few or no interior grid points, so the
    integration grid degenerates to nearly just the two endpoints.
    """
    curves, times, maturities = simulated
    t = times[-1]

    for T in [t + 0.01, t + 0.05, t + 0.12]:
        fast = bond_prices_vectorised(curves[:, -1, :], maturities, t, T)
        slow = _reference(curves[:, -1, :], maturities, t, T)
        assert np.allclose(fast, slow, rtol=1e-12, atol=1e-15), f"Mismatch at T={T}"


def test_matches_reference_constant_vol(simulated):
    """
    Repeat against a different volatility structure, so the test isn't
    passing by accident on one particular curve shape.
    """
    maturities = np.linspace(0.0, 10.0, 201)
    initial_curve = np.full(maturities.size, 0.04)
    curves, times, _ = simulate_forward_curves(
        initial_curve, maturities, ConstantVol(0.012),
        T_horizon=1.0, n_steps=50, n_paths=40, seed=5,
    )

    t = times[-1]
    for T in [2.0, 4.77, 8.5]:
        fast = bond_prices_vectorised(curves[:, -1, :], maturities, t, T)
        slow = _reference(curves[:, -1, :], maturities, t, T)
        assert np.allclose(fast, slow, rtol=1e-12, atol=1e-15)


def test_flat_curve_analytic():
    """
    Analytic check independent of the reference implementation: a flat
    forward curve at level c gives P(t,T) = exp(-c*(T-t)) exactly.

    Guards against both implementations being wrong in the same way.
    """
    c = 0.04
    maturities = np.linspace(0.0, 10.0, 401)
    curves_slice = np.full((7, maturities.size), c)

    t, T = 1.0, 5.0
    prices = bond_prices_vectorised(curves_slice, maturities, t, T)

    assert np.allclose(prices, np.exp(-c * (T - t)), rtol=1e-12)


def test_returns_one_at_maturity():
    """P(T,T) = 1 for every path."""
    maturities = np.linspace(0.0, 10.0, 201)
    curves_slice = np.full((5, maturities.size), 0.04)

    assert np.allclose(bond_prices_vectorised(curves_slice, maturities, 3.0, 3.0), 1.0)


def test_rejects_maturity_before_now():
    maturities = np.linspace(0.0, 10.0, 201)
    curves_slice = np.full((5, maturities.size), 0.04)

    with pytest.raises(ValueError):
        bond_prices_vectorised(curves_slice, maturities, 5.0, 2.0)


def test_handles_single_path():
    """
    A 1-D input should work, via atleast_2d. Worth pinning: a shape
    error here would only surface when someone prices one path.
    """
    maturities = np.linspace(0.0, 10.0, 201)
    row = np.full(maturities.size, 0.04)

    result = bond_prices_vectorised(row, maturities, 1.0, 5.0)
    assert result.shape == (1,)
    assert result[0] == pytest.approx(np.exp(-0.04 * 4.0), rel=1e-12)