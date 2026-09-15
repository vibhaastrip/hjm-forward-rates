import numpy as np
import pytest

from src.curve import(
    forward_rate_from_bond_prices,
    bond_price_from_forward_curve,
    spot_rate
)

@pytest.fixture

def sample_curve():
    maturities = np.array([0.5, 1.0, 2.0, 3.0, 5.0, 7.0])
    P_values = np.array([0.985, 0.970, 0.930, 0.885, 0.790, 0.700])
    return maturities, P_values

def test_forward_rates_positive(sample_curve):
    maturities, P_values = sample_curve
    f = forward_rate_from_bond_prices(P_values, maturities)

    assert np.all(f > 0), f"Expected all forward rates positive, got {f}"

def test_forward_rate_matches_analytic_flat_curve():
    c = 0.04
    maturities = np.linspace(0.5, 10.0, 20)
    P_values = np.exp(-c*maturities)

    f = forward_rate_from_bond_prices(P_values, maturities)

    assert np.allclose(f,c), f"Expected all forward rates = {c}, got {f}"

def test_bond_price_recovers_flat_curve():
    c = 0.04
    maturities = np.linspace(0.0, 10.0, 200)  # fine grid -> small trapz error
    forward_curve = np.full_like(maturities, c)

    t, T = 1.0, 5.0
    recovered = bond_price_from_forward_curve(forward_curve, maturities, t, T)
    expected = np.exp(-c * (T - t))

    assert recovered == pytest.approx(expected, rel=1e-4), (
        f"Expected {expected}, got {recovered}"
    )

def test_roundtrip_forward_and_back(sample_curve):
    maturities, P_values = sample_curve
    f = forward_rate_from_bond_prices(P_values, maturities)

    t_idx, T_idx = 0, 2  # maturities 0.5 and 2.0
    t, T = maturities[t_idx], maturities[T_idx]

    recovered = bond_price_from_forward_curve(f, maturities, t, T)
    expected_ratio = P_values[T_idx] / P_values[t_idx]

    assert recovered == pytest.approx(expected_ratio, rel=1e-2), (
        f"Expected ~{expected_ratio}, got {recovered}"
    )
def test_spot_rate_equals_short_end(sample_curve):
    maturities, P_values = sample_curve
    f = forward_rate_from_bond_prices(P_values, maturities)

    t = maturities[0]
    assert spot_rate(f, maturities, t) == pytest.approx(f[0])

def test_spot_rate_interpolates_between_grid(sample_curve):
    maturities, P_values = sample_curve
    f = forward_rate_from_bond_prices(P_values, maturities)

    t_between = 0.75  # between grid points 0.5 and 1.0
    r = spot_rate(f, maturities, t_between)

    # Should lie strictly between the two neighbouring forward rates.
    assert f[0] < r < f[1], f"Expected {f[0]} < {r} < {f[1]}"