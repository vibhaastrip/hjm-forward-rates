"""
maturity_convergence.py

Isolates the maturity-axis discretization error.

The diagnostics figure showed that the discrepancy between the
simulated bond price (integrating a 401-point forward curve) and the
Hull-White closed form (using the short rate alone) is NOT noise: it
has mean ~2e-07 against a standard deviation of ~2e-08, and correlates
-0.999 with r(t). A deterministic, state-dependent offset.

The convergence study in validate.py cannot see this, because it varies
the TIME step while holding n_maturities fixed at 401. The trapezoidal
integration along the maturity axis is O(h^2) in the maturity grid
spacing, and refining the time step does nothing to it.

This script varies n_maturities with everything else held fixed. If the
hypothesis is right, the mean discrepancy should fall by ~4x per
doubling of the grid density, and the correlation with r(t) should stay
near -1 throughout (the error shrinks but keeps its shape).

If instead the discrepancy is insensitive to n_maturities, the cause is
something else and the hypothesis is wrong.
"""

import numpy as np

from src.curve import bond_prices_vectorised
from src.hullwhite import bond_price, make_flat_curve
from src.simulate import ExponentialVol, simulate_forward_curves


SIGMA, A, F_FLAT = 0.01, 0.1, 0.04


def discrepancy(n_maturities, T=5.0, t_horizon=1.0, n_steps=200,
                n_paths=2000, max_maturity=10.0, seed=42):
    """
    Mean and spread of (simulated price - closed form) at a given
    maturity-grid density.

    The same seed and path count throughout, so rows differ only in the
    maturity grid. Note the underlying Brownian path is identical across
    rows: shocks have shape (n_paths, n_steps, n_factors), which does
    not depend on n_maturities at all.
    """
    maturities = np.linspace(0.0, max_maturity, n_maturities)
    initial = np.full(maturities.size, F_FLAT)
    P0 = make_flat_curve(F_FLAT)

    rng = np.random.default_rng(seed)
    shocks = rng.standard_normal((n_paths, n_steps, 1))

    curves, times, sr = simulate_forward_curves(
        initial, maturities, ExponentialVol(SIGMA, A),
        t_horizon, n_steps, n_paths, shocks=shocks, store_history=False,
    )

    t = times[-1]
    P_sim = bond_prices_vectorised(curves[:, -1, :], maturities, t, T)
    P_hw = bond_price(t, T, sr[:, -1], P0, F_FLAT, SIGMA, A)

    err = P_sim - P_hw
    h = max_maturity / (n_maturities - 1)

    return {
        "n_maturities": n_maturities,
        "h": h,
        "mean": float(err.mean()),
        "sd": float(err.std(ddof=1)),
        "corr_with_r": float(np.corrcoef(sr[:, -1], err)[0, 1]),
    }

if __name__ == "__main__":
    # --- Part 1: maturity grid (the original question) --------------
    grids = [101, 201, 401, 801, 1601]

    print("=" * 78)
    print("1. Discrepancy vs maturity-grid density (T = 5, t = 1)")
    print("=" * 78)
    print("\nTrapezoidal error on the maturity axis is O(h^2), so halving h")
    print("should cut any component it causes by ~4x.\n")
    print(f"{'n_mat':>7}  {'h':>8}  {'mean err':>12}  {'sd':>11}  "
          f"{'corr':>7}  {'mean ratio':>10}  {'sd ratio':>9}")
    print("-" * 78)

    prev_mean = prev_sd = None
    for n in grids:
        d = discrepancy(n)
        mr = f"{prev_mean / d['mean']:10.2f}" if prev_mean else "         -"
        sr_ = f"{prev_sd / d['sd']:9.2f}" if prev_sd else "        -"
        print(f"{d['n_maturities']:7d}  {d['h']:8.4f}  {d['mean']:12.3e}  "
              f"{d['sd']:11.3e}  {d['corr_with_r']:7.3f}  {mr}  {sr_}")
        prev_mean, prev_sd = d["mean"], d["sd"]

    # --- Part 2: time step (the remaining candidate) ----------------
    steps = [50, 100, 200, 400, 800]

    print("\n" + "=" * 78)
    print("2. Discrepancy vs Euler time step (n_maturities = 401)")
    print("=" * 78)
    print("\nPart 1 showed the SPREAD converges with the maturity grid but the")
    print("MEAN does not -- a constant offset of ~2.1e-07 survives a 16-fold")
    print("refinement. Euler bias in the simulated curve is the remaining")
    print("candidate: it is invisible to the maturity grid and should shrink")
    print("as O(dt).\n")
    print(f"{'n_steps':>8}  {'dt':>8}  {'mean err':>12}  {'sd':>11}  "
          f"{'corr':>7}  {'mean ratio':>10}")
    print("-" * 78)

    prev_mean = None
    for n in steps:
        d = discrepancy(401, n_steps=n)
        mr = f"{prev_mean / d['mean']:10.2f}" if prev_mean else "         -"
        print(f"{n:8d}  {1.0/n:8.4f}  {d['mean']:12.3e}  {d['sd']:11.3e}  "
              f"{d['corr_with_r']:7.3f}  {mr}")
        prev_mean = d["mean"]

    print("\n" + "=" * 78)
    print("How to read part 2")
    print("=" * 78)
    print("""
  Mean ratios near 2.0:
      Euler bias, O(dt), as expected. The simulated curve carries a
      small time-discretisation error that the closed form does not,
      and it shows up as a constant offset because it enters the bond
      price multiplicatively through the accumulated forward rate.
      Benign, and the README's original attribution stands.

  Mean ratios near 1.0:
      the offset survives BOTH refinements, which points at a genuine
      inconsistency between the two price formulas rather than any
      discretisation. Worth finding before publishing a number that
      depends on them agreeing.
""")