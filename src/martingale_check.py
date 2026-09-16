import numpy as np

from src.curve import bond_price_from_forward_curve, bond_prices_vectorised
from src.hullwhite import make_flat_curve
from src.simulate import ExponentialVol, simulate_forward_curves


# ----------------------------------------------------------------------
# Core quantities
# ----------------------------------------------------------------------


def money_market_account(short_rates, times):
    """
    B(t_k) = exp(integral_0^{t_k} r(y) dy), per path, at every time step.

    The integral is approximated by the cumulative trapezoidal rule along
    the time axis. This is a distinct error source from the Euler
    stepping and from the maturity-axis integration -- worth naming,
    since all three contribute to the residual bias measured below.

    Parameters
    ----------
    short_rates : np.ndarray, shape (n_paths, n_times)
    times : np.ndarray, shape (n_times,)

    Returns
    -------
    np.ndarray, shape (n_paths, n_times)
        B(t_k) per path. B[:, 0] = 1 by construction.
    """
    dt = np.diff(times)

    trapezoids = 0.5 * (short_rates[:, 1:] + short_rates[:, :-1]) * dt

    # Leading zero: the integral from 0 to 0 is 0, so B(0) = exp(0) = 1.
    integral = np.concatenate(
        [np.zeros((short_rates.shape[0], 1)), np.cumsum(trapezoids, axis=1)],
        axis=1,
    )

    return np.exp(integral)


def discounted_bond_prices(curves, times, maturities, short_rates, T, k):
    """
    Z(t_k, T) = P(t_k, T) / B(t_k) for every path.

    P comes from integrating the simulated forward curve (equation 2);
    B from accumulating the simulated short rate. Both are path-
    dependent, hence the per-path loop.

    Returns
    -------
    np.ndarray, shape (n_paths,)
    """
    n_paths = curves.shape[0]
    t = times[k]

    if T <= t:
        raise ValueError(f"Maturity {T} must exceed evaluation time {t}")

    B = money_market_account(short_rates, times)[:, k]
    P = bond_prices_vectorised(curves[:, k, :], maturities, t, T)

    return P / B


# ----------------------------------------------------------------------
# Shock handling -- the machinery that makes pairing possible
# ----------------------------------------------------------------------


def _aggregate_shocks(shocks_fine, n_steps_coarse):
    """
    Aggregate fine-resolution shocks into coarse-step shocks describing
    the SAME underlying Brownian path.

    A Brownian increment over a coarse step is the sum of the fine
    increments inside it:

        dW_coarse = sum_i dW_fine,i

    The simulator uses standard normals scaled by sqrt(dt), so with
    dt_coarse = m * dt_fine:

        sqrt(m*dt_f) * Z_coarse = sqrt(dt_f) * sum_i Z_fine,i
        =>  Z_coarse = (1/sqrt(m)) * sum_i Z_fine,i

    The 1/sqrt(m) preserves unit variance (a sum of m independent
    standard normals has variance m).

    This is what makes the convergence study a genuine paired
    comparison. Reusing a seed across different step counts does NOT
    pair anything: a different n_steps consumes a differently shaped
    draw, so the runs follow different Brownian paths and nothing
    cancels in the difference.

    Parameters
    ----------
    shocks_fine : np.ndarray, shape (n_paths, n_steps_fine, n_factors)
    n_steps_coarse : int
        Must divide n_steps_fine exactly.

    Returns
    -------
    np.ndarray, shape (n_paths, n_steps_coarse, n_factors)
    """
    n_paths, n_steps_fine, n_factors = shocks_fine.shape

    if n_steps_fine % n_steps_coarse != 0:
        raise ValueError(
            f"n_steps_coarse ({n_steps_coarse}) must divide "
            f"n_steps_fine ({n_steps_fine})"
        )

    m = n_steps_fine // n_steps_coarse

    # Reshape so each coarse step's block of m fine steps occupies its
    # own axis, then sum over that axis.
    blocks = shocks_fine.reshape(n_paths, n_steps_coarse, m, n_factors)
    return blocks.sum(axis=2) / np.sqrt(m)


def _Z_from_shocks(
    vol_structure,
    shocks,
    T,
    drift_on=True,
    f_flat=0.04,
    T_horizon=1.0,
    n_maturities=401,
    max_maturity=10.0,
):
    """
    Run the simulator on a supplied shock array and return Z(t,T) per
    path at the final time.

    Setting drift_on=False suppresses the no-arbitrage drift. That is a
    testing device only: a zero-drift HJM model admits arbitrage, which
    is precisely why equation (18) exists. It is monkeypatched here
    rather than exposed as a simulator option, so the production API
    cannot produce an arbitrageable model.
    """
    import src.simulate as sim

    maturities = np.linspace(0.0, max_maturity, n_maturities)
    initial_curve = np.full(maturities.size, f_flat)
    n_paths, n_steps, _ = shocks.shape

    original = sim.hjm_drift
    if not drift_on:
        sim.hjm_drift = lambda vol, t, mats: np.zeros(np.asarray(mats).size)
    try:
        curves, times, short_rates = sim.simulate_forward_curves(
            initial_curve=initial_curve,
            maturities=maturities,
            vol_structure=vol_structure,
            T_horizon=T_horizon,
            n_steps=n_steps,
            n_paths=n_paths,
            shocks=shocks,
            store_history=False,
        )
    finally:
        sim.hjm_drift = original

    return discounted_bond_prices(curves, times, maturities, short_rates, T, -1)


# ----------------------------------------------------------------------
# 1. The martingale test, with antithetic variates
# ----------------------------------------------------------------------


def martingale_test_antithetic(
    vol_structure,
    f_flat=0.04,
    T_horizon=1.0,
    n_steps=200,
    n_paths=5000,
    n_maturities=401,
    max_maturity=10.0,
    test_maturities=(3.0, 5.0, 8.0),
    seed=42,
):
    """
    E~[Z(t,T)] = P(0,T), with antithetic variates for variance
    reduction.

    For every path driven by shocks Z, a companion path is driven by -Z.
    Over a one-year horizon Z(t,T) is nearly linear in the shocks, so
    the pair is strongly negatively correlated and their average has
    substantially lower variance than either alone.

    The estimator averages within each pair; the standard error is
    computed ACROSS PAIRS, since paths within a pair are dependent by
    construction and treating them as independent would understate the
    error.

    Note antithetics remove VARIANCE, not BIAS. If Euler discretization
    introduces a systematic bias, this tightens the error bars around a
    slightly-off centre rather than moving the centre. Section 2 is what
    diagnoses that.
    """
    maturities = np.linspace(0.0, max_maturity, n_maturities)
    initial_curve = np.full(maturities.size, f_flat)
    P0 = make_flat_curve(f_flat)

    n_factors = vol_structure.n_factors
    rng = np.random.default_rng(seed)
    shocks = rng.standard_normal((n_paths, n_steps, n_factors))

    c_plus, times, r_plus = simulate_forward_curves(
        initial_curve=initial_curve,
        maturities=maturities,
        vol_structure=vol_structure,
        T_horizon=T_horizon,
        n_steps=n_steps,
        n_paths=n_paths,
        shocks=shocks,
        store_history=False,
    )
    c_minus, _, r_minus = simulate_forward_curves(
        initial_curve=initial_curve,
        maturities=maturities,
        vol_structure=vol_structure,
        T_horizon=T_horizon,
        n_steps=n_steps,
        n_paths=n_paths,
        shocks=-shocks,
        store_history=False,
    )

    results = {}
    for T in test_maturities:
        Z_plus = discounted_bond_prices(c_plus, times, maturities, r_plus, T, -1)
        Z_minus = discounted_bond_prices(c_minus, times, maturities, r_minus, T, -1)

        Z_pair = 0.5 * (Z_plus + Z_minus)

        target = float(P0(T))
        mean_Z = float(np.mean(Z_pair))
        bias = mean_Z - target
        se = float(np.std(Z_pair, ddof=1) / np.sqrt(Z_pair.size))
        se_plain = float(np.std(Z_plus, ddof=1) / np.sqrt(Z_plus.size))

        results[T] = {
            "mean_Z": mean_Z,
            "target": target,
            "bias": bias,
            "std_err": se,
            "std_err_plain": se_plain,
            "variance_reduction": se_plain / se if se > 0 else np.nan,
            "t_stat": bias / se if se > 0 else np.nan,
        }

    return results


# ----------------------------------------------------------------------
# 2. Convergence in dt, on a shared Brownian path
# ----------------------------------------------------------------------


def paired_convergence(
    vol_structure,
    n_steps_list=(25, 50, 100, 200, 400),
    n_steps_ref=800,
    T=5.0,
    n_paths=500,
    seed=42,
    **kwargs,
):
    """
    Discretization error against a high-accuracy reference, measured
    path by path on a SHARED Brownian path.

    Shocks are drawn once at the reference resolution, then aggregated
    into coarser step counts by _aggregate_shocks. Every run therefore
    follows the same underlying Brownian motion, so differencing cancels
    the sampling noise almost entirely.

    Euler-Maruyama has weak order O(dt), so halving dt should roughly
    halve the difference from the reference.

    Caveat: the reference carries its own O(dt) error at n_steps_ref. As
    the coarse runs approach that accuracy, the measured difference
    stops halving cleanly -- expect the ratio to degrade in the last row
    or two. That is the reference's error becoming comparable to what is
    being measured, not a failure of the method.

    Every entry in n_steps_list must divide n_steps_ref.

    Returns
    -------
    list of (n_steps, mean difference vs reference, standard error)
    """
    for n in n_steps_list:
        if n_steps_ref % n != 0:
            raise ValueError(f"{n} does not divide n_steps_ref ({n_steps_ref})")

    n_factors = vol_structure.n_factors
    rng = np.random.default_rng(seed)
    shocks_ref = rng.standard_normal((n_paths, n_steps_ref, n_factors))

    Z_ref = _Z_from_shocks(vol_structure, shocks_ref, T, **kwargs)

    rows = []
    for n_steps in n_steps_list:
        shocks = _aggregate_shocks(shocks_ref, n_steps)
        Z = _Z_from_shocks(vol_structure, shocks, T, **kwargs)

        D = Z - Z_ref
        rows.append((
            n_steps,
            float(np.mean(D)),
            float(np.std(D, ddof=1) / np.sqrt(D.size)),
        ))

    return rows


# ----------------------------------------------------------------------
# 3. Paired drift control -- does the test have power?
# ----------------------------------------------------------------------


def paired_drift_control(
    vol_structure, T=5.0, n_paths=5000, n_steps=200, seed=42, **kwargs
):
    """
    Path-wise paired control: does removing the no-arbitrage drift shift
    Z by a detectable amount?

    Both runs use the identical shock array, so the diffusion cancels
    within each path:

        D_p = Z_p(drift on) - Z_p(drift off)

    D_p therefore has variance orders of magnitude below Z_p itself,
    making the drift's effect measurable at a few thousand paths rather
    than the millions an unpaired comparison would need.

    This establishes the test's SENSITIVITY. Without it, a small bias in
    section 1 would be ambiguous between "no violation" and "a test too
    blunt to see one".
    """
    n_factors = vol_structure.n_factors
    rng = np.random.default_rng(seed)
    shocks = rng.standard_normal((n_paths, n_steps, n_factors))

    Z_on = _Z_from_shocks(vol_structure, shocks, T, drift_on=True, **kwargs)
    Z_off = _Z_from_shocks(vol_structure, shocks, T, drift_on=False, **kwargs)

    D = Z_on - Z_off
    sd = float(np.std(D, ddof=1))
    se = sd / np.sqrt(D.size)

    return {
        "mean_diff": float(np.mean(D)),
        "std_diff": sd,
        "se_diff": float(se),
        "t_stat": float(np.mean(D) / se) if se > 0 else np.nan,
        "mean_Z_on": float(np.mean(Z_on)),
        "std_Z_on": float(np.std(Z_on, ddof=1)),
    }


# ----------------------------------------------------------------------


if __name__ == "__main__":
    vol = ExponentialVol(sigma=0.01, a=0.1)
    N = 5000

    print("=" * 78)
    print("1. Martingale property, with antithetic variates")
    print("=" * 78)
    print(f"\nExponentialVol(sigma=0.01, a=0.1), {N} pairs, 200 steps, t = 1.0\n")

    res = martingale_test_antithetic(vol, n_paths=N)

    print(f"{'T':>5}  {'mean Z':>12}  {'P(0,T)':>12}  {'bias':>10}  "
          f"{'std err':>10}  {'var red':>8}  {'t':>7}")
    print("-" * 78)
    for T, r in res.items():
        print(f"{T:5.1f}  {r['mean_Z']:12.8f}  {r['target']:12.8f}  "
              f"{r['bias']:10.2e}  {r['std_err']:10.2e}  "
              f"{r['variance_reduction']:7.1f}x  {r['t_stat']:7.2f}")

    print("\n" + "=" * 78)
    print("2. Discretization error vs an 800-step reference (T = 5.0)")
    print("=" * 78)
    print("\nAll runs share one Brownian path, sampled at different")
    print("frequencies, so path-wise differencing cancels the sampling noise.")
    print("Euler is weak order O(dt): halving dt should roughly halve the error.\n")
    print(f"{'n_steps':>9}  {'mean diff':>11}  {'std err':>10}  {'ratio':>8}")
    print("-" * 44)

    prev = None
    for n_steps, diff, se in paired_convergence(vol, n_paths=500, n_steps_ref=800):
        ratio = f"{abs(prev / diff):8.2f}" if prev is not None else "       -"
        print(f"{n_steps:9d}  {diff:11.2e}  {se:10.2e}  {ratio}")
        prev = diff

    print("\n" + "=" * 78)
    print("3. Paired control: the drift's effect on Z (T = 5.0)")
    print("=" * 78)
    print("\nPath-wise difference between drift-on and drift-off runs sharing")
    print("the same shocks. Establishes the test can detect a broken drift.\n")

    ctrl = paired_drift_control(vol, n_paths=N)

    print(f"  mean Z (drift on):      {ctrl['mean_Z_on']:.8f}")
    print(f"  sd of Z across paths:   {ctrl['std_Z_on']:.2e}")
    print(f"  mean paired difference: {ctrl['mean_diff']:.4e}")
    print(f"  sd of the difference:   {ctrl['std_diff']:.2e}")
    print(f"  standard error:         {ctrl['se_diff']:.2e}")
    print(f"  t-statistic:            {ctrl['t_stat']:.1f}")
    print(f"\n  Variance reduction from pairing: "
          f"{ctrl['std_Z_on'] / ctrl['std_diff']:.0f}x")

    print("\n" + "=" * 78)
    print("Reading these together")
    print("=" * 78)
    b1 = abs(res[5.0]["bias"])
    b3 = abs(ctrl["mean_diff"])
    print(f"\n  Residual bias with the drift on (T=5):  {b1:.2e}")
    print(f"  Bias induced by removing the drift:    {b3:.2e}")
    print(f"  Ratio: the test detects a violation {b3 / b1:.0f}x larger")
    print("  than the residual it reports -- so 'no violation observed'")
    print("  reflects the model, not the test's blindness.\n")