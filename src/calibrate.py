import numpy as np

from src.hullwhite import make_flat_curve
from src.simulate import ConstantVol, ExponentialVol, TwoFactorVol
from src.caplet import caplet_monte_carlo, caplet_closed_form, forward_libor


# ----------------------------------------------------------------------
# The matching variance
# ----------------------------------------------------------------------


def integrated_forward_variance_numeric(vol_structure, T1, T2, n_v=4001, n_s=2001):
    """
    V = sum_i integral_0^T1 [ integral_T1^T2 sigma_i(v,s) ds ]^2 dv,
    by direct numerical integration.

    Deliberately written without reference to any particular volatility
    structure, so it can cross-check the analytic formulas below. Slow
    but general.
    """
    v_grid = np.linspace(0.0, T1, n_v)
    s_grid = np.linspace(T1, T2, n_s)

    total = 0.0
    for v in v_grid[:1]:
        pass  # placeholder to keep structure clear; real loop below

    # For each v, integrate sigma_i(v, s) over s, square, sum over
    # factors. Vectorised over s; looped over v.
    inner = np.zeros((v_grid.size,))
    for k, v in enumerate(v_grid):
        sig = vol_structure(v, s_grid)              # (n_factors, n_s)
        per_factor = np.trapezoid(sig, s_grid, axis=1)   # (n_factors,)
        inner[k] = np.sum(per_factor**2)

    return float(np.trapezoid(inner, v_grid))


def integrated_forward_variance_exponential(sigma, a, T1, T2):
    """
    Analytic V for sigma(v,s) = sigma*exp(-a(s-v)).

    Inner integral:
        integral_T1^T2 sigma*exp(-a(s-v)) ds
          = (sigma/a) * exp(a*v) * [exp(-a*T1) - exp(-a*T2)]
          = C * exp(a*v),   C = (sigma/a)*(exp(-a*T1) - exp(-a*T2))

    Outer:
        integral_0^T1 C^2 exp(2*a*v) dv = C^2 * (exp(2*a*T1) - 1)/(2a)

    Note V is exactly quadratic in sigma, which makes calibration a
    closed-form rescaling rather than a root-find.
    """
    C = (sigma / a) * (np.exp(-a * T1) - np.exp(-a * T2))
    return C**2 * (np.exp(2 * a * T1) - 1.0) / (2.0 * a)


def integrated_forward_variance_constant(sigma, T1, T2):
    """
    Analytic V for constant sigma.

    Inner integral is sigma*(T2-T1), independent of v, so
        V = sigma^2 * (T2-T1)^2 * T1
    """
    return sigma**2 * (T2 - T1) ** 2 * T1


def integrated_forward_variance_two_factor(sigma_1, sigma_2, lam, T1, T2):
    """
    Analytic V for the two-factor structure.

    Factor 1 is constant, factor 2 is exponential with decay rate
    lam/2. Independent factors contribute additively -- their cross
    terms vanish because dW_1 dW_2 = 0.
    """
    V1 = integrated_forward_variance_constant(sigma_1, T1, T2)
    V2 = integrated_forward_variance_exponential(sigma_2, lam / 2.0, T1, T2)
    return V1 + V2


def calibrate_exponential_to_variance(V_target, a, T1, T2):
    """
    Find the sigma making ExponentialVol(sigma, a) match V_target.

    V is exactly quadratic in sigma, so with V(1) the variance at
    sigma = 1:

        sigma = sqrt(V_target / V(1))

    No root-finding needed. The mean-reversion speed `a` is held fixed:
    it controls the SHAPE of the volatility term structure, and we are
    deliberately varying only the overall level so the comparison
    isolates shape from magnitude.
    """
    V_unit = integrated_forward_variance_exponential(1.0, a, T1, T2)
    return float(np.sqrt(V_target / V_unit))


# ----------------------------------------------------------------------


def run_comparison(
    sigma_1=0.008,
    sigma_2=0.006,
    lam=0.3,
    a=0.1,
    T1=1.0,
    T2=1.5,
    f_flat=0.04,
    n_paths=20000,
    n_steps=200,
    seed=42,
):
    """
    Price the same caplets under the two-factor structure and under a
    one-factor structure calibrated to the same integrated forward
    variance.
    """
    P0 = make_flat_curve(f_flat)
    tau = T2 - T1
    K_atm = forward_libor(P0(T2) / P0(T1), tau)
    strikes = [K_atm - 0.01, K_atm - 0.005, K_atm, K_atm + 0.005, K_atm + 0.01]

    vol_2f = TwoFactorVol(sigma_1, sigma_2, lam)

    V_2f = integrated_forward_variance_two_factor(sigma_1, sigma_2, lam, T1, T2)
    sigma_matched = calibrate_exponential_to_variance(V_2f, a, T1, T2)
    vol_1f = ExponentialVol(sigma_matched, a)
    V_1f = integrated_forward_variance_exponential(sigma_matched, a, T1, T2)

    # Uncalibrated baseline, for reference against the caplet.py table.
    vol_base = ExponentialVol(0.01, a)
    V_base = integrated_forward_variance_exponential(0.01, a, T1, T2)

    rows = []
    for K in strikes:
        p2 = caplet_monte_carlo(vol_2f, T1, T2, K, f_flat=f_flat,
                                n_paths=n_paths, n_steps=n_steps, seed=seed)
        p1 = caplet_monte_carlo(vol_1f, T1, T2, K, f_flat=f_flat,
                                n_paths=n_paths, n_steps=n_steps, seed=seed)
        cf = caplet_closed_form(T1, T2, K, P0, sigma_matched, a)
        rows.append((K, p2, p1, cf))

    return {
        "rows": rows,
        "K_atm": K_atm,
        "V_2f": V_2f,
        "V_1f": V_1f,
        "V_base": V_base,
        "sigma_matched": sigma_matched,
    }


if __name__ == "__main__":
    T1, T2 = 1.0, 1.5
    a = 0.1
    sigma_1, sigma_2, lam = 0.008, 0.006, 0.3

    print("=" * 76)
    print("Verifying the analytic variance formulas numerically")
    print("=" * 76)
    print()

    checks = [
        ("ConstantVol(0.01)", ConstantVol(0.01),
         integrated_forward_variance_constant(0.01, T1, T2)),
        ("ExponentialVol(0.01, 0.1)", ExponentialVol(0.01, a),
         integrated_forward_variance_exponential(0.01, a, T1, T2)),
        ("TwoFactorVol(0.008, 0.006, 0.3)", TwoFactorVol(sigma_1, sigma_2, lam),
         integrated_forward_variance_two_factor(sigma_1, sigma_2, lam, T1, T2)),
    ]

    print(f"{'structure':>34}  {'analytic':>13}  {'numeric':>13}  {'rel diff':>10}")
    print("-" * 76)
    for name, vol, V_analytic in checks:
        V_numeric = integrated_forward_variance_numeric(vol, T1, T2)
        rel = abs(V_analytic - V_numeric) / V_analytic
        print(f"{name:>34}  {V_analytic:13.6e}  {V_numeric:13.6e}  {rel:10.2e}")

    print("\n" + "=" * 76)
    print("Calibrated comparison: one factor vs two, matched on variance")
    print("=" * 76)

    res = run_comparison(sigma_1, sigma_2, lam, a, T1, T2)

    print(f"\n  Two-factor V:              {res['V_2f']:.6e}")
    print(f"  Calibrated one-factor V:   {res['V_1f']:.6e}")
    print(f"  Calibrated sigma:          {res['sigma_matched']:.6f}")
    print(f"  (uncalibrated sigma = 0.01 gives V = {res['V_base']:.6e}, "
          f"{res['V_base'] / res['V_2f']:.3f}x the two-factor variance)")

    print(f"\n{'strike':>8}  {'two-factor':>12}  {'one-factor':>12}  "
          f"{'difference':>11}  {'combined s.e.':>13}  {'t':>6}")
    print("-" * 76)

    for K, p2, p1, cf in res["rows"]:
        d = p2["price"] - p1["price"]
        se = np.sqrt(p2["std_err"] ** 2 + p1["std_err"] ** 2)
        print(f"{K:8.4f}  {p2['price']:12.8f}  {p1['price']:12.8f}  "
              f"{d:11.2e}  {se:13.2e}  {d / se:6.2f}")

    print("\n" + "=" * 76)
    print("One-factor Monte Carlo vs its own closed form (a sanity check)")
    print("=" * 76)
    print("\nConfirms the calibrated sigma is being used consistently on both")
    print("sides, so any two-factor difference above is not an artefact.\n")
    print(f"{'strike':>8}  {'MC':>12}  {'closed form':>12}  {'diff':>11}  {'s.e.':>10}")
    print("-" * 76)
    for K, p2, p1, cf in res["rows"]:
        print(f"{K:8.4f}  {p1['price']:12.8f}  {cf:12.8f}  "
              f"{p1['price'] - cf:11.2e}  {p1['std_err']:10.2e}")