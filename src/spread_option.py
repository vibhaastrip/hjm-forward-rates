"""
spread_option.py

Prices an option on the SPREAD between two forward rates observed at the
same date, and compares a two-factor HJM structure against a one-factor
structure calibrated to reproduce both marginal variances.

Payoff at T1:   max( L(T1, T1+delta_A) - L(T1, T1+delta_B) - K, 0 )

This is a steepener: it pays when the short rate exceeds the long rate
by more than K, so its value depends on the SLOPE of the curve.

Why this instrument and not a caplet. A caplet depends on one forward
rate, so its price is fixed by that rate's marginal distribution -- which
is why matching a single variance made the two structures
indistinguishable (see calibrate.py). A spread option depends on

    Var(L_A - L_B) = Var(L_A) + Var(L_B) - 2 Cov(L_A, L_B)

so it is sensitive to the CORRELATION, which the marginals do not pin
down. A one-factor model drives both rates from the same Brownian
motion, forcing their correlation near 1 and compressing the spread; two
factors with different maturity profiles let them decorrelate.

The calibration matches BOTH marginal variances, using the two free
parameters (sigma, a) of the exponential structure. Any remaining price
difference is therefore attributable to correlation alone, not to the
rates having different volatilities.

Prediction, stated before running: the one-factor model should UNDERPRICE
the spread option, because it cannot produce enough decorrelation.
"""

import numpy as np

from src.calibrate import (
    calibrate_exponential_two_variances,
    integrated_forward_variance_two_factor,
)
from src.caplet import forward_libor
from src.curve import bond_prices_vectorised
from src.hullwhite import make_flat_curve
from src.martingale_check import money_market_account
from src.simulate import ExponentialVol, TwoFactorVol, simulate_forward_curves


F_FLAT = 0.04


def spread_option_monte_carlo(
    vol_structure,
    T1,
    delta_A,
    delta_B,
    K,
    f_flat=F_FLAT,
    n_steps=200,
    n_paths=20000,
    n_maturities=401,
    max_maturity=15.0,
    seed=42,
    antithetic=True,
    return_rates=False,
):
    """
    Price max(L_A - L_B - K, 0) paid at T1, by Monte Carlo.

    Both rates are read off the SAME simulated curve at T1, so their
    joint dependence is whatever the volatility structure implies -- no
    correlation parameter is imposed anywhere. That is the point: the
    correlation is an output of the model, not an input.

    Note max_maturity defaults to 15 rather than 10, since the long
    accrual can extend to T1 + 9 and the curve must cover it with room
    for the drift integral beyond.

    Returns
    -------
    dict with the price, standard error, the realised correlation
    between the two rates, and diagnostics.
    """
    if delta_B <= delta_A:
        raise ValueError("delta_B must exceed delta_A (long vs short accrual)")
    if T1 + delta_B > max_maturity:
        raise ValueError(
            f"T1 + delta_B ({T1 + delta_B}) exceeds the grid ({max_maturity})"
        )

    maturities = np.linspace(0.0, max_maturity, n_maturities)
    initial_curve = np.full(maturities.size, f_flat)

    rng = np.random.default_rng(seed)
    shocks = rng.standard_normal((n_paths, n_steps, vol_structure.n_factors))
    shock_sets = [shocks, -shocks] if antithetic else [shocks]

    discounted, rates_A, rates_B = [], [], []

    for s in shock_sets:
        curves, times, short_rates = simulate_forward_curves(
            initial_curve=initial_curve,
            maturities=maturities,
            vol_structure=vol_structure,
            T_horizon=T1,
            n_steps=n_steps,
            n_paths=n_paths,
            shocks=s,
            store_history=False,
        )

        B_T1 = money_market_account(short_rates, times)[:, -1]
        slice_ = curves[:, -1, :]

        P_A = bond_prices_vectorised(slice_, maturities, T1, T1 + delta_A)
        P_B = bond_prices_vectorised(slice_, maturities, T1, T1 + delta_B)

        L_A = forward_libor(P_A, delta_A)
        L_B = forward_libor(P_B, delta_B)

        payoff = np.maximum(L_A - L_B - K, 0.0)
        discounted.append(payoff / B_T1)
        rates_A.append(L_A)
        rates_B.append(L_B)

    if antithetic:
        est = 0.5 * (discounted[0] + discounted[1])
        se_plain = float(np.std(discounted[0], ddof=1) / np.sqrt(n_paths))
    else:
        est = discounted[0]
        se_plain = None

    price = float(np.mean(est))
    se = float(np.std(est, ddof=1) / np.sqrt(est.size))

    L_A, L_B = rates_A[0], rates_B[0]
    spread = L_A - L_B

    out = {
        "price": price,
        "std_err": se,
        "variance_reduction": se_plain / se if se_plain and se > 0 else None,
        "prob_itm": float(np.mean(discounted[0] > 0)),
        "corr": float(np.corrcoef(L_A, L_B)[0, 1]),
        "sd_A": float(np.std(L_A, ddof=1)),
        "sd_B": float(np.std(L_B, ddof=1)),
        "sd_spread": float(np.std(spread, ddof=1)),
        "mean_spread": float(np.mean(spread)),
    }
    if return_rates:
        out["L_A"], out["L_B"] = L_A, L_B
    return out


def calibrated_pair(sigma_1, sigma_2, lam, T1, delta_A, delta_B):
    """
    Build the two-factor structure and the one-factor structure
    calibrated to reproduce both of its marginal variances.
    """
    V_A = integrated_forward_variance_two_factor(
        sigma_1, sigma_2, lam, T1, T1 + delta_A)
    V_B = integrated_forward_variance_two_factor(
        sigma_1, sigma_2, lam, T1, T1 + delta_B)

    cal = calibrate_exponential_two_variances(V_A, V_B, T1, delta_A, delta_B)
    if not cal["feasible"]:
        raise ValueError(
            f"No one-factor structure matches both marginals: target ratio "
            f"{cal['target_ratio']:.6f}, achievable "
            f"{cal['achievable_range']}"
        )

    return (TwoFactorVol(sigma_1, sigma_2, lam),
            ExponentialVol(cal["sigma"], cal["a"]),
            cal)


if __name__ == "__main__":
    sigma_1, sigma_2, lam = 0.008, 0.006, 0.3
    T1 = 1.0
    N = 20000

    # --- Part 1: realistic tenors -----------------------------------
    # 6-month against 5-year: a standard steepener pair, chosen for
    # realism rather than to maximise the effect.
    delta_A, delta_B = 0.5, 5.0

    vol_2f, vol_1f, cal = calibrated_pair(
        sigma_1, sigma_2, lam, T1, delta_A, delta_B)

    print("=" * 78)
    print(f"Spread option: {delta_A}y rate minus {delta_B}y rate, "
          f"observed at T1 = {T1}")
    print("=" * 78)
    print(f"\n  Two-factor:  sigma_1={sigma_1}, sigma_2={sigma_2}, "
          f"lambda={lam}")
    print(f"  One-factor:  sigma={cal['sigma']:.6f}, a={cal['a']:.4f}")
    print(f"               (calibrated to match BOTH marginal variances)")
    print(f"\n  Marginal variance targets vs achieved:")
    print(f"    short:  {cal['V_A_target']:.6e}  ->  "
          f"{cal['V_A_achieved']:.6e}")
    print(f"    long:   {cal['V_B_target']:.6e}  ->  "
          f"{cal['V_B_achieved']:.6e}")

    # Price at the money first, to fix a sensible strike scale.
    base_2f = spread_option_monte_carlo(vol_2f, T1, delta_A, delta_B,
                                        K=0.0, n_paths=N)
    K_ref = base_2f["mean_spread"]

    print(f"\n  Realised correlation between the two rates:")
    base_1f = spread_option_monte_carlo(vol_1f, T1, delta_A, delta_B,
                                        K=0.0, n_paths=N)
    print(f"    two-factor: {base_2f['corr']:.6f}")
    print(f"    one-factor: {base_1f['corr']:.6f}")
    print(f"\n  Spread standard deviation:")
    print(f"    two-factor: {base_2f['sd_spread']:.6e}")
    print(f"    one-factor: {base_1f['sd_spread']:.6e}")
    print(f"    ratio:      {base_2f['sd_spread']/base_1f['sd_spread']:.4f}")

    strikes = K_ref + np.array([-0.002, -0.001, 0.0, 0.001, 0.002])

    print(f"\n{'strike':>9}  {'two-factor':>12}  {'one-factor':>12}  "
          f"{'difference':>11}  {'comb s.e.':>10}  {'t':>7}  {'P(ITM)':>7}")
    print("-" * 78)

    for K in strikes:
        p2 = spread_option_monte_carlo(vol_2f, T1, delta_A, delta_B, K,
                                       n_paths=N)
        p1 = spread_option_monte_carlo(vol_1f, T1, delta_A, delta_B, K,
                                       n_paths=N)
        d = p2["price"] - p1["price"]
        se = np.sqrt(p2["std_err"] ** 2 + p1["std_err"] ** 2)
        t = d / se if se > 0 else np.nan
        print(f"{K:9.5f}  {p2['price']:12.8f}  {p1['price']:12.8f}  "
              f"{d:11.2e}  {se:10.2e}  {t:7.2f}  {p2['prob_itm']:7.3f}")

    # --- Part 2: sensitivity to the tenor gap -----------------------
    print("\n" + "=" * 78)
    print("Sensitivity: how far apart must the tenors be?")
    print("=" * 78)
    print("\nAt-the-money spread options across widening tenor gaps. If the")
    print("second factor matters through decorrelation, the effect should")
    print("grow as the two accruals separate.\n")
    print(f"{'short':>6}  {'long':>6}  {'corr 2f':>9}  {'corr 1f':>9}  "
          f"{'2f price':>11}  {'1f price':>11}  {'ratio':>7}  {'t':>7}")
    print("-" * 78)

    for dA, dB in [(0.5, 1.0), (0.5, 2.0), (0.5, 5.0), (0.5, 9.0),
                   (0.25, 9.0)]:
        try:
            v2, v1, c = calibrated_pair(sigma_1, sigma_2, lam, T1, dA, dB)
        except ValueError:
            print(f"{dA:6.2f}  {dB:6.2f}  {'infeasible calibration':>60}")
            continue

        b2 = spread_option_monte_carlo(v2, T1, dA, dB, K=0.0, n_paths=N)
        K = b2["mean_spread"]

        p2 = spread_option_monte_carlo(v2, T1, dA, dB, K, n_paths=N)
        p1 = spread_option_monte_carlo(v1, T1, dA, dB, K, n_paths=N)

        d = p2["price"] - p1["price"]
        se = np.sqrt(p2["std_err"] ** 2 + p1["std_err"] ** 2)
        ratio = p2["price"] / p1["price"] if p1["price"] > 0 else np.nan
        t = d / se if se > 0 else np.nan

        print(f"{dA:6.2f}  {dB:6.2f}  {p2['corr']:9.5f}  {p1['corr']:9.5f}  "
              f"{p2['price']:11.8f}  {p1['price']:11.8f}  "
              f"{ratio:7.4f}  {t:7.2f}")

    print("\n" + "=" * 78)
    print("How to read this")
    print("=" * 78)
    print("""
  The correlation columns are the mechanism. A one-factor model drives
  both rates from the same Brownian motion, so its correlation should
  sit very close to 1 regardless of tenor. The two-factor correlation
  should fall as the accruals separate, because the slope factor hits
  the short rate harder than the long one.

  If the price ratio rises above 1 and grows with the tenor gap, the
  second factor is adding value through decorrelation -- the effect a
  caplet was structurally unable to see.

  If the ratio stays at 1 throughout, then matching both marginals is
  enough to match the spread too, and the two-factor structure adds
  nothing this instrument can detect either.
""")