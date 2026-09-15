"""
caplet.py

Caplet pricing by Monte Carlo under the HJM framework, with a
Hull-White closed-form validation and a two-factor extension where no
closed form exists.

A caplet is a one-period interest rate option: the rate is fixed at the
reset date T1, and the payoff

    tau * max(L(T1,T2) - K, 0)

is paid at T2, where tau = T2 - T1 and L is the simply-compounded
forward rate over that period.

Two modelling points worth stating up front:

  1. We simulate only to T1, not T2. The payoff is known at T1, and
     P(T1,T2) is available exactly from the simulated curve, so
     discounting one period back by multiplying by P(T1,T2) is exact.
     Simulating onward to T2 would add Euler steps and money-market
     integration error for no benefit.

  2. Under ExponentialVol the caplet has a closed form (it is a put on
     a zero-coupon bond). Under TwoFactorVol it does not -- two factors
     with different maturity profiles leave the forward bond price
     outside the one-dimensional lognormal family. That contrast is the
     point of this module: validate against the tractable case, then
     price the intractable one.
"""

import numpy as np

from src.curve import bond_price_from_forward_curve
from src.hullwhite import bond_option_price, make_flat_curve
from src.martingale_check import money_market_account
from src.simulate import ExponentialVol, TwoFactorVol, simulate_forward_curves


def forward_libor(P_T1_T2, tau):
    """
    The simply-compounded forward rate implied by a bond price:

        L = (1/tau) * (1/P(T1,T2) - 1)

    The simple-compounding counterpart of the instantaneous forward rate
    f(t,T) = -d(lnP)/dT from equation (1). Caplets are written on
    simply-compounded rates because that is market convention for
    floating-rate payments, so this conversion is needed even though the
    model works internally in continuous compounding.

    Sanity check: P = 0.98 over tau = 0.5 gives L = 4.08%.
    """
    return (1.0 / P_T1_T2 - 1.0) / tau


def caplet_payoff(P_T1_T2, K, tau):
    """
    Caplet payoff, discounted from T2 back to T1.

    Paid at T2:              tau * max(L - K, 0)
    Discounted back to T1:   P(T1,T2) * tau * max(L - K, 0)

    Substituting L and simplifying (see module notes) this equals

        max(1 - P(T1,T2)*(1 + tau*K), 0)

    which is the form used here: fewer operations, and it makes the
    put-on-a-bond structure explicit.
    """
    return np.maximum(1.0 - P_T1_T2 * (1.0 + tau * K), 0.0)


def caplet_closed_form(T1, T2, K, P0, sigma, a):
    """
    Hull-White closed-form caplet price, via the put-on-a-bond identity.

    A caplet equals (1 + tau*K) puts on the T2-maturity zero-coupon
    bond, expiring at T1, struck at 1/(1 + tau*K):

        caplet = (1 + tau*K) * Put(T1, T2, strike = 1/(1+tau*K))

    Derivation, starting from the payoff discounted to T1 (writing
    P = P(T1,T2)):

        P * tau * max(L - K, 0)
          = P * max( (1/P - 1) - tau*K, 0 )        [substituting L]
          = max( 1 - P*(1 + tau*K), 0 )            [P > 0]
          = (1 + tau*K) * max( 1/(1+tau*K) - P, 0 )

    The last line is exactly (1 + tau*K) puts struck at 1/(1+tau*K).

    Only valid for ExponentialVol -- it relies on the Hull-White
    one-factor structure underlying bond_option_price.
    """
    tau = T2 - T1
    strike = 1.0 / (1.0 + tau * K)
    put = bond_option_price(T1, T2, strike, P0, sigma, a, option_type="put")
    return (1.0 + tau * K) * put


def caplet_monte_carlo(
    vol_structure,
    T1,
    T2,
    K,
    f_flat=0.04,
    n_steps=200,
    n_paths=20000,
    n_maturities=401,
    max_maturity=10.0,
    seed=42,
    antithetic=True,
):
    """
    Price a caplet by Monte Carlo under any HJM volatility structure.

    Simulates the forward curve to T1, reads P(T1,T2) off each
    simulated curve, forms the payoff, and discounts by B(T1).

    Antithetic variates are on by default. As in the martingale check
    they reduce variance, not bias -- the payoff's kink at the strike
    makes them somewhat less effective here than for a linear
    functional, since the max() breaks the near-linearity in the shocks
    that antithetics exploit.

    Returns
    -------
    dict with the price, its standard error, and diagnostics.
    """
    if T2 <= T1:
        raise ValueError(f"T2 ({T2}) must exceed T1 ({T1})")
    if T2 > max_maturity:
        raise ValueError(f"T2 ({T2}) exceeds the maturity grid ({max_maturity})")

    tau = T2 - T1
    maturities = np.linspace(0.0, max_maturity, n_maturities)
    initial_curve = np.full(maturities.size, f_flat)

    n_factors = vol_structure.n_factors
    rng = np.random.default_rng(seed)
    shocks = rng.standard_normal((n_paths, n_steps, n_factors))

    shock_sets = [shocks, -shocks] if antithetic else [shocks]
    discounted = []

    for s in shock_sets:
        curves, times, short_rates = simulate_forward_curves(
            initial_curve=initial_curve,
            maturities=maturities,
            vol_structure=vol_structure,
            T_horizon=T1,
            n_steps=n_steps,
            n_paths=n_paths,
            shocks=s,
        )

        B_T1 = money_market_account(short_rates, times)[:, -1]

        P = np.zeros(n_paths)
        for p in range(n_paths):
            row = curves[p, -1, :]
            live = ~np.isnan(row)
            P[p] = bond_price_from_forward_curve(row[live], maturities[live], T1, T2)

        discounted.append(caplet_payoff(P, K, tau) / B_T1)

    if antithetic:
        estimates = 0.5 * (discounted[0] + discounted[1])
        se_plain = float(np.std(discounted[0], ddof=1) / np.sqrt(n_paths))
    else:
        estimates = discounted[0]
        se_plain = None

    price = float(np.mean(estimates))
    se = float(np.std(estimates, ddof=1) / np.sqrt(estimates.size))

    # Fraction of paths finishing in the money -- a useful diagnostic.
    # If it's near 0 or 1 the estimator is effectively pricing a
    # degenerate payoff and the standard error understates the
    # uncertainty in the tail.
    itm = float(np.mean(discounted[0] > 0))

    return {
        "price": price,
        "std_err": se,
        "std_err_plain": se_plain,
        "variance_reduction": se_plain / se if se_plain and se > 0 else None,
        "prob_itm": itm,
        "tau": tau,
    }


if __name__ == "__main__":
    f_flat = 0.04
    P0 = make_flat_curve(f_flat)
    T1, T2 = 1.0, 1.5
    tau = T2 - T1
    N = 20000

    # ATM strike: the forward rate the initial curve implies for
    # [T1, T2]. Parameter status: DEFENSIBLE -- at-the-money is the
    # natural reference point, and the least favourable case for Monte
    # Carlo accuracy, since the payoff kink sits where the density peaks.
    P_fwd = P0(T2) / P0(T1)
    K_atm = forward_libor(P_fwd, tau)

    print("=" * 76)
    print("1. Hull-White: Monte Carlo vs closed form")
    print("=" * 76)
    print(f"\nCaplet on [{T1}, {T2}], sigma=0.01, a=0.1, {N} paths\n")
    print(f"{'strike':>8}  {'moneyness':>11}  {'MC price':>12}  "
          f"{'closed form':>12}  {'diff':>10}  {'MC s.e.':>10}")
    print("-" * 76)

    vol_hw = ExponentialVol(sigma=0.01, a=0.1)

    for K in [K_atm - 0.01, K_atm - 0.005, K_atm, K_atm + 0.005, K_atm + 0.01]:
        mc = caplet_monte_carlo(vol_hw, T1, T2, K, f_flat=f_flat, n_paths=N)
        cf = caplet_closed_form(T1, T2, K, P0, 0.01, 0.1)
        label = {True: "ITM", False: "OTM"}[K < K_atm] if K != K_atm else "ATM"
        print(f"{K:8.4f}  {label:>11}  {mc['price']:12.8f}  {cf:12.8f}  "
              f"{mc['price'] - cf:10.2e}  {mc['std_err']:10.2e}")

    print("\n" + "=" * 76)
    print("2. Two-factor extension: no closed form exists")
    print("=" * 76)
    print("\nsigma_1 = 0.008 (level), sigma_2 = 0.006, lambda = 0.3 (slope)")
    print("\nTwo factors with different maturity profiles leave the forward")
    print("bond price outside the one-dimensional lognormal family, so the")
    print("put-on-a-bond identity no longer yields a closed form. Monte")
    print("Carlo under the general HJM framework still prices it directly.\n")

    vol_2f = TwoFactorVol(sigma_1=0.008, sigma_2=0.006, lam=0.3)

    print(f"{'strike':>8}  {'2-factor MC':>13}  {'s.e.':>10}  "
          f"{'P(ITM)':>8}  {'var red':>8}")
    print("-" * 76)
    for K in [K_atm - 0.01, K_atm - 0.005, K_atm, K_atm + 0.005, K_atm + 0.01]:
        mc = caplet_monte_carlo(vol_2f, T1, T2, K, f_flat=f_flat, n_paths=N)
        vr = mc["variance_reduction"]
        print(f"{K:8.4f}  {mc['price']:13.8f}  {mc['std_err']:10.2e}  "
              f"{mc['prob_itm']:8.3f}  {vr:7.1f}x")