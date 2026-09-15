import numpy as np


def forward_rate_from_bond_prices(P_values, maturities, dT = None):
    log_P = np.log(P_values)

    if dT is None:
        d_log_P = np.gradient(log_P, maturities)
    else:
        d_log_P = np.gradient(log_P, dT)

    return -d_log_P

def bond_price_from_forward_curve(forward_curve, maturities, t, T):
    if T < t:
        raise ValueError(f"T ({T}) must be >= t ({t})")
    if T == t:
        return 1.0  

    interior_mask = (maturities > t) & (maturities < T)
    interior_s = maturities[interior_mask]
    s_values = np.concatenate(([t], interior_s, [T]))

    f_values = np.interp(s_values, maturities, forward_curve)

    integral = np.trapezoid(f_values, s_values)

    return np.exp(-integral)

def spot_rate(forward_curve, maturities, t):
    return np.interp(t, maturities, forward_curve)

if __name__ == "__main__":
    maturities = np.array([0.5, 1.0, 2.0, 3.0, 5.0, 7.0])

    # Made-up but realistic-looking zero-coupon prices -- decreasing,
    # roughly consistent with an upward-sloping yield curve.
    P_values = np.array([0.985, 0.970, 0.930, 0.885, 0.790, 0.700])

    f = forward_rate_from_bond_prices(P_values, maturities)

    print("Maturities:     ", maturities)
    print("Bond prices:    ", P_values)
    print("Forward rates:  ", np.round(f, 5))

    # Quick sanity checks:
    # 1. All forward rates should be positive, since P_values is
    #    strictly decreasing (an upward-sloping/normal curve here).
    print("\nAll forward rates positive:", np.all(f > 0))

    # 2. Test bond_price_from_forward_curve recovers something close
    #    to the original P(0, 2.0), by integrating the forward curve
    #    we just derived from it.
    recovered_P = bond_price_from_forward_curve(f, maturities, t=0.5, T=2.0)
    print("\nOriginal P(0.5, 2.0)-ish input value:", P_values[2])
    print("Recovered via integration:          ", round(recovered_P, 5))

    # 3. Spot rate check -- should equal f at the shortest maturity,
    #    since r(t) = f(t,t) and our shortest maturity acts as "now".
    r = spot_rate(f, maturities, t=0.5)
    print("\nSpot rate at t=0.5:", round(r, 5))
    print("f at shortest maturity (0.5):", round(f[0], 5))