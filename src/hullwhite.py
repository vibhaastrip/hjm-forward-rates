import numpy as np
from scipy.stats import norm

def B_function(t,T, a):
    tau = np.asarray(T, dtype=float) - np.asarray(t, dtype=float)
    return (1.0 - np.exp(-a * tau)) / a

def bond_price(t, T, r_t, P0, f0_t, sigma, a):
    if T < t:
        raise ValueError(f"T ({T}) must be >= t ({t})")
    if T == t:
        return np.ones_like(np.asarray(r_t, dtype=float))

    B = B_function(t, T, a)

    log_A = (
        np.log(P0(T) / P0(t))
        + B * f0_t
        - (sigma**2 / (4.0 * a)) * (1.0 - np.exp(-2.0 * a * t)) * B**2
    )

    return np.exp(log_A - B * r_t)

def short_rate_variance(t, sigma, a):
    return sigma**2 * (1.0 - np.exp(-2.0 *a*t)) / (2.0 *a)

def bond_option_price(t_star, T, K, P0, sigma, a, option_type="call"):
    if T <= t_star:
        raise ValueError(f"Bond maturity T ({T}) must exceed expiry ({t_star})")
    if K <= 0:
        raise ValueError("Strike must be positive")

    B = B_function(t_star, T,a)
    nu_squared = short_rate_variance(t_star, sigma, a) * B**2
    nu = np.sqrt(nu_squared)

    P_T = P0(T)
    P_star = P0(t_star)

    h = np.log(P_T / (K * P_star)) / nu + nu / 2.0

    if option_type == "call":
        return P_T * norm.cdf(h) - K * P_star * norm.cdf(h - nu)
    elif option_type == "put":
        return K * P_star * norm.cdf(-(h - nu)) - P_T * norm.cdf(-h)
    else:
        raise ValueError(f"option_type must be 'call' or 'put', got {option_type}")

def make_flat_curve(f_flat):
    return lambda u: np.exp(-f_flat * np.asarray(u, dtype=float))


if __name__=="__main__":
    sigma, a = 0.01, 0.1
    f_flat = 0.04
    P0 = make_flat_curve(f_flat)

    # Check 1: at t=0, r(0)=f(0,0), the formula must return the observed
    # discount factor exactly. If this fails, A(t,T) is wrong.
    for T in [1.0, 5.0, 10.0]:
        model = bond_price(0.0, T, f_flat, P0, f_flat, sigma, a)
        observed = P0(T)
        print(f"T={T:5.1f}  model={model:.10f}  observed={observed:.10f}  "
              f"diff={abs(model - observed):.2e}")

    # Check 2: P(T,T) = 1 regardless of the short rate.
    print("\nP(5,5) at r=0.06:", bond_price(5.0, 5.0, 0.06, P0, f_flat, sigma, a))

    # Check 3: put-call parity.
    t_star, T, K = 1.0, 5.0, 0.80
    c = bond_option_price(t_star, T, K, P0, sigma, a, "call")
    p = bond_option_price(t_star, T, K, P0, sigma, a, "put")
    print(f"\ncall={c:.8f}  put={p:.8f}")
    print(f"C - P      = {c - p:.10f}")
    print(f"P(0,T)-K*P(0,t*) = {P0(T) - K * P0(t_star):.10f}")
