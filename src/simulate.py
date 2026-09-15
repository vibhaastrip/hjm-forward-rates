import numpy as np


class ConstantVol:
    n_factors = 1

    def __init__(self,sigma):
        if sigma <= 0:
            raise ValueError("Sigma must be positive")
        self.sigma = sigma

    def __call__(self, t, T):
        T = np.asarray(T, dtype=float)
        return np.full((1, T.size), self.sigma)

class ExponentialVol:
    n_factors = 1
    def __init__(self, sigma, a):
        if sigma <= 0:
            raise ValueError("Sigma must be positive")
        if a <= 0:
            raise ValueError("a must be positive")
        self.sigma = sigma
        self.a = a

    def __call__(self, t, T):
        T = np.asarray(T, dtype=float)
        return (self.sigma * np.exp(-self.a * (T-t)))[np.newaxis,:]

class TwoFactorVol:
    n_factors = 2
    def __init__(self, sigma_1, sigma_2, lam):
        self.sigma_1 = sigma_1
        self.sigma_2 = sigma_2
        self.lam = lam

    def __call__(self,t,T):
        T = np.asarray(T, dtype=float)
        s1 = np.full(T.size, self.sigma_1)
        s2 = self.sigma_2 * np.exp(-(self.lam /2.0)*(T-t))
        return np.vstack([s1,s2])

def hjm_drift(vol_structure, t, maturities):
    maturities = np.asarray(maturities, dtype= float)
    if maturities[0] >t:
        grid = np.concatenate(([t],maturities))
        drop_first = True
    else:
        grid = maturities
        drop_first = False

    sig_grid = vol_structure(t,grid)
    dv = np.diff(grid)
    trapezoids = 0.5 * (sig_grid[:, 1:]+ sig_grid[:,:-1]) *dv
    cum_integral = np.concatenate(
        [np.zeros((sig_grid.shape[0],1)), np.cumsum(trapezoids, axis=1)],
        axis = 1,
    )
    if drop_first:
        sig_at_t = sig_grid[:, 1:]
        cum_at_t = cum_integral[:, 1:]
    else:
        sig_at_t = sig_grid
        cum_at_t = cum_integral

    return np.sum(sig_at_t * cum_at_t, axis = 0)

def simulate_forward_curves(
        initial_curve,
        maturities,
        vol_structure,
        T_horizon,
        n_steps,
        n_paths,
        seed= None,
        shocks = None
):
    initial_curve = np.asarray(initial_curve, dtype = float)
    maturities = np.asarray(maturities, dtype = float)

    if initial_curve.shape != maturities.shape:
        raise ValueError("initial curve and maturities must have the same shape")

    dt = T_horizon/ n_steps
    sqrt_dt = np.sqrt(dt)
    times = np.linspace(0.0, T_horizon, n_steps +1)
    n_mat = maturities.size
    n_factors = vol_structure.n_factors

    rng = np.random.default_rng(seed)

    if shocks is None:
        shocks = rng.standard_normal((n_paths, n_steps, n_factors))
    else:
        shocks = np.asarray(shocks, dtype=float)
        expected = (n_paths, n_steps, n_factors)
        if shocks.shape != expected:
            raise ValueError(f"shocks must have shape {expected}, got {shocks.shape}")

    curves = np.full((n_paths, n_steps+1, n_mat) , np.nan)
    curves[:,0,:]= initial_curve

    short_rates = np.full((n_paths, n_steps+1),np.nan)
    short_rates[:,0] = np.interp(0.0, maturities, initial_curve)

    for k in range(n_steps):
        t = times[k]
        t_next = times[k+1]

        active = maturities >= t_next
        if not np.any(active):
            break
        T_act = maturities[active]

        alpha = hjm_drift(vol_structure, t, T_act)
        sig = vol_structure(t, T_act)


        Z = shocks[:,k,:]

        diffusion = sqrt_dt * np.einsum("pi,ia -> pa", Z,sig)

        curves[:, k+1, active] =(
            curves[:, k , active] + alpha* dt + diffusion
        )

        for p in range(n_paths):
            row = curves[p, k+1, :]
            live = ~np.isnan(row)
            short_rates[p,k+1] = np.interp(
                t_next, maturities[live], row[live]
            )

    return curves, times, short_rates

if __name__== "__main__":
    maturities = np.linspace(0.0, 10.0, 101)

    # Flat 4% initial curve. Parameter status: ARBITRARY -- a real
    # implementation would bootstrap this from market bond prices.
    initial_curve = np.full(maturities.size, 0.04)

    vol = ExponentialVol(sigma=0.01, a=0.1)

    curves, times, short_rates = simulate_forward_curves(
        initial_curve=initial_curve,
        maturities=maturities,
        vol_structure=vol,
        T_horizon=1.0,
        n_steps=100,
        n_paths=1000,
        seed=42,
    )

    print("curves shape:     ", curves.shape)
    print("r(0):             ", round(short_rates[0, 0], 6))
    print("mean r(1) across paths:", round(np.nanmean(short_rates[:, -1]), 6))
    print("std  r(1) across paths:", round(np.nanstd(short_rates[:, -1]), 6))

    # Analytic check on the diffusion: for the exponential structure,
    # r(t) has variance sigma^2 * (1 - exp(-2*a*t)) / (2*a) -- the
    # standard OU variance, from the X(t) SDE we derived.
    t = 1.0
    expected_sd = np.sqrt(
        vol.sigma**2 * (1 - np.exp(-2 * vol.a * t)) / (2 * vol.a)
    )
    print("expected OU sd at t=1:", round(expected_sd, 6))