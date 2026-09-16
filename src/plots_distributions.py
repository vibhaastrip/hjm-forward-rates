"""
plots_distributions.py

Distribution-level figures, in the style of the market-making project's
diagnostics: 2x2 panels showing the full spread across paths rather than
collapsing everything to a mean and a standard error.

This matters more here than it might seem. The headline results --
"bias 9.2e-06", "variance reduction 1519x" -- are summary statistics
computed from 20,000-path arrays that are otherwise never looked at.
Plotting them shows things a table cannot: how wide the raw
distribution is relative to the effect being measured, why antithetic
variates lose effectiveness on kinked payoffs, and what the paired
difference actually looks like.

Run:  python3 -m src.plots_distributions
"""

import os

import matplotlib.pyplot as plt
import numpy as np

from src.caplet import caplet_payoff, forward_libor
from src.curve import bond_prices_vectorised
from src.hullwhite import make_flat_curve
from src.martingale_check import discounted_bond_prices, money_market_account
from src.simulate import ExponentialVol, TwoFactorVol, simulate_forward_curves


FIGDIR = "figures"
F_FLAT = 0.04
SIGMA, A = 0.01, 0.1


def _save(fig, name):
    os.makedirs(FIGDIR, exist_ok=True)
    path = os.path.join(FIGDIR, name)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"  wrote {path}")
    plt.close(fig)


def _simulate(vol, T_horizon, n_paths, n_steps=200, seed=42,
              shocks=None, drift_on=True, n_maturities=401):
    """Run a simulation and return curves, times, short rates, maturities."""
    import src.simulate as sim

    maturities = np.linspace(0.0, 10.0, n_maturities)
    initial = np.full(maturities.size, F_FLAT)

    original = sim.hjm_drift
    if not drift_on:
        sim.hjm_drift = lambda v, t, m: np.zeros(np.asarray(m).size)
    try:
        curves, times, sr = sim.simulate_forward_curves(
            initial, maturities, vol, T_horizon, n_steps, n_paths,
            seed=seed, shocks=shocks, store_history=False,
        )
    finally:
        sim.hjm_drift = original

    return curves, times, sr, maturities


# ----------------------------------------------------------------------


def plot_martingale_distributions(n_paths=20000, n_steps=200, seed=42):
    """
    Four views of the martingale verification.

    The table reports a bias of ~9e-06 against a standard error of
    ~1e-05. What it does not convey is that the underlying Z
    distribution has a standard deviation of ~3e-02 -- roughly three
    thousand times the effect being measured. Panels 1 and 2 put those
    on the same page.
    """
    maturities_test = [3.0, 5.0, 8.0]
    P0 = make_flat_curve(F_FLAT)

    n_factors = 1
    rng = np.random.default_rng(seed)
    shocks = rng.standard_normal((n_paths, n_steps, n_factors))

    vol = ExponentialVol(SIGMA, A)
    c_plus, times, r_plus, mats = _simulate(vol, 1.0, n_paths, n_steps,
                                            shocks=shocks)
    c_minus, _, r_minus, _ = _simulate(vol, 1.0, n_paths, n_steps,
                                       shocks=-shocks)
    c_off, _, r_off, _ = _simulate(vol, 1.0, n_paths, n_steps,
                                   shocks=shocks, drift_on=False)

    Z = {T: discounted_bond_prices(c_plus, times, mats, r_plus, T, -1)
         for T in maturities_test}
    Z_anti = {T: discounted_bond_prices(c_minus, times, mats, r_minus, T, -1)
              for T in maturities_test}
    Z_off = discounted_bond_prices(c_off, times, mats, r_off, 5.0, -1)

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))

    # --- Panel 1: raw Z distributions, one per maturity -------------
    ax = axes[0, 0]
    for i, T in enumerate(maturities_test):
        ax.hist(Z[T], bins=60, alpha=0.5, label=f"T = {T}", color=f"C{i}")
        ax.axvline(P0(T), color=f"C{i}", ls="--", lw=1.5)
    ax.set_xlabel("$Z(t,T) = P(t,T)/B(t)$")
    ax.set_ylabel("Frequency")
    ax.set_title("Discounted bond prices at $t=1$\n"
                 "(dashed lines: the martingale targets $P(0,T)$)")
    ax.legend(fontsize=9)

    # --- Panel 2: the paired drift difference -----------------------
    # The figure that makes the 1519x variance reduction visible: the
    # spread here is ~2e-05, against ~3e-02 for Z itself in panel 1.
    ax = axes[0, 1]
    D = Z[5.0] - Z_off
    ax.hist(D, bins=60, color="C4", alpha=0.85)
    ax.axvline(0, color="k", ls="--", lw=1.5)
    ax.axvline(D.mean(), color="C3", lw=2,
               label=f"mean = {D.mean():.2e}")
    ax.set_xlabel("$Z$ (drift on) $-$ $Z$ (drift off)")
    ax.set_ylabel("Frequency")
    ax.set_title("Paired drift difference, $T=5$\n"
                 "Same shocks, so the diffusion cancels within each path")
    ax.legend(fontsize=9)
    ax.ticklabel_format(axis="x", style="sci", scilimits=(0, 0))

    # --- Panel 3: bias relative to noise, by maturity ---------------
    ax = axes[1, 0]
    labels, biases, errs = [], [], []
    for T in maturities_test:
        pair = 0.5 * (Z[T] + Z_anti[T])
        labels.append(f"T = {T}")
        biases.append(pair.mean() - P0(T))
        errs.append(pair.std(ddof=1) / np.sqrt(pair.size))
    x = np.arange(len(labels))
    ax.axhline(0, color="k", lw=1.5)
    ax.errorbar(x, biases, yerr=errs, fmt="o", ms=8, capsize=5,
                color="C0", lw=2)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("mean $Z$ $-$ $P(0,T)$")
    ax.set_title("Residual bias with $\\pm1$ standard error\n"
                 "Every maturity straddles zero")
    ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    ax.grid(alpha=0.3, axis="y")

    # --- Panel 4: what antithetic variates do -----------------------
    # Two stacked axes rather than one shared. The distributions differ
    # in width by ~40x, so any shared scale renders one as a spike, and
    # a log scale distorts the bin heights. Separate axes with matching
    # shapes but different ranges show both honestly.
    ax = axes[1, 1]
    ax.remove()

    gs = axes[1, 1].get_subplotspec() if hasattr(axes[1, 1], "get_subplotspec") else None
    sub = fig.add_gridspec(2, 2)[1, 1].subgridspec(2, 1, hspace=0.45)
    ax_a = fig.add_subplot(sub[0])
    ax_b = fig.add_subplot(sub[1])

    T = 5.0
    pair = 0.5 * (Z[T] + Z_anti[T])
    sd_single, sd_pair = Z[T].std(ddof=1), pair.std(ddof=1)

    ax_a.hist(Z[T] - P0(T), bins=60, color="C0", alpha=0.75)
    ax_a.axvline(0, color="k", ls="--", lw=1.5)
    ax_a.set_title(f"single path, sd = {sd_single:.1e}", fontsize=9)
    ax_a.ticklabel_format(axis="x", style="sci", scilimits=(0, 0))

    ax_b.hist(pair - P0(T), bins=60, color="C3", alpha=0.8)
    ax_b.axvline(0, color="k", ls="--", lw=1.5)
    ax_b.set_title(f"antithetic pair, sd = {sd_pair:.1e}  "
                   f"({sd_single/sd_pair:.0f}x narrower)", fontsize=9)
    ax_b.set_xlabel("$Z(t,T) - P(0,T)$")
    ax_b.ticklabel_format(axis="x", style="sci", scilimits=(0, 0))

    fig.suptitle("Martingale verification: $E[Z(t,T)] = P(0,T)$",
                 fontsize=13, y=0.995)
    fig.tight_layout()
    _save(fig, "martingale_distributions.png")


# ----------------------------------------------------------------------


def plot_caplet_distributions(T1=1.0, T2=1.5, n_paths=20000,
                              n_steps=200, seed=42):
    """
    Four views of the caplet Monte Carlo.

    Panel 2 is the one that explains a result the tables only assert:
    antithetic variance reduction falls from 5.3x to 1.5x as the option
    moves out of the money. The payoff's kink is visible directly.
    """
    P0 = make_flat_curve(F_FLAT)
    tau = T2 - T1
    K_atm = forward_libor(P0(T2) / P0(T1), tau)
    strikes = [K_atm - 0.01, K_atm - 0.005, K_atm,
               K_atm + 0.005, K_atm + 0.01]

    rng = np.random.default_rng(seed)
    shocks = rng.standard_normal((n_paths, n_steps, 1))

    vol = ExponentialVol(SIGMA, A)
    curves, times, sr, mats = _simulate(vol, T1, n_paths, n_steps,
                                        shocks=shocks)
    P = bond_prices_vectorised(curves[:, -1, :], mats, T1, T2)
    B = money_market_account(sr, times)[:, -1]
    L = forward_libor(P, tau)

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))

    # --- Panel 1: the forward rate at reset -------------------------
    ax = axes[0, 0]
    ax.hist(L * 100, bins=70, color="C0", alpha=0.8)
    for i, K in enumerate(strikes):
        ax.axvline(K * 100, color="C3", ls="--", lw=1,
                   alpha=0.4 + 0.15 * abs(i - 2))
    ax.axvline(K_atm * 100, color="C3", ls="-", lw=2, label="ATM strike")
    ax.set_xlabel("$L(T_1,T_2)$ at reset (%)")
    ax.set_ylabel("Frequency")
    ax.set_title(f"Forward rate at the reset date $T_1={T1}$\n"
                 "Dashed lines mark the five strikes")
    ax.legend(fontsize=9)

    # --- Panel 2: the payoff kink -----------------------------------
    ax = axes[0, 1]
    idx = rng.choice(n_paths, size=3000, replace=False)
    payoff_atm = caplet_payoff(P, K_atm, tau)
    ax.scatter(L[idx] * 100, payoff_atm[idx] * 1e3, s=4, alpha=0.25,
               color="C0", edgecolors="none")
    ax.axvline(K_atm * 100, color="C3", ls="--", lw=1.5, label="strike")
    ax.set_xlabel("$L(T_1,T_2)$ (%)")
    ax.set_ylabel("payoff at $T_1$ ($\\times 10^{-3}$)")
    ax.set_title("The kink: payoff against the underlying rate\n"
                 "Flat below the strike -- why antithetics lose power OTM")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    # --- Panel 3: payoff distributions by strike --------------------
    ax = axes[1, 0]
    data, labels, itm = [], [], []
    for K in strikes:
        pay = caplet_payoff(P, K, tau) / B
        data.append(pay[pay > 0] * 1e3)   # in-the-money paths only
        labels.append(f"{K*100:.2f}%")
        itm.append(np.mean(pay > 0))
    bp = ax.boxplot(data, tick_labels=labels, showfliers=False, patch_artist=True)
    for patch in bp["boxes"]:
        patch.set_facecolor("C0")
        patch.set_alpha(0.55)
    for i, f in enumerate(itm):
        ax.text(i + 1, ax.get_ylim()[1] * 0.92, f"{f:.0%}",
                ha="center", fontsize=8, color="C3")
    ax.set_xlabel("strike")
    ax.set_ylabel("discounted payoff ($\\times 10^{-3}$)")
    ax.set_title("Payoff distribution, in-the-money paths only\n"
                 "Red: fraction of paths finishing in the money")
    ax.grid(alpha=0.3, axis="y")

        # --- Panel 4: calibrated comparison, per strike -----------------
    # Prices both structures on the SAME shocks, matching how the result
    # was established in calibrate.py. Plotting raw payoff histograms
    # instead would be dominated by the ~50% of paths finishing out of
    # the money, and drawing fresh shocks for each structure would
    # reintroduce exactly the sampling noise the pairing removes.
    ax = axes[1, 1]
    from src.calibrate import (
        calibrate_exponential_to_variance,
        integrated_forward_variance_two_factor,
    )
    from src.caplet import caplet_monte_carlo

    s1, s2, lam = 0.008, 0.006, 0.3
    V = integrated_forward_variance_two_factor(s1, s2, lam, T1, T2)
    sigma_cal = calibrate_exponential_to_variance(V, A, T1, T2)

    d, se = [], []
    for K in strikes:
        p2 = caplet_monte_carlo(TwoFactorVol(s1, s2, lam), T1, T2, K,
                                f_flat=F_FLAT, n_paths=n_paths, seed=seed)
        p1 = caplet_monte_carlo(ExponentialVol(sigma_cal, A), T1, T2, K,
                                f_flat=F_FLAT, n_paths=n_paths, seed=seed)
        d.append(p2["price"] - p1["price"])
        se.append(np.sqrt(p2["std_err"]**2 + p1["std_err"]**2))

    ax.axhline(0, color="k", lw=1.5)
    ax.errorbar([k * 100 for k in strikes], d, yerr=se, fmt="o", ms=7,
                capsize=4, color="C1", lw=2)
    ax.set_xlabel("strike (%)")
    ax.set_ylabel("two-factor $-$ one-factor")
    ax.set_title(f"Matched on integrated forward variance\n"
                 f"($\\sigma$ = {sigma_cal:.4f}): every strike straddles zero")
    ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    ax.grid(alpha=0.3)

    fig.suptitle(f"Caplet on $[{T1}, {T2}]$, {n_paths:,} paths",
                 fontsize=13, y=0.995)
    fig.tight_layout()
    _save(fig, "caplet_distributions.png")


# ----------------------------------------------------------------------


def plot_simulation_diagnostics(n_paths=20000, n_steps=200, seed=42):
    """
    Does the simulation reproduce the distributions theory predicts?

    Panels 1 and 2 overlay the analytic densities on the simulated
    histograms. These are checks the test suite makes numerically; here
    they are visible.
    """
    from src.hullwhite import bond_price, short_rate_variance

    P0 = make_flat_curve(F_FLAT)
    vol = ExponentialVol(SIGMA, A)
    curves, times, sr, mats = _simulate(vol, 1.0, n_paths, n_steps, seed=seed)

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))

    # --- Panel 1: short rate against the analytic OU density --------
    ax = axes[0, 0]
    r1 = sr[:, -1]
    ax.hist(r1 * 100, bins=70, density=True, alpha=0.6, color="C0",
            label="simulated")
    sd = np.sqrt(short_rate_variance(1.0, SIGMA, A))
    mu = r1.mean()
    xs = np.linspace(r1.min(), r1.max(), 300)
    dens = np.exp(-0.5 * ((xs - mu) / sd) ** 2) / (sd * np.sqrt(2 * np.pi))
    ax.plot(xs * 100, dens / 100, "C3", lw=2,
            label=f"OU normal, sd = {sd:.2e}")
    ax.set_xlabel("$r(1)$ (%)")
    ax.set_ylabel("Density")
    ax.set_title("Short rate at $t=1$\n"
                 "Hull-White predicts $\\sigma^2(1-e^{-2at})/2a$")
    ax.legend(fontsize=9)

    # --- Panel 2: simulated vs closed-form bond prices --------------
    ax = axes[0, 1]
    T = 5.0
    P_sim = bond_prices_vectorised(curves[:, -1, :], mats, 1.0, T)
    P_hw = bond_price(1.0, T, sr[:, -1], P0, F_FLAT, SIGMA, A)
    ax.hist(P_sim, bins=70, alpha=0.5, color="C0",
            label="from the simulated curve (401 points)")
    ax.hist(P_hw, bins=70, alpha=0.5, color="C3",
            label="Hull-White, from $r(t)$ alone (1 number)")
    ax.set_xlabel(f"$P(1, {T})$")
    ax.set_ylabel("Frequency")
    ax.set_title("Two routes to the same price\n"
                 f"Max difference: {np.max(np.abs(P_sim - P_hw)):.2e}")
    ax.legend(fontsize=8)

    # --- Panel 3: the per-path error -------------------------------
    ax = axes[1, 0]
    err = P_sim - P_hw
    ax.hist(err, bins=70, color="C2", alpha=0.85)
    ax.axvline(0, color="k", ls="--", lw=1.5)
    ax.set_xlabel("simulated $-$ closed form")
    ax.set_ylabel("Frequency")
    ax.set_title(f"Per-path discrepancy, $T={T}$\n"
                 f"mean {err.mean():.2e}, sd {err.std(ddof=1):.2e}")
    ax.ticklabel_format(axis="x", style="sci", scilimits=(0, 0))

    # --- Panel 4: error against the short rate ---------------------
    # If the discrepancy were a real modelling failure rather than
    # discretization, it would likely correlate with the state.
    ax = axes[1, 1]
    rng = np.random.default_rng(0)
    idx = rng.choice(n_paths, size=4000, replace=False)
    ax.scatter(sr[idx, -1] * 100, err[idx], s=4, alpha=0.25, color="C0",
               edgecolors="none")
    ax.axhline(0, color="k", ls="--", lw=1.5)
    ax.set_xlabel("$r(1)$ (%)")
    ax.set_ylabel("simulated $-$ closed form")
    ax.set_title("Discrepancy against the state\n"
                 f"correlation: {np.corrcoef(sr[:, -1], err)[0,1]:.3f}")
    ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    ax.grid(alpha=0.3)

    fig.suptitle("Simulation diagnostics: does it reproduce the theory?",
                 fontsize=13, y=0.995)
    fig.tight_layout()
    _save(fig, "simulation_diagnostics.png")


if __name__ == "__main__":
    print("Generating distribution figures...")
    plot_martingale_distributions()
    plot_caplet_distributions()
    plot_simulation_diagnostics()
    print("Done.")