"""
plots.py

Figures for the HJM project.

Deliberately 2D. The obvious candidate for a 3D surface -- the forward
curve over calendar time and maturity -- reads as a featureless sheet,
because the accumulated drift (~2e-4) is two orders of magnitude below
the curve level (~4e-2). 3D also costs accuracy in reading values:
occlusion, perspective distortion, and no clean axis to read against.
Each figure below answers one question that a 2D plot answers better.

Run:  python3 -m src.plots
Saves PNGs to figures/ in the project root.
"""

import os

import matplotlib.pyplot as plt
import numpy as np

from src.caplet import caplet_closed_form, caplet_monte_carlo, forward_libor
from src.hullwhite import make_flat_curve
from src.simulate import ExponentialVol, TwoFactorVol, simulate_forward_curves
from src.validate import convergence_study


FIGDIR = "figures"


def _save(fig, name):
    os.makedirs(FIGDIR, exist_ok=True)
    path = os.path.join(FIGDIR, name)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"  wrote {path}")
    plt.close(fig)


# ----------------------------------------------------------------------


def plot_volatility_structures(sigma=0.01, a=0.1,
                               sigma_1=0.008, sigma_2=0.006, lam=0.3):
    """
    sigma_i(t,T) against time to maturity, T - t.

    The most explanatory figure in the set: it shows at a glance WHY the
    two-factor structure is described as level-plus-slope, and why the
    exponential structure is the one that reduces to Hull-White.

    Plotted against T-t rather than T because all three structures
    depend on the two arguments only through that difference, so this is
    the natural axis and the curves are then time-invariant.
    """
    tau = np.linspace(0.0, 10.0, 400)
    t = 0.0
    T = t + tau

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))

    # Left: the Hull-White-reducing structure.
    ax1.plot(tau, ExponentialVol(sigma, a)(t, T)[0], color="C0", lw=2)
    ax1.set_title(f"Exponential: $\\sigma e^{{-a(T-t)}}$\n"
                  f"$\\sigma$={sigma}, $a$={a}")
    ax1.set_xlabel("time to maturity, $T-t$ (years)")
    ax1.set_ylabel("$\\sigma(t,T)$")
    ax1.set_ylim(bottom=0)
    ax1.grid(alpha=0.3)

    # Right: the two-factor structure, decomposed.
    two = TwoFactorVol(sigma_1, sigma_2, lam)(t, T)
    ax2.plot(tau, two[0], color="C1", lw=2,
             label=f"factor 1: level, $\\sigma_1$={sigma_1}")
    ax2.plot(tau, two[1], color="C2", lw=2,
             label=f"factor 2: slope, $\\sigma_2$={sigma_2}, $\\lambda$={lam}")
    # Total volatility: independent factors, so variances add.
    ax2.plot(tau, np.sqrt(two[0]**2 + two[1]**2), color="0.4", lw=1.5, ls="--",
             label="total, $\\sqrt{\\sigma_1^2+\\sigma_2^2}$")
    ax2.set_title("Two-factor structure")
    ax2.set_xlabel("time to maturity, $T-t$ (years)")
    ax2.set_ylabel("$\\sigma_i(t,T)$")
    ax2.set_ylim(bottom=0)
    ax2.legend(fontsize=8)
    ax2.grid(alpha=0.3)

    fig.suptitle("Volatility structures: the only thing you choose in HJM "
                 "(the drift is then forced by equation 18)", fontsize=10)
    fig.tight_layout()
    _save(fig, "volatility_structures.png")


def plot_convergence():
    """
    Validation error against Euler-Maruyama step count, log-log.

    O(dt) convergence appears as a straight line of slope -1 on log-log
    axes, so the reference line makes the claim checkable by eye rather
    than requiring the reader to divide consecutive errors.

    The measured points drift slightly shallower than the reference at
    fine step sizes: as the Euler error shrinks it approaches the fixed
    trapezoidal error on the maturity axis, which refining the time step
    does not improve.
    """
    rows = convergence_study()
    steps = np.array([r[0] for r in rows], dtype=float)
    errs = np.array([r[1] for r in rows])

    fig, ax = plt.subplots(figsize=(6, 4.5))

    ax.loglog(steps, errs, "o-", color="C0", lw=2, ms=7, label="measured")

    # Reference slope -1, anchored at the first point.
    ref = errs[0] * steps[0] / steps
    ax.loglog(steps, ref, "--", color="0.5", lw=1.5,
              label="$O(\\Delta t)$ reference (slope $-1$)")

    ax.set_xlabel("Euler-Maruyama steps")
    ax.set_ylabel("mean absolute error in $P(t,T)$")
    ax.set_title("Simulation vs Hull-White closed form, $T=5$")
    ax.grid(alpha=0.3, which="both")
    ax.legend()

    # Annotate the observed ratios.
    ratios = errs[:-1] / errs[1:]
    txt = "error ratio per halving of $\\Delta t$:\n" + "  ".join(
        f"{r:.2f}" for r in ratios
    )
    ax.text(0.04, 0.06, txt, transform=ax.transAxes, fontsize=8,
            va="bottom", bbox=dict(boxstyle="round", fc="white", alpha=0.85))

    _save(fig, "convergence.png")


def plot_caplet_validation(T1=1.0, T2=1.5, f_flat=0.04,
                           sigma=0.01, a=0.1, n_paths=20000, seed=42):
    """
    Caplet prices across strikes: Monte Carlo against closed form.

    Two panels. The left shows prices, where agreement is unsurprising
    and the eye cannot resolve differences of 1e-6 on prices of 5e-3.
    The right shows the DIFFERENCE with error bars, which is where the
    validation actually lives: every point should straddle zero.

    A price plot alone would be close to uninformative here -- the
    curves overlay exactly -- so the residual panel is the point.
    """
    P0 = make_flat_curve(f_flat)
    tau = T2 - T1
    K_atm = forward_libor(P0(T2) / P0(T1), tau)
    strikes = np.array([K_atm - 0.01, K_atm - 0.005, K_atm,
                        K_atm + 0.005, K_atm + 0.01])

    mc, se, cf = [], [], []
    for K in strikes:
        r = caplet_monte_carlo(ExponentialVol(sigma, a), T1, T2, K,
                               f_flat=f_flat, n_paths=n_paths, seed=seed)
        mc.append(r["price"])
        se.append(r["std_err"])
        cf.append(caplet_closed_form(T1, T2, K, P0, sigma, a))

    mc, se, cf = np.array(mc), np.array(se), np.array(cf)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))

    # Left: prices. Closed form on a fine grid, MC as points.
    K_fine = np.linspace(strikes[0], strikes[-1], 100)
    cf_fine = [caplet_closed_form(T1, T2, K, P0, sigma, a) for K in K_fine]
    ax1.plot(K_fine, cf_fine, "-", color="0.4", lw=1.5, label="closed form")
    ax1.errorbar(strikes, mc, yerr=se, fmt="o", color="C0", ms=6,
                 capsize=3, label="Monte Carlo")
    ax1.axvline(K_atm, color="0.7", ls=":", lw=1)
    ax1.text(K_atm, ax1.get_ylim()[1] * 0.95, " ATM", fontsize=8, color="0.4")
    ax1.set_xlabel("strike")
    ax1.set_ylabel("caplet price")
    ax1.set_title(f"Caplet on $[{T1}, {T2}]$, Hull-White")
    ax1.legend()
    ax1.grid(alpha=0.3)

    # Right: residuals, where the validation is actually visible.
    ax2.axhline(0, color="0.4", lw=1.5)
    ax2.errorbar(strikes, mc - cf, yerr=se, fmt="o", color="C3", ms=6,
                 capsize=3)
    ax2.set_xlabel("strike")
    ax2.set_ylabel("Monte Carlo $-$ closed form")
    ax2.set_title(f"Residuals with $\\pm1$ standard error "
                  f"({n_paths:,} paths)")
    ax2.grid(alpha=0.3)
    ax2.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))

    _save(fig, "caplet_validation.png")


def plot_calibrated_comparison(T1=1.0, T2=1.5, f_flat=0.04, a=0.1,
                               sigma_1=0.008, sigma_2=0.006, lam=0.3,
                               n_paths=20000, seed=42):
    """
    The controlled comparison: two-factor against one-factor, before and
    after matching integrated forward variance.

    Left panel, uncalibrated: two-factor prices sit above one-factor at
    every strike, with the gap growing out of the money. Tempting to
    read as the second factor adding tail value.

    Right panel, calibrated: once the structures carry the same
    variance, the gap vanishes. The apparent effect was a
    volatility-magnitude confound.

    Both structures are Gaussian, so P(T1,T2) is lognormal under each
    and the mean is pinned by the martingale property -- matching the
    variance matches the whole distribution. A single-tenor caplet
    therefore cannot distinguish them.
    """
    from src.calibrate import (
        calibrate_exponential_to_variance,
        integrated_forward_variance_two_factor,
    )

    P0 = make_flat_curve(f_flat)
    tau = T2 - T1
    K_atm = forward_libor(P0(T2) / P0(T1), tau)
    strikes = np.array([K_atm - 0.01, K_atm - 0.005, K_atm,
                        K_atm + 0.005, K_atm + 0.01])

    V_2f = integrated_forward_variance_two_factor(sigma_1, sigma_2, lam, T1, T2)
    sigma_matched = calibrate_exponential_to_variance(V_2f, a, T1, T2)

    vol_2f = TwoFactorVol(sigma_1, sigma_2, lam)
    vol_base = ExponentialVol(0.01, a)
    vol_cal = ExponentialVol(sigma_matched, a)

    def price_all(vol):
        out = [caplet_monte_carlo(vol, T1, T2, K, f_flat=f_flat,
                                  n_paths=n_paths, seed=seed) for K in strikes]
        return (np.array([o["price"] for o in out]),
                np.array([o["std_err"] for o in out]))

    p2f, s2f = price_all(vol_2f)
    pbase, sbase = price_all(vol_base)
    pcal, scal = price_all(vol_cal)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)

    for ax, p1, s1, label, title in [
        (ax1, pbase, sbase, "one factor, $\\sigma=0.010$",
         "Uncalibrated: two-factor appears richer"),
        (ax2, pcal, scal, f"one factor, $\\sigma={sigma_matched:.4f}$",
         "Matched variance: the difference vanishes"),
    ]:
        d = (p2f - p1) / p1 * 100
        se = np.sqrt(s2f**2 + s1**2) / p1 * 100
        ax.axhline(0, color="0.4", lw=1.5)
        ax.errorbar(strikes, d, yerr=se, fmt="o-", color="C1", ms=6, capsize=3)
        ax.axvline(K_atm, color="0.7", ls=":", lw=1)
        ax.set_xlabel("strike")
        ax.set_title(title, fontsize=10)
        ax.grid(alpha=0.3)
        ax.text(0.03, 0.95, label, transform=ax.transAxes, fontsize=8,
                va="top", bbox=dict(boxstyle="round", fc="white", alpha=0.85))

    ax1.set_ylabel("two-factor premium over one-factor (%)")

    fig.suptitle("Controlling for total volatility removes the two-factor "
                 "premium entirely", fontsize=10)
    _save(fig, "calibrated_comparison.png")


def plot_simulated_curves(sigma=0.01, a=0.1, f_flat=0.04,
                          n_steps=200, seed=7):
    """
    A handful of simulated forward curves, at several calendar times.

    Shows what the curve dynamics actually look like: the whole curve
    is shifted by each shock rather than each maturity moving
    independently, because condition C.1 drives the entire curve from a
    single Brownian motion here. The curves stay smooth in maturity and
    rough only in time.

    Also visible: the live region shortens as calendar time advances,
    since maturities expire.
    """
    maturities = np.linspace(0.0, 10.0, 401)
    initial = np.full(maturities.size, f_flat)

    curves, times, short_rates = simulate_forward_curves(
        initial, maturities, ExponentialVol(sigma, a),
        T_horizon=3.0, n_steps=n_steps, n_paths=3, seed=seed,
        store_history=True,
    )

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))

    # Left: one path's curve at several times.
    snapshots = [0, n_steps // 4, n_steps // 2, 3 * n_steps // 4, n_steps]
    cmap = plt.cm.viridis(np.linspace(0.15, 0.9, len(snapshots)))
    for c, k in zip(cmap, snapshots):
        row = curves[0, k, :]
        live = ~np.isnan(row)
        ax1.plot(maturities[live], row[live] * 100, color=c, lw=1.8,
                 label=f"$t={times[k]:.2f}$")
    ax1.set_xlabel("maturity $T$ (years)")
    ax1.set_ylabel("forward rate $f(t,T)$ (%)")
    ax1.set_title("One path: the whole curve moves together")
    ax1.legend(fontsize=8)
    ax1.grid(alpha=0.3)

    # Right: short rate paths.
    for p in range(3):
        ax2.plot(times, short_rates[p] * 100, lw=1.4, label=f"path {p+1}")
    ax2.axhline(f_flat * 100, color="0.5", ls="--", lw=1,
                label="$f(0,\\cdot)$")
    ax2.set_xlabel("calendar time $t$ (years)")
    ax2.set_ylabel("short rate $r(t)$ (%)")
    ax2.set_title("Short rate $r(t)=f(t,t)$, three paths")
    ax2.legend(fontsize=8)
    ax2.grid(alpha=0.3)

    _save(fig, "simulated_curves.png")


if __name__ == "__main__":
    print("Generating figures...")
    plot_volatility_structures()
    plot_simulated_curves()
    plot_convergence()
    plot_caplet_validation()
    plot_calibrated_comparison()
    print("Done.")