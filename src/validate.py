import numpy as np

from src.curve import bond_price_from_forward_curve
from src.hullwhite import bond_price, make_flat_curve
from src.simulate import ExponentialVol, simulate_forward_curves



def run_validation(
        sigma = 0.01,
        a = 0.1,
        f_flat = 0.04,
        T_horizon = 1.0,
        n_steps = 200,
        n_paths = 200,
        n_maturities = 401,
        max_maturity = 10.0,
        compare_maturities =(3.0, 5.0, 8.0),
        seed = 42,
):
    maturities = np.linspace(0.0, max_maturity, n_maturities)
    initial_curve = np.full(maturities.size , f_flat)
    vol = ExponentialVol(sigma=sigma, a=a)
    P0 = make_flat_curve(f_flat)

    curves, times, short_rates = simulate_forward_curves(
        initial_curve=initial_curve,
        maturities= maturities,
        vol_structure=  vol,
        T_horizon=T_horizon,
        n_steps=n_steps,
        n_paths=n_paths,
        seed=seed,
    )

    k =-1
    t = times[-1]

    results = {}

    for T in compare_maturities:
        if T<=t:
            raise ValueError(f"Comparison maturity {T} must exceed horizon {t}")

        sim_prices = np.zeros(n_paths)
        hw_prices = np.zeros(n_paths)

        for p in range(n_paths):
            row = curves[p,k,:]
            live = ~np.isnan(row)

            sim_prices[p] = bond_price_from_forward_curve(
                row[live], maturities[live], t, T
            )
            hw_prices[p] = bond_price(t, T, short_rates[p, k], P0, f_flat, sigma, a)

        abs_err = np.abs(sim_prices - hw_prices)
        rel_err = abs_err/hw_prices

        results[T] = {
            "sim_prices": sim_prices,
            "hw_prices": hw_prices,
            "max_abs_err": np.max(abs_err),
            "mean_abs_err": np.mean(abs_err),
            "max_rel_err": np.max(rel_err),
            "mean_rel_err": np.mean(rel_err),
            "mean_sim": np.mean(sim_prices),
            "mean_hw": np.mean(hw_prices),
        }

    return results, times, short_rates

def convergence_study(n_steps_list=(50,100,200,400,800), **kwargs):
    kwargs.setdefault("compare_maturities", (5.0,))
    rows = []

    for n_steps in n_steps_list:
        results, _, _ = run_validation(n_steps=n_steps, **kwargs)
        err = results[5.0]["mean_abs_err"]
        rows.append((n_steps, err))

    return rows


if __name__ == "__main__":
    print("=" * 68)
    print("HJM simulation vs Hull-White closed form")
    print("=" * 68)

    results, times, short_rates = run_validation()

    print(f"\nComparison at t = {times[-1]:.2f}, across 200 paths\n")
    print(f"{'T':>6}  {'mean sim':>12}  {'mean HW':>12}  "
          f"{'mean |err|':>12}  {'max rel err':>12}")
    print("-" * 68)
    for T, r in results.items():
        print(f"{T:6.1f}  {r['mean_sim']:12.8f}  {r['mean_hw']:12.8f}  "
              f"{r['mean_abs_err']:12.2e}  {r['max_rel_err']:12.2e}")

    print("\n" + "=" * 68)
    print("Convergence in the number of Euler-Maruyama steps (T = 5.0)")
    print("=" * 68)
    print("\nWeak order O(dt): doubling n_steps should roughly halve the error.\n")
    print(f"{'n_steps':>9}  {'mean |err|':>12}  {'ratio':>8}")
    print("-" * 34)

    rows = convergence_study()
    prev = None
    for n_steps, err in rows:
        ratio = f"{prev / err:8.2f}" if prev is not None else "       -"
        print(f"{n_steps:9d}  {err:12.2e}  {ratio}")
        prev = err