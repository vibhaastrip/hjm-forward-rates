# HJM Forward Rate Model

A replication of Heath, Jarrow & Morton (1992), *"Bond Pricing and the Term
Structure of Interest Rates: A New Methodology for Contingent Claims
Valuation,"* Econometrica 60(1), 77–105.

The no-arbitrage drift restriction is derived from first principles, implemented
as a Monte Carlo simulation of the full forward rate curve, validated against the
Hull-White closed-form special case, and then extended to price a caplet under a
two-factor volatility structure where no closed form exists.

---

## The result HJM establishes

Specify how volatile forward rates are, and no-arbitrage **forces** their drift:


You do not get to choose the drift independently. Under the risk-neutral measure
the market price of risk `phi(t)` cancels out entirely, leaving contingent claim
prices determined by the volatility structure alone — a quantity that can be
calibrated from market data, unlike the market price of risk.

The framework also takes today's observed forward curve as a direct input rather
than deriving it, which avoids the "inversion of the term structure" problem the
paper identifies in equilibrium models such as Cox-Ingersoll-Ross (Section 8).

---

## Results

### Hull-White validation

The volatility structure `sigma(t,T) = sigma*exp(-a(T-t))` reduces general HJM to
Hull-White, a Markov short-rate model with closed-form prices. Bond prices were
computed two ways at `t = 1.0`, across 200 paths:

- **(A)** integrating the full 401-point simulated forward curve
- **(B)** the Hull-White affine formula, using only the scalar short rate `r(t)`

| maturity | mean (simulated) | mean (closed form) | mean abs error | max rel error |
|---------:|-----------------:|-------------------:|---------------:|--------------:|
| 3.0 | 0.92202877 | 0.92202870 | 7.02e-08 | 9.83e-08 |
| 5.0 | 0.85030234 | 0.85030213 | 2.12e-07 | 2.89e-07 |
| 8.0 | 0.75326512 | 0.75326468 | 4.35e-07 | 6.39e-07 |

That these agree is not a tautology. Side (A) uses the entire infinite-dimensional
curve; side (B) uses one number. Their agreement is the numerical confirmation
that the exponential volatility structure collapses the curve into a
one-dimensional Markov state — the content of the Hull-White reduction.

**Convergence** (T = 5.0). Euler-Maruyama is weak order O(dt), so halving the step
size should halve the error:

| steps | mean abs error | ratio |
|------:|---------------:|------:|
| 50 | 8.45e-07 | – |
| 100 | 4.21e-07 | 2.01 |
| 200 | 2.12e-07 | 1.99 |
| 400 | 1.09e-07 | 1.94 |
| 800 | 5.71e-08 | 1.91 |

Ratios of 2.01 and 1.99 at the coarse end match the theoretical rate. The drift
below 2.0 at finer steps is expected: as the Euler error shrinks it approaches the
fixed trapezoidal error on the maturity axis, which refining the time step does
not improve.

This table is also the diagnostic that caught two real bugs during development
(an uninitialised output array and a misplaced `return`). Both produced
plausible-looking prices but an error floor that refinement could not move —
which is what a structural error looks like, as opposed to discretization.

### Martingale verification

Under the risk-neutral measure, discounted bond prices `Z(t,T) = P(t,T)/B(t)`
must be martingales, so `E[Z(t,T)] = P(0,T)`. With antithetic variates, 5,000
pairs, at `t = 1.0`:

| maturity | mean Z | P(0,T) | bias | std err | variance reduction | t |
|---------:|-------:|-------:|-----:|--------:|-------------------:|--:|
| 3.0 | 0.88691654 | 0.88692044 | -3.90e-06 | 4.22e-06 | 65.4x | -0.92 |
| 5.0 | 0.81872156 | 0.81873075 | -9.20e-06 | 1.04e-05 | 40.1x | -0.89 |
| 8.0 | 0.72613190 | 0.72614904 | -1.71e-05 | 1.95e-05 | 27.6x | -0.88 |

No violation detectable at this precision.

**Establishing that the test has power.** A small bias means little unless the
test could detect a large one. Re-running with the drift forced to zero, over the
*same* Brownian shocks, and differencing path by path:

| | value |
|---|---:|
| sd of Z across paths | 2.94e-02 |
| sd of the paired difference | 1.94e-05 |
| variance reduction from pairing | 1519x |
| mean difference (drift on − off) | -5.39e-04 |
| t-statistic | -1966 |

Removing the drift shifts Z by an amount **59x larger** than the residual bias
observed with the drift in place. The martingale result therefore reflects the
model, not an insensitive test.

**On discretization.** Comparing coarse runs against an 800-step reference on a
shared Brownian path (coarse shocks built by block-aggregating the fine ones, so
every run follows the same underlying path) showed no measurable dt-dependence:
differences stayed below 2e-06 and within their own standard errors even at 25
steps. Euler discretization is not the limiting error source for Z at these
parameters.

### Caplet pricing

**Validation.** Under Hull-White a caplet has a closed form, via the identity that
a caplet equals `(1 + tau*K)` puts on a zero-coupon bond struck at
`1/(1 + tau*K)`. Caplet on [1.0, 1.5], 20,000 paths:

| strike | | Monte Carlo | closed form | difference | MC std err |
|-------:|:--|------------:|------------:|-----------:|-----------:|
| 0.0304 | ITM | 0.00503709 | 0.00504031 | -3.22e-06 | 4.89e-06 |
| 0.0354 | ITM | 0.00319884 | 0.00319726 | +1.58e-06 | 7.64e-06 |
| 0.0404 | ATM | 0.00178589 | 0.00177968 | +6.20e-06 | 9.27e-06 |
| 0.0454 | OTM | 0.00084850 | 0.00084664 | +1.86e-06 | 7.81e-06 |
| 0.0504 | OTM | 0.00033341 | 0.00033649 | -3.08e-06 | 5.14e-06 |

Every difference is within one standard error, with alternating signs — noise,
not bias.

**Extension.** Under a two-factor structure (`sigma_1` constant, a level factor;
`sigma_2*exp(-(lam/2)(T-t))`, a slope factor) the forward bond price leaves the
one-dimensional lognormal family and the put-on-a-bond identity yields no closed
form. Monte Carlo under the general framework still prices it:

| strike | two-factor MC | std err | P(ITM) | uplift vs HW |
|-------:|--------------:|--------:|-------:|-------------:|
| 0.0304 | 0.00507599 | 5.34e-06 | 0.847 | +0.7% |
| 0.0354 | 0.00325092 | 8.09e-06 | 0.700 | +1.7% |
| 0.0404 | 0.00183659 | 9.72e-06 | 0.500 | +3.2% |
| 0.0454 | 0.00090023 | 8.26e-06 | 0.308 | +6.3% |
| 0.0504 | 0.00037243 | 5.61e-06 | 0.157 | +10.7% |

The uplift grows monotonically out of the money and is statistically significant
(6.4 standard errors at the highest strike). Extra dispersion in the forward rate
matters disproportionately for options whose value lies entirely in the tail.

**Caveat, stated plainly:** the two parameter sets were not calibrated to a common
total variance, so part of this gap reflects "more volatility" rather than "a
second factor specifically." Isolating the two-factor effect would require
matching the structures on forward-rate variance at T1 first.

---

## Repository layout


## Running it

```bash
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt

python3 -m pytest tests/ -v       # test suite
python3 -m src.validate           # Hull-White validation
python3 -m src.martingale_check   # martingale verification
python3 -m src.caplet             # caplet pricing
```

Scripts that import from `src/` must be run as modules (`python3 -m src.x`) from
the project root.

---

## Notes on method

**Paired-seed comparison.** Several quantities here are far smaller than the Monte
Carlo noise at any practical sample size — the drift's effect on the forward curve
is around 1e-5 against a standard error of 4e-4, a signal forty times below the
noise. Running two simulations over identical Brownian shocks and differencing
path by path cancels the diffusion within each path rather than merely averaging
it away, which made these effects measurable at a few thousand paths instead of
the hundreds of thousands an unpaired comparison would need. The drift control
above gained a factor of 1519 this way.

**Antithetic variates** reduce variance but not bias, and their effectiveness
degrades for payoffs with a kink: 40–65x on the martingale test (a smooth
functional), 1.5–5.3x on caplets, falling as the option moves out of the money.

**Parameter status.** Every parameter is labelled in the source as *paper
convention*, *arbitrary but defensible*, or *needs justification*. The flat 4%
initial curve is arbitrary, chosen so initial discount factors are exact in closed
form and the validation is not contaminated by bootstrapping error. `sigma` and
`a` are defensible in magnitude but uncalibrated; production use would fit them to
cap or swaption prices.

**Error sources**, in the order they matter here: Euler-Maruyama time stepping
(weak order O(dt)); trapezoidal integration along the maturity axis (O(h^2), and
the binding constraint once the time step is fine); trapezoidal accumulation of
the money-market account. The convergence tables distinguish these — an error that
shrinks at the predicted rate is discretization, one that plateaus is a bug.

---

## Reference

Heath, D., Jarrow, R., & Morton, A. (1992). Bond Pricing and the Term Structure of
Interest Rates: A New Methodology for Contingent Claims Valuation. *Econometrica*,
60(1), 77–105.