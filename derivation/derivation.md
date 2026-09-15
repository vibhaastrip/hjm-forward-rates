# The HJM No-Arbitrage Drift Restriction

A derivation from first principles, following Heath, Jarrow & Morton (1992),
*"Bond Pricing and the Term Structure of Interest Rates: A New Methodology for
Contingent Claims Valuation,"* Econometrica 60(1), 77–105.

---

## Contents

1. [The term structure](#1-the-term-structure)
2. [Mathematical toolkit](#2-mathematical-toolkit)
3. [The forward rate process (Condition C.1)](#3-the-forward-rate-process-condition-c1)
4. [The bond price dynamics](#4-the-bond-price-dynamics)
5. [The no-arbitrage drift restriction](#5-the-no-arbitrage-drift-restriction)
6. [Contingent claim valuation](#6-contingent-claim-valuation)
7. [Worked example: constant volatility](#7-worked-example-constant-volatility)
8. [The Hull-White reduction](#8-the-hull-white-reduction)
9. [Caplets](#9-caplets)
10. [Discretization](#10-discretization)

---

## 1. The term structure

### Zero-coupon bonds

The basic object is a **zero-coupon bond**: an IOU paying exactly $1 at maturity
$T$, with no coupons. Write $P(t,T)$ for its price at time $t$.

Three requirements (the paper's Section 2):

- $P(T,T) = 1$ — at maturity it is worth its face value.
- $P(t,T) > 0$ — otherwise a certain dollar could be had for free.
- $\partial \ln P(t,T)/\partial T$ exists — needed for forward rates to be defined.

The second property that matters for everything downstream: $P(t,T)$ is
**decreasing in $T$**. A dollar further away is worth less today, because money
in hand can earn interest.

### Forward rates, derived

A **forward rate** $f(t,T)$ is the rate you can lock in today, at time $t$, for
a loan starting at some future date $T$. It is not quoted anywhere — it is
implied by the shape of the bond price curve, and the derivation is a
replication argument.

Suppose you want to lock in borrowing between $T$ and $T + \Delta T$, using only
bonds. Two trades:

- **Buy** one $T$-maturity bond, costing $P(t,T)$ today, paying $1 at $T$.
- **Sell** $k$ units of the $(T+\Delta T)$-maturity bond, receiving
  $k \cdot P(t, T+\Delta T)$ today, obliging you to pay $k$ at $T+\Delta T$.

Choose $k$ so today's cash flows cancel exactly:

$$P(t,T) = k \cdot P(t,T+\Delta T) \quad \Longrightarrow \quad k = \frac{P(t,T)}{P(t,T+\Delta T)}$$

Net position: receive $1 at $T$, pay $k$ at $T+\Delta T$. You have synthetically
borrowed over $[T, T+\Delta T]$ at a rate implied by $k$. Writing that rate $F$
in simple compounding, $1 + F\,\Delta T = k$, so

$$F = \frac{1}{\Delta T}\left(\frac{P(t,T)}{P(t,T+\Delta T)} - 1\right)
  = \frac{1}{\Delta T} \cdot \frac{P(t,T) - P(t,T+\Delta T)}{P(t,T+\Delta T)}$$

**Worked example.** With $P(0,1) = 0.97$ and $P(0,2) = 0.93$:
$k = 0.97/0.93 \approx 1.043$, so the one-year forward rate starting in one year
is about 4.3%.

### The continuous limit

Let $\Delta T \to 0$. The numerator over $\Delta T$ is *minus* a derivative —
note the ordering is "old minus new," backwards from the usual difference
quotient:

$$\lim_{\Delta T \to 0}\frac{P(t,T) - P(t,T+\Delta T)}{\Delta T} = -\frac{\partial P(t,T)}{\partial T}$$

so

$$f(t,T) = -\frac{1}{P(t,T)}\frac{\partial P(t,T)}{\partial T}
         = -\frac{\partial \ln P(t,T)}{\partial T} \tag{1}$$

**Why the minus sign is necessary, not conventional.** $P$ is decreasing in $T$,
so $\partial \ln P/\partial T < 0$. Without the minus sign, $f$ would be negative
under normal conditions. The sign flips a naturally-negative derivative into a
positive rate.

Note this is the **continuously-compounded** forward rate, whereas the worked
example above gave a **simply-compounded** one. They are related by
$f = \ln(1+F)$: with $F = 4.3\%$, $f = \ln(1.043) \approx 4.21\%$. The paper works
throughout in continuous compounding, because that is what makes the calculus
clean.

### Recovering the bond price

Integrating (1) back up:

$$P(t,T) = \exp\left(-\int_t^T f(t,s)\,ds\right) \tag{2}$$

The bond price is the exponentiated, negated sum of all forward rates between
now and maturity — knowing the rate for every instant along the way is the same
as knowing the total discount factor.

### The spot rate

$$r(t) = f(t,t) \tag{3}$$

The instantaneous rate for a loan starting *now* — the shortest point on the
curve.

### Why model forward rates rather than bond prices

This is the paper's stated methodological innovation, and the reason is
technical rather than economic. A bond's price is pinned to exactly $1 at
maturity, so its volatility **must** shrink to zero as $t \to T$ — there is no
uncertainty left. Any constant-volatility model of $P(t,T)$ is therefore
inconsistent with the bond's own boundary condition. Forward rates carry no such
constraint: a constant forward-rate volatility is perfectly compatible with
$P(T,T) = 1$.

---

## 2. Mathematical toolkit

### Brownian motion

$W(t)$ is a standard Brownian motion if:

1. $W(0) = 0$
2. Increments over disjoint intervals are independent
3. $W(t) - W(s) \sim N(0, t-s)$ — variance equals elapsed time
4. Paths are continuous but **nowhere differentiable**

**Construction from a random walk.** Chop $[0,T]$ into $n$ steps of size
$\Delta t = T/n$. At each step move $\pm\sqrt{\Delta t}$ on a fair coin flip.

The $\sqrt{\Delta t}$ scaling is not arbitrary — it is the unique choice that
keeps randomness alive in the limit. Each step has variance
$(\sqrt{\Delta t})^2 = \Delta t$; summing $n = T/\Delta t$ independent steps gives
total variance $n\,\Delta t = T$, matching property 3 *regardless of how finely
we chop*. Had we scaled steps by $\Delta t$ instead, total variance would be
$n(\Delta t)^2 = T\Delta t \to 0$: all randomness would vanish and we would be
left with a flat deterministic line.

### Quadratic variation

This is the single structural fact that separates stochastic from ordinary
calculus. Sum the **squared** increments:

$$\sum_{i=1}^n (\Delta W_i)^2$$

Each $\Delta W_i = \pm\sqrt{\Delta t}$, so $(\Delta W_i)^2 = \Delta t$ —
deterministically, regardless of direction. Hence

$$\sum_{i=1}^n (\Delta W_i)^2 = n \cdot \Delta t = T$$

**every time**, not merely in expectation.

Contrast an ordinary smooth function: there, increments are $O(\Delta t)$, so
squared increments are $O(\Delta t^2)$, which vanishes as the partition refines.
That vanishing is essentially what "differentiable" means. For Brownian motion
squared increments are $O(\Delta t)$ — the same order as the increments
themselves — so they do not vanish; they accumulate to a finite, deterministic,
nonzero total.

Written informally: $(dW)^2 = dt$. Every correction term in what follows traces
back to this one fact.

### Itô integrals

To define $\int_0^T \sigma(t)\,dW(t)$, partition $[0,T]$ and form

$$I_n = \sum_{i=0}^{n-1}\sigma(t_i)\big(W(t_{i+1}) - W(t_i)\big)$$

evaluating $\sigma$ at the **left endpoint** $t_i$. Then $I = \lim_{n\to\infty} I_n$,
which exists provided $\int_0^T \sigma(t)^2\,dt < \infty$.

**The left endpoint is forced, not chosen.** For an ordinary Riemann integral any
evaluation point in $[t_i, t_{i+1}]$ gives the same limit. Here it does not:
Brownian motion is rough enough that the choice interacts with the increment size
in a way that survives the limit. Left-endpoint evaluation (Itô's convention)
corresponds to a realistic trading rule — you commit to a position *before*
seeing the next random move, never after. The midpoint convention gives a
different object (the Stratonovich integral), which recovers ordinary-calculus
answers but does not correspond to any implementable strategy.

**Worked example.** Compute $\int_0^T W(t)\,dW(t)$ directly from the sum. Use the
identity $a(b-a) = \tfrac12(b^2 - a^2) - \tfrac12(b-a)^2$ — verifiable by
expanding the right side:

$$\tfrac12(b^2-a^2) - \tfrac12(b^2 - 2ab + a^2) = ab - a^2 = a(b-a) \;\checkmark$$

Applying it term by term with $a = W(t_i)$, $b = W(t_{i+1})$:

$$I_n = \tfrac12\sum_i\left[W(t_{i+1})^2 - W(t_i)^2\right] - \tfrac12\sum_i\left[W(t_{i+1}) - W(t_i)\right]^2$$

The first sum **telescopes** — every interior term cancels — leaving
$\tfrac12\left[W(T)^2 - W(0)^2\right] = \tfrac12 W(T)^2$, exactly, for any $n$.
The second sum is precisely the quadratic-variation sum, converging to $T$.
Therefore

$$\int_0^T W(t)\,dW(t) = \tfrac12 W(T)^2 - \tfrac12 T$$

Ordinary calculus would predict $\tfrac12 W(T)^2$ by analogy with
$\int x\,dx$. The extra $-\tfrac12 T$ is the entire stochastic content of the
answer, and it comes from one place: the quadratic variation does not vanish.

### Itô's Lemma

For $dX = \mu\,dt + \sigma\,dW$ and smooth $g(X,t)$:

$$dg = \left(\frac{\partial g}{\partial t} + \mu\frac{\partial g}{\partial X}
    + \tfrac12\sigma^2\frac{\partial^2 g}{\partial X^2}\right)dt
    + \sigma\frac{\partial g}{\partial X}\,dW$$

Mechanically it is the ordinary chain rule plus a second-order term, expanded
using the multiplication rules

$$(dt)^2 = 0, \qquad dt\cdot dW = 0, \qquad (dW)^2 = dt$$

and, for independent Brownian motions,

$$dW_i \cdot dW_j = 0 \quad (i \neq j)$$

That last rule is why, in multi-factor models, each factor's contribution to the
drift can be derived independently and simply summed.

**Consistency check.** Apply Itô's Lemma with $g(x) = \tfrac12 x^2$, $X = W$:
$dg = W\,dW + \tfrac12\,dt$. Integrating gives
$\tfrac12 W(T)^2 = \int_0^T W\,dW + \tfrac12 T$ — the same answer as the
summation computation above, now from the rule rather than from first
principles.

**Practical point.** The summation is the *definition*; Itô's Lemma and the
multiplication table are what you actually compute with, exactly as one derives
$\int x\,dx$ from Riemann sums once and then uses the power rule forever after.

### Martingales

$X_t$ is a martingale with respect to a filtration $\{\mathcal{F}_t\}$ if

$$E[X_{t+1} \mid \mathcal{F}_t] = X_t$$

In words: given everything known now, the best forecast of the future value is
today's value. No drift, no edge.

**Discrete example.** Start with £100; each day a fair coin adds or subtracts
£10. Then $E[X_{t+1}\mid\mathcal F_t] = \tfrac12(X_t+10) + \tfrac12(X_t-10) = X_t$.
A martingale. Change the payoff to $+£11$ / $-£10$ and the expectation becomes
$X_t + 0.5$ — a submartingale, with a genuine edge.

**The fact that does most of the work here:** a *pure* Itô integral
$\int_0^t \sigma(s)\,dW(s)$, with no separate $dt$ term, is always a martingale
with expectation zero.

*Proof sketch.* Condition on the past:

$$E\big[\sigma(t_i)(W(t_{i+1}) - W(t_i)) \mid \mathcal F_{t_i}\big]
= \sigma(t_i)\,E\big[W(t_{i+1}) - W(t_i) \mid \mathcal F_{t_i}\big] = 0$$

using **adaptedness** (so $\sigma(t_i)$ comes outside the conditional
expectation) and **mean-zero independent increments**. Summing zeros gives zero,
and the same argument applied to any sub-interval $[s,t]$ gives the full
martingale property.

Note this is a *different* mechanism from quadratic variation. $(dW)^2 = dt$
explains why Itô's Lemma has an extra drift-generating term; adaptedness plus
mean-zero increments explains why a driftless stochastic integral is a
martingale. Conflating them is easy and leads to confusion later.

### Stochastic Fubini

For well-behaved functions, ordinary Fubini lets you swap the order of a double
integral freely. With a stochastic integral inside, this is not automatic: an
Itô integral's definition depends on the left-endpoint, non-anticipating
construction, and swapping must preserve that structure.

The theorem states that, given the integrability conditions in C.1–C.3,

$$\int_0^y\left(\int_t^T \sigma(s,a)\,da\right)dW(s)
= \int_t^T\left(\int_0^y \sigma(s,a)\,dW(s)\right)da$$

Read left-to-right: integrate over maturity first, then do the stochastic
integral. Right-to-left: do the stochastic integral for each fixed maturity, then
integrate the resulting random variables over maturity. Both are valid and equal.

This is the licence that makes the bond-price derivation in Section 4 possible.
The paper proves it in its Appendix (Lemma 0.1 and Corollaries 1–2), following
Ikeda & Watanabe.

### No-arbitrage and the martingale measure

**Fundamental Theorem of Asset Pricing** (Harrison–Kreps 1979, Harrison–Pliska
1981, both cited by HJM): a market admits no arbitrage **if and only if** there
exists a probability measure under which every *discounted* asset price is a
martingale.

**Discounting** means dividing by the money-market account

$$B(t) = \exp\left(\int_0^t r(y)\,dy\right)$$

the value of £1 rolled over continuously at the stochastic short rate. This
matters because a raw bond price is *not* a martingale even in an arbitrage-free
world — it drifts upward toward its face value as maturity approaches. Only
after stripping out that risk-free accrual does the martingale question become
meaningful.

**No-arbitrage is weaker than market efficiency.** Efficiency is an empirical
claim about the real-world measure: that prices already reflect available
information. No-arbitrage says only that no trading strategy can manufacture a
riskless profit from nothing. A market can be persistently mispriced and still
admit no arbitrage, if the mispricing cannot be locked in. Risk premia are fully
compatible with no-arbitrage — they just have to be *consistent* across assets,
which is exactly what the conditions in Section 5 pin down.

### Girsanov's theorem

We need to move between measures. The idea, stripped of continuous-time
machinery: a fair coin and a 0.7/0.3 coin describe the same two outcomes with
different weights. Reweighting does not change *what can happen*, only how much
probability each outcome carries.

**The construction.** Let $W(t)$ be Brownian under $P$. Define a new measure $Q$
by the Radon–Nikodym derivative

$$\frac{dQ}{dP} = \exp\left(\gamma W(T) - \tfrac12\gamma^2 T\right)$$

This is a likelihood ratio: paths where $W(T)$ happened to be large and positive
get up-weighted; paths that drifted down get down-weighted. The
$-\tfrac12\gamma^2T$ term is a normalizer.

**Step 1: it is a valid reweighting.** Requires $E_P[dQ/dP] = 1$:

$$E_P\left[e^{\gamma W(T) - \frac12\gamma^2T}\right]
= e^{-\frac12\gamma^2T}\cdot E_P\left[e^{\gamma W(T)}\right]$$

The Gaussian moment generating function supplies $E_P[e^{\gamma W(T)}]
= e^{\frac12\gamma^2 T}$, so the product is exactly 1.

**Deriving that MGF.** For $X \sim N(0,\sigma^2)$:

$$E[e^{\theta X}] = \frac{1}{\sqrt{2\pi\sigma^2}}\int_{-\infty}^{\infty}
\exp\left(\theta x - \frac{x^2}{2\sigma^2}\right)dx$$

Complete the square. Factor out $-\tfrac{1}{2\sigma^2}$:

$$\theta x - \frac{x^2}{2\sigma^2} = -\frac{1}{2\sigma^2}\left(x^2 - 2\sigma^2\theta x\right)$$

Using $x^2 - 2cx = (x-c)^2 - c^2$ with $c = \sigma^2\theta$:

$$= -\frac{1}{2\sigma^2}\left[(x - \sigma^2\theta)^2 - \sigma^4\theta^2\right]
= -\frac{(x-\sigma^2\theta)^2}{2\sigma^2} + \frac{\sigma^2\theta^2}{2}$$

The constant pulls outside the integral, and what remains is the density of a
$N(\sigma^2\theta, \sigma^2)$ variable, which integrates to 1 — shifting a bell
curve does not change the area beneath it. Hence

$$E[e^{\theta X}] = e^{\theta^2\sigma^2/2}$$

Applied to $W(T) \sim N(0,T)$, the variance is $T$ (not $T^2$ — $T$ *is* the
variance, occupying the slot labelled $\sigma^2$ in the general formula), giving
$e^{\gamma^2 T/2}$.

**Step 2: what happens to $W$.** Define $\tilde W(t) = W(t) - \gamma t$. Compute
its MGF under $Q$, using $E_Q[X] = E_P[X \cdot dQ/dP]$:

$$E_Q\left[e^{\theta\tilde W(T)}\right]
= E_P\left[e^{\theta(W(T)-\gamma T)}\cdot e^{\gamma W(T) - \frac12\gamma^2T}\right]$$

Collect the exponents — the random parts and the constants separately:

$$= e^{-\theta\gamma T - \frac12\gamma^2 T}\cdot E_P\left[e^{(\theta+\gamma)W(T)}\right]
= e^{-\theta\gamma T - \frac12\gamma^2 T}\cdot e^{\frac12(\theta+\gamma)^2T}$$

Expand $(\theta+\gamma)^2 = \theta^2 + 2\theta\gamma + \gamma^2$:

$$\text{exponent} = -\theta\gamma T - \tfrac12\gamma^2T
+ \tfrac12\theta^2T + \theta\gamma T + \tfrac12\gamma^2T$$

The $\pm\theta\gamma T$ terms cancel; the $\pm\tfrac12\gamma^2T$ terms cancel.
Everything except $\tfrac12\theta^2T$ vanishes:

$$E_Q\left[e^{\theta\tilde W(T)}\right] = e^{\frac12\theta^2 T}$$

which is exactly the MGF of $N(0,T)$. So under $Q$, $\tilde W$ is a standard
Brownian motion — and therefore $W(t) = \tilde W(t) + \gamma t$ carries drift
$\gamma$ under $Q$.

**The cancellation is engineered, not lucky.** The $-\tfrac12\gamma^2T$
normalizer in $dQ/dP$ and the $-\gamma t$ shift in $\tilde W$ are built as a
matched pair precisely so this works for any $\theta$ and $\gamma$.

**What is and is not preserved.** $Q$ is a genuinely different measure:
$E_Q[W(T)] = \gamma T \neq 0 = E_P[W(T)]$. It is not a relabelling. What
"equivalent" guarantees is only that $P$ and $Q$ agree on which events are
*impossible*; everything quantitative may differ.

**Why using the "wrong" measure gives the right price.** A derivative's price is
defined as the cost of *replicating* its payoff through trading, not as its
expected payoff. It is a theorem that this replication cost equals
$E_Q[\text{discounted payoff}]$ for any equivalent martingale measure $Q$ —
precisely because $Q$ was constructed to make discounted tradeable assets
driftless, which is the condition the hedging argument needs. Using $P$ would
answer a different question (what will actually happen on average) and give the
wrong price.

**Mechanically**, in everything below: a process with drift $b$ and volatility
$a$ becomes driftless under $Q$ when $b + a\gamma = 0$. Solving for $\gamma$
constructs the measure.

---

## 3. The forward rate process (Condition C.1)

The paper's modelling assumption, its equation (4):

$$f(t,T) - f(0,T) = \int_0^t \alpha(v,T,\omega)\,dv
+ \sum_{i=1}^n\int_0^t \sigma_i(v,T,\omega)\,dW_i(v) \tag{4}$$

**Every symbol:**

| symbol | meaning |
|---|---|
| $f(t,T)$ | forward rate at calendar time $t$ for maturity $T$ (random) |
| $f(0,T)$ | today's observed forward curve — a fixed, known **input** |
| $v$ | dummy integration variable over calendar time |
| $T$ | fixed maturity parameter, held constant throughout (4) |
| $\omega$ | which random outcome; signals that $\alpha, \sigma_i$ may depend on the whole realized path |
| $\alpha$ | drift |
| $\sigma_i$ | loading of the $i$-th Brownian factor on this maturity |
| $W_i$ | $n$ independent Brownian motions driving the whole curve |

In words: *the forward rate for maturity $T$ equals its starting value, plus
accumulated drift, plus accumulated shocks from $n$ independent random sources.*

**Technical conditions.** $f(0,\cdot)$ fixed and non-random;
$\int_0^T|\alpha|\,dt < \infty$; $\int_0^T\sigma_i^2\,dt < \infty$. The last is
exactly the condition that makes the Itô integrals in (4) well-defined.

**C.1 is posited, not derived.** This is worth being explicit about. It is a
modelling choice, in the same way that $dS = \mu S\,dt + \sigma S\,dW$ is a
choice in Black-Scholes rather than a theorem. Everything that follows is
conditional on it.

The paper names its two substantive restrictions (page 79): forward rates have
**continuous sample paths** (no jumps) and depend on **finitely many random
shocks** across the entire curve. The first rules out jump-diffusion rate models;
the second means the whole curve moves in a highly correlated way at each
instant, rather than each maturity wandering independently.

At this stage $\alpha$ and $\sigma_i$ are otherwise **completely unrestricted**.
That is deliberate — Section 5 is where no-arbitrage starts constraining $\alpha$.

**Two immediate consequences.** Setting $T = t$ in (4) gives the spot rate
process (equation 5); and $B(t) = \exp(\int_0^t r(y)\,dy)$ (equation 6) defines
the money-market account, with condition C.2 ensuring it stays finite and
strictly positive — necessary because we are about to divide by it.

---

## 4. The bond price dynamics

**Goal:** get from (4), which describes forward rates, to an SDE for $P(t,T)$.
Single factor throughout for clarity; the multi-factor case adds a sum over $i$
at every step and nothing else.

### Setting up

Start from $\ln P(t,T) = -\int_t^T f(t,s)\,ds$ (equation 2) and substitute (4):

$$\ln P(t,T) = \underbrace{-\int_t^T f(0,s)\,ds}_{A}
\underbrace{- \int_t^T\left[\int_0^t\alpha(v,s)\,dv\right]ds}_{B}
\underbrace{- \int_t^T\left[\int_0^t\sigma(v,s)\,dW(v)\right]ds}_{C}$$

### Step 1: swap the integration order

Term $B$ is an ordinary double integral — ordinary Fubini applies. Term $C$ has a
stochastic integral inside, so this is where **stochastic Fubini** is invoked
(the rectangular case, the paper's Corollary 1):

$$\ln P(t,T) = -\int_t^T f(0,s)\,ds
- \int_0^t\left[\int_t^T\alpha(v,s)\,ds\right]dv
- \int_0^t\left[\int_t^T\sigma(v,s)\,ds\right]dW(v)$$

Both remaining double integrals now have $v$ outermost, which is the shape we
want. But the inner limits run from $t$, whereas we need them running from $v$.

### Step 2: split at $v$

Since $v \le t \le T$ always holds here:

$$\int_t^T = \int_v^T - \int_v^t$$

Applying this to both terms splits each into a "$\int_v^T$" piece and a
"$\int_v^t$" leftover:

- $B \to B_1 + B_2$ where
  $B_1 = -\int_0^t\left[\int_v^T\alpha\,ds\right]dv$ and
  $B_2 = +\int_0^t\left[\int_v^t\alpha\,ds\right]dv$
- $C \to C_1 + C_2$ analogously.

### Step 3: the leftovers reconstruct the accumulated short rate

This is the step that makes the derivation work. Take the spot rate process
(equation 5) and integrate it over $y \in [0,t]$:

$$\int_0^t r(y)\,dy = \int_0^t f(0,y)\,dy
+ \int_0^t\left[\int_0^y\alpha(v,y)\,dv\right]dy
+ \int_0^t\left[\int_0^y\sigma(v,y)\,dW(v)\right]dy$$

Apply stochastic Fubini again to the last term — this time over the **triangular**
region $\{0 \le v \le y \le t\}$, which is the paper's Corollary 2:

$$\int_0^t\left[\int_0^y\sigma(v,y)\,dW(v)\right]dy
= \int_0^t\left[\int_v^t\sigma(v,y)\,dy\right]dW(v)$$

The right-hand side is **exactly $C_2$** (with the dummy variable renamed). The
same rearrangement matches $B_2$. Therefore

$$B_2 + C_2 = \int_0^t r(y)\,dy - \int_0^t f(0,y)\,dy$$

**What this means.** The part of the bond's log-price movement attributable to
"time simply passing" separates out as exactly the accumulated risk-free rate.
That is intuitively what should happen: some of a bond's return is just the going
short rate, earned as maturity approaches.

### Step 4: assemble

Substituting, and using $\ln P(0,T) = -\int_0^T f(0,s)\,ds$ split at $t$ to absorb
the remaining initial-curve terms (they cancel exactly), we get

$$\ln P(t,T) = \ln P(0,T) + \int_0^t\left[r(v) + b(v,T)\right]dv
- \tfrac12\int_0^t a(v,T)^2\,dv + \int_0^t a(v,T)\,dW(v) \tag{8}$$

with the definitions

$$a(v,T) \equiv -\int_v^T \sigma(v,s)\,ds, \qquad
b(v,T) \equiv -\int_v^T\alpha(v,s)\,ds + \tfrac12 a(v,T)^2$$

### Step 5: exponentiate

Apply Itô's Lemma with $g(x) = e^x$ to $P = e^{\ln P}$. Writing
$Y = \ln P(t,T)$, we have $(dY)^2 = a(t,T)^2\,dt$, so

$$dP = P\,dY + \tfrac12 P (dY)^2
= P\left[r + b - \tfrac12 a^2\right]dt + \tfrac12 P a^2\,dt + P a\,dW$$

The $\mp\tfrac12 a^2$ terms **cancel exactly** — which is precisely why $b$ was
defined with that extra $+\tfrac12 a^2$ built in. Result:

$$dP(t,T) = \left[r(t) + b(t,T)\right]P(t,T)\,dt + a(t,T)P(t,T)\,dW(t) \tag{9}$$

**Reading this off:** the bond's own volatility is
$a(t,T) = -\int_t^T\sigma(t,s)\,ds$ — the **accumulated forward-rate volatility
between now and maturity**. That is a genuine structural result, not bookkeeping:
a bond's riskiness is inherited from how volatile the forward curve is across the
maturities it spans. It also explains automatically why $a(T,T) = 0$: at maturity
there is nothing left to accumulate.

### Discounting

With $Z(t,T) = P(t,T)/B(t)$ and $\ln B(t) = \int_0^t r(y)\,dy$, the $r(v)$ terms
in (8) cancel exactly — which is the entire purpose of dividing by $B(t)$:

$$\ln Z(t,T) = \ln Z(0,T) + \int_0^t b(v,T)\,dv
- \tfrac12\int_0^t a(v,T)^2\,dv + \int_0^t a(v,T)\,dW(v) \tag{10}$$

$Z$ is **not yet** a martingale: the $b(v,T)$ drift survives. It represents the
bond's expected return *above* the risk-free rate. Eliminating it is Section 5.

### An aside on the Markov property

A process is **Markov** if its future distribution, given the entire history,
depends only on its current value — the present fully summarizes the past.

Because C.1 permits $\alpha$ and $\sigma_i$ to depend on the whole realized path
(the $\omega$ argument), $P(t,T)$ **need not** be Markov. If forward-rate
volatility depends on, say, how much the curve has moved over the past month,
then knowing today's bond price is not enough to know tomorrow's distribution.

The paper flags this as a feature. Earlier models (Brennan–Schwartz, Langetieg)
*assumed* bond prices were functions of a few current state variables; HJM does
not need that restriction. The cost, as Section 8 shows, is that closed forms
generally disappear.

---

## 5. The no-arbitrage drift restriction

### Converting to $Z$'s own SDE

Applying Itô to $Z = e^{\ln Z}$ with (10), the two $\tfrac12 a^2$ terms cancel as
before, leaving

$$dZ(t,T) = Z(t,T)\left[b(t,T)\,dt + a(t,T)\,dW(t)\right]$$

so $Z$'s drift is simply $b(t,T)$.

### Killing the drift

Substitute the Girsanov shift $dW = d\tilde W + \gamma(t)\,dt$:

$$dZ = Z\left[\left(b(t,T) + a(t,T)\gamma(t)\right)dt + a(t,T)\,d\tilde W(t)\right]$$

For $Z$ to be a martingale under $\tilde Q$ the drift must vanish — a pure Itô
integral is a martingale, but only if there is no $dt$ term left. Hence

$$b(t,T) = -a(t,T)\gamma(t) \tag{13}$$

This is the paper's equation (13), and it has a clean reading: the left side is
the bond's excess expected return over the risk-free rate; the right side is
(minus) the market price of risk times the bond's exposure to the random factor.

### Conditions C.4, C.5, C.6

With $n$ factors there are $n$ unknown market prices of risk
$\gamma_1,\dots,\gamma_n$, but a single bond supplies only one equation. The three
conditions address three distinct failure modes.

**C.4 (existence).** Pick $n$ bond maturities $S_1 < \cdots < S_n$. Each supplies
its own version of (13), giving $n$ equations for $n$ unknowns — the paper's
equation (11), a matrix system. C.4 assumes this system is solvable (plus
integrability conditions 12a–c that keep the resulting Girsanov exponential
well-behaved). **Proposition 1**: solvability is necessary *and sufficient* for an
equivalent martingale measure to exist for those bonds.

**C.5 (uniqueness).** A solvable linear system can still have infinitely many
solutions if its coefficient matrix is singular. C.5 requires the matrix
$[a_i(t,S_j)]$ to be **nonsingular**. **Proposition 2**: this is necessary and
sufficient for the measure to be unique.

Intuitively, singularity would mean two factors load proportionally onto every
chosen bond, making their risk prices indistinguishable from those bonds alone.
The single-factor case reduces to $\sigma_1(t,S_1) > 0$ — you cannot back out a
risk price from a bond with no volatility.

Economically, uniqueness corresponds to **market completeness**: every contingent
claim can be replicated, so every claim has a well-defined price.

**C.6 (consistency).** C.4 and C.5 fix a measure for *one chosen set* of $n$
bonds. Nothing yet stops a different choice giving a *different* measure. That
would be incoherent: "the" risk-neutral measure would depend on which bonds
happened to be used to construct it.

C.6 assumes all choices give the same $\tilde Q$. **Proposition 3** shows three
statements are equivalent:

- **(16)** one universal $\tilde Q$ makes $Z(t,T)$ a martingale for *every* $T$;
- **(17)** the $\gamma_i$ do not depend on which bonds were used — they are
  genuinely functions of $t$ alone, written $\phi_i(t)$ (the "standard finance
  condition");
- **(18)** a direct restriction on $\alpha(t,T)$, holding for all $T$ at once.

### Deriving equation (18)

Take (13) with the universal $\phi(t)$, and substitute $b$'s definition:

$$-\int_t^T\alpha(t,s)\,ds + \tfrac12 a(t,T)^2 = -a(t,T)\phi(t)$$

$$\int_t^T\alpha(t,s)\,ds = \tfrac12 a(t,T)^2 + a(t,T)\phi(t)$$

Differentiate both sides with respect to $T$. The left side gives $\alpha(t,T)$
by the Fundamental Theorem of Calculus. The right side needs the **chain rule**,
because $a(t,T)$ itself depends on $T$ through its own integral — with

$$\frac{\partial a(t,T)}{\partial T} = -\sigma(t,T)$$

we get

$$\alpha(t,T) = a(t,T)\cdot(-\sigma(t,T)) + \phi(t)\cdot(-\sigma(t,T))
= -\sigma(t,T)\left[a(t,T) + \phi(t)\right]$$

Substituting $a(t,T) = -\int_t^T\sigma(t,v)\,dv$:

$$\boxed{\;\alpha(t,T) = \sigma(t,T)\int_t^T\sigma(t,v)\,dv - \sigma(t,T)\phi(t)\;} \tag{18}$$

**This is the paper's central result.** Once you specify a volatility structure,
the drift is *forced*. There is no remaining freedom — an arbitrary
drift-and-volatility pair will generally admit arbitrage.

The multi-factor version sums over $i$, with no cross terms, because independent
Brownian motions satisfy $dW_i\,dW_j = 0$.

---

## 6. Contingent claim valuation

### Bond price as an expectation

$Z(t,T)$ being a $\tilde Q$-martingale means $Z(t,T) = \tilde E[Z(T,T)\mid\mathcal F_t]$.
At maturity $Z(T,T) = P(T,T)/B(T) = 1/B(T)$, so

$$P(t,T) = B(t)\,\tilde E\left[\frac{1}{B(T)}\,\Big|\,\mathcal F_t\right] \tag{19}$$

Pure martingale algebra — no new content.

### Replication

This is the conceptual heart of Section 5, and it is a different idea from
"expected payoff."

**The analogy.** A contract paying exactly $110 in a year, with 10% interest
available, must cost $100 today. Not because we forecast anything, but because
$100 in a savings account *replicates* the payoff exactly. Any other price is a
free lunch.

Generalizing: the building blocks are the money-market account and a set of zero
coupon bonds. Equation (20) states that some combination of them matches the
claim's payoff $X$ at time $S_1$:

$$N_0(S_1)B(S_1) + \sum_{i=1}^n N_{S_i}(S_1)P(S_1,S_i) = X \tag{20}$$

**Completeness** (from C.5's uniqueness) is the guarantee that such a combination
always exists. Given it, no-arbitrage forces

$$C(t) = \tilde E\left[\frac{X}{B(S_1)}\,\Big|\,\mathcal F_t\right]B(t) \tag{21}$$

— the same "discount, take the risk-neutral expectation, re-grow" recipe as (19),
now for an arbitrary payoff. A bond is just the special case $X = 1$. Equation
(22) substitutes (20) into (21) and rewrites in terms of $Z$, purely to set up the
computation.

### What you need to compute it

**Equation (23)** is the spot rate process under $\tilde Q$ — equation (5) with the
Girsanov substitution applied.

**Equation (24)** is $Z(t,u)$ in closed form. Since $dZ = a\,Z\,d\tilde W$ (zero
drift), the solution is the **stochastic exponential**

$$Z(t) = Z(0)\exp\left(-\tfrac12\int_0^t a(s)^2\,ds + \int_0^t a(s)\,d\tilde W(s)\right)$$

*Verification.* Let $Y$ be the exponent, so $dY = -\tfrac12 a^2\,dt + a\,d\tilde W$
and $(dY)^2 = a^2\,dt$. Then
$dZ = Z\,dY + \tfrac12 Z(dY)^2 = Z[-\tfrac12a^2\,dt + a\,d\tilde W] + \tfrac12 Za^2\,dt
= Za\,d\tilde W$ — drift-free, as required.

### The market price of risk cancels

Note the asymmetry: equation (24) contains **no** $\gamma_i$, but equation (23)
does. The bond price is clean by construction — eliminating its drift was the
whole point — while the spot rate merely inherited whatever $\gamma$-dependence
was in $\alpha$.

Substituting the drift restriction into (23) resolves this. Integrating (18) with
$s = t$ over $v \in [0,t]$ gives equation (25); its $\phi$ term is

$$-\sum_i\int_0^t\sigma_i(v,t)\phi_i(v)\,dv$$

while (23) already carries, from the Girsanov shift,

$$+\sum_i\int_0^t\phi_i(v)\sigma_i(v,t)\,dv$$

These are **exact negatives**. They cancel completely, leaving

$$r(t) = f(0,t) + \sum_i\int_0^t\sigma_i(v,t)\int_v^t\sigma_i(v,y)\,dy\,dv
+ \sum_i\int_0^t\sigma_i(v,t)\,d\tilde W_i(v) \tag{26}$$

**This is not a coincidence.** $\phi$ was *defined* as the quantity that makes the
risk-neutral drift vanish; substituting it back into a related quantity cancels
exactly what it was built to cancel. The cancellation is the signature of
internal consistency across the bond price and the short rate.

**The practical payoff, and the paper's main selling point:** specify a
volatility structure — something estimable from data — and price everything,
never needing to estimate a market price of risk.

---

## 7. Worked example: constant volatility

Set $\sigma(t,T) \equiv \sigma > 0$. This is the paper's Section 6 Example 1, and
it is the continuous-time limit of Ho–Lee.

**Drift.** $\int_t^T\sigma\,dv = \sigma(T-t)$, so (18) gives

$$\alpha(t,T) = \sigma^2(T-t) - \sigma\phi(t) \tag{27}$$

**Forward rate.** Substituting into (4) under $\tilde Q$, the $\phi$ terms cancel
as in Section 6, and $\int_0^t(T-v)\,dv = t(T - t/2)$:

$$f(t,T) = f(0,T) + \sigma^2 t\left(T - \tfrac{t}{2}\right) + \sigma\tilde W(t) \tag{28}$$

**Spot rate.** Set $T = t$:

$$r(t) = f(0,t) + \tfrac{\sigma^2t^2}{2} + \sigma\tilde W(t) \tag{29}$$

**Bond price.** Integrating (28) via equation (2):

$$P(t,T) = \frac{P(0,T)}{P(0,t)}\exp\left\{-\tfrac{\sigma^2}{2}tT(T-t) - \sigma(T-t)\tilde W(t)\right\} \tag{30}$$

**A known limitation.** Both (28) and (29) are Gaussian — linear in $\tilde W$ —
so rates can go negative with positive probability, for any parameter choice. The
paper's Section 7 addresses this by making volatility depend on the rate's own
level, with the caveat that an unbounded specification ($\sigma \propto f$)
produces explosive feedback and no finite solution; a capped version,
$\sigma\min(f,\lambda)$, is both Lipschitz and bounded and keeps rates
non-negative.

---

## 8. The Hull-White reduction

**Goal:** find the volatility structure under which general HJM collapses to
Hull-White,

$$dr(t) = \left[\theta(t) - a\,r(t)\right]dt + \sigma\,d\tilde W(t)$$

a mean-reverting Ornstein–Uhlenbeck short rate with closed-form prices.

**The answer:**

$$\sigma(t,T) = \sigma e^{-a(T-t)}$$

The property that matters is **separability**: $e^{-a(T-t)} = e^{-aT}\cdot e^{at}$
factors into a function of $T$ alone times a function of $t$ alone.

### Step 1: accumulated volatility

$$\int_v^T\sigma e^{-a(y-v)}\,dy = \frac{\sigma}{a}\left(1 - e^{-a(T-v)}\right)$$

### Step 2: the drift term

$$\int_0^t \sigma e^{-a(T-v)}\cdot\frac{\sigma}{a}\left(1 - e^{-a(T-v)}\right)dv
= \frac{\sigma^2}{a^2}\left(e^{-a(T-t)} - e^{-aT}\right)
- \frac{\sigma^2}{2a^2}\left(e^{-2a(T-t)} - e^{-2aT}\right)$$

### Step 3: set $T = t$

Every $e^{-a(T-t)} \to 1$, giving

$$r(t) = \underbrace{f(0,t) + \frac{\sigma^2}{a^2}(1-e^{-at})
- \frac{\sigma^2}{2a^2}(1-e^{-2at})}_{g(t),\ \text{deterministic}}
\;+\; \underbrace{\sigma e^{-at}\int_0^t e^{av}\,d\tilde W(v)}_{X(t),\ \text{random}}$$

### Step 4: the random part closes on itself

Write $X(t) = \sigma e^{-at}Y(t)$ with $Y(t) = \int_0^t e^{av}\,d\tilde W(v)$, so
$dY = e^{at}\,d\tilde W$. The product rule (no Itô correction — $e^{-at}$ is
deterministic) gives

$$dX = -a\sigma e^{-at}Y\,dt + \sigma e^{-at}e^{at}\,d\tilde W
= -aX\,dt + \sigma\,d\tilde W$$

**This is the crux.** $X$ satisfies a self-contained SDE depending only on its
own current value — an Ornstein–Uhlenbeck process. The separability of
$\sigma(t,T)$ is what makes this happen; a non-separable structure would leave
$X$ depending on its whole history.

### Step 5: assemble

With $r = g(t) + X(t)$ and $X = r - g$:

$$dr = g'(t)\,dt - a(r - g)\,dt + \sigma\,d\tilde W
= \left[\underbrace{g'(t) + ag(t)}_{\theta(t)} - a\,r(t)\right]dt + \sigma\,d\tilde W$$

exactly Hull-White, with $\theta(t)$ built from the observed initial curve and the
two parameters.

### Closed forms

Bond prices are **affine**: $P(t,T) = A(t,T)e^{-B(t,T)r(t)}$ with

$$B(t,T) = \frac{1 - e^{-a(T-t)}}{a}$$

$$\ln A(t,T) = \ln\frac{P(0,T)}{P(0,t)} + B(t,T)f(0,t)
- \frac{\sigma^2}{4a}\left(1 - e^{-2at}\right)B(t,T)^2$$

The three terms in $\ln A$: reproduce today's curve; correct for $r(t)$ differing
from the forward rate the curve implies for $t$; and a convexity adjustment from
Itô.

Note $\sigma B(t,T)$ is exactly the accumulated forward-rate volatility from
Step 1 — the explicit link confirming Hull-White *is* the HJM special case rather
than a separate model. $B(t,t) = 0$, and $B \to 1/a$ as $T \to \infty$: mean
reversion caps how much long bonds respond to today's short rate.

**Bond options** take Black-Scholes form via a change of numeraire to the
$t^*$-maturity bond, under which the forward bond price $P(t,T)/P(t,t^*)$ is a
driftless lognormal martingale:

$$C(0) = P(0,T)\Phi(h) - K P(0,t^*)\Phi(h - \nu)$$

$$h = \frac{1}{\nu}\ln\frac{P(0,T)}{K P(0,t^*)} + \frac{\nu}{2}, \qquad
\nu^2 = \frac{\sigma^2}{2a}\left(1 - e^{-2at^*}\right)B(t^*,T)^2$$

$\nu$ factors as (variance of $r(t^*)$) × (duration $B(t^*,T)$)² — short-rate
uncertainty converted into bond-price uncertainty. It is the volatility of the
**forward** bond price, *not* the bond's own return volatility; the paper flags
this distinction explicitly after its equation (34), and it is the easiest thing
to get wrong when implementing.

### Why this avoids the CIR problem

Section 8 of the paper contrasts HJM with the equilibrium approach of
Cox–Ingersoll–Ross. CIR derives bond dynamics from an economic model, which fixes
a functional form for the initial forward curve. Matching real market data then
requires "inverting" that relation for $\theta(t)$ — and the paper shows (its
equation 50) that not every observed curve is even attainable, because of CIR's
own internal consistency requirement $2K\theta(t) \ge \sigma^2$.

HJM never faces this: $f(0,\cdot)$ is an input from the start. The $\theta(t)$
derived above is computed *from* the observed curve, not fitted to it.

---

## 9. Caplets

A **caplet** is a one-period rate option. The rate is fixed at the reset date
$T_1$; the payoff is paid at $T_2$, with $\tau = T_2 - T_1$:

$$\tau\max\left(L(T_1,T_2) - K,\ 0\right)$$

where $L$ is the **simply-compounded** forward rate — market convention for
floating payments — implied by the bond price:

$$L(T_1,T_2) = \frac{1}{\tau}\left(\frac{1}{P(T_1,T_2)} - 1\right)$$

### The put-on-a-bond identity

Discount the payoff back to $T_1$ by multiplying by $P \equiv P(T_1,T_2)$:

$$P\cdot\tau\max(L - K, 0) = P\max\left(\frac{1}{P} - 1 - \tau K,\ 0\right)
= \max\left(1 - P(1+\tau K),\ 0\right)$$

(valid since $P > 0$). Factoring:

$$= (1+\tau K)\max\left(\frac{1}{1+\tau K} - P,\ 0\right)$$

So a caplet is exactly $(1+\tau K)$ **puts** on the $T_2$-bond, expiring at $T_1$,
struck at $1/(1+\tau K)$ — priceable in closed form under Hull-White using the
bond option formula above.

### Why the two-factor case has no closed form

With two factors of different maturity profiles — one constant (a level factor),
one decaying (a slope factor) — the forward bond price is no longer a
one-dimensional lognormal, and the identity yields nothing tractable. Monte Carlo
under the general HJM framework prices it directly. That contrast is the point:
the general framework earns its complexity precisely where the special case
cannot reach.

### An implementation note

Simulate only to $T_1$, not $T_2$. The payoff is known at $T_1$, and $P(T_1,T_2)$
is available exactly from the simulated curve, so discounting one period back by
multiplying by $P(T_1,T_2)$ is exact. Simulating onward would add Euler steps and
money-market integration error for no benefit.

---

## 10. Discretization

### Euler-Maruyama

$$f(t_{k+1},T) = f(t_k,T) + \alpha(t_k,T)\,\Delta t
+ \sum_i \sigma_i(t_k,T)\sqrt{\Delta t}\,Z_{i,k}$$

with $Z_{i,k} \sim N(0,1)$ independent across factors and steps, and $\alpha$
computed from (18) rather than supplied.

**One draw per factor per step, shared across all maturities.** This is a
modelling requirement, not an optimization. C.1 says the entire curve is driven
by $n$ Brownian motions; every maturity is shocked by the *same* $Z_i$ at each
step, scaled by its own $\sigma_i(t,T)$. Drawing independent noise per maturity
would be a different, arbitrage-inconsistent model.

### Error sources

Three, distinguishable by how they scale:

1. **Euler time stepping** — weak order $O(\Delta t)$.
2. **Trapezoidal integration along the maturity axis** — $O(h^2)$, where $h$ is
   the maturity grid spacing. Enters both the drift computation and the bond
   price integral.
3. **Trapezoidal accumulation of $B(t)$** along the time axis.

The trapezoidal rule is **exact** for constant integrands, so the constant
volatility case has no error from source 2 at all.

### Diagnosing errors by convergence rate

The practical technique used throughout this project: an error that shrinks at
its predicted rate under refinement is discretization; one that plateaus is a
bug. A halving of $\Delta t$ should halve an $O(\Delta t)$ error; a halving of $h$
should quarter an $O(h^2)$ error. A stubborn floor means something structural.

This diagnostic caught two real implementation errors during development, both
of which produced plausible-looking prices.

### Paired comparison

Many quantities of interest here are far below Monte Carlo noise at any practical
sample size. The drift's effect on the curve is around $10^{-5}$ against a
standard error near $4\times10^{-4}$ — a signal forty times smaller than the
noise it sits in.

Running two simulations over **identical Brownian shocks** and differencing path
by path cancels the diffusion *within* each path rather than averaging it away
across paths. In this project that bought a variance reduction factor of roughly
1500, making effects measurable at a few thousand paths that would otherwise
have needed hundreds of thousands.

For comparisons across different step counts, the shocks must be **block
aggregated** rather than merely seeded identically: a coarse increment is the sum
of the fine increments it contains, rescaled by $1/\sqrt{m}$ to preserve unit
variance. Reusing a seed across different step counts pairs nothing, because a
different step count consumes a differently-shaped draw and the runs follow
different paths.

---

## Reference

Heath, D., Jarrow, R., & Morton, A. (1992). Bond Pricing and the Term Structure
of Interest Rates: A New Methodology for Contingent Claims Valuation.
*Econometrica*, 60(1), 77–105.