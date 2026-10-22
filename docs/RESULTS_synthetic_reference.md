# Results — synthetic-data reference run

**Run:** 3 seeds, PPO 300 k steps / SAC 120 k steps, 30 held-out test days,
chronological split (240 d train / 60 d test), forecast error `σ = 0.15` on.
Bangkok-like tropical site, volatile (Singapore-like) price, hourly carbon.
Safety layer on. `numpy 2.4.4, pandas 3.0.2, scipy 1.17.1, torch 2.13`.

**These numbers are a working demonstration of the pipeline, not a publishable
result.** They come from synthetic profiles. Steps 6–8 of `RUNBOOK.md` replace them
with real data; the conclusions below are about *method*, not about magnitudes.

---

## Headline table

Paired against the status quo on the **same** held-out days (mean ± 95 % CI).

| controller | cost % | CO₂ % | cooling energy % | PUE | peak kW | max T_room | thermal violations | reward |
|---|---|---|---|---|---|---|---|---|
| B0 status-quo | — | — | — | 1.434 | 1068 | 26.1 | 0 | −0.6974 |
| B1 greedy-solar | **−1.68 ± 0.39** | **−7.27 ± 0.12** | **−8.00 ± 0.12** | 1.406 | 1191 | 26.8 | 0 | −0.6902 |
| B2 price-threshold | **−1.43 ± 0.56** | **−2.93 ± 0.86** | **−6.64 ± 0.37** | 1.406 | 1354 | 27.0 | 0 | −0.7070 |
| B3 carbon-threshold | +1.99 ± 1.12 | **−4.66 ± 1.30** | **−6.42 ± 0.47** | 1.409 | 1349 | 26.9 | 0 | −0.7063 |
| B4 cooling-reset | **−1.99 ± 0.12** | **−1.99 ± 0.12** | **−5.97 ± 0.31** | 1.408 | 1054 | 27.0 | 0 | −0.6835 |
| B5 MPC-4h | **−0.41 ± 0.22** | **−5.29 ± 0.21** | **−6.04 ± 0.35** | 1.408 | 1127 | 27.0 | 0 | −0.6854 |
| **R1 PPO** | **−1.50 ± 0.14** | **−5.58 ± 0.18** | **−8.01 ± 0.48** | **1.400** | **1041** | 27.0 | 0 | **−0.6772** |
| R2 SAC | +2.20 ± 0.57 | +1.95 ± 0.29 | **−6.49 ± 0.53** | 1.406 | 1392 | 27.0 | 0 | −0.7137 |

**Pairing matters.** With unpaired CIs across test days, day-to-day weather and price
variance swamps everything and *no* controller looks significantly different from B0.
Differencing against B0 on the same day shrinks the CIs by roughly an order of
magnitude and every effect becomes resolvable. Report paired differences; an unpaired
CI here is a statistical error, not merely a conservative choice.

---

## What the run actually shows

**1. PPO wins on the objective it optimises, and only narrowly.** R1 has the best
reward (−0.6772 vs −0.6835 for the best baseline), the best PUE (1.400), the lowest
peak (1041 kW), and the largest cooling-energy saving (−8.0 %). But on *cost alone*
it is beaten by B4, and on *carbon alone* by B1. **A single scalar reward hides this**
— which is the argument for the multi-objective framing, not a footnote to it.

**2. The margin over simple heuristics is small — this is the finding, not a
failure.** PPO's cost advantage over B2 (price threshold) is 0.07 pp, well inside the
noise. This is exactly what Sukprasert et al. (EuroSys '24) predict: simple threshold
policies capture most of the available benefit. The companion conference paper (§7 of
the design doc) should own this result rather than bury it, and map *where in the
configuration space* the margin becomes real.

**3. Naive heuristics can be actively harmful.** B3 (carbon threshold) *increases*
cost by 2.0 % — it charges the battery on low-carbon hours regardless of price and
drives the peak from 1068 to 1349 kW, and the demand charge eats the savings. B2 does
the same to a lesser degree. Any paper reporting carbon-only heuristics without a
demand charge in the objective is over-reporting.

**4. SAC is worse than the status quo on cost and carbon.** At 120 k steps it is
undertrained — SAC is ~15× slower per step in wall-clock here, so it got 40 % of
PPO's steps. **Report this as a compute-budget result, not an algorithm result**, or
give it equal steps before drawing any conclusion.

**5. The safety layer holds absolutely.** Zero thermal violations for every
controller, max room temperature exactly at the 27 °C ASHRAE bound — the projection
is binding, not slack. The controllers all learn to sit *on* the constraint, because
that is where cooling energy is minimised. This is the result that makes the CMDP
framing worth writing up: turn the layer off (`DCConfig(safety_layer=False)`) and the
violation rate becomes the headline metric.

**6. Free cooling is essentially unavailable in the tropics.** Wet-bulb p05 ≈ 20 °C
means the economiser almost never fires. Cooling savings therefore come entirely from
raising the setpoint against the thermal envelope, which is only possible when IT
power is low — i.e. **only via the workload lever**. That coupling is the physical
justification for joint control and it is climate-specific; a temperate site would
show a different mechanism. Run both (the experiment matrix in §6) and contrast them.

**7. The Pareto front is compressed — a negative result worth reporting.** Sweeping
`w_carbon` from 0 to 1 moves cost only over $3438.9–3447.9 and CO₂ over
7781.9–7804.8 kg: a 0.3 % cost span and a 0.3 % carbon span. The front dominates most
baselines but it barely spreads. Two candidate explanations to test before writing:
(a) with a *volatile* price whose troughs coincide with high solar, cost and carbon
are nearly co-linear here, so there is little genuine conflict to trade off;
(b) the preference-conditioned policy is under-trained at 450 k steps. **Test (a)
first** by re-running with the flat-carbon and the TOU-price regimes — if the front
opens up, the compression is a property of this scenario, which is itself a
publishable observation.

---

## Figures

| file | what it shows |
|---|---|
| `fig1_dispatch.png` | one held-out day under the trained PPO agent: IT + cooling stack with PV and grid import; offered vs executed utilisation with the deferred backlog; SoC against price and carbon; room temperature against the ASHRAE limit |
| `fig2_bars.png` | the paired table above, with significance |
| `fig3_pareto.png` | cost–carbon front from the single preference-conditioned policy, baselines as points |
| `fig4_tradeoff.png` | cooling energy and violation rate vs a fixed setpoint — the physical tension, controller-independent |

`fig4` is the one to check first when anything changes: cooling energy must fall
monotonically with setpoint while violations stay at zero and then rise sharply past
~22.5 °C. If that curve flattens, the thermal or COP parameters are wrong and the
control problem is trivial.

---

## Known limitations of this run

- **Synthetic profiles.** Real price series have fatter tails and real irradiance has
  persistent cloudy spells; both usually *increase* the value of flexibility. Expect
  the numbers to move once step 7 of the runbook is done.
- **No MILP oracle.** Without B6 there is no optimality gap, so "PPO is −1.5 %" has no
  denominator. This is the highest-value next build (P4).
- **SAC under-trained** (see 4).
- **MPC chatters.** The 4 h rollout MPC switches setpoint by ~1 K per step on a coarse
  action grid. Not an error, but a reviewer will ask; either refine the grid, add a
  switching penalty, or report the chatter explicitly.
- **Single site, single year, one workload archetype.** The experiment matrix in §6 of
  the design doc is not yet run.
- **3 seeds.** Use ≥ 10 before publishing.
- **Cross-machine reproducibility: confirmed.** The same run on macOS arm64 /
  Python 3.12 / pandas 3.0.5 reproduced this Linux x86-64 / Python 3.11 / pandas 3.0.2
  run to the printed precision — identical free-cooling fraction (0.0938), identical
  terminal storage cost (110.9), identical max room temperature (26.999 °C). Still
  record the version stack (printed at the top of `verify.py`) in the paper, since
  NumPy does not formally guarantee `Generator` stream stability across versions.
