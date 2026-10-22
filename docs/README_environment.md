# dcmg-gym — a data-centre microgrid RL environment

Simulation code accompanying the research design in [`docs/research_design.md`](docs/research_design.md).

A grid-connected microgrid (PV + battery + grid) supplies a data centre. One agent
jointly controls **IT workload deferral**, **CRAC supply-air setpoint**, and
**battery dispatch**, trading electricity cost against carbon, service level, and
the ASHRAE thermal envelope.

This fills a hole in the existing tooling: SustainDC models the data centre
(workload, cooling, battery) but explicitly has **no on-site PV, grid dispatch, or
electricity market**; pymgrid, CityLearn and Grid2Op model the electrical side but
have **no data-centre thermal or IT model**.

## Layout

```
docs/research_design.md   the science: gap, CMDP formulation, data plan, paper split
src/profiles.py           exogenous profiles (synthetic) + real-dataset loader stubs
src/env.py                DCMicrogridEnv — the Gymnasium environment
src/baselines.py          B0–B5 controllers (status quo … rollout MPC)
src/train.py              PPO / SAC training + held-out evaluation + MORL Pareto
src/plots.py              publication figures
src/verify.py             physical and accounting sanity checks — run this first
```

## Install and run

```bash
pip install gymnasium stable-baselines3 numpy pandas scipy matplotlib
cd src
python verify.py                                    # 40+ physical checks
python train.py --steps 300000 --sac-steps 120000 \
                --seeds 3 --test-episodes 30 --pareto --out ../results
python plots.py --results ../results --out ../figures
```

## The environment

**Control interval** 15 min. **Episode** 1 day (96 steps). Chronological
train/test split: 240 days train, 60 days held out.

**Action** — `Box(-1, 1, (3,))`

| idx | lever | mapped range |
|---|---|---|
| 0 | fraction of the deferrable backlog dispatched now | `[0, 1]` |
| 1 | battery power (+ discharge, − charge) | `±800 kW` |
| 2 | CRAC supply-air setpoint | `[18, 27] °C` |

**Observation** — time encodings, exogenous signals now (PV, dry/wet bulb, price,
carbon), noisy 1 h and 4 h forecasts, and internal state (SoC, room temperature,
utilisation, backlog, time-to-deadline, rolling peak). Optionally the 4-vector of
objective preference weights, for the preference-conditioned MORL policy.

**Objectives** — `cost`, `carbon`, `sla`, `thermal`, each normalised then combined
with weights `w`. `info["objectives"]` returns the raw vector.

**Constraints** — `info["cost_thermal"] / ["cost_sla"] / ["cost_pcc"]` are per-step
CMDP constraint costs, ready for a Lagrangian solver.

**Safety layer** (`DCConfig.safety_layer`, default `True`) projects actions onto the
feasible set before they are applied: the setpoint is capped so the one-step-ahead
room temperature stays inside the ASHRAE envelope, and battery charging is capped so
the point-of-common-coupling import limit is not breached. Turning it off is the
ablation arm that shows what an unconstrained agent does.

## Modelling notes

- **IT power** from a monotone interpolation through the measured
  `SPECpower_ssj2008` load points of a reference server (77.4 W idle → 347 W at
  100 %). The strong non-linearity near idle is why consolidation matters.
- **Thermal** lumped first-order RC, integrated exactly (`τ = C/k = 15 min`), sized
  so that at nameplate IT load the room-to-supply lift is ≈ 4.2 K. A 22 °C setpoint
  is just feasible; 26 °C is not *unless the agent first sheds IT power*. That
  coupling is the whole point of joint control.
- **Cooling** chiller COP rising with setpoint and falling with wet-bulb, plus an
  economiser mode. In the Bangkok base case free cooling is essentially never
  available — a finding, not a bug.
- **Battery** round-trip efficiency, SoC limits, and a throughput degradation cost.
- **Terminal storage value.** At episode end the agent is charged (or credited) for
  the change in stored energy at the episode-mean price. Without this a policy that
  simply drains the battery harvests ~640 kWh of free energy every episode, because
  `reset()` restores the SoC. `verify.py` checks this explicitly.

## Baselines

| id | controller |
|---|---|
| B0 | status quo — fixed 22 °C, no deferral, battery idle |
| B1 | greedy solar — run the backlog under sun, bank surplus, cover the night |
| B2 | price threshold on the trailing-24 h quantiles |
| B3 | carbon threshold, same logic |
| B4 | rule-based supply-air reset from wet-bulb |
| B5 | rollout MPC, 4 h horizon, noisy forecast, storage valued at the trailing-24 h mean price |
| R1/R2 | PPO / SAC |
| R4 | preference-conditioned MORL → a Pareto front from one policy |

B5 sees the **same noisy forecast** the RL agent sees, so the comparison is fair.
B6 (MILP oracle with perfect foresight) is the next thing to build — see the build
plan in the design doc.

## Swapping in real data

`profiles.py` has loader stubs for the verified dataset stack: Azure Public Dataset
V2 for workload, NSRDB-Himawari on AWS Open Data for irradiance (covers Thailand, no
API key), Open-Meteo for weather, and SustainCluster's standardised Singapore price
and carbon-intensity CSVs. Licences, access methods, citations and the time-zone /
resolution traps are documented in §4 of the design doc.

## Caveats

Synthetic profiles by default — the numbers below are a working demonstration of the
pipeline, not a publishable result. Publishable results need the real data (P3), the
MILP oracle (P4), and validation of IT-power realism against Google PowerData2019 or
the Acme IPMI traces (P8).
