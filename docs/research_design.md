# Safety-Constrained Multi-Objective Deep RL for a Data-Centre Microgrid

**Research design document — v1.0, 17 August 2026**
Project: *Microgrid RL and energy in manufacturing process optimization* (Publication 2)

---

## 0. What this document is

A complete, reviewable research design for a Q1–Q3 journal paper plus a companion energy/sustainability conference paper, on the theme:

> A grid-connected microgrid (PV + battery + grid) supplies a data centre. A reinforcement-learning agent jointly controls **IT workload deferral**, **cooling setpoints**, and **battery dispatch** to reduce electricity cost, carbon, and cooling energy / peak temperature — subject to hard thermal and service-level constraints.

Sections 1–4 fix the science (gap, formulation, models, data). Sections 5–7 fix the experiments and the paper split. Section 8 is the build plan. Section 9 lists the honest weaknesses a reviewer will attack.

---

## 1. The research gap

### 1.1 What is already done

Three literatures exist and barely talk to each other.

**(a) RL for data-centre cooling — mature, now deployed.** Zhan et al. (ICLR 2025, arXiv:2501.15085) run physics-informed offline RL with a T-symmetry-enforced GNN thermal model in a *live commercial data centre* for >2,000 hours across 11 ACUs: 14–21 % cooling-energy saving, zero safety violations. Meta reports a simulator-based offline-RL supply-airflow controller (−20 % fan energy). Wang et al. ("Safari", ACM TCPS 2023, doi:10.1145/3582577) do safe RL for CRAC control with 18–26.6 % savings and ~99 % fewer safety violations. **None of these control anything on the energy-supply side** — no PV, no battery, no grid dispatch, no price or carbon signal.

**(b) DRL for microgrid EMS — mature, but the load is generic.** Large literature on PV + BESS + grid dispatch (e.g. Wu et al., *Applied Energy* 406, 2026, doi:10.1016/j.apenergy.2025.127223). The flexible load is almost always an abstract deferrable demand, never a data centre with its own internal thermodynamics.

**(c) Carbon-aware workload shifting — mature, and partly debunked.** Google's Carbon-Intelligent Computing System (Radovanović et al., *IEEE Trans. Power Systems* 38(2), 2023) and Meta's Carbon Explorer (ASPLOS 2023, doi:10.1145/3575693.3575754) are the industrial reference points. **Crucially**, Sukprasert et al. (*EuroSys '24*, arXiv:2306.06502) show across 123 regions that (i) real carbon savings are far below the ideal, (ii) **simple scheduling policies capture most of the benefit and sophisticated ones add little**, and (iii) the advantage shrinks as grids decarbonise. Any RL paper in this space that does not engage with this result will be rejected.

### 1.2 The joint papers — and what they leave out

| Paper | Joint over | Missing |
|---|---|---|
| Ran et al., *DeepEE*, ICDCS 2019, doi:10.1109/ICDCS.2019.00070 | jobs + cooling | no PV/BESS/grid, no carbon, no safety guarantee |
| Wang & Nguyen, *EA*, AAAI-26 | deferrable jobs + active servers + AC/free-cooling mode, under solar | no BESS or grid dispatch, no tariff, no Pareto, no CMDP |
| Rafique et al., *Energies* 19(7):1736, 2026, doi:10.3390/en19071736 | thermal-aware jobs + BESS + PV/wind + grid peak | cooling is implicit (no HVAC actuation), no carbon signal, no safe RL |
| Zhang Y. et al., *Applied Energy* 402, 2026, doi:10.1016/j.apenergy.2025.126984 | cooling + waste-heat recovery + ATES | no IT workload, no PV/BESS/grid |
| Pan, Liu, Wei, Jing, Qiu — *Uncertainty-Aware RL for Joint Optimization of DC Operations and Electricity Market Bidding*, SSRN 5412618 → *Applied Energy* | **IT + thermal + on-site DER + market bidding** | **Closest competitor.** Appears to lack explicit Pareto/MORL, formal constraint satisfaction, and multi-site. ⚠️ *Retrieve and read this before writing related work.* |
| *Mixture-of-experts multi-critic DRL for sustainable management of data center microgrids*, *Applied Energy* 2026, PII S0306261926002138 | DC microgrids | ⚠️ **Title verified only; authors/volume/DOI not confirmed. Retrieve manually — this may be your nearest competitor.** |

### 1.3 The environment gap (verifiable in one click)

No open RL environment couples **DC internal thermodynamics + IT workload scheduling + a genuine microgrid**:

| Environment | Has | Lacks |
|---|---|---|
| **SustainDC** (HPE, NeurIPS 2024 D&B, arXiv:2408.07841) | workload scheduling, CRAC cooling, battery, CI traces | **explicitly no on-site PV, no grid dispatch, no electricity market** |
| **SustainCluster** (HPE, NeurIPS 2025 D&B) | geo-distributed multi-DC, 15-min, price + CI + weather for 20+ regions incl. Singapore | no single-site microgrid asset dispatch, HVAC is a proxy |
| **Sinergym** (*Energy and Buildings* 327:115075, 2025) | EnergyPlus DC building thermal | no IT job scheduling, no DER |
| **CityLearn v2** (BuildSys '23, doi:10.1145/3600100.3626257) | district buildings, PV, batteries | no data-centre model |
| **pymgrid** (arXiv:2011.08004) | microgrid PV/BESS/genset | frozen; no thermal or IT model |
| **Grid2Op** | transmission-level | no microgrid, no DC |

### 1.4 Positioning — the two papers

**Journal (Q1, target *Applied Energy*):**
> *Safety-constrained multi-objective deep reinforcement learning for joint IT-workload, cooling and PV–battery dispatch in a data-centre microgrid.*

Contributions: (1) a CMDP formulation whose constraints span three physical domains simultaneously — ASHRAE thermal envelope, job deadlines/SLA, and SoC/point-of-common-coupling limits; (2) a **preference-conditioned multi-objective** policy that returns a *Pareto front*, not a single scalarised point; (3) an MILP oracle upper bound; (4) a released environment as the artifact.

*Why it is publishable:* Kahil et al. (*Applied Energy* 389:125734, 2025), a PRISMA systematic review of 65 studies, names exactly these gaps — absence of real-time validation, no standard reporting metrics, and **"joint optimization of multi-system objectives"** as the top future direction. You are citing the review that asks for your paper.

**Conference (energy/sustainability, target ACM e-Energy or IEEE IGSC / Applied Energy Symposium):**
> *When is reinforcement learning actually worth it? A skeptical study of RL versus threshold heuristics for data-centre microgrids.*

Contribution: a well-powered ablation over PV penetration, battery sizing, tariff structure and carbon-signal volatility, identifying the **regime boundary** where RL beats a simple price/carbon threshold. Directly answers Sukprasert et al. Uses marginal (not average) emissions factors and reports confidence intervals over ≥10 seeds. This paper is the one that survives a hostile reviewer, and it is cheap once the environment exists.

---

## 2. Problem formulation

### 2.1 Constrained multi-objective MDP

Control interval Δt = **15 min** (matches SustainCluster, matches HVAC thermal inertia, matches typical grid data granularity). Episode = 1 day (96 steps) for training, 1 month rolling for evaluation.

**State** `s_t ∈ R^n`:

| Group | Variables |
|---|---|
| Time | sin/cos of hour-of-day, sin/cos of day-of-year, weekday flag |
| Exogenous now | PV available power `P_pv,t`, ambient dry-bulb `T_amb,t`, wet-bulb `T_wb,t`, price `π_t`, carbon intensity `c_t` |
| Exogenous forecast | `P_pv`, `π`, `c` over horizon H = 4 steps (1 h) and H = 16 (4 h), with injected forecast error |
| Internal | battery SoC, IT-room temperature `T_room,t`, current IT utilisation `u_t`, deferrable backlog `B_t`, time-to-deadline of oldest job, rolling peak demand `P̂_t` |
| Preference (MORL only) | weight vector `w = (w_cost, w_carbon, w_sla, w_therm)` on the simplex |

**Action** `a_t ∈ [−1,1]^3` (continuous, then affine-mapped):

1. `a_shift → φ_t ∈ [0,1]` — fraction of the deferrable backlog dispatched this interval.
2. `a_bess → P_bess,t ∈ [−P̄_ch, P̄_dis]` — battery charge (−) / discharge (+).
3. `a_T → T_sup,t ∈ [18, 27] °C` — CRAC supply-air setpoint.

*(Optional 4th action for the extended version: `a_cap` — GPU/CPU power cap ∈ [0.6, 1.0], the lever that Colangelo et al., Nature Energy 2026, doi:10.1038/s41560-025-01927-1 demonstrate physically on a 256-GPU cluster.)*

**Objective vector** `J = (J_cost, J_carbon, J_SLA, J_therm)`:

```
J_cost   = Σ_t Δt · [ π_t · max(P_grid,t, 0) − π_exp,t · max(−P_grid,t, 0) ] + λ_peak · max_t P_grid,t
J_carbon = Σ_t Δt · c_t · max(P_grid,t, 0)
J_SLA    = Σ_t ( deadline violations + queueing delay penalty )
J_therm  = Σ_t max(0, T_room,t − T_max)²          (soft) ,  plus a hard CMDP constraint
```

**Constraints** (this is the safe-RL contribution — three domains, one CMDP):

| Domain | Constraint | Source |
|---|---|---|
| Thermal | `T_room,t ≤ 27 °C` recommended, hard limit 32 °C (ASHRAE class A1 allowable 15–32 °C) | ASHRAE TC 9.9 Thermal Guidelines, 5th ed. |
| Service | every job completes within its deadline `d_j`; backlog bounded | SLA definition |
| Electrical | `SoC ∈ [0.1, 0.9]`; `P_grid,t ≤ P_PCC` (contracted capacity) | BESS spec + utility contract |

Solved as a CMDP with **Lagrangian PPO** (PPO-Lagrangian) and, as an ablation, **action projection / safety layer** so that thermal-envelope violations are structurally impossible. Report **constraint-violation rate**, not just reward — that is the number an operator cares about and the number no competing paper reports for the joint problem.

### 2.2 Multi-objective handling

Scalarise as `R = −(w_cost Ĵ_cost + w_carbon Ĵ_carbon + w_SLA Ĵ_SLA + w_therm Ĵ_therm)` with each term z-normalised by a baseline run, **and condition the policy on `w`** (append `w` to the observation, sample `w ~ Dirichlet(1)` each episode). One trained policy → a whole Pareto front at inference by sweeping `w`.

Report **hypervolume** and the Pareto front figure. This is the figure that sells the paper: a cost–carbon trade-off curve with the heuristic baselines plotted as single dominated points.

---

## 3. System models

### 3.1 IT power

`P_IT(u) = N · [ P_idle + (P_max − P_idle) · f(u) ]`, with `f` fitted to a published **SPECpower_ssj2008** result — free, public, gives measured watts at every 10 % load step. Verified example: Dell PowerEdge R7615 / EPYC 9654P — idle 77.4 W, 10 % 145 W, 50 % 231 W, 100 % 347 W. Cite the exact result ID; the strong non-linearity near idle is what makes consolidation worth modelling.

Workload split: fraction `α` **interactive** (must run now, follows the trace) and `1−α` **deferrable/batch** (has a deadline `d`, can be queued). Sweep `α ∈ {0.5, 0.7, 0.9}` — flexibility fraction is the single most important sensitivity, and Sukprasert et al. imply the result is mostly a function of it.

### 3.2 Thermal model (lumped RC, first-order)

```
C_room · dT_room/dt = P_IT + P_aux − ṁ · c_p · (T_room − T_sup)
```

Discretised at 15 min. `ṁ` from CRAC fan law. Calibrate `C_room`, `ṁ` so the time constant is ~10–20 min (physically standard for a server room). **Validate** the resulting IT-power magnitude and diurnal shape against **Google PowerData2019** (real measured power across 57 power domains, CC-BY 4.0, ASPLOS '20) and/or **Acme** IPMI power at 15 s (CC-BY 4.0, NSDI '24). This validation step is what separates a credible paper from a toy.

### 3.3 Cooling power

Chiller with temperature-dependent COP plus a free-cooling/economiser mode:

```
COP(T_sup, T_wb) = COP_nom · g(T_sup − T_wb)      # rises as setpoint rises and wet-bulb falls
P_cool = P_IT / COP + P_fan(ṁ)                     # P_fan ∝ ṁ³ (fan affinity law)
free cooling active when T_wb < T_sup − ΔT_ec
```

This creates the genuine tension the RL agent must resolve: **raising `T_sup` cuts cooling power but pushes `T_room` toward the ASHRAE limit**, and the safe headroom depends on IT power, which the agent also controls via workload deferral. In a tropical climate (Bangkok, wet-bulb often >25 °C) free cooling is rarely available — that is a *finding*, not a flaw, and it is the Thailand-specific angle.

### 3.4 Microgrid

```
P_pv,t   = η · A · G_t · (1 − β(T_cell − 25))      # simple PV model
SoC_{t+1} = SoC_t + Δt·(η_ch·P_ch − P_dis/η_dis)/E_cap
P_grid,t = P_IT,t + P_cool,t − P_pv,t + P_ch,t − P_dis,t
```

Battery degradation as a per-throughput cost term (keeps the agent from cycling pathologically — reviewers check for this).

---

## 4. Data plan

All sources verified accessible on **17 Aug 2026**. Every element is CC-BY / MIT / US-federal — no paywall, no sales conversation.

| Layer | Primary | Why | Fallback |
|---|---|---|---|
| **Workload** | **Azure Public Dataset V2** (2019 VM trace) — 5-min CPU util (min/max/avg), 30 continuous days, 2.7 M VMs, **CC-BY 4.0**, direct GitHub-Releases download. Cite Cortez et al., SOSP '17 | Only trace with a clean, regular, calendar-anchorable utilisation time series and an unambiguous licence | Alibaba cluster-trace-v2018 `machine_usage.csv` (1.7 GB; survey-gated, no LICENSE file) |
| **AI overlay** | **Azure LLM Inference Trace 2024** (1 week, CC-BY 4.0, cite Stojkovic et al., HPCA '25) | Tiny CSV; gives the topical AI-data-centre scenario at near-zero cost | **Acme** (InternLM, NSDI '24) — 15-s utilisation **and measured IPMI watts**, CC-BY 4.0 |
| **Util→power** | **SPECpower_ssj2008** published results (free, public, per-10 %-load watts) | Measured, citable, no assumptions | — |
| **Solar** | **NSRDB Himawari via AWS Open Data**, `s3://nrel-pds-nsrdb/himawari/` — 4 km, 10-min, 2015+, **CC BY 3.0 US**, `aws s3 --no-sign-request`, **no key** | **Confirmed to cover Thailand.** No registration at all | NREL NSRDB REST API (free key, 10-min, 2 km) |
| **Weather** | **Open-Meteo Historical API** (ERA5-Land, hourly, global, no key) — dry-bulb + humidity → wet-bulb | Free, global, reproducible | NSRDB bundled met variables |
| **Price** | **Singapore USEP hourly 2023** from the SustainCluster repo (`data/electricity_prices/standardized/SG/2023/`, verified 8,760 rows, MIT) — **plus** a synthesised **MEA/PEA TOU** series as scenario 2 | SG = real tropical wholesale market at UTC+8; TH TOU = the policy-relevant case | ENTSO-E (CC-BY 4.0, free token) or CAISO OASIS (no registration) |
| **Carbon** | **Singapore hourly CI** from SustainCluster (2021–2024); **Ember** (CC-BY 4.0) for the Thailand annual/monthly factor and magnitude sanity-check | Avoids the Electricity Maps paywall while keeping an hourly signal | UK Carbon Intensity API (30-min, no key, CC-BY 4.0) — the lowest-friction carbon dataset that exists |
| **Validation** | **Google PowerData2019** (CC-BY 4.0) and/or **Acme** IPMI power | Real measured DC power to validate magnitude/variance/shape | — |
| **Baseline env** | **SustainDC** (MIT + CC-BY-NC-4.0 portions) and **SustainCluster** (clean MIT) | Peer-reviewed NeurIPS D&B artifacts reviewers accept | — |

### 4.1 The Thailand problem — stated honestly

**There is no public time-varying electricity-price dataset for Thailand.** MEA and PEA publish a *tariff schedule*, not a time series: on-peak Mon–Fri 09:00–22:00, off-peak 22:00–09:00 + weekends + public holidays; indicative 2026 medium/large business energy charge ≈ 4.10 THB/kWh on-peak vs 2.58 off-peak, plus an Ft surcharge ≈ 0.16 THB/kWh revised quarterly by the ERC.

Consequence: a Thai price series must be *synthesised deterministically* from the published schedule + the Thai public-holiday calendar + quarterly Ft. Defensible and reproducible if you cite the tariff announcements with an access date — **but it is a two-block step function with no intra-day volatility**, which materially weakens any "RL learns to arbitrage price" narrative.

**Fix:** run the primary experiment on **Singapore USEP** (real half-hourly market, same UTC+8, tropical) and the **Thai TOU tariff as a second scenario**. You get a real-market result *and* a Thailand policy result. Same for carbon: use SG hourly CI as an explicitly-labelled regional proxy; do not silently upsample Ember's annual Thai factor into an hourly signal.

### 4.2 Data gotchas that will silently break the simulation

- **Alibaba GPU traces have falsified calendar dates** (time-of-day and day-of-week preserved, UTC+8). You *cannot* join them to real solar/price by date; you must map onto a reference year. GPU-v2026 has no dates at all (`day=0..184`).
- **Google ClusterData2011-2 is US Eastern**; Acme mixes `+08:00` and `+00:00` *within one dataset*.
- **Alibaba v2018 encodes invalid values as −1 or 101** in [0,100]-normalised fields.
- **Alibaba GPU-v2020 `machine_metric` / `pai_sensor_table` are per-instance-lifetime averages, not fixed-interval samples.** Treating them as a time series is a silent, serious error.
- **Electricity Maps' free data portal has moved** — `portal.electricitymaps.com/datasets` now redirects to the live map. Older papers cite URLs that no longer serve data. Upstream licence is **ODbL** (share-alike), stricter than CC-BY.
- **PJM prohibits redistribution of raw data without membership** — do not use it in a reproducibility artifact.
- **SustainDC is effectively non-commercial** (CC-BY-NC-4.0 portions from Carbon Explorer). SustainCluster is cleanly MIT — prefer it if any industry partner is involved.
- **Pick one control interval (15 min) and document every resampling rule.** Never upsample carbon or price and present it as measured.
- **PVGIS coverage of Thailand is doubtful** (JRC describes SARAH-3 as "Europe and surrounding regions"). Test one call before relying on it; NSRDB-Himawari is the safe choice.

---

## 5. Baselines

A paper without a strong baseline set will be rejected, especially given Sukprasert et al.

| # | Baseline | Purpose |
|---|---|---|
| B0 | **Status quo**: fixed CRAC setpoint 22 °C, no deferral, battery idle | Reference for all % savings |
| B1 | **Greedy solar**: run backlog whenever PV surplus > 0; charge battery on surplus | The obvious industrial heuristic |
| B2 | **Price threshold**: defer when `π_t` > rolling median; discharge battery on-peak | The heuristic Sukprasert et al. say already captures most of the benefit |
| B3 | **Carbon threshold**: same on `c_t` | " |
| B4 | **Rule-based cooling**: setpoint reset schedule from ambient (standard practice) | Isolates the cooling contribution |
| B5 | **MPC** with perfect 4-h forecast | Realistic strong controller |
| B6 | **MILP oracle** with perfect foresight over the whole episode (Gurobi/HiGHS; cf. Hao et al., *Front. Energy Res.* 12, 2024, doi:10.3389/fenrg.2024.1344837) | **Upper bound.** Report optimality gap. SustainDC's own paper lists the absence of an oracle as a limitation — supplying one is a contribution |
| R1 | **PPO** (scalarised) | Standard DRL |
| R2 | **SAC** (scalarised) | Standard DRL, continuous |
| R3 | **PPO-Lagrangian** (CMDP) | The safety contribution |
| R4 | **Preference-conditioned MORL** | The multi-objective contribution |

---

## 6. Metrics and experimental protocol

**Primary:** total electricity cost (THB or USD), total operational CO₂ (kg), cooling energy (kWh), PUE, peak grid demand (kW), self-consumption ratio, SLA violation rate, **thermal constraint-violation rate and worst-case excursion**.

**Statistical hygiene — non-negotiable:**

- ≥ **10 random seeds** per configuration; report **mean ± 95 % CI**, not best run.
- **Chronological train/test split** (e.g. train on months 1–8, test on 9–12). Never random-split a time series.
- **Held-out year or held-out site** for generalisation.
- **Forecast error injected** into the observation (perfect-foresight RL is the single most common reviewer complaint in this field).
- Report **inference latency** per decision — relevant to Kahil et al.'s "absence of real-time validation" gap.
- Multi-objective: **hypervolume** + Pareto front, with baselines plotted as dominated points.

**How long must the evaluation period be?** Kang, Guo, Wang, Li, Zheng, Tao, Chen & Yan, *Quantification of building demand uncertainty on the operational performance evaluation and capacity optimizations in building microgrid system design*, **Renewable Energy 275:126296 (2026), doi:10.1016/j.renene.2026.126296** (Tsinghua BERC) answer exactly this question for building microgrids, and the answer transfers:

- The simulation length required for a stable estimate of microgrid operating performance **grows exponentially with demand uncertainty** — a handful of "typical days" is not enough, which is what most of the DC-RL papers in §1.2 actually use.
- Optimal component sizing moves with uncertainty too: **+1 % demand uncertainty → −0.0260 % optimal PV capacity and −0.0587 % optimal battery capacity.**

Two consequences for this design. (i) Justify the 60-day held-out test period explicitly by a convergence check — plot the running mean of cost and carbon against evaluation length and show where it settles — rather than asserting it. (ii) The PV/battery sizing sweep in the matrix below should be read as *conditional on the assumed demand uncertainty*; report the uncertainty level of the workload trace you use, and cite Kang et al. for why the optimum shifts. This is a cheap, citable rebuttal to "why this test length?" and "why these component sizes?", and it connects the paper to the building-microgrid design literature rather than only the DC literature.

**Experiment matrix (drives the conference paper):**

| Factor | Levels |
|---|---|
| Flexible workload fraction `1−α` | 10 %, 30 %, 50 % |
| PV penetration (PV peak / DC peak) | 0 %, 30 %, 60 %, 100 % |
| Battery energy (h of DC peak) | 0, 1 h, 2 h, 4 h |
| Price regime | SG USEP (volatile) / TH TOU (two-block) / flat |
| Carbon signal | SG hourly / flat annual factor |
| Climate | Bangkok (tropical, no free cooling) / a temperate site |

The headline conference result is the **regime boundary**: the region of this grid where RL's margin over B2/B3 exceeds the seed noise. Expect it to be narrow when the tariff is flat and flexibility is low — say so.

---

## 7. Paper split

**Paper A — journal (Applied Energy / Energy / IEEE TSG).**
Sections: intro & gap (cite Kahil et al. 2025 SLR) → CMDP formulation → system models → environment (artifact) → PPO-Lagrangian + preference-conditioned MORL → results vs B0–B6 incl. MILP oracle → Pareto fronts → constraint-violation analysis → Bangkok vs temperate case study → limitations.
Headline claims to aim for: X % cost and Y % carbon reduction vs B0 at **zero** thermal violations, with an explicit optimality gap to the MILP oracle, and a Pareto front dominating all heuristics.

**Paper B — conference (ACM e-Energy / IEEE IGSC / Applied Energy Symposium ICAE).**
*When is RL worth it?* The full experiment matrix, marginal emission factors, ≥10 seeds with CIs, an explicit reply to Sukprasert et al., and a decision rule for practitioners: *under these conditions use a threshold policy; only beyond this boundary is RL justified.* Deliberately includes negative results — which is why it is credible.

**Artifact.** Release the environment (`dcmg-gym`) under MIT with the SustainCluster-derived data pipeline. Consider a NeurIPS Climate Change AI workshop paper for the environment itself — both pymgrid and Green-DCC debuted that way.

**Relation to Publication 1 (manufacturing).** The MDP skeleton, the safe-RL machinery and the microgrid model are shared. The data centre is a second *flexible load archetype*: it differs from a factory in having (i) continuous rather than discrete deferability, (ii) a fast internal thermal loop coupled to the electrical load, and (iii) an exogenous carbon signal worth optimising against. State this explicitly — it makes the two publications a coherent programme rather than salami slicing.

---

## 8. Build plan

| Phase | Deliverable | Notes |
|---|---|---|
| P1 | `dcmg-gym` Gymnasium env: microgrid + IT + cooling + thermal, synthetic profiles | ✅ *implemented alongside this document* |
| P2 | Baselines B0–B5 + PPO/SAC training loop + plots | ✅ *implemented* |
| P3 | Swap synthetic profiles for real data (Azure V2, NSRDB Himawari, SG USEP + CI) | loader stubs provided |
| P4 | MILP oracle (B6) in HiGHS/Gurobi; report optimality gap | needs a linearised cooling model |
| P5 | PPO-Lagrangian (CMDP) + safety layer ablation | the journal contribution |
| P6 | Preference-conditioned MORL + Pareto/hypervolume | the second journal contribution |
| P7 | Experiment matrix, 10 seeds, CIs | the conference paper |
| P8 | Validate IT-power realism against Google PowerData2019 / Acme | credibility |

---

## 9. Weaknesses a reviewer will attack — plan the rebuttal now

1. **"Simple heuristics already do this"** (Sukprasert et al.). *Rebuttal:* Paper B is exactly this study; Paper A reports the margin over B2/B3 with CIs and does not overclaim.
2. **"Simulation only."** *Rebuttal:* validate IT power against real measured traces (PowerData2019 / Acme), inject forecast error, report inference latency, and cite Zhan et al. (ICLR 2025) on why sim-to-real is hard — then be explicit that hardware validation is future work. Do not claim deployment you do not have.
3. **"Your thermal model is a single RC node."** *Rebuttal:* justify by control-relevant time constant, show sensitivity to `C_room` and `ṁ`, and cite Safari's EnergyPlus+OpenFOAM testbed as the higher-fidelity path. Consider a Sinergym/EnergyPlus cross-check for one scenario — that single experiment buys a lot of credibility.
4. **"Thailand data are synthesised."** *Rebuttal:* the SG-primary / TH-scenario design in §4.1, stated up front rather than buried.
5. **"Reward weights are arbitrary."** *Rebuttal:* that is precisely why the paper is multi-objective and reports a Pareto front.
6. **"Where is the oracle?"** *Rebuttal:* B6 MILP with perfect foresight, and report the gap.
7. **Carbon accounting.** Use **marginal** emission factors where available, state the average-vs-marginal choice explicitly, and do not claim embodied carbon you have not modelled.

---

## 10. Citations to verify before writing related work

⚠️ The following could not be fully verified and **must be checked in a library database before citing**:

1. *Mixture-of-experts based multi-critic DRL for sustainable management of data center microgrids*, *Applied Energy* 2026 (PII S0306261926002138) — authors, volume, DOI unknown. **Possibly your closest competitor.**
2. Pan et al., SSRN 5412618 → *Applied Energy* (PII S0306261926008603) — volume/DOI unverified.
3. Author lists for the ACM e-Energy 2020 joint IT+cooling MADRL paper (doi:10.1145/3396851.3402658) and the e-Energy 2023 Lyapunov safe-RL paper (doi:10.1145/3575813.3597343).
4. DOI string for Radovanović et al., *IEEE TPWRS* 38(2), 2023 (journal/volume/issue verified from the PDF; DOI not).
5. "Joint IT-Facility Optimization for Green Data Centers via Deep RL" (IEEE doc 9427293) — journal/volume/DOI unverified.
6. Nakabi & Toivanen, *SEGAN* — volume/DOI unverified.
7. Meta's "Simulator-Based RL for Data Center Cooling Optimization" — no venue/year on the OpenReview PDF; treat as a tech report.
8. Luo et al. 2022 (DeepMind/Trane, arXiv:2211.07357) — confirm NeurIPS main track vs workshop.

Also note: **"Sustainable Data Center Gym" and a maintained "GreenDC" simulator do not appear to exist** — the former is a conflation of SustainDC and SustainGym's `DatacenterEnv`; GREENDC is an EU H2020 project (grant 734273), not an open RL codebase.
