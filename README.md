# RL-microgrid-dc

Simulation environment, data-preparation pipeline, trained policies and analysis
scripts for the manuscript

> Yanbenjawong V., Tseng S.-H., et al. *Safe reinforcement learning for data center
> microgrids: when does a learned policy beat a rule?* (submitted to Operations
> Research Perspectives, 2026).

A grid-connected microgrid (PV, battery, grid connection) supplies a data center.
One controller jointly decides, every 15 minutes, how much deferrable IT work to
release, how much battery power to charge or discharge, and the cooling supply-air
setpoint. A safety layer projects every command onto the thermal and
grid-connection limits before it reaches the plant. The paper compares a
safe PPO policy against seven reference strategies (B0 to B6) and a
perfect-foresight MILP benchmark under three real tariffs: Singapore spot
(USEP 2023), Thailand time-of-use (MEA/PEA Rate 4.2, LANTA-calibrated load) and
California spot (CAISO 2023).

The companion conference paper (IEEE PES ISGT Asia 2026, paper 6299) uses the
uncalibrated SPECpower server curve and the Singapore scenario only; its runs are
`results_real/` and `results_slafloor/`.

## Layout

```
src/                 environment, controllers, training, benchmark and analysis
  env.py             DCMicrogridEnv (Gymnasium), DCConfig, calibrate()
  profiles.py        exogenous profiles and loaders (workload, PV, weather, price, carbon)
  baselines.py       reference strategies B0 to B6
  train.py           PPO training, held-out evaluation, paired statistics
  run_baselines.py   evaluate the rules alone on a profile set
  oracle3.py         perfect-foresight MILP benchmark (PuLP/CBC) with simulator replay
  oracle.py, oracle2.py   earlier benchmark versions kept for provenance
  reeval.py          re-score saved policies under a changed cost accounting
  sweep.py           reward-weight sweep (Pareto)
  fig_regime.py      regime map and daily-cost figures
  plots.py           figure helpers
  verify.py          physical and accounting sanity checks (run first)
  data_prep.py       builds data/profiles_*.csv from the raw sources
  make_thai_profiles.py   Thai tariff and LANTA utilisation profile
data/                processed profiles with provenance notes, small raw inputs
docs/                research design, environment README, regime table, figures
paper/               figure scripts and PNGs used in the journal manuscript
results_*/           run outputs: config.json, results_raw.csv, summary.csv, policies R1_s*.zip
logs/                training and benchmark logs
```

## Install

```bash
conda create -n dcmg python=3.12 -y
conda activate dcmg
pip install -r requirements.txt
cd src && python verify.py
```

See `RUNBOOK.md` for the step-by-step procedure, including the manual data
downloads.

## Reproducing the journal results

All journal numbers use one accounting: LANTA-calibrated power curve
(`p_idle_w=180.44`), dense demand-charge signal, terminal battery energy and
terminal backlog valued, safety layer on, soft SLA (`sla_floor=false`), five
seeds, 300k steps per seed, 30 held-out days paired with B0.

```bash
cd src
# Thailand (MEA/PEA Rate 4.2, demand charge 3.91 USD/kW-month)
python train.py --profiles ../data/profiles_th.csv   --seeds 5 --steps 300000 --skip-sac \
       --idle-frac 0.52 --peak-charge 3.91 --dense-peak --out ../results_th_final
# Singapore (USEP 2023, demand charge 8.0 USD/kW-month)
python train.py --profiles ../data/profiles_real.csv --seeds 5 --steps 300000 --skip-sac \
       --idle-frac 0.52 --peak-charge 8.0  --dense-peak --out ../results_sg_final
# California: policies from results_ca_dp re-scored under the final accounting
python reeval.py --profiles ../data/profiles_ca.csv --models ../results_ca_dp \
       --idle-frac 0.52 --peak-charge 8.0 --dense-peak --out ../results_ca_dp_v2
# Perfect-foresight benchmark, each scenario
python oracle3.py --profiles ../data/profiles_th.csv   --idle-frac 0.52 --peak-charge 3.91 --out ../results_oracle3_th
python oracle3.py --profiles ../data/profiles_real.csv --idle-frac 0.52 --peak-charge 8.0  --out ../results_oracle3_sg
python oracle3.py --profiles ../data/profiles_ca.csv   --idle-frac 0.52 --peak-charge 8.0  --out ../results_oracle3_ca
# Regime map and daily-cost figures
python fig_regime.py --out ../docs
```

Exact flag names are documented in each script's `--help`; the `config.json` in
every results folder records the settings that produced it.

### Which folder is which

| Folder | Scenario | Power curve | Seeds | Used for |
|---|---|---|---|---|
| `results_th_final` | Thailand TOU | LANTA-calibrated | 5 | Journal Tables 2 to 4, Figs. 3, 5, 6, 8 |
| `results_sg_final` | Singapore spot | LANTA-calibrated | 5 | Journal Tables 2 to 4, Figs. 3, 5, 6, 8 |
| `results_ca_dp_v2` | California spot | LANTA-calibrated | 5 | Journal Tables 2 to 4, Fig. 8 (re-scored from `results_ca_dp`) |
| `results_oracle3_{th,sg,ca}` | all three | LANTA-calibrated | n/a | Perfect-foresight benchmark, simulator-verified |
| `results_th_dp`, `results_sg_cal_dp`, `results_ca_dp` | all three | LANTA-calibrated | 5 | Same objective without terminal backlog valuation (robustness check) |
| `results_th`, `results_sg_cal` | TH, SG | LANTA-calibrated | 3 | Early calibrated runs without the dense demand-charge signal |
| `results_real` | Singapore spot | SPECpower reference | 10 (PPO and SAC) | Conference paper; journal Section 4.2 (reference curve) |
| `results_slafloor` | Singapore spot | SPECpower reference | 5 | SLA-floor variant (conference paper) |
| `results_ca` | California spot | SPECpower reference | 5 | Reference-curve comparison |
| `results_reference`, `results` | synthetic profiles | SPECpower reference | 3 | Pipeline demonstration (`docs/RESULTS_synthetic_reference.md`) |
| `results_oracle`, `results_oracle_ca` | SG, CA | SPECpower reference | n/a | Earlier benchmark versions |
| `results_smoke` | synthetic | SPECpower reference | 1 | 20k-step smoke test |
| `test_oracle` | SG day 0 | LANTA-calibrated | n/a | B0 reproduction check for the MILP replay |

The safety-layer ablation (journal Section 4.5, `results_j1_nosafety`) was run on
a separate machine and will be added in a later commit.

### Result files

- `results_raw.csv`: one row per (controller, seed, held-out day) with cost,
  carbon, peak kW, SLA misses, thermal violations and the reward components.
- `summary.csv`: paired differences against B0 with 95 % confidence intervals.
- `R1_s<k>.zip`: the trained Stable-Baselines3 PPO policy for seed k; load with
  `PPO.load("R1_s0.zip")`.
- `oracle3_paired.csv`: per-day MILP cost, replayed simulator cost, and the B0
  and B6 costs on the same day.

## Data

Processed profiles are in `data/` with a provenance file next to each one.
Raw sources:

- Workload: Azure Public Dataset V2 (`vmtable`), reconstructed to 15-minute
  utilisation; the 7.7 GB raw archive is not in the repository (see
  `data/profiles_real_provenance.txt` and `src/data_prep.py`).
- Thai utilisation: LANTA (ThaiSC) daily means, see `src/make_thai_profiles.py`.
- Irradiance and weather: NSRDB (Himawari) Bangkok 2020, calendar-transplanted
  to 2023. The hourly CSV is `data/raw/solar/nsrdb_bangkok_2020.csv`; download
  instructions for other sites are in `RUNBOOK.md` step 6.
- Price and carbon: Singapore USEP and grid emission factor 2023, CAISO price
  and carbon 2023 (`data/raw/`, `data/raw_ca/`); Thai tariff from
  `data/EN_Electricity_Tariffs_May_2023.pdf`.
- Server power: SPECpower_ssj2008 result for Dell PowerEdge R7615
  (`data/raw/spec/specpower_result.txt`).

## Citation

See `CITATION.cff`. Please cite the journal manuscript once it is published;
until then cite this repository and the ISGT Asia 2026 paper.

## License

Code is released under the MIT License (`LICENSE`). Third-party data keep
their own licences, listed in the provenance files.
