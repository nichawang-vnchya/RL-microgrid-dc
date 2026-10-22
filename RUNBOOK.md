# RUNBOOK — running the microgrid × RL data-centre simulation end to end

Every step, in order, with the exact command, what you should see, and how to tell
it worked. Steps 1–5 run on synthetic data and need no downloads. Step 6 is the
manual data download. Steps 7–10 are the research extensions.

Nothing here needs a GPU. The whole synthetic pipeline runs on a laptop.

---

## Step 0 — Install

> **Use conda with Python 3.12. Do not build a venv from your system `python3`.**
> On many machines `python3` is now **3.14**, which has no working NumPy or PyTorch
> wheels; a venv built from it fails at import with
> `symbol not found in flat namespace '_PyUnstable_Object_IsUniqueReferencedTemporary'`.
> Conda pins the interpreter version explicitly, so it cannot go wrong this way.

```bash
conda create -n dcmg python=3.12 -y
conda activate dcmg
pip install gymnasium stable-baselines3 numpy pandas scipy matplotlib
```

**Check — this must pass before you go any further:**

```bash
python -c "import sys; print(sys.version)"    # must start with 3.11, 3.12 or 3.13
python -c "import gymnasium, stable_baselines3, torch; print(gymnasium.__version__, stable_baselines3.__version__, torch.__version__)"
```

Tested with gymnasium 1.3.0, stable-baselines3 2.9.0, torch 2.13, numpy 2.4, pandas 3.0.
Any recent gymnasium ≥ 0.29 and SB3 ≥ 2.0 should work.

**Environment activations stack.** If your prompt shows two names —
`(.venv) (dcmg)` — the leftmost one wins and it may be the broken one. Run
`deactivate` until only `(dcmg)` remains, then re-run the check above. If you already
created a bad venv, delete it: `rm -rf "$DCMG/src/.venv"`.

Every shell you open needs exactly two lines before you run anything:

```bash
conda activate dcmg
cd "$DCMG/src"
```

*(No conda? Install Miniforge, or use `pyenv install 3.12` — but do not use a system
Python 3.14 venv. There is no workaround; the wheels do not exist yet.)*

---

## Step 1 — Set one path variable, then never use chained relative paths

Unzip `dcmg.zip` wherever you keep the project. The layout inside is:

```text
dcmg/
├── docs/research_design.md     read this before running anything
├── RESULTS.md                  reference run to compare your numbers against
├── src/                        all Python commands run from here
├── figures/
├── results_reference/
└── data/raw/                   downloaded datasets go here in step 6
```

> That block is a **picture of the folder layout**, not something to type.

Set `DCMG` to the folder that *contains* `src/`, and every command in this runbook
will work regardless of where your shell happens to be:

```bash
export DCMG="$HOME/Documents/Academic_Paper_2_Conf/dcmg"     # adjust to your path
cd "$DCMG/src"
pwd                                                          # must end in /dcmg/src
ls                                                           # env.py baselines.py train.py verify.py plots.py profiles.py data_prep.py
```

Add the `export` line to your `~/.zshrc` (or `~/.bashrc`) so it survives new terminals.
If your project path contains spaces, `&`, or a trailing `.`, **always keep the quotes**
— and consider renaming it, since `&` is a shell metacharacter and breaks some tooling.

Read `docs/research_design.md` before running anything: §2 is the MDP, §3 the physical
models, §4 the data plan, §5 the baselines, §6 the evaluation protocol. If you change
the code and not the document, the paper drifts out of sync with the experiments.

---

## Step 2 — Verify the physics before trusting any number

```bash
python verify.py
```

Runs ~45 checks: power balance closes to machine precision, SoC integrates
consistently, work is conserved (offered = executed + queued + missed), the room
temperature stays between its start value and its steady state, COP rises with
setpoint, PUE is physically plausible, cost is reproducible from price × import,
the same seed reproduces the same trajectory, and cost is monotonically
non-increasing in PV capacity.

**Expected last line:** `ALL CHECKS PASSED`

If anything fails, stop. A failing check here means every downstream number is
wrong. This is also the file to extend whenever you add a component — one check per
claim you would have to defend to a reviewer.

---

## Step 3 — Run the baselines only (≈ 1 minute)

```bash
python -c "
import pandas as pd, warnings; warnings.filterwarnings('ignore')
from env import DCMicrogridEnv, DCConfig, calibrate
from baselines import REGISTRY, run_controller
from train import build_profiles
tr, te = build_profiles(seed=1)
cfg = calibrate(DCConfig(), profiles=tr, n_episodes=10)
env = DCMicrogridEnv(cfg=cfg, profiles=te, seed=0, eval_mode=True)
rows = []
for k, C in REGISTRY.items():
    r = pd.DataFrame(run_controller(C, env, n_episodes=20)).mean(numeric_only=True)
    r['ctrl'] = k; rows.append(r)
t = pd.DataFrame(rows).set_index('ctrl')
print(t[['cost','carbon_kg','energy_cool_kwh','peak_kw','t_room_max','reward']].round(2))
"
```

**What you should see:** B0 (status quo) is the most expensive; B1–B5 come in 1–8 %
below it on cost, carbon and cooling energy; `t_room_max` never exceeds 27 °C
(the safety layer holds); `sla_missed` is 0.

**Sanity rule:** if any controller beats B0 by more than ~20 % on cost, you have a
bug or an accidentally free resource — check the terminal storage value and the
demand charge first.

---

## Step 4 — Train the RL agents

Quick smoke run first (≈ 3 minutes) to prove the loop works:

```bash
cd "$DCMG/src"
python train.py --steps 20000 --sac-steps 10000 --seeds 1 --test-episodes 10 \
                --out "$DCMG/results_smoke"
```

Then the real run (≈ 1.5–2 hours on 4 CPU cores):

```bash
python train.py --steps 300000 --sac-steps 120000 --seeds 3 --test-episodes 30 \
                --pareto --out "$DCMG/results"
```

| flag | meaning |
|---|---|
| `--steps` | PPO training steps per seed |
| `--sac-steps` | SAC steps per seed. SAC is far more sample-efficient but ~15× slower in wall-clock, so give it fewer steps |
| `--seeds` | number of random seeds. **Use ≥ 5 for anything you publish**; 3 is the minimum for a working run |
| `--test-episodes` | held-out evaluation days |
| `--pareto` | also train the preference-conditioned MORL policy and sweep the weight vector |

**Outputs in `$DCMG/results/`:** `results_raw.csv` (one row per controller × seed ×
episode), `summary.csv` (aggregated with 95 % CIs and % change vs B0),
`pareto_raw.csv`, `config.json`, and the saved model zips.

**Run it in the background and watch it:**

```bash
cd "$DCMG/src"
nohup python train.py --steps 300000 --sac-steps 120000 --seeds 3 \
      --test-episodes 30 --pareto --out "$DCMG/results" > "$DCMG/train.log" 2>&1 &
tail -f "$DCMG/train.log"
```

**Expected:** PPO should land at or below the best baseline on *reward* (the thing
it optimises), typically ~2 % on cost and ~6 % on carbon versus B0. SAC usually
needs more steps than PPO here; if SAC is clearly worse, that is a compute budget
result, not an algorithm result — say so, or give it more steps.

---

## Step 5 — Figures

```bash
python plots.py --results "$DCMG/results" --out "$DCMG/figures"
```

| file | what it shows |
|---|---|
| `fig1_dispatch.png` | one held-out day: IT + cooling stack, PV, grid import; offered vs executed utilisation and the deferred backlog; SoC against price and carbon; room temperature against the ASHRAE limit |
| `fig2_bars.png` | cost / carbon / cooling energy vs the status quo, **paired on the same test days**, mean ± 95 % CI, non-significant bars faded and marked n.s. |
| `fig3_pareto.png` | cost–carbon Pareto front from the single preference-conditioned policy, with the baselines plotted as dominated points |
| `fig4_tradeoff.png` | cooling energy and constraint-violation rate vs a fixed setpoint — the physical tension, independent of any RL result |

`fig4` is the figure to check first: cooling energy must fall monotonically with
setpoint while violations stay at zero and then rise sharply. If that curve is flat,
your thermal or COP parameters are wrong and the whole problem is trivial.

---

## Step 6 — Download the real data manually

This is the step that turns a demonstration into a publishable experiment. Create
the directories first:

```bash
mkdir -p "$DCMG"/data/raw/{azure,solar,weather,price,carbon,spec}
```

Licences, citations and the full list of traps are in §4 of the design document.
**Do all of this in a browser or with `curl`/`aws` yourself** — none of it is
automated on purpose, because several sources need you to accept terms.

### 6a. Workload — Azure Public Dataset V2 (CC-BY 4.0)

Repository: <https://github.com/Azure/AzurePublicDataset>
The file `AzurePublicDatasetLinksV2.txt` lists 195 direct download URLs.

```bash
cd "$DCMG/data/raw/azure"
curl -sO https://raw.githubusercontent.com/Azure/AzurePublicDataset/master/AzurePublicDatasetLinksV2.txt
grep vmtable AzurePublicDatasetLinksV2.txt | xargs -n1 curl -sOL     # ONE file, full 30 days
ls -lh
```

**Use `vmtable`, not the readings shards.** The 195 `vm_cpu_readings` files are
CHRONOLOGICAL slices (~3.7 h each), so any affordable subset covers hours, not the
month. `vmtable.csv.gz` is a single ~800 MB file listing every VM's lifetime and
average CPU; `data_prep.py` reconstructs the full 30-day aggregate utilisation from
it (core-weighted mean of live VMs) and prefers it automatically when present. The
readings shards remain a fallback and are cropped to whole days before tiling.

**The first ~11 lines of the links file are metadata** (notebook, schema, aggregate
stats), not trace data — hence the `grep vm_cpu_readings` filter above. The shards are
gzip-compressed (`data_prep.py` reads .csv.gz directly, no need to decompress).
Schema (no header row): `timestamp_seconds, vm_id, min_cpu, max_cpu, avg_cpu`.
5-minute readings over 30 continuous days, 2.7 M VMs total.
**Cite:** Cortez et al., *Resource Central*, SOSP '17.

> **Trap.** The full V2 set is 235 GB. You do not need it — 8 shards give a stable
> aggregate. Also note the files moved from Azure Blob Storage to GitHub Releases,
> so URLs quoted in older papers look broken.

*AI-data-centre variant (optional, tiny):* `AzureLLMInferenceTrace_conv_1week.csv`
from the `dataset-llm-2024` release — one week of `TIMESTAMP, ContextTokens,
GeneratedTokens`. Cite Stojkovic et al., *DynamoLLM*, HPCA '25.

### 6b. Solar — NSRDB Himawari (covers Thailand, CC BY 3.0 US)

**Easiest path, no API key, no registration:**

```bash
aws s3 ls --no-sign-request s3://nrel-pds-nsrdb/himawari/
```

4 km, 10-minute, 2015 onward, HDF5. Read with `rex`/`h5pyd`, export the nearest
grid cell to your site as CSV into `$DCMG/data/raw/solar/`.

**Simpler path if you would rather have a CSV directly** — the NREL API (free key
from <https://developer.nrel.gov/signup/>):

```bash
curl -L -o "$DCMG/data/raw/solar/nsrdb_bangkok_2020.csv" \
 "https://developer.nrel.gov/api/nsrdb/v2/solar/himawari-download.csv?api_key=YOUR_KEY&wkt=POINT(100.50%2013.75)&names=2020&attributes=ghi,dni,dhi,air_temperature,relative_humidity,wind_speed&interval=60&utc=true&email=YOU@EXAMPLE.COM&full_name=Your+Name&affiliation=Your+Uni&reason=research"
```

The `-L` matters: the endpoint replies with a **redirect to S3**, and without `-L` you
save a one-line "Found. Redirecting to ..." message instead of the data. Verify with
`head -4` — you want two metadata rows then `Year,Month,Day,...` headers, ~400 KB.
NREL is mid-rebrand to "National Laboratory of the Rockies"; if `developer.nrel.gov`
does not resolve on your network, `developer.nlr.gov` mirrors it — sign up and
download on whichever resolves, same key works on both.

**Include `air_temperature` and `relative_humidity` in the attributes** (as above) and
you can skip step 6c entirely: `data_prep.py` derives dry-bulb and wet-bulb from the
NSRDB file itself when `weather/` is empty — same grid cell, same satellite record,
more internally consistent than pairing with another year's reanalysis.

**Cite:** Sengupta et al., *The National Solar Radiation Data Base (NSRDB)*, RSER 2018.

> **Trap.** Himawari coverage is 2016–2020 (2011–2015 is a separate endpoint).
> PVGIS coverage of Thailand is **doubtful** — the JRC describes SARAH-3 as "Europe
> and surrounding regions". Test one call before relying on it. NREL's
> *Solar Power Data for Integration Studies* is 2006, synthetic and **US-only** — it
> cannot serve a Thailand study.

### 6c. Weather — Open-Meteo (free, no key, global) — OPTIONAL if 6b included temperature+humidity

```bash
curl -o "$DCMG/data/raw/weather/openmeteo_2023.csv" \
 "https://archive-api.open-meteo.com/v1/archive?latitude=13.75&longitude=100.50&start_date=2023-01-01&end_date=2023-12-31&hourly=temperature_2m,relative_humidity_2m&timezone=Asia%2FBangkok&format=csv"
```

You need **both** temperature and relative humidity — humidity is what gives the
wet-bulb, and wet-bulb is what determines whether free cooling is ever available.
**Cite:** Zippenfenig, *Open-Meteo.com Weather API* (Zenodo DOI) **and** Hersbach
et al., ERA5, QJRMS 2020.

### 6d. Electricity price — Singapore USEP via SustainCluster (MIT)

```bash
cd "$DCMG/data/raw/price"
curl -O https://raw.githubusercontent.com/HewlettPackard/sustain-cluster/main/data/electricity_prices/standardized/SG/2023/SG_electricity_prices_2023.csv
wc -l SG_electricity_prices_2023.csv        # expect 8761 (8760 + header)
```

Schema: `Datetime (UTC), Datetime (Local), Price (USD/MWh)`. Real half-hourly
tropical wholesale market at UTC+8. **Cite:** Guillen et al., *SustainCluster*,
NeurIPS 2025 Datasets & Benchmarks.

**Thailand tariff, second scenario.** There is **no public time-varying price
dataset for Thailand** — MEA and PEA publish a *tariff schedule*, not a time series.
Synthesise it deterministically (the code already does this via
`make_profiles(price_regime="tou")`):

- On-peak Mon–Fri 09:00–22:00; off-peak 22:00–09:00 + weekends + public holidays
- Indicative 2026 medium/large business: 4.1025 THB/kWh on-peak, 2.5849 off-peak
- Ft surcharge ≈ 0.1623 THB/kWh, revised quarterly by the ERC

Cite the MEA/PEA tariff announcements with an access date, and state in the paper
that it is synthesised. Be aware that a two-block step function has almost no
intra-day volatility, so the "RL learns to arbitrage price" story is much weaker
there — that is exactly why Singapore is the primary scenario.

> **Alternatives if you prefer a large studied market:** ENTSO-E (CC-BY 4.0, free
> token) or CAISO OASIS (no registration at all). **Avoid PJM as your primary** —
> its terms prohibit redistributing raw data without membership, which breaks a
> reproducibility artifact.

### 6e. Carbon intensity

Singapore hourly, from the same SustainCluster repo:

```bash
cd "$DCMG/data/raw/carbon"
curl -O https://raw.githubusercontent.com/HewlettPackard/sustain-cluster/main/data/carbon_intensity/SG/2023/SG_2023_hourly.csv
```

(Browse `.../data/carbon_intensity/` in the repo if the path has moved.)

**Absolute lowest-friction alternative** — the UK NESO API, no key at all:

```bash
curl "https://api.carbonintensity.org.uk/intensity/2023-01-01T00:00Z/2023-01-14T00:00Z"
```

30-minute settlement periods, CC-BY 4.0. Paginate in ≤ 14-day chunks.

For **Thailand**: Ember (CC-BY 4.0, free, no registration) gives you a yearly and
monthly emission factor — enough for a magnitude check, **not** enough for hourly
carbon-aware control. Use Singapore hourly as an explicitly-labelled regional proxy.

> **Trap.** Electricity Maps' free data portal has **moved** —
> `portal.electricitymaps.com/datasets` now redirects to the live map, so URLs cited
> in older papers no longer serve data. Its upstream licence is ODbL (share-alike),
> stricter than CC-BY. WattTime's free tier is `CAISO_NORTH` only and blocks the
> historical endpoint, so do not plan around it.

### 6f. Server power curve — SPECpower_ssj2008 (free, public)

Pick a server close to your assumed fleet and save the `.txt` result:

```bash
curl -o "$DCMG/data/raw/spec/specpower_result.txt" \
 https://www.spec.org/power_ssj2008/results/res2023q3/power_ssj2008-20230829-01308.txt
```

That one is a Dell PowerEdge R7615 / EPYC 9654P: **idle 77.4 W, 10 % 145 W,
50 % 231 W, 100 % 347 W.** Browse <https://www.spec.org/power_ssj2008/results/> for
others. There is no bulk CSV — you take the per-result file.
**Cite:** SPEC, *SPECpower_ssj2008*, plus the exact result ID.

### 6g. Optional — real measured power, for validating realism

- **Google PowerData2019** (CC-BY 4.0): measured power across 57 power domains, 8 of
  which map onto the ClusterData2019 Borg cells. BigQuery access. Cite Sakalkar
  et al., ASPLOS '20.
- **Acme** (CC-BY 4.0): `github.com/InternLM/AcmeTrace` — 15-second node utilisation
  **and IPMI-measured watts** from an LLM training cluster. The better choice if
  your framing is AI data centres. Cite Hu et al., NSDI '24.

You are not simulating with these; you are checking that your synthetic IT-power
series has a realistic magnitude, variance and diurnal shape. Reviewers ask.

---

## Step 7 — Build the real profile table

```bash
cd "$DCMG/src"
python data_prep.py --raw "$DCMG/data/raw" --out "$DCMG/data/profiles_real.csv" \
                    --year 2023 --tz Asia/Bangkok --pv-kw 600
```

The script prints one line per column saying exactly where the data came from and
what resampling rule it applied, then writes `profiles_real_provenance.txt` next to
the output. Any column it could not find falls back to synthetic **and is printed in
capitals**:

```
!! STILL SYNTHETIC: price, carbon -- results are NOT publishable
```

**Expected on a complete download:** `6/6 exogenous columns from real data`.

Resampling rules it applies, which you must state in the paper:

- workload 5 min → 15 min by **mean**
- irradiance 10/60 min → 15 min by **mean**
- temperature and humidity hourly → 15 min by **time interpolation**
- price and carbon hourly → 15 min by **forward fill** — these are piecewise-constant
  settlement quantities; interpolating them and calling the result 15-minute data is
  a real error a reviewer will catch

**Year mismatches are handled, but state them in the paper.** The layers cannot all
come from one year (NSRDB Himawari ends 2020; SustainCluster price/CI start
2020/2021). When a source's year differs from `--year`, `data_prep.py`
calendar-transplants it (month-day-hour alignment, leap days dropped) and records the
shift in the provenance file. Diurnal and seasonal structure is preserved;
day-of-week alignment and cross-year weather-price correlation are not — say so in
the limitations section.

> **The single biggest trap in this whole step is calendar alignment.** The Azure
> trace has no calendar anchor, so it is tiled onto the target year. Alibaba's GPU
> traces go further: the dates are deliberately falsified (time-of-day and
> day-of-week are preserved in UTC+8, the calendar is not), so you *cannot* join
> them to real solar or price data by date. Google's 2011 trace is in US Eastern.
> Acme mixes `+08:00` and `+00:00` inside one dataset. Decide on one timezone —
> `Asia/Bangkok` here — convert everything to it explicitly, and never let pandas
> silently align naive timestamps.

---

## Step 8 — Re-run everything on real data

```bash
python -c "
import pandas as pd
from env import DCMicrogridEnv, DCConfig, calibrate
from baselines import REGISTRY, run_controller
df = pd.read_csv('../data/profiles_real.csv', index_col=0, parse_dates=True)
cut = 240*96
tr, te = df.iloc[:cut], df.iloc[cut:cut+60*96]
cfg = calibrate(DCConfig(), profiles=tr, n_episodes=10)
env = DCMicrogridEnv(cfg=cfg, profiles=te, seed=0, eval_mode=True)
for k, C in REGISTRY.items():
    r = pd.DataFrame(run_controller(C, env, n_episodes=20)).mean(numeric_only=True)
    print(k, round(r.cost,1), round(r.carbon_kg,1), round(r.pue,3))
"
```

Then re-run `verify.py` (it will use synthetic profiles — also spot-check the real
table by hand: no NaNs, no negative prices unless the market genuinely has them, PV
exactly zero at night, wet-bulb ≤ dry-bulb everywhere), and re-run `train.py` with
the real profiles wired into `build_profiles`.

**Compare the real-data result against the synthetic one and expect it to differ.**
Real price series have fat tails and real irradiance has persistent cloudy spells,
both of which usually *increase* the value of flexibility. If your numbers barely
move, check that the real data actually loaded.

---

## Step 9 — The three research extensions, in priority order

These are what turn a working simulation into the two papers (§7 of the design doc).

**P4 — MILP oracle (B6).** Perfect-foresight upper bound over a whole episode, in
HiGHS or Gurobi. Linearise the cooling model (piecewise-linear COP in the setpoint)
and report the **optimality gap** of every controller against it. SustainDC's own
NeurIPS paper lists the absence of an oracle as a limitation; supplying one is a
contribution, and "where is the oracle?" is the first thing a reviewer asks.

**P5 — PPO-Lagrangian (CMDP).** The environment already emits `info["cost_thermal"]`,
`info["cost_sla"]` and `info["cost_pcc"]` per step. Feed them to a Lagrangian solver,
run with `DCConfig(safety_layer=False)`, and report **constraint-violation rates**
alongside reward. The safety-layer-on run is the ablation that shows what the
projection buys you. No competing paper reports violation rates for the joint
three-domain problem.

**P6 — Preference-conditioned MORL.** Already scaffolded: pass
`preference_conditioned=True`, sample `w ~ Dirichlet(1)` per episode, and sweep `w`
at inference to trace a Pareto front from one policy. Report **hypervolume**, and
plot the heuristics as dominated points. This is the figure that answers "your
reward weights are arbitrary".

---

## Step 10 — Protocol discipline before you write the paper

Non-negotiables from §6 of the design doc:

- **≥ 10 seeds**, report mean ± 95 % CI. Never report a best run.
- **Chronological** train/test split. Never random-split a time series.
- **Forecast error stays on.** `DCConfig.forecast_err_std = 0.15` by default. Perfect
  foresight is the most common reviewer complaint in this literature.
- **Report inference latency** per decision — it speaks to the "no real-time
  validation" gap that Kahil et al. (Applied Energy 389:125734, 2025) identify.
- **Justify the test length.** Kang et al. (Renewable Energy 275:126296, 2026) show
  the simulation length needed for a stable microgrid performance estimate grows
  *exponentially* with demand uncertainty. Plot the running mean of cost and carbon
  against evaluation length and show where it settles, rather than asserting that
  60 days is enough.
- **Engage with Sukprasert et al.** (EuroSys '24): simple threshold policies capture
  most of the carbon benefit, and the advantage shrinks as grids decarbonise. If your
  RL margin over B2/B3 is inside the seed noise, report that. It is the honest result
  and it is the whole point of the companion conference paper.

---

## Troubleshooting

| symptom | cause |
|---|---|
| `symbol not found ... _PyUnstable_Object_IsUniqueReferencedTemporary` | you are on Python 3.14. `deactivate` until only `(dcmg)` shows, `rm -rf "$DCMG/src/.venv"`, and use the conda env from step 0 |
| two names in your prompt, e.g. `(.venv) (dcmg)` | activations stack and the leftmost wins. `deactivate` until only `(dcmg)` remains |
| `ModuleNotFoundError: gymnasium` in a new terminal | you are in the wrong env. `conda activate dcmg` before anything else |
| `no such file or directory: dcmg/` | you pasted the folder-layout diagram from step 1. It is a picture, not a command |
| `cd: no such file or directory` after a `cd` | `$DCMG` is unset in this shell. Re-run the `export DCMG=...` line, or add it to `~/.zshrc` |
| `ModuleNotFoundError: profiles` | run from inside `src/`, or set `PYTHONPATH=.` |
| `verify.py` fails "power balance" | you changed the dispatch order in `step()`; the balance must be computed after the battery is clipped |
| a controller beats B0 by > 20 % | usually free energy — check the terminal storage value and the demand charge are still applied at episode end |
| `t_room_max` above 27 °C | `safety_layer=False`; that is the ablation arm, not a bug |
| SAC much worse than PPO | compute budget, not algorithm quality — SAC is ~15× slower per step here. Give it more steps or say so explicitly |
| PPO plateaus at baseline reward | objectives are normalised so rewards sit near −0.7 and differences are ~0.03; train longer, or widen the weight on the objective you care about |
| `pyarrow` ImportError | write CSV instead of parquet, or `pip install pyarrow` |
| real-data run identical to synthetic | the loaders silently fell back — read the provenance file |
