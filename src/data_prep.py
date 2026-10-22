"""
Turn manually-downloaded public datasets into the exogenous profile table that
DCMicrogridEnv consumes.

    python data_prep.py --raw ../data/raw --out ../data/profiles_real.csv \
                        --year 2023 --lat 13.75 --lon 100.50 --tz Asia/Bangkok

Expected layout under --raw (see RUNBOOK.md for the download commands):

    raw/
      azure/            vm_cpu_readings-file-1-of-195.csv[.gz] ...   (Azure V2, CC-BY 4.0)
      solar/            nsrdb_bangkok_<year>.csv                      (NSRDB Himawari)
      weather/          openmeteo_<year>.csv                          (Open-Meteo ERA5)
      price/            SG_electricity_prices_<year>.csv              (SustainCluster, MIT)
      carbon/           SG_<year>_hourly.csv                          (SustainCluster / EM)
      spec/             specpower_result.txt                          (SPECpower_ssj2008)

Every loader is defensive: it prints what it found, states the resampling rule it
applied, and refuses to silently invent data. Missing inputs fall back to the
synthetic generator for that column and say so loudly, so a half-real run can
never be mistaken for a fully-real one.
"""
from __future__ import annotations

import argparse
import glob
import os
import re
import sys

import numpy as np
import pandas as pd

from profiles import DT_H, make_profiles, stull_wet_bulb

FREQ = f"{int(DT_H * 60)}min"
PROVENANCE: dict[str, str] = {}


def _log(col: str, msg: str):
    PROVENANCE[col] = msg
    print(f"  [{col:9s}] {msg}")


def _grid(year: int, tz: str) -> pd.DatetimeIndex:
    return pd.date_range(f"{year}-01-01", f"{year}-12-31 23:45", freq=FREQ, tz=tz)


def retarget_year(s: pd.Series, idx: pd.DatetimeIndex) -> tuple[pd.Series, str | None]:
    """Transplant a series from its source year onto the target year's calendar.

    Needed because the layers cannot all come from the same year: NSRDB Himawari
    ends in 2020 while SustainCluster price/CI start in 2020/2021. Alignment is by
    calendar position (month-day-hour-minute), which preserves diurnal and seasonal
    structure; day-of-week alignment is NOT preserved, and leap days without a
    counterpart are dropped. The transplant is recorded in the provenance file and
    must be stated in the paper.
    """
    tgt_year = int(idx[0].year)
    src_years = pd.Index(s.index.year)
    if (src_years == tgt_year).any():
        return s, None
    src_year = int(src_years.value_counts().idxmax())
    delta = tgt_year - src_year
    ts, vals = [], []
    for t, v in s.items():
        try:
            ts.append(t.replace(year=t.year + delta))
            vals.append(v)
        except ValueError:          # Feb 29 with no counterpart in the target year
            continue
    s2 = pd.Series(vals, index=pd.DatetimeIndex(ts)).sort_index()
    s2 = s2[~s2.index.duplicated(keep="first")]
    return s2, f"calendar-transplanted {src_year}->{tgt_year} (leap days dropped)"


# --------------------------------------------------------------------------
# 1. Workload  ->  u_demand
# --------------------------------------------------------------------------
STEPS_PER_DAY_ = int(24 / DT_H)


def _shape_to_operating_point(s: pd.Series, target_mean: float) -> tuple[pd.Series, str]:
    """Rescale the trace's SHAPE onto a realistic DC operating point.

    Azure per-VM average CPU sits near 10 % (Cortez et al., SOSP '17) -- true for
    cloud VMs, but a DC running at a flat 10 % has no interesting energy problem.
    Standard practice is to keep the trace's relative variation and rescale the
    level to the modelled facility. Multiplicative scaling preserves relative
    amplitude; the factor is recorded for the paper.
    """
    scale = target_mean / max(float(s.mean()), 1e-9)
    return s * scale, f"shape rescaled x{scale:.2f} to mean {target_mean:.2f}"


def _tile_whole_days(s: pd.Series, idx: pd.DatetimeIndex) -> tuple[np.ndarray, str]:
    """Tile a calendar-less trace over the target year WITHOUT drifting the clock.

    The block being tiled must be a whole number of days, or the diurnal pattern
    shifts by the remainder on every repetition (a 29.75 h block drifts ~6 h per
    tile). Crop to whole days first; refuse if less than one full day survives.
    """
    n = (len(s) // STEPS_PER_DAY_) * STEPS_PER_DAY_
    if n == 0:
        raise ValueError("trace shorter than one full day -- cannot tile without "
                         "clock drift; download more of the trace")
    days = n // STEPS_PER_DAY_
    return np.resize(s.to_numpy()[:n], len(idx)), f"{days} whole day(s) tiled"


def load_workload(raw: str, idx: pd.DatetimeIndex, target_mean: float = 0.55,
                  seed: int = 0) -> pd.Series | None:
    """Azure Public Dataset V2 -> aggregate utilisation shape at 15 min.

    Two paths, best available wins:

    PREFERRED -- vmtable.csv.gz (ONE ~800 MB file, covers the FULL 30 days).
      Columns: vmid, subscription, deployment, created_s, deleted_s, maxcpu,
      avgcpu, p95maxcpu, category, corebucket, membucket.  The aggregate
      utilisation at time t is the core-weighted mean avgcpu of the VMs alive at
      t; VM churn carries the real diurnal/weekly signal.  This is the whole-
      trace reconstruction recommended when the readings shards are impractical.

    FALLBACK -- *vm_cpu_readings* shards.  BEWARE: the 195 shards are
      CHRONOLOGICAL slices (~3.7 h each), so 8 contiguous shards cover only
      ~30 hours, not 30 days.  The loader crops to whole days before tiling and
      says so; for a publishable series use vmtable or many more shards.

    Either way the trace has no calendar anchor: it is tiled onto the target
    year (whole days only) and rescaled to `target_mean`, both logged.
    """
    vmtable = sorted(glob.glob(os.path.join(raw, "azure", "*vmtable*.csv*")))
    readings = sorted(glob.glob(os.path.join(raw, "azure", "*vm_cpu_readings*.csv*")))

    if vmtable:
        print(f"  [u_demand ] reconstructing from {os.path.basename(vmtable[0])} ...",
              flush=True)
        cols = ["vmid", "sub", "dep", "created", "deleted", "maxcpu", "avgcpu",
                "p95", "cat", "cores", "mem"]
        starts, ends, weights, cores_l = [], [], [], []
        for chunk in pd.read_csv(vmtable[0], header=None, names=cols,
                                 usecols=["created", "deleted", "avgcpu", "cores"],
                                 compression="infer", chunksize=500_000):
            c = pd.to_numeric(chunk["cores"], errors="coerce").fillna(1.0).clip(lower=1)
            a = pd.to_numeric(chunk["avgcpu"], errors="coerce").fillna(0.0).clip(0, 100)
            t0 = pd.to_numeric(chunk["created"], errors="coerce").fillna(0.0)
            t1 = pd.to_numeric(chunk["deleted"], errors="coerce")
            starts.append(t0.to_numpy())
            ends.append(t1.to_numpy())
            weights.append((c * a).to_numpy())
            cores_l.append(c.to_numpy())
        t0 = np.concatenate(starts)
        t1 = np.concatenate(ends)
        w = np.concatenate(weights)
        cr = np.concatenate(cores_l)
        horizon = np.nanmax(t1[np.isfinite(t1)]) if np.isfinite(t1).any() else t0.max()
        horizon = float(max(horizon, t0.max()))
        step_s = int(DT_H * 3600)
        nbins = int(horizon // step_s) + 2
        b0 = np.clip((t0 // step_s).astype(int), 0, nbins - 1)
        t1f = np.where(np.isfinite(t1), t1, horizon)
        b1 = np.clip((t1f // step_s).astype(int), 0, nbins - 1)
        dw = np.zeros(nbins + 1)
        dc = np.zeros(nbins + 1)
        np.add.at(dw, b0, w);  np.add.at(dw, b1, -w)
        np.add.at(dc, b0, cr); np.add.at(dc, b1, -cr)
        util = np.cumsum(dw)[:nbins] / np.maximum(np.cumsum(dc)[:nbins], 1.0) / 100.0
        # Boundary bins hold partial VM populations (trace start/end) and dip
        # toward zero; left as-is they tile into one-step load dips every 30 d.
        if nbins > 4:
            util[0], util[1] = util[2], util[2]
            util[-1], util[-2] = util[-3], util[-3]
        s = pd.Series(util)
        src = (f"Azure V2 vmtable reconstruction ({len(t0):,} VMs, core-weighted "
               f"avgcpu of live VMs, {step_s//60}-min bins over {horizon/86400:.1f} d)")
    elif readings:
        # Chunked streaming with a stable hash-sample of VMs (deterministic
        # across runs and chunk boundaries -- no RNG).
        frac_pct = 2
        parts = []
        for fi, f in enumerate(readings[:8]):
            print(f"  [u_demand ] reading shard {fi+1}/{min(len(readings),8)} "
                  f"({os.path.basename(f)[-28:]}) ...", flush=True)
            for chunk in pd.read_csv(f, header=None, compression="infer",
                                     names=["ts", "vm", "cpu_min", "cpu_max", "cpu_avg"],
                                     usecols=["ts", "vm", "cpu_avg"],
                                     chunksize=2_000_000):
                h = pd.util.hash_pandas_object(chunk["vm"], index=False) % 100
                kept = chunk[h < frac_pct]
                if len(kept):
                    parts.append(kept.groupby("ts").cpu_avg.agg(["sum", "count"]))
        agg = pd.concat(parts).groupby(level=0).sum()
        s = (agg["sum"] / agg["count"]).sort_index() / 100.0
        s.index = pd.to_datetime(s.index, unit="s", origin="unix", utc=True)
        s = s.resample(FREQ).mean().interpolate("time")
        src = (f"Azure V2 readings, {len(readings[:8])} CHRONOLOGICAL shards "
               f"(~{len(s)*DT_H:.0f} h of trace -- shards are time slices, "
               f"use vmtable for the full 30 d), {frac_pct}% hash-sample")
    else:
        return None

    s, scale_note = _shape_to_operating_point(s, target_mean)
    vals, tile_note = _tile_whole_days(s, idx)
    _log("u_demand", f"{src}; {scale_note}; {tile_note} over {len(idx)} steps")
    return pd.Series(np.clip(vals, 0.02, 0.99), index=idx)


# --------------------------------------------------------------------------
# 2. Solar  ->  ghi_norm, pv_kw
# --------------------------------------------------------------------------
def load_solar(raw: str, idx: pd.DatetimeIndex, tz: str) -> pd.Series | None:
    """NSRDB (Himawari or PSM3) CSV export -> normalised GHI, 10min/30min -> 15min.

    NREL CSVs carry two metadata rows before the data header. Handles both the
    API export (Year,Month,Day,Hour,Minute,GHI,...) and a plain GHI time series.
    """
    files = sorted(glob.glob(os.path.join(raw, "solar", "*.csv")))
    if not files:
        return None
    d = pd.read_csv(files[0], skiprows=2)
    if {"Year", "Month", "Day", "Hour", "Minute"}.issubset(d.columns):
        ts = pd.to_datetime(d[["Year", "Month", "Day", "Hour", "Minute"]])
    else:
        ts = pd.to_datetime(d.iloc[:, 0])
    ghi_col = next(c for c in d.columns if c.upper().startswith("GHI"))
    s = pd.Series(d[ghi_col].to_numpy(float), index=ts)
    # NSRDB timestamps are UTC unless you asked for local time; localise then convert.
    s.index = (s.index.tz_localize("UTC") if s.index.tz is None else s.index).tz_convert(tz)
    native = pd.infer_freq(s.index[:20]) or "?"
    s, shift_note = retarget_year(s, idx)
    s = s.resample(FREQ).mean().interpolate("time").reindex(idx).ffill().bfill()
    ghi = np.clip(s.to_numpy() / 1000.0, 0.0, 1.2)              # W/m2 -> ~[0,1]
    _log("ghi_norm", f"NSRDB {os.path.basename(files[0])}, native {native}, "
                     f"-> {FREQ} mean, normalised by 1000 W/m2"
                     + (f"; {shift_note}" if shift_note else ""))
    return pd.Series(ghi, index=idx)


# --------------------------------------------------------------------------
# 3. Weather  ->  t_amb, t_wb
# --------------------------------------------------------------------------
def load_weather(raw: str, idx: pd.DatetimeIndex, tz: str):
    """Open-Meteo historical CSV -> dry-bulb and (via Stull) wet-bulb, hourly -> 15min.

    Temperature is INTERPOLATED from hourly, not upsampled-and-held; the rule is
    recorded so the paper can state it.
    """
    files = sorted(glob.glob(os.path.join(raw, "weather", "*.csv")))
    if not files:
        return _weather_from_nsrdb(raw, idx, tz)
    d = pd.read_csv(files[0], comment="#")
    tcol = next(c for c in d.columns if c.lower().startswith(("time", "date")))
    ta = next(c for c in d.columns if "temperature" in c.lower())
    rh = next((c for c in d.columns if "humidity" in c.lower()), None)
    ts = pd.to_datetime(d[tcol])
    ts = (ts.dt.tz_localize(tz) if ts.dt.tz is None else ts.dt.tz_convert(tz))
    t = pd.Series(d[ta].to_numpy(float), index=ts).resample(FREQ).interpolate("time")
    t = t.reindex(idx).ffill().bfill()
    if rh is not None:
        h = pd.Series(d[rh].to_numpy(float), index=ts).resample(FREQ).interpolate("time")
        h = h.reindex(idx).ffill().bfill()
        wb = pd.Series(stull_wet_bulb(t.to_numpy(), h.to_numpy()), index=idx)
        _log("t_wb", "Stull (2011) approximation from Open-Meteo relative humidity")
    else:
        wb = t - 4.0
        _log("t_wb", "NO HUMIDITY COLUMN -- crude t_amb - 4 K fallback, DO NOT PUBLISH")
    _log("t_amb", f"Open-Meteo {os.path.basename(files[0])}, hourly -> {FREQ} "
                  f"time-interpolated")
    return t, wb


def _weather_from_nsrdb(raw: str, idx: pd.DatetimeIndex, tz: str):
    """Derive dry-bulb and wet-bulb from the NSRDB solar file itself.

    The NSRDB export already carries hourly Temperature and Relative Humidity for
    the same grid cell and the same satellite record as the irradiance, which makes
    it MORE internally consistent than pairing 2020 irradiance with another year's
    Open-Meteo weather. Used automatically when data/raw/weather/ is empty.
    """
    files = sorted(glob.glob(os.path.join(raw, "solar", "*.csv")))
    if not files:
        return None, None
    d = pd.read_csv(files[0], skiprows=2)
    if not {"Year", "Month", "Day", "Hour", "Minute"}.issubset(d.columns):
        return None, None
    tcol = next((c for c in d.columns if c.lower().startswith("temperature")), None)
    rcol = next((c for c in d.columns if "humidity" in c.lower()), None)
    if tcol is None or rcol is None:
        return None, None
    ts = pd.to_datetime(d[["Year", "Month", "Day", "Hour", "Minute"]])
    t = pd.Series(d[tcol].to_numpy(float), index=ts)
    h = pd.Series(d[rcol].to_numpy(float), index=ts)
    t.index = t.index.tz_localize("UTC").tz_convert(tz)
    h.index = h.index.tz_localize("UTC").tz_convert(tz)
    t, note = retarget_year(t, idx)
    h, _ = retarget_year(h, idx)
    t = t.resample(FREQ).interpolate("time").reindex(idx).ffill().bfill()
    h = h.resample(FREQ).interpolate("time").reindex(idx).ffill().bfill()
    wb = pd.Series(stull_wet_bulb(t.to_numpy(), np.clip(h.to_numpy(), 5, 99)), index=idx)
    src = os.path.basename(files[0])
    _log("t_amb", f"NSRDB {src} Temperature, hourly -> {FREQ} interpolated"
                  + (f"; {note}" if note else ""))
    _log("t_wb", f"Stull (2011) from NSRDB {src} Relative Humidity (same satellite "
                 f"record as the irradiance -- internally consistent)")
    return t, wb


# --------------------------------------------------------------------------
# 4. Price and 5. carbon
# --------------------------------------------------------------------------
def _hourly_series(path: str, idx: pd.DatetimeIndex, tz: str, value_hint: str):
    d = pd.read_csv(path)
    tcol = next((c for c in d.columns if "datetime" in c.lower() or "time" in c.lower()
                 or "date" in c.lower()), d.columns[0])
    vcol = next((c for c in d.columns if value_hint in c.lower()),
                d.select_dtypes("number").columns[-1])
    ts = pd.to_datetime(d[tcol], utc=True, errors="coerce")
    s = pd.Series(d[vcol].to_numpy(float), index=ts).dropna().sort_index()
    s.index = s.index.tz_convert(tz)
    s, shift_note = retarget_year(s, idx)
    if shift_note:
        _log(os.path.basename(path), shift_note)
    # Hourly -> 15 min by FORWARD FILL: a price or an emission factor is a piecewise-
    # constant settlement quantity, not something to interpolate. Never present the
    # upsampled series as measured at 15 min.
    return s.resample(FREQ).ffill().reindex(idx).ffill().bfill()


def load_price(raw: str, idx, tz, usd_per_local: float = 1.0):
    files = sorted(glob.glob(os.path.join(raw, "price", "*.csv")))
    if not files:
        return None
    s = _hourly_series(files[0], idx, tz, "price") / usd_per_local / 1000.0  # $/MWh -> $/kWh
    _log("price", f"{os.path.basename(files[0])}, hourly -> {FREQ} FORWARD-FILL "
                  f"(piecewise-constant settlement price), $/MWh -> $/kWh")
    return s.clip(lower=0.005)


def load_carbon(raw: str, idx, tz):
    files = sorted(glob.glob(os.path.join(raw, "carbon", "*.csv")))
    if not files:
        return None
    s = _hourly_series(files[0], idx, tz, "carbon")
    if s.median() > 5:                       # gCO2/kWh -> kgCO2/kWh
        s = s / 1000.0
    _log("carbon", f"{os.path.basename(files[0])}, hourly -> {FREQ} FORWARD-FILL, "
                   f"gCO2/kWh -> kgCO2/kWh")
    return s.clip(0.01, 1.5)


# --------------------------------------------------------------------------
# 6. SPECpower -> server power curve
# --------------------------------------------------------------------------
def load_spec_curve(raw: str) -> dict[float, float] | None:
    """Parse the load/watts table out of a SPECpower_ssj2008 .txt result file."""
    files = sorted(glob.glob(os.path.join(raw, "spec", "*.txt")))
    if not files:
        return None
    txt = open(files[0], errors="ignore").read()
    pts: dict[float, float] = {}
    for m in re.finditer(r"^\s*(\d{1,3})%\s+.*?([\d,]+\.?\d*)\s*$", txt, re.M):
        pts[int(m.group(1)) / 100.0] = float(m.group(2).replace(",", ""))
    idle = re.search(r"Active Idle\s+.*?([\d,]+\.?\d*)\s*$", txt, re.M)
    if idle:
        pts[0.0] = float(idle.group(1).replace(",", ""))
    if len(pts) < 3:
        print("  [spec     ] could not parse a load/watts table -- check the file format")
        return None
    _log("spec", f"{os.path.basename(files[0])}: {len(pts)} measured load points, "
                 f"idle {pts.get(0.0, float('nan')):.1f} W")
    return pts


# --------------------------------------------------------------------------
def build(raw: str, year: int, tz: str, pv_kw: float, seed: int = 0) -> pd.DataFrame:
    idx = _grid(year, tz)
    n = len(idx)
    print(f"\nBuilding {n} steps of {FREQ} profiles for {year} ({tz})")
    syn = make_profiles(n_days=n // int(24 / DT_H) + 2, seed=seed,
                        pv_capacity_kw=pv_kw).iloc[:n]
    syn.index = idx

    out = pd.DataFrame(index=idx)

    def _try(name, fn, *a, **kw):
        """Never let a truncated, empty or malformed download kill the whole build.

        A partial `curl`, a zero-byte placeholder, or an HTML error page saved with a
        .csv extension are all common. Report the failure clearly and fall back to
        synthetic for that column only.
        """
        try:
            return fn(*a, **kw)
        except Exception as e:
            print(f"  [{name:9s}] *** FAILED to parse: {type(e).__name__}: {e} ***")
            print(f"  [{name:9s}]     check the file is a complete, non-empty CSV "
                  f"(not an HTML error page)")
            return (None, None) if name == "weather" else None

    u = _try("u_demand", load_workload, raw, idx, seed=seed)
    ghi = _try("ghi_norm", load_solar, raw, idx, tz)
    ta, twb = _try("weather", load_weather, raw, idx, tz)
    pr = _try("price", load_price, raw, idx, tz)
    ci = _try("carbon", load_carbon, raw, idx, tz)

    for name, real, fallback in [
        ("u_demand", u, syn.u_demand), ("ghi_norm", ghi, syn.ghi_norm),
        ("t_amb", ta, syn.t_amb), ("t_wb", twb, syn.t_wb),
        ("price", pr, syn.price), ("carbon", ci, syn.carbon),
    ]:
        if real is None:
            print(f"  [{name:9s}] *** SYNTHETIC FALLBACK -- no usable file "
                  f"(missing, empty, or failed to parse above) ***")
            PROVENANCE[name] = "SYNTHETIC FALLBACK"
            out[name] = fallback.to_numpy()
        else:
            out[name] = np.asarray(real, dtype=float)

    out["pv_kw"] = np.clip(pv_kw * out.ghi_norm * (1 - 0.004 * (out.t_amb - 25.0)), 0, None)
    PROVENANCE["pv_kw"] = f"{pv_kw} kW STC x ghi_norm x (1 - 0.004(T_amb - 25))"

    cols = ["u_demand", "ghi_norm", "t_amb", "t_wb", "price", "carbon"]
    fake = [k for k in cols if "SYNTHETIC" in PROVENANCE.get(k, "SYNTHETIC")]
    print(f"\n{len(cols) - len(fake)}/{len(cols)} exogenous columns from real data")
    if fake:
        print(f"!! STILL SYNTHETIC: {', '.join(fake)} -- results are NOT publishable")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="../data/raw")
    ap.add_argument("--out", default="../data/profiles_real.csv")
    ap.add_argument("--year", type=int, default=2023)
    ap.add_argument("--tz", default="Asia/Bangkok")
    ap.add_argument("--pv-kw", type=float, default=600.0)
    ap.add_argument("--usd-per-local", type=float, default=1.0,
                    help="divide prices by this (e.g. 34.0 for THB -> USD)")
    a = ap.parse_args()
    if not os.path.isdir(a.raw):
        sys.exit(f"raw directory not found: {a.raw} -- see RUNBOOK.md step 6")
    df = build(a.raw, a.year, a.tz, a.pv_kw)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    if a.out.endswith(".parquet"):
        df.to_parquet(a.out)          # needs pyarrow
    else:
        df.to_csv(a.out)
    with open(os.path.splitext(a.out)[0] + "_provenance.txt", "w") as f:
        for k, v in PROVENANCE.items():
            f.write(f"{k}\t{v}\n")
    print(f"\nwrote {a.out} and the provenance file next to it")
    print(df.describe().round(3).to_string())
