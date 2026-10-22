"""
Exogenous profile generation and real-data loaders for the DC-microgrid environment.

Two modes:
  * synthetic  -- reproducible parametric profiles, no downloads. Used for P1/P2.
  * real       -- loader stubs for the verified dataset stack (see docs/research_design.md §4).

All profiles are returned at the control interval DT_H (default 0.25 h = 15 min)
and are calendar-anchored to a reference year so that solar, price and carbon
align by timestamp.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

DT_H = 0.25                      # control interval, hours
STEPS_PER_DAY = int(24 / DT_H)   # 96


# --------------------------------------------------------------------------
# Synthetic profiles
# --------------------------------------------------------------------------
def _hours(n_days: int) -> np.ndarray:
    return np.arange(n_days * STEPS_PER_DAY) * DT_H


def make_profiles(
    n_days: int = 365,
    seed: int = 0,
    site: str = "bangkok",
    price_regime: str = "volatile",
    carbon_regime: str = "hourly",
    pv_capacity_kw: float = 600.0,
) -> pd.DataFrame:
    """Build a calendar-anchored exogenous dataframe.

    Columns
    -------
    ghi_norm   : clear-sky-normalised global horizontal irradiance in [0, 1]
    t_amb      : ambient dry-bulb temperature, degC
    t_wb       : ambient wet-bulb temperature, degC
    price      : energy price, currency/kWh
    carbon     : grid carbon intensity, kgCO2/kWh
    u_demand   : offered data-centre utilisation in [0, 1]
    pv_kw      : available PV power, kW
    """
    rng = np.random.default_rng(seed)
    h = _hours(n_days)
    n = h.size
    tod = h % 24.0                      # hour of day
    doy = (h // 24.0) % 365.0           # day of year
    dow = (h // 24.0) % 7.0             # 0 = Monday
    weekend = (dow >= 5).astype(float)

    # ---- solar -----------------------------------------------------------
    # Tropical: near-constant daylength, high cloud noise in the wet season.
    if site == "bangkok":
        daylen, noon, seasonal_amp = 12.2, 12.3, 0.10
        t_mean, t_amp_day, t_amp_year = 29.0, 4.0, 2.0
        rh_mean = 0.74
    else:                                # temperate reference site
        daylen, noon, seasonal_amp = 12.0, 12.5, 0.35
        t_mean, t_amp_day, t_amp_year = 11.0, 7.0, 10.0
        rh_mean = 0.62

    solar_elev = np.cos(np.pi * (tod - noon) / daylen)
    clear = np.clip(solar_elev, 0.0, None) ** 1.3
    seasonal = 1.0 + seasonal_amp * np.cos(2 * np.pi * (doy - 172) / 365.0)
    # Cloud: AR(1) so cloudy spells persist for hours, as in reality.
    cloud = np.zeros(n)
    eps = rng.normal(0, 1, n)
    for i in range(1, n):
        cloud[i] = 0.965 * cloud[i - 1] + 0.28 * eps[i]
    cloud_factor = np.clip(1.0 - 0.55 / (1.0 + np.exp(-cloud)), 0.15, 1.0)
    if site == "bangkok":                # wet season May-Oct
        wet = 0.5 * (1 + np.cos(2 * np.pi * (doy - 220) / 365.0))
        cloud_factor *= 1.0 - 0.30 * wet
    ghi_norm = np.clip(clear * seasonal * cloud_factor, 0.0, 1.0)

    # ---- temperature -----------------------------------------------------
    t_amb = (
        t_mean
        + t_amp_year * np.cos(2 * np.pi * (doy - 200) / 365.0)
        + t_amp_day * np.cos(2 * np.pi * (tod - 15) / 24.0)
        + rng.normal(0, 0.6, n)
    )
    # Wet-bulb via a Stull-style approximation with an AR(1) humidity signal.
    rh_noise = np.zeros(n)
    e2 = rng.normal(0, 1, n)
    for i in range(1, n):
        rh_noise[i] = 0.97 * rh_noise[i - 1] + 0.24 * e2[i]
    rh = np.clip(rh_mean + 0.10 * np.tanh(rh_noise) - 0.10 * (t_amb - t_mean) / 10.0,
                 0.25, 0.99)
    t_wb = stull_wet_bulb(t_amb, rh * 100.0)

    # ---- price -----------------------------------------------------------
    if price_regime == "volatile":       # wholesale-market-like (e.g. SG USEP)
        base = 0.11
        shape = (0.55 * np.clip(np.cos(2 * np.pi * (tod - 19) / 24.0), 0, None)
                 + 0.35 * np.clip(np.cos(2 * np.pi * (tod - 10) / 24.0), 0, None))
        ar = np.zeros(n)
        e3 = rng.normal(0, 1, n)
        for i in range(1, n):
            ar[i] = 0.92 * ar[i - 1] + 0.30 * e3[i]
        spike = (rng.random(n) < 0.004) * rng.lognormal(0.6, 0.5, n)
        price = base * (1 + 1.5 * shape) * np.exp(0.30 * ar) * (1 + spike)
        price *= 1.0 - 0.12 * weekend
        price = np.clip(price, 0.02, 2.0)
    elif price_regime == "tou":          # Thai MEA/PEA two-block TOU (synthesised)
        on_peak = ((tod >= 9) & (tod < 22) & (weekend == 0))
        price = np.where(on_peak, 4.1025, 2.5849) + 0.1623   # THB/kWh incl. Ft
        price = price / 34.0                                  # -> USD/kWh
    else:                                # flat
        price = np.full(n, 0.13)

    # ---- carbon ----------------------------------------------------------
    if carbon_regime == "hourly":
        ci = 0.45 - 0.16 * ghi_norm      # solar depresses grid CI in daylight
        ar = np.zeros(n)
        e4 = rng.normal(0, 1, n)
        for i in range(1, n):
            ar[i] = 0.94 * ar[i - 1] + 0.25 * e4[i]
        carbon = np.clip(ci * np.exp(0.12 * ar), 0.10, 0.85)
    else:
        carbon = np.full(n, 0.40)        # flat annual average factor

    # ---- offered workload ------------------------------------------------
    # Diurnal interactive shape + weekly seasonality + AR(1) noise, from the
    # 5-min Azure V2 style utilisation signature.
    diurnal = 0.55 + 0.30 * np.clip(np.cos(2 * np.pi * (tod - 14) / 24.0), -1, 1)
    diurnal *= 1.0 - 0.18 * weekend
    ar = np.zeros(n)
    e5 = rng.normal(0, 1, n)
    for i in range(1, n):
        ar[i] = 0.90 * ar[i - 1] + 0.35 * e5[i]
    u_demand = np.clip(diurnal * np.exp(0.10 * ar), 0.05, 0.98)

    pv_kw = pv_capacity_kw * ghi_norm * (1.0 - 0.004 * (t_amb - 25.0))
    pv_kw = np.clip(pv_kw, 0.0, None)

    idx = pd.date_range("2023-01-01", periods=n, freq=f"{int(DT_H*60)}min", tz="Asia/Bangkok")
    return pd.DataFrame(
        {
            "ghi_norm": ghi_norm,
            "t_amb": t_amb,
            "t_wb": t_wb,
            "price": price,
            "carbon": carbon,
            "u_demand": u_demand,
            "pv_kw": pv_kw,
        },
        index=idx,
    )


def stull_wet_bulb(t_c: np.ndarray, rh_pct: np.ndarray) -> np.ndarray:
    """Stull (2011) wet-bulb approximation. Valid roughly -20..50 degC, RH 5..99 %."""
    rh = np.clip(rh_pct, 5.0, 99.0)
    return (
        t_c * np.arctan(0.151977 * np.sqrt(rh + 8.313659))
        + np.arctan(t_c + rh)
        - np.arctan(rh - 1.676331)
        + 0.00391838 * rh ** 1.5 * np.arctan(0.023101 * rh)
        - 4.686035
    )


# --------------------------------------------------------------------------
# Real-data loaders  (P3 -- see docs/research_design.md section 4)
# --------------------------------------------------------------------------
def load_azure_v2(path: str, n_vms: int = 3000, seed: int = 0) -> pd.Series:
    """Azure Public Dataset V2 (2019 VM trace) -> aggregate utilisation at 15 min.

    Source : https://github.com/Azure/AzurePublicDataset  (CC-BY 4.0)
    Cite   : Cortez et al., "Resource Central", SOSP '17.
    Raw is 5-min per-VM min/max/avg CPU. Subsample `n_vms`, take the avg column,
    aggregate, then resample 5min -> 15min by mean.
    """
    raise NotImplementedError("P3: point `path` at the downloaded Azure V2 CSVs.")


def load_nsrdb_himawari(lat: float, lon: float, year: int) -> pd.DataFrame:
    """NSRDB Himawari via AWS Open Data (no API key, CC BY 3.0 US).

        aws s3 ls --no-sign-request s3://nrel-pds-nsrdb/himawari/

    4 km, 10-min, 2015-. Confirmed to cover Thailand. Resample 10min -> 15min.
    """
    raise NotImplementedError("P3: read the HDF5 via rex/HSDS or a local pull.")


def load_sustaincluster_price_ci(region: str = "SG", year: int = 2023) -> pd.DataFrame:
    """Hourly price + carbon intensity from the SustainCluster repo (MIT).

    price : data/electricity_prices/standardized/<REGION>/<YEAR>/...csv
    carbon: data/carbon_intensity/...  (upstream Electricity Maps, ODbL)
    Cite  : Guillen et al., SustainCluster, NeurIPS 2025 D&B.
    Interpolate hourly -> 15 min and LABEL IT as interpolated in the paper.
    """
    raise NotImplementedError("P3: clone HewlettPackard/sustain-cluster and point here.")


def fit_specpower_curve(watts_by_load: dict[float, float]) -> callable:
    """Fit P(u) from a published SPECpower_ssj2008 result.

    Example (Dell PowerEdge R7615 / EPYC 9654P, res2023q3-20230829-01308):
        {0.0: 77.4, 0.1: 145.0, 0.5: 231.0, 1.0: 347.0}
    Returns a callable u -> watts using a monotone cubic through the measured points.
    """
    from scipy.interpolate import PchipInterpolator

    loads = np.array(sorted(watts_by_load))
    watts = np.array([watts_by_load[k] for k in loads])
    interp = PchipInterpolator(loads, watts, extrapolate=True)
    return lambda u: np.clip(interp(np.clip(u, 0.0, 1.0)), watts[0], watts[-1] * 1.05)
