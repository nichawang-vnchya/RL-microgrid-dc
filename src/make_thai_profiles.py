"""
Build the Thailand scenario profile table.

    python make_thai_profiles.py --base ../data/profiles_real.csv \
        --out ../data/profiles_th.csv --ft 0.20 --gef 0.4999 --thb-per-usd 34.0

Solar, weather and workload shape are already Thai in the base table (NSRDB
Himawari at 13.75 N / 100.50 E). This script replaces the two series that are
not:

  price   MEA/PEA Rate 4.2, Large General Service, Time-of-Use, 22-33 kV.
          On-peak  09:00-22:00 Mon-Fri (and the Royal Ploughing Ceremony Day);
          off-peak all other hours, plus all day Sat/Sun and public holidays.
          Energy charge + Ft adjustment, converted THB -> USD.
          NOTE: the demand charge (132.93 THB/kW/month at 22-33 kV) is applied
          in the environment, not here -- set it with --peak-charge on the run.

  carbon  TGO grid emission factor, a single national average. Thailand's mix is
          gas-dominated and close to flat hour-to-hour, so a constant factor is
          a defensible approximation; it is also what Thai carbon accounting
          uses. Stated as an assumption in the paper.

Optionally rescales the workload to a measured operating point and imposes the
weekday / weekend structure observed at the LANTA facility.
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

# MEA/PEA Rate 4.2, Large General Service TOU, 22-33 kV (THB/kWh, before Ft)
ON_PEAK_THB = 4.1839
OFF_PEAK_THB = 2.6037
DEMAND_CHARGE_THB_PER_KW = 132.93          # per kW of monthly peak, 22-33 kV

# Thai public holidays 2023 (the base table's calendar year)
TH_HOLIDAYS_2023 = {
    "2023-01-01", "2023-01-02", "2023-03-06", "2023-04-06", "2023-04-13",
    "2023-04-14", "2023-04-15", "2023-05-01", "2023-05-04", "2023-05-05",
    "2023-06-03", "2023-06-05", "2023-07-28", "2023-08-01", "2023-08-12",
    "2023-08-14", "2023-10-13", "2023-10-23", "2023-12-05", "2023-12-10",
    "2023-12-11", "2023-12-31",
}


def thai_tou_price(idx: pd.DatetimeIndex, ft: float, thb_per_usd: float) -> pd.Series:
    """Piecewise-constant TOU price in USD/kWh."""
    hol = pd.to_datetime(sorted(TH_HOLIDAYS_2023)).date
    is_weekend = idx.dayofweek >= 5
    is_holiday = np.isin(idx.date, hol)
    in_window = (idx.hour >= 9) & (idx.hour < 22)
    on_peak = in_window & ~is_weekend & ~is_holiday
    thb = np.where(on_peak, ON_PEAK_THB, OFF_PEAK_THB) + ft
    return pd.Series(thb / thb_per_usd, index=idx)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="../data/profiles_real.csv")
    ap.add_argument("--out", default="../data/profiles_th.csv")
    ap.add_argument("--ft", type=float, required=True,
                    help="current Ft fuel adjustment, THB/kWh (from the ERC/MEA "
                         "announcement; revised every four months)")
    ap.add_argument("--gef", type=float, required=True,
                    help="TGO grid emission factor, kgCO2e/kWh")
    ap.add_argument("--thb-per-usd", type=float, default=34.0)
    ap.add_argument("--util-mean", type=float, default=None,
                    help="rescale workload to this mean utilisation "
                         "(LANTA measures 0.753)")
    ap.add_argument("--weekend-ratio", type=float, default=None,
                    help="weekend/holiday utilisation as a fraction of weekday "
                         "(LANTA measures 1.05 -- an HPC centre runs flat)")
    args = ap.parse_args()

    df = pd.read_csv(args.base, index_col=0, parse_dates=True)
    idx = df.index
    print(f"base table: {len(df)} steps, {idx[0]} to {idx[-1]}")

    price = thai_tou_price(idx, args.ft, args.thb_per_usd)
    df["price"] = price.to_numpy()
    df["carbon"] = args.gef

    if args.weekend_ratio is not None:
        wknd = (idx.dayofweek >= 5) | np.isin(idx.date,
                                              pd.to_datetime(sorted(TH_HOLIDAYS_2023)).date)
        df.loc[wknd, "u_demand"] *= args.weekend_ratio
    if args.util_mean is not None:
        df["u_demand"] = np.clip(df["u_demand"] * (args.util_mean / df["u_demand"].mean()),
                                 0.02, 0.99)

    on = (df["price"] > df["price"].min() + 1e-9).mean()
    print(f"on-peak fraction of the year: {on:.3f}")
    print(f"price  USD/kWh: on-peak {(ON_PEAK_THB+args.ft)/args.thb_per_usd:.4f}  "
          f"off-peak {(OFF_PEAK_THB+args.ft)/args.thb_per_usd:.4f}  "
          f"ratio {(ON_PEAK_THB+args.ft)/(OFF_PEAK_THB+args.ft):.2f}x")
    print(f"carbon kgCO2/kWh: {args.gef} (constant)")
    print(f"u_demand mean {df['u_demand'].mean():.3f}")
    df.to_csv(args.out)

    with open(args.out.replace(".csv", "_provenance.txt"), "w") as f:
        f.write(f"price\tMEA/PEA Rate 4.2 TOU 22-33 kV, on-peak {ON_PEAK_THB} + Ft "
                f"{args.ft} THB/kWh 09:00-22:00 Mon-Fri, off-peak {OFF_PEAK_THB} + Ft; "
                f"THB->USD at {args.thb_per_usd}; demand charge "
                f"{DEMAND_CHARGE_THB_PER_KW} THB/kW/month applied in the environment\n")
        f.write(f"carbon\tTGO grid emission factor {args.gef} kgCO2e/kWh, constant\n")
        f.write(f"solar/weather\tunchanged from {args.base} (NSRDB Himawari, Bangkok)\n")
        f.write(f"u_demand\tunchanged shape from {args.base}"
                + (f"; rescaled to mean {args.util_mean}" if args.util_mean else "")
                + (f"; weekend x{args.weekend_ratio}" if args.weekend_ratio else "") + "\n")
    print(f"wrote {args.out} and its provenance file")


if __name__ == "__main__":
    main()
