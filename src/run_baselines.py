"""
Evaluate the rule-based controllers only (no RL training) on a profile table.

    python run_baselines.py --profiles ../data/profiles_th.csv --idle-frac 0.52 \
        --peak-charge 3.91 --out ../results_th_dp --controllers B0,B3,B6

Baselines are deterministic, so one pass over the test days is exact. Writes
baselines_raw.csv (one row per controller x day) and baselines_summary.csv into
--out, next to the RL results they are compared with. Pass --merge to also
append the new controllers into results_raw.csv in that folder (rows with the
same controller name are replaced).
"""
from __future__ import annotations

import argparse
import os
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from baselines import REGISTRY, run_controller                   # noqa: E402
from env import DCConfig, DCMicrogridEnv, calibrate               # noqa: E402
from train import build_profiles                                  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--profiles", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--test-episodes", type=int, default=30)
    ap.add_argument("--controllers", default=",".join(REGISTRY),
                    help="comma-separated keys, e.g. B0,B3,B6")
    ap.add_argument("--idle-frac", type=float, default=None)
    ap.add_argument("--peak-charge", type=float, default=None)
    ap.add_argument("--sla-floor", action="store_true")
    ap.add_argument("--merge", action="store_true")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    train_df, test_df = build_profiles(profiles_csv=args.profiles)
    kw = dict(sla_floor=args.sla_floor)
    if args.idle_frac is not None:
        kw["p_idle_w"] = args.idle_frac * DCConfig().p_max_w
    if args.peak_charge is not None:
        kw["peak_charge"] = args.peak_charge
    cfg = calibrate(DCConfig(**kw), profiles=train_df, n_episodes=12)
    env = DCMicrogridEnv(cfg=cfg, profiles=test_df, seed=0, eval_mode=True)

    rows = []
    for key in args.controllers.split(","):
        C = REGISTRY[key.strip()]
        res = run_controller(C, env, n_episodes=args.test_episodes)
        for r in res:
            r["seed"] = 0
            r["family"] = "baseline"
        rows += res
        print(f"{key:>3} {C.name:<20} cost={np.mean([r['cost'] for r in res]):9.1f} "
              f"co2={np.mean([r['carbon_kg'] for r in res]):9.1f} "
              f"peak={np.mean([r['peak_kw'] for r in res]):7.1f} "
              f"sla={np.mean([r['sla_missed'] for r in res]):.4f}", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(args.out, "baselines_raw.csv"), index=False)
    df.groupby("controller")[["cost", "carbon_kg", "peak_kw", "energy_cool_kwh",
                              "sla_missed"]].mean().to_csv(
        os.path.join(args.out, "baselines_summary.csv"))

    if args.merge:
        p = os.path.join(args.out, "results_raw.csv")
        if os.path.exists(p):
            old = pd.read_csv(p)
            old = old[~old.controller.isin(df.controller.unique())]
            pd.concat([old, df], ignore_index=True).to_csv(p, index=False)
            print(f"merged into {p}")
    print("done")


if __name__ == "__main__":
    main()
