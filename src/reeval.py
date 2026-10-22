"""
Re-score saved PPO policies and all rule-based controllers on the test days
under the current environment accounting (terminal backlog valued). No
training: loads R1_s*.zip from --models.

    python reeval.py --profiles ../data/profiles_th.csv --idle-frac 0.52 \
        --peak-charge 3.91 --dense-peak --models ../results_th_dp \
        --out ../results_th_dp_v2
"""
from __future__ import annotations

import argparse
import glob
import os
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from stable_baselines3 import PPO                                  # noqa: E402

from baselines import REGISTRY, run_controller                     # noqa: E402
from env import DCConfig, DCMicrogridEnv, calibrate                 # noqa: E402
from train import build_profiles, eval_policy                       # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--profiles", required=True)
    ap.add_argument("--models", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--test-episodes", type=int, default=30)
    ap.add_argument("--idle-frac", type=float, default=None)
    ap.add_argument("--peak-charge", type=float, default=None)
    ap.add_argument("--dense-peak", action="store_true")
    ap.add_argument("--controllers", default=",".join(REGISTRY))
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    train_df, test_df = build_profiles(profiles_csv=args.profiles)
    kw = dict(dense_peak=args.dense_peak)
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
            r["seed"] = 0; r["family"] = "baseline"
        rows += res
        print(f"{key:>3} {C.name:<20} cost={np.mean([r['cost'] for r in res]):9.1f} "
              f"backlog_cost={np.mean([r['terminal_backlog_cost'] for r in res]):6.1f}", flush=True)

    for path in sorted(glob.glob(os.path.join(args.models, "R1_s*.zip"))):
        seed = int(os.path.basename(path)[4:-4])
        model = PPO.load(path, device="cpu")
        res = eval_policy(model, cfg, test_df, args.test_episodes, "R1 PPO", seed=seed)
        for r in res:
            r["seed"] = seed; r["family"] = "rl"
        rows += res
        print(f"R1 PPO seed {seed}: cost={np.mean([r['cost'] for r in res]):9.1f} "
              f"backlog_cost={np.mean([r['terminal_backlog_cost'] for r in res]):6.1f}", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(args.out, "results_raw.csv"), index=False)
    print(df.groupby("controller")[["cost", "terminal_backlog_cost", "backlog_final",
                                    "peak_kw", "sla_missed"]].mean().round(2).to_string())


if __name__ == "__main__":
    main()
