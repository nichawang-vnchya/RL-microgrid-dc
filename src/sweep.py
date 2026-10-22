"""
Decision-boundary sweep: PV capacity x battery size x flexible-work fraction.

    python sweep.py --profiles ../data/profiles_real.csv --out ../results_sweep

For every cell of the grid, evaluates the rule-based managers (fast, deterministic)
on the held-out test days, and reports where active management helps vs hurts.
Optionally trains PPO on a subset of corner cells (--ppo-corners, slow).

Output: sweep_results.csv (one row per cell x controller) and a printed pivot of
the best manager per cell. This quantifies the screening rule in Section 4.7 of
the conference paper: where does active management stop paying?
"""
from __future__ import annotations

import argparse
import itertools
import os
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from baselines import REGISTRY, run_controller          # noqa: E402
from env import DCConfig, DCMicrogridEnv, calibrate      # noqa: E402
from train import build_profiles                         # noqa: E402

PV_LEVELS = [0.0, 300.0, 600.0, 1200.0]        # kW
BESS_LEVELS = [(0.001, 0.001), (800.0, 400.0), (1600.0, 800.0), (3200.0, 1600.0)]  # (kWh, kW)
ALPHA_LEVELS = [0.9, 0.7, 0.5]                 # interactive fraction -> flex = 1 - alpha
CONTROLLERS = ["B0", "B1", "B2", "B4", "B5"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--profiles", required=True)
    ap.add_argument("--out", default="../results_sweep")
    ap.add_argument("--episodes", type=int, default=30)
    ap.add_argument("--sla-floor", action="store_true", default=True)
    ap.add_argument("--ppo-corners", action="store_true",
                    help="also train PPO (3 seeds, 150k steps) on the 4 extreme "
                         "cells -- adds hours")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    train_df, test_df = build_profiles(profiles_csv=args.profiles)
    base_cfg = calibrate(DCConfig(sla_floor=True), profiles=train_df, n_episodes=8)

    rows = []
    cells = list(itertools.product(PV_LEVELS, BESS_LEVELS, ALPHA_LEVELS))
    for ci, (pv, (be, bp), alpha) in enumerate(cells):
        cfg = DCConfig(pv_capacity_kw=max(pv, 0.001), bess_energy_kwh=be,
                       bess_power_kw=bp, alpha_interactive=alpha, sla_floor=True)
        cfg.norm = base_cfg.norm
        te = test_df.copy()
        te["pv_kw"] = te["pv_kw"] * (pv / 600.0)   # rescale the real PV trace
        env = DCMicrogridEnv(cfg=cfg, profiles=te, seed=0, eval_mode=True)
        for key in CONTROLLERS:
            res = run_controller(REGISTRY[key], env, n_episodes=args.episodes)
            d = pd.DataFrame(res)
            rows.append(dict(pv_kw=pv, bess_kwh=be, flex=round(1 - alpha, 2),
                             controller=key, cost=d.cost.mean(),
                             carbon=d.carbon_kg.mean(), peak=d.peak_kw.mean(),
                             sla=d.sla_missed.mean()))
        print(f"cell {ci+1}/{len(cells)}: PV={pv:.0f} BESS={be:.0f} flex={1-alpha:.1f} done",
              flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(args.out, "sweep_results.csv"), index=False)

    # best active manager's margin over B0, per cell
    piv = []
    for (pv, be, fx), g in df.groupby(["pv_kw", "bess_kwh", "flex"]):
        b0 = g[g.controller == "B0"].cost.iloc[0]
        act = g[g.controller != "B0"]
        best = act.loc[act.cost.idxmin()]
        piv.append(dict(pv_kw=pv, bess_kwh=be, flex=fx,
                        best_active=best.controller,
                        margin_pct=round((best.cost / b0 - 1) * 100, 2)))
    pv_t = pd.DataFrame(piv)
    pv_t.to_csv(os.path.join(args.out, "sweep_boundary.csv"), index=False)
    print("\nBest active manager vs do-nothing (negative = active management pays):")
    print(pv_t.pivot_table(index=["pv_kw", "bess_kwh"], columns="flex",
                           values="margin_pct").round(2).to_string())

    if args.ppo_corners:
        from stable_baselines3 import PPO
        from stable_baselines3.common.vec_env import DummyVecEnv
        corners = [(PV_LEVELS[0], BESS_LEVELS[0], ALPHA_LEVELS[0]),
                   (PV_LEVELS[-1], BESS_LEVELS[-1], ALPHA_LEVELS[-1]),
                   (PV_LEVELS[-1], BESS_LEVELS[0], ALPHA_LEVELS[0]),
                   (PV_LEVELS[0], BESS_LEVELS[-1], ALPHA_LEVELS[-1])]
        for pv, (be, bp), alpha in corners:
            cfg = DCConfig(pv_capacity_kw=max(pv, 0.001), bess_energy_kwh=be,
                           bess_power_kw=bp, alpha_interactive=alpha, sla_floor=True)
            cfg.norm = base_cfg.norm
            tr = train_df.copy(); tr["pv_kw"] = tr["pv_kw"] * (pv / 600.0)
            te = test_df.copy(); te["pv_kw"] = te["pv_kw"] * (pv / 600.0)
            costs = []
            for seed in range(3):
                m = PPO("MlpPolicy",
                        DummyVecEnv([lambda: DCMicrogridEnv(cfg=cfg, profiles=tr, seed=seed)]),
                        seed=seed, verbose=0, n_steps=1024, batch_size=256,
                        gamma=0.995, ent_coef=0.004, learning_rate=3e-4,
                        policy_kwargs=dict(net_arch=[128, 128]))
                m.learn(total_timesteps=150_000)
                env = DCMicrogridEnv(cfg=cfg, profiles=te, seed=seed, eval_mode=True)
                cc = []
                for e in range(args.episodes):
                    env.start_index = e * env.ep_len
                    obs, _ = env.reset(seed=1000 + e)
                    done = False
                    while not done:
                        a, _ = m.predict(obs, deterministic=True)
                        obs, _, t1, t2, _ = env.step(a)
                        done = t1 or t2
                    cc.append(env.summary()["cost"])
                costs.append(np.mean(cc))
            print(f"PPO corner PV={pv:.0f} BESS={be:.0f} flex={1-alpha:.1f}: "
                  f"cost {np.mean(costs):.1f} ± {np.std(costs):.1f}", flush=True)


if __name__ == "__main__":
    main()
