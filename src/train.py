"""
Training and evaluation harness for dcmg-gym.

Usage
-----
    python train.py --steps 300000 --seeds 3 --out ../results

Trains PPO and SAC on the DC-microgrid environment, evaluates them against
baselines B0-B5 on a chronologically held-out test period, and writes tidy CSVs.

Protocol (docs/research_design.md §6)
    * chronological train/test split -- never random-split a time series
    * >= N seeds, report mean +/- 95 % CI
    * forecast error injected into the observation (no perfect foresight)
"""
from __future__ import annotations

import argparse
import json
import os
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from baselines import REGISTRY, run_controller          # noqa: E402
from env import DCConfig, DCMicrogridEnv, calibrate      # noqa: E402
from profiles import STEPS_PER_DAY, make_profiles        # noqa: E402


TRAIN_DAYS, TEST_DAYS = 240, 60


def build_profiles(seed: int = 1, profiles_csv: str | None = None,
                   **kw) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Chronological train/test split. If `profiles_csv` is given (the output of
    data_prep.py), the real profile table is used instead of synthetic."""
    if profiles_csv:
        df = pd.read_csv(profiles_csv, index_col=0, parse_dates=True)
        need = {"ghi_norm", "t_amb", "t_wb", "price", "carbon", "u_demand", "pv_kw"}
        missing = need - set(df.columns)
        if missing:
            raise ValueError(f"{profiles_csv} is missing columns: {missing}")
        n = (TRAIN_DAYS + TEST_DAYS) * STEPS_PER_DAY
        if len(df) < n:
            raise ValueError(f"{profiles_csv} has {len(df)} steps; "
                             f"need >= {n} for a {TRAIN_DAYS}/{TEST_DAYS} d split")
        df = df.iloc[:n]
        print(f"using real profiles: {profiles_csv} "
              f"({TRAIN_DAYS} d train / {TEST_DAYS} d test, chronological)")
    else:
        df = make_profiles(n_days=TRAIN_DAYS + TEST_DAYS, seed=seed, **kw)
    cut = TRAIN_DAYS * STEPS_PER_DAY
    return df.iloc[:cut].copy(), df.iloc[cut:].copy()


def eval_policy(model, cfg, test_df, n_episodes: int, label: str,
                preference=None, seed: int = 0) -> list[dict]:
    env = DCMicrogridEnv(cfg=cfg, profiles=test_df, seed=seed,
                         eval_mode=True,
                         preference_conditioned=preference is not None)
    out = []
    for e in range(n_episodes):
        env.start_index = e * env.ep_len
        obs, _ = env.reset(seed=1000 + e)
        if preference is not None:
            env.w = np.asarray(preference, dtype=float)
            obs = env._obs()
        done = False
        while not done:
            a, _ = model.predict(obs, deterministic=True)
            obs, r, term, trunc, _ = env.step(a)
            done = term or trunc
        s = env.summary()
        s.update(episode=e, controller=label)
        if preference is not None:
            s["w"] = json.dumps(list(map(float, preference)))
        out.append(s)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=200_000)
    ap.add_argument("--sac-steps", type=int, default=None,
                    help="SAC is far more sample-efficient per step but ~15x "
                         "slower in wall-clock; default = steps // 2")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--test-episodes", type=int, default=30)
    ap.add_argument("--out", default="../results")
    ap.add_argument("--pareto", action="store_true",
                    help="also train a preference-conditioned MORL policy")
    ap.add_argument("--profiles", default=None,
                    help="path to a profiles CSV from data_prep.py; omit for synthetic")
    ap.add_argument("--sla-floor", action="store_true",
                    help="structurally guarantee deadlines (urgent work always "
                         "dispatched) for every controller incl. RL")
    ap.add_argument("--no-safety", action="store_true",
                    help="J1 ablation: disable the action-projection safety "
                         "layer (thermal + PCC guards) for every controller")
    ap.add_argument("--forecast-sigma", type=float, default=None,
                    help="J8 sensitivity: relative std of the injected forecast "
                         "error (default keeps the config value, 0.15)")
    ap.add_argument("--idle-frac", type=float, default=None,
                    help="measured-facility calibration: idle power as a fraction "
                         "of nameplate (SPECpower server alone = 0.223; the LANTA "
                         "facility feed measures 0.52)")
    ap.add_argument("--dense-peak", action="store_true",
                    help="pay the demand charge incrementally (per kW added to "
                         "the running episode peak) instead of as a lump sum at "
                         "episode end. Same episodic return, denser credit "
                         "assignment -- addresses PPO's higher peaks under TOU")
    ap.add_argument("--no-terminal-backlog", action="store_true",
                    help="reproduce pre-October runs: leave deferrable work "
                         "queued at midnight unvalued (horizon artefact)")
    ap.add_argument("--skip-sac", action="store_true",
                    help="train PPO only. SAC is ~15x slower per step in this "
                         "setup and has lost on every scenario so far; skipping "
                         "it cuts wall-clock by roughly an order of magnitude")
    ap.add_argument("--peak-charge", type=float, default=None,
                    help="demand charge, currency per kW of monthly peak "
                         "(Singapore default 8.0; Thai Rate 4.2 at 22-33 kV is "
                         "132.93 THB = 3.91 USD at 34 THB/USD)")
    ap.add_argument("--util-mean", type=float, default=None,
                    help="rescale the workload profile to this mean utilisation "
                         "(the LANTA facility runs at 0.75, not 0.55)")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    from stable_baselines3 import PPO, SAC
    from stable_baselines3.common.vec_env import DummyVecEnv

    train_df, test_df = build_profiles(seed=1, profiles_csv=args.profiles)
    if args.util_mean is not None:
        for d in (train_df, test_df):
            d["u_demand"] = np.clip(d["u_demand"] * (args.util_mean / d["u_demand"].mean()),
                                    0.02, 0.99)
        print(f"workload rescaled to mean utilisation {args.util_mean}")
    cfg_kw = dict(sla_floor=args.sla_floor,
                  safety_layer=not args.no_safety,
                  dense_peak=args.dense_peak,
                  terminal_backlog=not args.no_terminal_backlog)
    if args.idle_frac is not None:
        cfg_kw["p_idle_w"] = args.idle_frac * DCConfig().p_max_w
    if args.peak_charge is not None:
        cfg_kw["peak_charge"] = args.peak_charge
    if args.forecast_sigma is not None:
        cfg_kw["forecast_err_std"] = args.forecast_sigma
    cfg = calibrate(DCConfig(**cfg_kw), profiles=train_df, n_episodes=12)
    if args.sla_floor:
        print("service-level floor ON: urgent work is always dispatched")
    if args.no_safety:
        print("SAFETY LAYER OFF (J1 ablation): thermal/PCC guards disabled")
    if args.forecast_sigma is not None:
        print(f"forecast error std overridden: {args.forecast_sigma}")
    if args.peak_charge is not None:
        print(f"demand charge set to {args.peak_charge} per kW of monthly peak")
    if args.dense_peak:
        print("demand charge paid incrementally (dense peak signal)")
    if args.idle_frac is not None:
        print(f"IT power curve recalibrated: idle = {args.idle_frac:.3f} x nameplate "
              f"({cfg.p_idle_w:.1f} W idle, {cfg.p_max_w:.1f} W max)")
    print("normalisation:", {k: round(v, 2) for k, v in cfg.norm.items()})
    with open(os.path.join(args.out, "config.json"), "w") as f:
        json.dump({"norm": cfg.norm, "steps": args.steps, "seeds": args.seeds,
                   "train_days": TRAIN_DAYS, "test_days": TEST_DAYS,
                   "profiles": args.profiles or "synthetic",
                   "sla_floor": args.sla_floor,
                   "safety_layer": not args.no_safety,
                   "forecast_err_std": cfg.forecast_err_std,
                   "p_idle_w": cfg.p_idle_w,
                   "util_mean": args.util_mean,
                   "peak_charge": cfg.peak_charge,
                   "dense_peak": cfg.dense_peak,
                   "terminal_backlog": cfg.terminal_backlog}, f, indent=2)

    rows: list[dict] = []

    # ---------------- baselines (deterministic -> one pass) ----------------
    base_env = DCMicrogridEnv(cfg=cfg, profiles=test_df, seed=0, eval_mode=True)
    for key, C in REGISTRY.items():
        res = run_controller(C, base_env, n_episodes=args.test_episodes)
        for r in res:
            r["seed"] = 0
            r["family"] = "baseline"
        rows += res
        print(f"{key:>3} {C.name:<20} cost={np.mean([r['cost'] for r in res]):9.1f} "
              f"co2={np.mean([r['carbon_kg'] for r in res]):9.1f}")

    # ---------------- RL agents -------------------------------------------
    for seed in range(args.seeds):
        def mk():
            return DCMicrogridEnv(cfg=cfg, profiles=train_df, seed=seed)

        sac_steps = args.sac_steps or args.steps // 2
        for name, Algo, nsteps, kw in [
            ("R1 PPO", PPO, args.steps,
             dict(policy="MlpPolicy", n_steps=1024, batch_size=256,
                  gae_lambda=0.95, gamma=0.995, ent_coef=0.004,
                  learning_rate=3e-4, policy_kwargs=dict(net_arch=[128, 128]))),
            ("R2 SAC", SAC, sac_steps,
             dict(policy="MlpPolicy", batch_size=256, gamma=0.995,
                  learning_rate=3e-4, train_freq=1, gradient_steps=1,
                  buffer_size=200_000, learning_starts=5_000,
                  policy_kwargs=dict(net_arch=[128, 128]))),
        ]:
            if args.skip_sac and name.startswith("R2"):
                continue
            print(f"\n=== training {name} seed {seed} ({nsteps} steps) ===", flush=True)
            model = Algo(env=DummyVecEnv([mk]), seed=seed, verbose=0, **kw)
            model.learn(total_timesteps=nsteps, progress_bar=False)
            model.save(os.path.join(args.out, f"{name.split()[0]}_s{seed}"))
            res = eval_policy(model, cfg, test_df, args.test_episodes, name, seed=seed)
            for r in res:
                r["seed"] = seed
                r["family"] = "rl"
            rows += res
            print(f"{name} seed {seed}: cost={np.mean([r['cost'] for r in res]):9.1f} "
                  f"co2={np.mean([r['carbon_kg'] for r in res]):9.1f} "
                  f"reward={np.mean([r['reward'] for r in res]):7.3f}")

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(args.out, "results_raw.csv"), index=False)

    # ---------------- preference-conditioned MORL (Pareto front) ----------
    if args.pareto:
        print("\n=== training preference-conditioned MORL policy ===")

        def mk_pref():
            return DCMicrogridEnv(cfg=cfg, profiles=train_df, seed=99,
                                  preference_conditioned=True)

        m = PPO("MlpPolicy", DummyVecEnv([mk_pref]), seed=99, verbose=0,
                n_steps=1024, batch_size=256, gamma=0.995, ent_coef=0.004,
                learning_rate=3e-4, policy_kwargs=dict(net_arch=[128, 128]))
        m.learn(total_timesteps=int(args.steps * 1.5), progress_bar=False)
        m.save(os.path.join(args.out, "R4_MORL"))

        prows = []
        for lam in np.linspace(0.0, 1.0, 11):
            w = [0.75 * (1 - lam), 0.75 * lam, 0.15, 0.10]
            res = eval_policy(m, cfg, test_df, min(args.test_episodes, 20),
                              f"R4 MORL w_carbon={lam:.2f}", preference=w, seed=99)
            prows += res
        pd.DataFrame(prows).to_csv(os.path.join(args.out, "pareto_raw.csv"), index=False)
        print(f"wrote pareto_raw.csv ({len(prows)} rows)")

    # ---------------- summary ---------------------------------------------
    g = (df.groupby("controller")
           .agg(cost_mean=("cost", "mean"), cost_ci=("cost", _ci),
                co2_mean=("carbon_kg", "mean"), co2_ci=("carbon_kg", _ci),
                cool_kwh=("energy_cool_kwh", "mean"),
                pue=("pue", "mean"), peak=("peak_kw", "mean"),
                sla=("sla_missed", "mean"),
                therm_viol=("thermal_viol_rate", "mean"),
                t_max=("t_room_max", "max"),
                self_cons=("self_consumption", "mean"),
                reward=("reward", "mean"))
           .round(3))
    b0 = g.loc["B0 status-quo"]
    g["cost_vs_B0_%"] = ((g.cost_mean / b0.cost_mean - 1) * 100).round(2)
    g["co2_vs_B0_%"] = ((g.co2_mean / b0.co2_mean - 1) * 100).round(2)
    g["cool_vs_B0_%"] = ((g.cool_kwh / b0.cool_kwh - 1) * 100).round(2)
    g.to_csv(os.path.join(args.out, "summary.csv"))
    print("\n" + g.to_string())


def _ci(x) -> float:
    x = np.asarray(x, dtype=float)
    return float(1.96 * x.std(ddof=1) / np.sqrt(len(x))) if len(x) > 1 else 0.0


if __name__ == "__main__":
    main()
