"""
Perfect-foresight MILP oracle (journal item J3).

Answers: of the cost saving that is *achievable at all* on these days, how much
does the learned policy actually capture?

    python oracle.py --profiles ../data/profiles_real.csv --episodes 30 \
                     --out ../results_oracle

Scope of the optimisation (stated honestly, and repeated in the paper):
  * decisions optimised with perfect foresight -- deferrable-workload dispatch,
    battery charge/discharge, and the resulting grid import/export and peak;
  * the cooling setpoint is enumerated over a grid of constant values rather
    than optimised per step (see COOLING_LEVELS); the best feasible level per
    day is reported;
  * deadlines are enforced as hard constraints (zero SLA misses), which is the
    regime every controller in the paper operates in;
  * the SPECpower utilisation->power curve is represented EXACTLY, as a
    non-convex piecewise-linear function with binary segment selection.

Because the cooling setpoint is restricted to constant trajectories, the value
reported is an *achievable* schedule (an upper bound on attainable cost, i.e. a
conservative oracle), not a lower bound. A policy that varies the setpoint over
the day could in principle do better; in practice the reported gaps are large
enough that this does not change any conclusion.
"""
from __future__ import annotations

import argparse
import os
import warnings

import numpy as np
import pandas as pd
import pulp

warnings.filterwarnings("ignore")

from env import DCConfig, DCMicrogridEnv, calibrate      # noqa: E402
from profiles import DT_H, STEPS_PER_DAY                 # noqa: E402
from train import build_profiles                         # noqa: E402

COOLING_LEVELS = [18.0, 20.0, 22.0, 24.0, 26.0, 27.0]    # constant t_sup candidates
SPEC_X = [0.0, 0.10, 0.50, 1.00]                         # SPECpower breakpoints
SPEC_Y = [0.0, 0.2504, 0.5704, 1.00]


def _cop(c: DCConfig, t_sup: float, t_wb: float) -> float:
    if t_wb < (t_sup - c.econ_delta_t):
        return c.cop_free
    v = (c.cop_nom * (1.0 + c.cop_gamma * (t_sup - c.t_sup_ref))
         * (1.0 - c.cop_delta * (t_wb - c.t_wb_ref)))
    return float(np.clip(v, c.cop_min, c.cop_max))


def _thermal_ok(c: DCConfig, t_sup: float, p_it: np.ndarray, t_room0: float) -> bool:
    """Replay the lumped-RC room model at a constant setpoint; reject the level
    if the mean room temperature ever leaves the ASHRAE envelope."""
    tau = c.c_room_kwh_per_k / c.k_cool_kw_per_k
    decay = float(np.exp(-DT_H / tau))
    t = t_room0
    for p in p_it:
        t_inf = t_sup + p / c.k_cool_kw_per_k
        t_mean = t_inf + (t - t_inf) * tau / DT_H * (1.0 - decay)
        if t_mean > c.t_room_max + 1e-6:
            return False
        t = t_inf + (t - t_inf) * decay
    return True


def solve_day(c: DCConfig, day: pd.DataFrame, t_sup: float,
              t_room0: float, msg: bool = False):
    """MILP for one episode at a fixed cooling setpoint. Returns (cost, detail)
    or (None, None) if infeasible."""
    T = len(day)
    price = day["price"].to_numpy()
    ci = day["carbon"].to_numpy()
    pv = day["pv_kw"].to_numpy()
    t_wb = day["t_wb"].to_numpy()
    u_off = day["u_demand"].to_numpy()

    u_now = c.alpha_interactive * u_off
    arrival = (1.0 - c.alpha_interactive) * u_off * DT_H      # util*h
    D = int(c.deadline_h / DT_H)                              # deadline, steps
    cop = np.array([_cop(c, t_sup, w) for w in t_wb])
    p_fan = c.fan_frac * c.p_it_max_kw
    P_span = c.p_it_max_kw - c.p_it_idle_kw
    P_base = c.p_it_idle_kw + c.p_aux_frac * c.p_it_max_kw

    m = pulp.LpProblem("oracle", pulp.LpMinimize)

    # --- utilisation, as an exact piecewise-linear (incremental) decomposition
    seg_w = [SPEC_X[i + 1] - SPEC_X[i] for i in range(3)]
    seg_m = [(SPEC_Y[i + 1] - SPEC_Y[i]) / seg_w[i] for i in range(3)]
    d = [[pulp.LpVariable(f"d{i}_{t}", lowBound=0, upBound=seg_w[i]) for i in range(3)]
         for t in range(T)]
    b = [[pulp.LpVariable(f"b{i}_{t}", cat="Binary") for i in range(2)]
         for t in range(T)]
    for t in range(T):
        m += d[t][0] >= seg_w[0] * b[t][0]
        m += d[t][1] <= seg_w[1] * b[t][0]
        m += d[t][1] >= seg_w[1] * b[t][1]
        m += d[t][2] <= seg_w[2] * b[t][1]

    u_exec = [pulp.lpSum(d[t]) for t in range(T)]
    f_u = [pulp.lpSum(seg_m[i] * d[t][i] for i in range(3)) for t in range(T)]
    p_it = [P_base + P_span * f_u[t] for t in range(T)]
    p_load = [p_it[t] * (1.0 + 1.0 / cop[t]) + p_fan for t in range(T)]

    # --- workload: dispatch >= interactive, capacity cap, arrival & deadline
    served = [pulp.LpVariable(f"s_{t}", lowBound=0, upBound=None) for t in range(T)]
    for t in range(T):
        m += u_exec[t] >= u_now[t]
        m += u_exec[t] <= c.u_cap
        m += served[t] == (u_exec[t] - u_now[t]) * DT_H
        m += pulp.lpSum(served[:t + 1]) <= float(arrival[:t + 1].sum())
        due = float(arrival[:max(t - D + 1, 0)].sum())
        m += pulp.lpSum(served[:t + 1]) >= due
    m += pulp.lpSum(served) == float(arrival.sum())        # nothing dropped

    # --- battery
    ch = [pulp.LpVariable(f"ch_{t}", lowBound=0, upBound=c.bess_power_kw) for t in range(T)]
    dis = [pulp.LpVariable(f"di_{t}", lowBound=0, upBound=c.bess_power_kw) for t in range(T)]
    soc = [pulp.LpVariable(f"soc_{t}", lowBound=c.soc_min, upBound=c.soc_max) for t in range(T)]
    for t in range(T):
        prev = c.soc_init if t == 0 else soc[t - 1]
        m += soc[t] == prev + (ch[t] * c.eta_ch - dis[t] * (1.0 / c.eta_dis)) \
            * (DT_H / c.bess_energy_kwh)

    # --- grid
    imp = [pulp.LpVariable(f"im_{t}", lowBound=0, upBound=None) for t in range(T)]
    exp = [pulp.LpVariable(f"ex_{t}", lowBound=0, upBound=None) for t in range(T)]
    pk = pulp.LpVariable("peak", lowBound=0, upBound=None)
    for t in range(T):
        m += imp[t] - exp[t] == p_load[t] - pv[t] + ch[t] - dis[t]
        m += imp[t] <= c.pcc_limit_kw
        m += pk >= imp[t]

    # --- objective: energy + export credit + degradation + peak + terminal SoC
    price_ref = float(price.mean())
    energy = pulp.lpSum(price[t] * imp[t] * DT_H
                        - price[t] * c.export_price_frac * exp[t] * DT_H
                        + c.degrade_cost * (ch[t] + dis[t]) * DT_H
                        for t in range(T))
    peak_cost = c.peak_charge * pk / 30.0 * c.episode_days
    soc_cost = (c.soc_init - soc[T - 1]) * c.bess_energy_kwh * price_ref
    m += energy + peak_cost + soc_cost

    m.solve(pulp.PULP_CBC_CMD(msg=msg))
    if pulp.LpStatus[m.status] != "Optimal":
        return None, None

    p_it_v = np.array([P_base + P_span * sum(seg_m[i] * d[t][i].value()
                                             for i in range(3)) for t in range(T)])
    if not _thermal_ok(c, t_sup, p_it_v, t_room0):
        return None, None

    imp_v = np.array([imp[t].value() for t in range(T)])
    detail = dict(t_sup=t_sup,
                  cost=float(pulp.value(m.objective)),
                  carbon=float((ci * imp_v * DT_H).sum()),
                  peak_kw=float(imp_v.max()),
                  energy_it_kwh=float((p_it_v * DT_H).sum()),
                  grid_import_kwh=float((imp_v * DT_H).sum()))
    return detail["cost"], detail


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--profiles", required=True)
    ap.add_argument("--episodes", type=int, default=30)
    ap.add_argument("--out", default="../results_oracle")
    ap.add_argument("--sla-floor", action="store_true")
    ap.add_argument("--idle-frac", type=float, default=None)
    ap.add_argument("--peak-charge", type=float, default=None)
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    train_df, test_df = build_profiles(profiles_csv=args.profiles)
    kw = {}
    if args.idle_frac is not None:
        kw["p_idle_w"] = args.idle_frac * DCConfig().p_max_w
    if args.peak_charge is not None:
        kw["peak_charge"] = args.peak_charge
    cfg = calibrate(DCConfig(sla_floor=args.sla_floor, **kw), profiles=train_df, n_episodes=8)

    # the room temperature each episode starts from, taken from the environment
    env = DCMicrogridEnv(cfg=cfg, profiles=test_df, seed=0, eval_mode=True)
    env.reset(seed=1000)
    t_room0 = env.t_room

    rows = []
    for e in range(args.episodes):
        day = test_df.iloc[e * STEPS_PER_DAY: (e + 1) * STEPS_PER_DAY]
        if len(day) < STEPS_PER_DAY:
            break
        best = None
        for lvl in COOLING_LEVELS:
            cost, det = solve_day(cfg, day, lvl, t_room0, msg=args.verbose)
            if cost is not None and (best is None or cost < best["cost"]):
                best = det
        if best is None:
            print(f"episode {e}: no feasible schedule", flush=True)
            continue
        best["episode"] = e
        rows.append(best)
        print(f"episode {e:3d}: oracle cost {best['cost']:9.1f} "
              f"(t_sup {best['t_sup']:.0f} C, peak {best['peak_kw']:.0f} kW)",
              flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(args.out, "oracle_raw.csv"), index=False)
    print(f"\noracle mean cost over {len(df)} episodes: {df.cost.mean():.1f} "
          f"(+/- {1.96 * df.cost.std(ddof=1) / np.sqrt(len(df)):.1f})")


if __name__ == "__main__":
    main()
