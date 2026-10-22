"""
Perfect-foresight optimum, version 2: the cooling set-point is a decision.

    python oracle2.py --profiles ../data/profiles_th.csv --episodes 30 \
        --idle-frac 0.52 --peak-charge 3.91 --out ../results_oracle_th

Version 1 (oracle.py) optimised workload dispatch and the battery while holding
the cooling set-point constant over the day. Under the Singapore spot tariff
that restriction was free -- 22 C won on all thirty days. Under a flat
time-of-use tariff it is not: arbitrage is worth almost nothing, cooling is
worth everything, and a constant-set-point "optimum" is beaten by a rule that
moves the set-point with the weather. This version therefore makes the
set-point a per-interval decision, which restores the value's meaning as an
upper bound on what any controller could achieve.

Formulation notes
    * the set-point is discretised over LEVELS; binary z[t,n] selects one per
      interval, so the coefficient of performance is known per (t, level);
    * cooling power is p_it[t] / COP, a product of a continuous variable and a
      binary -- linearised exactly by McCormick envelopes on w[t,n];
    * the lumped-RC room model is linear in the set-point and in IT power, so
      the room temperature enters as ordinary linear recursions and the ASHRAE
      limit as a linear constraint on the within-interval mean temperature;
    * the SPECpower curve remains exact (non-convex piecewise linear, two
      binaries per interval).
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

# Cooling set-point choices, deg C. The COP rises with the set-point, so the
# optimum sits at the highest level the thermal constraint allows; a coarse grid
# (18/21/24/27) therefore under-estimates what a controller with a continuous
# set-point achieves. 0.5 K resolution over 22-27 C with two cold fall-backs.
LEVELS = [18.0, 20.0, 22.0, 23.0, 24.0, 24.5, 25.0, 25.5, 26.0, 26.5, 27.0]
SPEC_X = [0.0, 0.10, 0.50, 1.00]
SPEC_Y = [0.0, 0.2504, 0.5704, 1.00]


def cop_at(c: DCConfig, t_sup: float, t_wb: float) -> float:
    if t_wb < (t_sup - c.econ_delta_t):
        return c.cop_free
    v = (c.cop_nom * (1.0 + c.cop_gamma * (t_sup - c.t_sup_ref))
         * (1.0 - c.cop_delta * (t_wb - c.t_wb_ref)))
    return float(np.clip(v, c.cop_min, c.cop_max))


def solve_day(c: DCConfig, day: pd.DataFrame, t_room0: float,
              msg: bool = False, time_limit: int = 120, gap_rel: float = 0.001):
    T = len(day)
    price = day["price"].to_numpy()
    ci = day["carbon"].to_numpy()
    pv = day["pv_kw"].to_numpy()
    t_wb = day["t_wb"].to_numpy()
    u_off = day["u_demand"].to_numpy()

    u_now = c.alpha_interactive * u_off
    arrival = (1.0 - c.alpha_interactive) * u_off * DT_H
    D = int(c.deadline_h / DT_H)
    p_fan = c.fan_frac * c.p_it_max_kw
    P_span = c.p_it_max_kw - c.p_it_idle_kw
    P_base = c.p_it_idle_kw + c.p_aux_frac * c.p_it_max_kw
    P_UB = c.p_it_max_kw + c.p_aux_frac * c.p_it_max_kw      # McCormick bound

    tau = c.c_room_kwh_per_k / c.k_cool_kw_per_k
    decay = float(np.exp(-DT_H / tau))
    a_mean = tau / DT_H * (1.0 - decay)                      # weight on t_room[t]

    m = pulp.LpProblem("oracle2", pulp.LpMinimize)

    # ---- IT power via the exact SPECpower piecewise-linear curve ----------
    seg_w = [SPEC_X[i + 1] - SPEC_X[i] for i in range(3)]
    seg_m = [(SPEC_Y[i + 1] - SPEC_Y[i]) / seg_w[i] for i in range(3)]
    d = [[pulp.LpVariable(f"d{i}_{t}", lowBound=0, upBound=seg_w[i]) for i in range(3)] for t in range(T)]
    bb = [[pulp.LpVariable(f"bb{i}_{t}", cat="Binary") for i in range(2)] for t in range(T)]
    for t in range(T):
        m += d[t][0] >= seg_w[0] * bb[t][0]
        m += d[t][1] <= seg_w[1] * bb[t][0]
        m += d[t][1] >= seg_w[1] * bb[t][1]
        m += d[t][2] <= seg_w[2] * bb[t][1]
    u_exec = [pulp.lpSum(d[t]) for t in range(T)]
    p_it = [P_base + P_span * pulp.lpSum(seg_m[i] * d[t][i] for i in range(3))
            for t in range(T)]

    # ---- cooling set-point choice + McCormick product --------------------
    N = len(LEVELS)
    z = [[pulp.LpVariable(f"z{n}_{t}", cat="Binary") for n in range(N)] for t in range(T)]
    w = [[pulp.LpVariable(f"w{n}_{t}", lowBound=0, upBound=P_UB) for n in range(N)] for t in range(T)]
    p_cool = []
    for t in range(T):
        m += pulp.lpSum(z[t]) == 1
        for n in range(N):
            m += w[t][n] <= P_UB * z[t][n]
            m += w[t][n] <= p_it[t]
            m += w[t][n] >= p_it[t] - P_UB * (1 - z[t][n])
        m += pulp.lpSum(w[t]) == p_it[t]
        p_cool.append(pulp.lpSum(w[t][n] * (1.0 / cop_at(c, LEVELS[n], t_wb[t]))
                                 for n in range(N)) + p_fan)
    t_sup = [pulp.lpSum(LEVELS[n] * z[t][n] for n in range(N)) for t in range(T)]

    # ---- room temperature: linear recursion, ASHRAE limit on the mean ----
    t_room = [pulp.LpVariable(f"tr_{t}", lowBound=10.0, upBound=c.t_room_hard) for t in range(T + 1)]
    m += t_room[0] == t_room0
    for t in range(T):
        t_inf = t_sup[t] + p_it[t] * (1.0 / c.k_cool_kw_per_k)
        m += t_room[t + 1] == decay * t_room[t] + (1 - decay) * t_inf
        m += (1 - a_mean) * t_inf + a_mean * t_room[t] <= c.t_room_max

    # ---- workload -------------------------------------------------------
    served = [pulp.LpVariable(f"s_{t}", lowBound=0, upBound=None) for t in range(T)]
    for t in range(T):
        m += u_exec[t] >= u_now[t]
        m += u_exec[t] <= c.u_cap
        m += served[t] == (u_exec[t] - u_now[t]) * DT_H
        m += pulp.lpSum(served[:t + 1]) <= float(arrival[:t + 1].sum())
        m += pulp.lpSum(served[:t + 1]) >= float(arrival[:max(t - D + 1, 0)].sum())
    m += pulp.lpSum(served) == float(arrival.sum())

    # ---- battery and grid -----------------------------------------------
    ch = [pulp.LpVariable(f"ch_{t}", lowBound=0, upBound=c.bess_power_kw) for t in range(T)]
    dis = [pulp.LpVariable(f"di_{t}", lowBound=0, upBound=c.bess_power_kw) for t in range(T)]
    soc = [pulp.LpVariable(f"soc_{t}", lowBound=c.soc_min, upBound=c.soc_max) for t in range(T)]
    imp = [pulp.LpVariable(f"im_{t}", lowBound=0, upBound=None) for t in range(T)]
    exp = [pulp.LpVariable(f"ex_{t}", lowBound=0, upBound=None) for t in range(T)]
    pk = pulp.LpVariable("peak", lowBound=0, upBound=None)
    for t in range(T):
        prev = c.soc_init if t == 0 else soc[t - 1]
        m += soc[t] == prev + (ch[t] * c.eta_ch - dis[t] * (1.0 / c.eta_dis)) * (DT_H / c.bess_energy_kwh)
        m += imp[t] - exp[t] == p_it[t] + p_cool[t] - pv[t] + ch[t] - dis[t]
        m += imp[t] <= c.pcc_limit_kw
        m += pk >= imp[t]

    price_ref = float(price.mean())
    m += (pulp.lpSum(price[t] * imp[t] * DT_H
                     - price[t] * c.export_price_frac * exp[t] * DT_H
                     + c.degrade_cost * (ch[t] + dis[t]) * DT_H for t in range(T))
          + c.peak_charge * pk / 30.0 * c.episode_days
          + (c.soc_init - soc[T - 1]) * c.bess_energy_kwh * price_ref)

    m.solve(pulp.PULP_CBC_CMD(msg=msg, timeLimit=time_limit, gapRel=gap_rel))
    if pulp.LpStatus[m.status] not in ("Optimal",):
        return None, None
    # PuLP reports "Optimal" for a feasible incumbent when CBC stops on the time
    # limit; keep the solver's own verdict so a time-limited day is visible.
    sol_status = pulp.LpSolution.get(m.sol_status, str(m.sol_status))

    imp_v = np.array([imp[t].value() for t in range(T)])
    tsup_v = np.array([sum(LEVELS[n] * z[t][n].value() for n in range(N)) for t in range(T)])
    return float(pulp.value(m.objective)), dict(
        cost=float(pulp.value(m.objective)),
        carbon=float((ci * imp_v * DT_H).sum()),
        peak_kw=float(imp_v.max()),
        grid_import_kwh=float((imp_v * DT_H).sum()),
        t_sup_mean=float(tsup_v.mean()),
        t_sup_changes=int((np.diff(tsup_v) != 0).sum()),
        status=sol_status)


def main():
    from baselines import REGISTRY, run_controller
    ap = argparse.ArgumentParser()
    ap.add_argument("--profiles", required=True)
    ap.add_argument("--episodes", type=int, default=30)
    ap.add_argument("--out", required=True)
    ap.add_argument("--idle-frac", type=float, default=None)
    ap.add_argument("--peak-charge", type=float, default=None)
    ap.add_argument("--time-limit", type=int, default=120)
    ap.add_argument("--gap", type=float, default=0.001, help="relative MIP gap")
    ap.add_argument("--levels", default=None,
                    help="comma-separated set-point levels to override LEVELS")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    if args.levels:
        LEVELS[:] = [float(x) for x in args.levels.split(",")]

    tr, te = build_profiles(profiles_csv=args.profiles)
    kw = {}
    if args.idle_frac is not None:
        kw["p_idle_w"] = args.idle_frac * DCConfig().p_max_w
    if args.peak_charge is not None:
        kw["peak_charge"] = args.peak_charge
    cfg = calibrate(DCConfig(sla_floor=False, **kw), profiles=tr, n_episodes=8)
    env = DCMicrogridEnv(cfg=cfg, profiles=te, seed=0, eval_mode=True)
    env.reset(seed=1000)
    t0 = env.t_room
    base = {k: pd.DataFrame(run_controller(REGISTRY[k], env, n_episodes=args.episodes))
            for k in REGISTRY}

    rows = []
    for e in range(args.episodes):
        day = te.iloc[e * STEPS_PER_DAY:(e + 1) * STEPS_PER_DAY]
        cost, det = solve_day(cfg, day, t0, msg=args.verbose,
                              time_limit=args.time_limit, gap_rel=args.gap)
        if cost is None:
            print(f"ep {e}: no solution", flush=True)
            continue
        det["episode"] = e
        for k in base:
            det[f"{k}_cost"] = float(base[k].cost.iloc[e])
        rows.append(det)
        print(f"ep {e:3d}: oracle {cost:8.1f}  B0 {det['B0_cost']:8.1f}  "
              f"gap {100*(cost/det['B0_cost']-1):6.2f}%  "
              f"t_sup mean {det['t_sup_mean']:.1f} C, {det['t_sup_changes']} changes, {det['status']}",
              flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(args.out, "oracle2_paired.csv"), index=False)
    print(f"\n=== paired over {len(df)} days ===")
    for k in base:
        g = 100 * (df.cost / df[f"{k}_cost"] - 1)
        print(f"oracle vs {k}: {g.mean():6.2f}% +/- "
              f"{1.96*g.std(ddof=1)/np.sqrt(len(g)):.2f}")
    g = 100 * (df.cost / df.B0_cost - 1)
    print(f"vs B0: median {g.median():.2f}  min {g.min():.2f}  max {g.max():.2f}")


if __name__ == "__main__":
    main()
