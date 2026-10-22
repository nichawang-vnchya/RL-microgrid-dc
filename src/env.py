"""
dcmg-gym : a Gymnasium environment for a grid-connected microgrid (PV + BESS + grid)
supplying a data centre whose IT workload, cooling setpoint and battery are jointly
controlled by a single agent.

Fills the environment gap identified in docs/research_design.md §1.3: SustainDC has
the DC side but explicitly no PV / grid dispatch / market; pymgrid and CityLearn have
the electrical side but no DC thermal or IT model.

Action (Box, 3-dim, in [-1, 1])
    0  a_shift : fraction of the deferrable backlog dispatched this interval  -> [0, 1]
    1  a_bess  : battery power                                    -> [-P_ch_max, +P_dis_max]
    2  a_tsup  : CRAC supply-air setpoint                         -> [18, 27] degC

Objectives (vector, all "lower is better"; scalarised by weights w)
    cost, carbon, sla, thermal

Constraints (CMDP; `info["cost_*"]` gives per-step constraint costs for a
Lagrangian solver such as PPO-Lagrangian)
    thermal : T_room <= T_ROOM_MAX  (ASHRAE class A1 recommended upper bound 27 degC,
                                     allowable 32 degC)
    sla     : deferrable work completes before its deadline
    soc     : SoC within [SOC_MIN, SOC_MAX]  (enforced structurally)
    pcc     : grid import <= contracted capacity
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from profiles import DT_H, STEPS_PER_DAY, make_profiles


# ==========================================================================
# Configuration
# ==========================================================================
@dataclass
class DCConfig:
    # ---- IT ----------------------------------------------------------------
    n_servers: int = 2500
    p_idle_w: float = 77.4          # SPECpower_ssj2008: Dell R7615 / EPYC 9654P
    p_max_w: float = 347.0
    p_aux_frac: float = 0.06        # PDU / UPS / network losses as a fraction of P_IT_max
    u_cap: float = 0.95             # max achievable aggregate utilisation
    alpha_interactive: float = 0.70 # fraction of offered load that must run immediately
    deadline_h: float = 6.0         # deadline for deferrable work

    # ---- thermal (lumped first-order RC) -----------------------------------
    # Sized so that at nameplate IT load the room-to-supply lift is ~4.2 K:
    # T_sup = 22 degC is just feasible, T_sup = 26 degC is not unless the agent
    # first sheds IT power. That coupling is the joint-optimisation story.
    c_room_kwh_per_k: float = 55.0  # thermal capacitance  -> tau = C/k = 0.25 h
    k_cool_kw_per_k: float = 220.0  # effective UA of the air loop
    t_room_init: float = 24.0
    t_room_max: float = 27.0        # ASHRAE recommended upper bound (soft/CMDP)
    t_room_hard: float = 32.0       # ASHRAE class A1 allowable upper bound
    t_sup_min: float = 18.0
    t_sup_max: float = 27.0

    # ---- cooling plant -----------------------------------------------------
    cop_nom: float = 3.2            # at t_sup_ref, t_wb_ref -> design PUE ~1.4,
                                    # typical of an air-cooled tropical-climate site
    t_sup_ref: float = 22.0
    t_wb_ref: float = 20.0
    cop_gamma: float = 0.060        # dCOP/dT_sup, per K  (higher setpoint -> better COP)
    cop_delta: float = 0.030        # dCOP/dT_wb,  per K  (hotter/wetter air -> worse COP)
    cop_min: float = 1.5
    cop_max: float = 8.0
    econ_delta_t: float = 6.0       # free cooling when t_wb < t_sup - econ_delta_t
    cop_free: float = 15.0
    fan_frac: float = 0.030         # fan power as a fraction of IT nameplate

    # ---- microgrid ---------------------------------------------------------
    pv_capacity_kw: float = 600.0
    bess_energy_kwh: float = 1600.0
    bess_power_kw: float = 800.0
    eta_ch: float = 0.95
    eta_dis: float = 0.95
    soc_min: float = 0.10
    soc_max: float = 0.90
    soc_init: float = 0.50
    degrade_cost: float = 0.02      # currency per kWh of throughput
    pcc_limit_kw: float = 1400.0    # contracted grid import capacity
    export_price_frac: float = 0.30
    peak_charge: float = 8.0        # currency per kW of monthly peak import
    # True  = the demand charge is paid incrementally: at each step the agent is
    #         charged for the kW by which it raises the running episode peak.
    #         Sum over the episode == the lump-sum charge, so the episodic return
    #         (and hence the optimal policy) is unchanged; only the credit
    #         assignment is denser.
    # False = lump sum at episode end (original formulation).
    dense_peak: bool = False
    # TERMINAL BACKLOG VALUE. Deferrable work still queued at midnight is neither
    # served nor penalised when the episode resets, so every controller can push
    # up to one deadline window of work past the horizon for free. Charging the
    # leftover at the energy it will cost (marginal IT power x the episode-mean
    # price) removes the artefact, exactly as the terminal SoC term does for the
    # battery. Reported separately in summary() as terminal_backlog_cost.
    terminal_backlog: bool = True

    # ---- safety layer (action projection) ----------------------------------
    # True  = realistic operation: actions are projected onto the feasible set
    #         before they are applied (this is the "safety layer" ablation arm).
    # False = raw actions; constraint violations are learnable and reported.
    safety_layer: bool = True

    # ---- service-level floor -----------------------------------------------
    # True = the dispatch fraction is floored so that work whose deadline falls
    # within the next third of the deadline window is always released -- the same
    # structural guarantee the rule-based controllers implement via their
    # _urgent() helper. Makes deadline misses (nearly) impossible for ANY policy,
    # at the cost of removing the agent's freedom to gamble with service.
    sla_floor: bool = False

    # ---- episode -----------------------------------------------------------
    episode_days: int = 1
    forecast_h: tuple[int, ...] = (1, 4)   # forecast horizons, hours
    forecast_err_std: float = 0.15         # relative std of injected forecast error

    # ---- reward weights (used when preference_conditioned=False) -----------
    w_cost: float = 0.40
    w_carbon: float = 0.25
    w_sla: float = 0.20
    w_thermal: float = 0.15

    # ---- normalisation constants (set by calibrate()) ----------------------
    norm: dict = field(default_factory=lambda: {
        "cost": 1.0, "carbon": 1.0, "sla": 1.0, "thermal": 1.0})

    # ---- derived -----------------------------------------------------------
    @property
    def p_it_max_kw(self) -> float:
        return self.n_servers * self.p_max_w / 1000.0

    @property
    def p_it_idle_kw(self) -> float:
        return self.n_servers * self.p_idle_w / 1000.0


# ==========================================================================
# Environment
# ==========================================================================
class DCMicrogridEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(
        self,
        cfg: DCConfig | None = None,
        profiles=None,
        seed: int | None = None,
        preference_conditioned: bool = False,
        eval_mode: bool = False,
        start_index: int | None = None,
        profile_kwargs: dict | None = None,
    ):
        super().__init__()
        self.cfg = cfg or DCConfig()
        self.preference_conditioned = preference_conditioned
        self.eval_mode = eval_mode
        self.start_index = start_index
        pk = dict(n_days=365, seed=0, site="bangkok",
                  price_regime="volatile", carbon_regime="hourly",
                  pv_capacity_kw=self.cfg.pv_capacity_kw)
        pk.update(profile_kwargs or {})
        self.df = profiles if profiles is not None else make_profiles(**pk)
        self._arr = {c: self.df[c].to_numpy(dtype=np.float64) for c in self.df.columns}
        self._n = len(self.df)
        self.ep_len = self.cfg.episode_days * STEPS_PER_DAY

        self.action_space = spaces.Box(-1.0, 1.0, shape=(3,), dtype=np.float32)
        obs_dim = self._obs_dim()
        self.observation_space = spaces.Box(-10.0, 10.0, shape=(obs_dim,), dtype=np.float32)
        self._rng = np.random.default_rng(seed)
        self.w = np.array([self.cfg.w_cost, self.cfg.w_carbon,
                           self.cfg.w_sla, self.cfg.w_thermal])

    # ---------------------------------------------------------------- spaces
    def _obs_dim(self) -> int:
        # 5 time + 5 exogenous-now + 2 horizons x 3 signals + 6 internal
        base = 5 + 5 + 2 * len(self.cfg.forecast_h) * 3 + 6
        return base + (4 if self.preference_conditioned else 0)

    # ---------------------------------------------------------------- reset
    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        c = self.cfg
        max_start = self._n - self.ep_len - int(max(c.forecast_h) / DT_H) - 1
        if self.start_index is not None:
            self.t0 = int(self.start_index)
        elif self.eval_mode:
            self.t0 = 0
        else:
            self.t0 = int(self._rng.integers(0, max_start) // STEPS_PER_DAY * STEPS_PER_DAY)
        self.k = 0
        self.soc = c.soc_init
        self.t_room = c.t_room_init
        self.peak_kw = 0.0
        # backlog: deque of [amount (util*h), deadline step index]
        self.backlog: deque[list[float]] = deque()
        self.sla_missed = 0.0
        if self.preference_conditioned:
            self.w = self._rng.dirichlet(np.ones(4))
        self.log: list[dict] = []
        return self._obs(), {}

    # ---------------------------------------------------------------- step
    def step(self, action):
        c = self.cfg
        a = np.clip(np.asarray(action, dtype=np.float64), -1.0, 1.0)
        i = self.t0 + self.k

        phi = 0.5 * (a[0] + 1.0)                                  # [0, 1]
        if c.sla_floor and self.backlog:
            total_bk = sum(b[0] for b in self.backlog)
            soon = sum(b[0] for b in self.backlog
                       if (b[1] - self.k) * DT_H <= c.deadline_h * 0.34)
            phi = max(phi, min(soon / max(total_bk, 1e-9), 1.0))
        p_bess_cmd = a[1] * c.bess_power_kw
        t_sup = c.t_sup_min + 0.5 * (a[2] + 1.0) * (c.t_sup_max - c.t_sup_min)
        t_sup_raw = t_sup

        # ---- 1. workload -------------------------------------------------
        u_off = self._arr["u_demand"][i]
        u_now = c.alpha_interactive * u_off
        arrival = (1.0 - c.alpha_interactive) * u_off * DT_H       # util*h of deferrable work
        self.backlog.append([arrival, self.k + int(c.deadline_h / DT_H)])

        backlog_total = sum(b[0] for b in self.backlog)
        headroom = max(c.u_cap - u_now, 0.0) * DT_H               # util*h that can be run now
        served = min(phi * backlog_total, headroom, backlog_total)
        # FIFO drain (earliest deadline first)
        rem = served
        while rem > 1e-12 and self.backlog:
            take = min(self.backlog[0][0], rem)
            self.backlog[0][0] -= take
            rem -= take
            if self.backlog[0][0] <= 1e-12:
                self.backlog.popleft()

        # deadline misses: work still queued past its deadline is dropped and penalised
        missed = 0.0
        while self.backlog and self.backlog[0][1] <= self.k:
            missed += self.backlog.popleft()[0]
        self.sla_missed += missed

        u_exec = np.clip(u_now + served / DT_H, 0.0, c.u_cap)

        # ---- 2. IT power -------------------------------------------------
        p_it = (c.p_it_idle_kw
                + (c.p_it_max_kw - c.p_it_idle_kw) * self._spec_curve(u_exec))
        p_it += c.p_aux_frac * c.p_it_max_kw

        # ---- 3. thermal --------------------------------------------------
        # C dT/dt = P_IT - k (T_room - T_sup)   ->  exact exponential integration
        tau = c.c_room_kwh_per_k / c.k_cool_kw_per_k
        decay = np.exp(-DT_H / tau)
        t_room_start = self.t_room

        # Safety layer, part 1 -- project T_sup onto the set that keeps the
        # one-step-ahead room temperature inside the ASHRAE envelope.
        if c.safety_layer:
            t_sup_feas = ((c.t_room_max - self.t_room * decay) / (1.0 - decay)
                          - p_it / c.k_cool_kw_per_k)
            t_sup = float(np.clip(min(t_sup, t_sup_feas), c.t_sup_min, c.t_sup_max))

        t_inf = t_sup + p_it / c.k_cool_kw_per_k
        t_room_next = t_inf + (self.t_room - t_inf) * decay
        t_room_mean = t_inf + (self.t_room - t_inf) * tau / DT_H * (1.0 - decay)

        # ---- 4. cooling power --------------------------------------------
        t_wb = self._arr["t_wb"][i]
        free = t_wb < (t_sup - c.econ_delta_t)
        if free:
            cop = c.cop_free
        else:
            cop = c.cop_nom * (1.0 + c.cop_gamma * (t_sup - c.t_sup_ref)) \
                            * (1.0 - c.cop_delta * (t_wb - c.t_wb_ref))
            cop = float(np.clip(cop, c.cop_min, c.cop_max))
        p_fan = c.fan_frac * c.p_it_max_kw
        p_cool = p_it / cop + p_fan

        # ---- 5. battery (structurally clipped to SoC and power limits) ----
        p_load = p_it + p_cool
        p_pv = self._arr["pv_kw"][i]

        # Safety layer, part 2 -- never charge so hard that the point-of-common-
        # coupling import limit is breached.
        if c.safety_layer and p_bess_cmd < 0:
            headroom_kw = c.pcc_limit_kw - (p_load - p_pv)
            p_bess_cmd = -min(-p_bess_cmd, max(headroom_kw, 0.0))

        if p_bess_cmd >= 0:   # discharge
            e_avail = (self.soc - c.soc_min) * c.bess_energy_kwh * c.eta_dis
            p_dis = min(p_bess_cmd, c.bess_power_kw, e_avail / DT_H)
            p_ch = 0.0
            self.soc -= p_dis * DT_H / (c.eta_dis * c.bess_energy_kwh)
        else:                 # charge
            e_room = (c.soc_max - self.soc) * c.bess_energy_kwh / c.eta_ch
            p_ch = min(-p_bess_cmd, c.bess_power_kw, e_room / DT_H)
            p_dis = 0.0
            self.soc += p_ch * DT_H * c.eta_ch / c.bess_energy_kwh
        self.soc = float(np.clip(self.soc, c.soc_min, c.soc_max))
        throughput = (p_ch + p_dis) * DT_H

        # ---- 6. grid -----------------------------------------------------
        p_grid = p_load - p_pv + p_ch - p_dis
        imp = max(p_grid, 0.0)
        exp = max(-p_grid, 0.0)
        peak_inc = max(0.0, imp - self.peak_kw)          # kW added to the running peak
        self.peak_kw = max(self.peak_kw, imp)

        price = self._arr["price"][i]
        ci = self._arr["carbon"][i]
        cost = (price * imp * DT_H
                - price * c.export_price_frac * exp * DT_H
                + c.degrade_cost * throughput)
        carbon = ci * imp * DT_H

        # ---- 7. objectives and constraints -------------------------------
        thermal_exc = max(0.0, t_room_mean - c.t_room_max)
        sla_pen = missed + 0.02 * sum(b[0] for b in self.backlog)   # miss + queueing
        pcc_exc = max(0.0, imp - c.pcc_limit_kw)

        j = np.array([cost, carbon, sla_pen, thermal_exc ** 2])
        nrm = np.array([self.cfg.norm["cost"], self.cfg.norm["carbon"],
                        self.cfg.norm["sla"], self.cfg.norm["thermal"]])
        reward = float(-np.dot(self.w, j / np.maximum(nrm, 1e-9)))
        if c.dense_peak:
            # incremental demand charge; telescopes to the lump sum below
            reward -= self.w[0] * (c.peak_charge * peak_inc / 30.0 * c.episode_days) \
                / max(self.cfg.norm["cost"], 1e-9)

        self.t_room = float(t_room_next)
        self.k += 1
        terminated = False
        truncated = self.k >= self.ep_len

        # Episode-end accounting.
        #  (a) demand charge on the episode peak.
        #  (b) TERMINAL STORAGE VALUE. Without this, a policy that simply drains
        #      the battery from soc_init to soc_min collects
        #      (soc_init - soc_min) * E_cap kWh of free energy every episode,
        #      because reset() restores the SoC. Charging the agent (or crediting
        #      it) for the change in stored energy at the episode-mean price
        #      removes that artefact and makes short episodes unbiased.
        if truncated:
            peak_cost = c.peak_charge * self.peak_kw / 30.0 * c.episode_days
            price_ref = float(np.mean(self._arr["price"][self.t0: self.t0 + self.ep_len]))
            soc_cost = (c.soc_init - self.soc) * c.bess_energy_kwh * price_ref
            backlog_cost = 0.0
            if c.terminal_backlog:
                leftover = sum(b[0] for b in self.backlog)                 # util*h
                slope = (1.0 - 0.5704) / 0.5                               # SPECpower upper segment
                backlog_cost = leftover * (c.p_it_max_kw - c.p_it_idle_kw) * slope * price_ref
            cost += peak_cost + soc_cost + backlog_cost
            # the reported cost always carries the full charge; the reward only
            # carries it here when it was not already paid incrementally
            lump = soc_cost + backlog_cost + (0.0 if c.dense_peak else peak_cost)
            reward -= self.w[0] * lump / max(self.cfg.norm["cost"], 1e-9)
            rec_extra = dict(peak_cost=peak_cost, soc_cost=soc_cost, backlog_cost=backlog_cost)
        else:
            rec_extra = dict(peak_cost=0.0, soc_cost=0.0, backlog_cost=0.0)

        rec = dict(
            t_room_start=float(t_room_start),
            step=self.k, u_off=u_off, u_exec=u_exec, backlog=sum(b[0] for b in self.backlog),
            missed=missed, p_it=p_it, p_cool=p_cool, p_pv=p_pv, p_ch=p_ch, p_dis=p_dis,
            p_grid=p_grid, imp=imp, exp=exp, soc=self.soc, t_room=t_room_mean,
            t_sup=t_sup, t_sup_raw=t_sup_raw, safety_int=float(t_sup_raw - t_sup > 1e-6),
            t_amb=self._arr["t_amb"][i], t_wb=t_wb, cop=cop, free=float(free),
            price=price, carbon_int=ci, cost=cost, carbon=carbon,
            sla=sla_pen, thermal_exc=thermal_exc, pcc_exc=pcc_exc,
            pue=(p_it + p_cool) / max(p_it, 1e-6), reward=reward, **rec_extra,
        )
        self.log.append(rec)

        info = dict(rec)
        info["cost_thermal"] = thermal_exc          # CMDP constraint costs
        info["cost_sla"] = missed
        info["cost_pcc"] = pcc_exc
        info["objectives"] = j
        if truncated:
            info["episode_summary"] = self.summary()
        return self._obs(), reward, terminated, truncated, info

    # ---------------------------------------------------------------- helpers
    def _spec_curve(self, u: float) -> float:
        """Normalised SPECpower shape: strongly convex-then-linear, matches the
        measured 0/10/50/100 % points of the reference server."""
        u = float(np.clip(u, 0.0, 1.0))
        # measured watts 77.4 / 145 / 231 / 347 -> normalised (P - Pidle)/(Pmax - Pidle)
        xs = np.array([0.0, 0.10, 0.50, 1.00])
        ys = np.array([0.0, 0.2504, 0.5704, 1.0])
        return float(np.interp(u, xs, ys))

    def _fc(self, key: str, i: int, hours: int) -> float:
        """Mean of `key` over the next `hours`, with multiplicative forecast error."""
        n = max(int(hours / DT_H), 1)
        seg = self._arr[key][i: i + n]
        if seg.size == 0:
            seg = self._arr[key][i: i + 1]
        v = float(seg.mean())
        if self.cfg.forecast_err_std > 0:
            v *= float(np.exp(self._rng.normal(0, self.cfg.forecast_err_std * np.sqrt(hours / 4))))
        return v

    def _obs(self) -> np.ndarray:
        c = self.cfg
        i = min(self.t0 + self.k, self._n - 1)
        tod = (i % STEPS_PER_DAY) / STEPS_PER_DAY
        doy = ((i // STEPS_PER_DAY) % 365) / 365.0
        dow = float(((i // STEPS_PER_DAY) % 7) >= 5)
        o = [
            np.sin(2 * np.pi * tod), np.cos(2 * np.pi * tod),
            np.sin(2 * np.pi * doy), np.cos(2 * np.pi * doy), dow,
            self._arr["pv_kw"][i] / max(c.pv_capacity_kw, 1e-6),
            (self._arr["t_amb"][i] - 25.0) / 10.0,
            (self._arr["t_wb"][i] - 20.0) / 10.0,
            self._arr["price"][i] / 0.15,
            self._arr["carbon"][i] / 0.5,
        ]
        for h in c.forecast_h:
            o += [self._fc("pv_kw", i, h) / max(c.pv_capacity_kw, 1e-6),
                  self._fc("price", i, h) / 0.15,
                  self._fc("carbon", i, h) / 0.5]
            o += [self._fc("u_demand", i, h),
                  (self._fc("t_wb", i, h) - 20.0) / 10.0,
                  0.0]
        backlog = sum(b[0] for b in self.backlog)
        oldest = (min((b[1] for b in self.backlog), default=self.k) - self.k) * DT_H
        o += [
            (self.soc - 0.5) * 2.0,
            (self.t_room - 24.0) / 5.0,
            self._arr["u_demand"][i],
            backlog / max(c.deadline_h * 0.5, 1e-6),
            oldest / max(c.deadline_h, 1e-6),
            self.peak_kw / max(c.pcc_limit_kw, 1e-6),
        ]
        if self.preference_conditioned:
            o += list(self.w)
        return np.asarray(o, dtype=np.float32)

    # ---------------------------------------------------------------- report
    def summary(self) -> dict:
        import pandas as pd
        d = pd.DataFrame(self.log)
        it = (d.p_it * DT_H).sum()
        cool = (d.p_cool * DT_H).sum()
        return {
            "cost": d.cost.sum(),
            "carbon_kg": d.carbon.sum(),
            "energy_it_kwh": it,
            "energy_cool_kwh": cool,
            "grid_import_kwh": (d.imp * DT_H).sum(),
            "grid_export_kwh": (d.exp * DT_H).sum(),
            "pv_used_kwh": (np.minimum(d.p_pv, d.p_it + d.p_cool + d.p_ch) * DT_H).sum(),
            "pue": (it + cool) / max(it, 1e-9),
            "peak_kw": d.imp.max(),
            "self_consumption": float(
                (np.minimum(d.p_pv, d.p_it + d.p_cool + d.p_ch) * DT_H).sum()
                / max((d.p_pv * DT_H).sum(), 1e-9)),
            "peak_cost": float(d.peak_cost.sum()),
            "terminal_soc_cost": float(d.soc_cost.sum()),
            "terminal_backlog_cost": float(d.backlog_cost.sum()),
            "backlog_final": float(sum(b[0] for b in self.backlog)),
            "soc_final": self.soc,
            "sla_missed": self.sla_missed,
            "thermal_viol_rate": float((d.thermal_exc > 1e-6).mean()),
            "thermal_max_exc": float(d.thermal_exc.max()),
            "pcc_viol_rate": float((d.pcc_exc > 1e-6).mean()),
            "t_room_max": float(d.t_room.max()),
            "free_cooling_frac": float(d.free.mean()),
            "safety_intervention_rate": float(d.safety_int.mean()),
            "reward": d.reward.sum(),
        }


# ==========================================================================
# Normalisation calibration
# ==========================================================================
def calibrate(cfg: DCConfig, profiles=None, n_episodes: int = 20, seed: int = 0) -> DCConfig:
    """Set per-objective normalisation constants so the reward weights are
    comparable across objectives (research_design §2.2).

    cost / carbon : measured from a status-quo (B0) rollout -> a weight of 1.0
                    means "one status-quo episode's worth".
    sla           : total deferrable work arriving per episode (util*h) -> a
                    weight of 1.0 means "the entire flexible workload missed".
    thermal       : a sustained 1 K excursion over the whole episode.

    Physical scales are used for SLA and thermal rather than a policy rollout,
    because with the safety layer on both are ~0 for any sensible policy and a
    rollout-derived constant would be a meaningless near-zero divisor.
    Calibrated once, then frozen for all controllers and all seeds.
    """
    from baselines import StatusQuo

    env = DCMicrogridEnv(cfg=cfg, profiles=profiles, seed=seed)
    pol = StatusQuo(env)
    tot = np.zeros(4)
    for e in range(n_episodes):
        obs, _ = env.reset(seed=seed + e)
        done, acc = False, np.zeros(4)
        while not done:
            obs, _, term, trunc, info = env.step(pol.act(obs, env))
            acc += info["objectives"]
            done = term or trunc
        tot += acc
    m = tot / n_episodes

    u_mean = float(np.mean(env._arr["u_demand"]))
    defer_per_ep = (1.0 - cfg.alpha_interactive) * u_mean * cfg.episode_days * 24.0
    cfg.norm = {
        "cost": max(float(m[0]), 1e-3),
        "carbon": max(float(m[1]), 1e-3),
        "sla": max(defer_per_ep, 1e-3),
        "thermal": float(env.ep_len) * 1.0,
    }
    return cfg
