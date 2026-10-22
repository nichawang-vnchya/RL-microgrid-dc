import pandas as pd, numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
plt.rcParams['font.family'] = 'Liberation Sans'
D = '../data/'; O = 'figures/'
BLUE = '#2E75B6'; ORANGE = '#E07B39'; GREEN = '#2E9E6B'; GREY = '#666666'; DARK = '#1F2937'; LIGHT = '#EEF3F7'; RED = '#C0392B'
SC = {'th': ('Thailand TOU (MEA/PEA 4.2)', 'profiles_th.csv'), 'sg': ('Singapore spot (USEP 2023)', 'profiles_real.csv'), 'ca': ('California spot (CAISO 2023)', 'profiles_ca.csv')}
import shutil; shutil.copy('/mnt/user-data/uploads/Documents/Academic_Paper_2_Conf/dcmg/data/profiles_real.csv', D + 'profiles_real.csv')

# ---------- Fig 1: system architecture and control loop
fig = plt.figure(figsize=(9, 5.2), dpi=300); ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 9); ax.set_ylim(-0.3, 5.2); ax.axis('off')
def box(x, y, w, h, text, fc=LIGHT, ec=DARK, size=9, bold=False, color=DARK):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0,rounding_size=0.08', fc=fc, ec=ec, lw=1.0))
    ax.text(x + w / 2, y + h / 2, text, ha='center', va='center', fontsize=size, weight='bold' if bold else 'normal', color=color, linespacing=1.25)
def arr(x0, y0, x1, y1, color=DARK, style='-|>', lw=1.2):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle=style, mutation_scale=11, color=color, lw=lw))
# plant
ax.add_patch(FancyBboxPatch((0.3, 0.3), 5.0, 3.1, boxstyle='round,pad=0,rounding_size=0.1', fc='white', ec=GREY, lw=1.0, ls='--'))
ax.text(0.45, 3.25, 'Data-center microgrid (Section 2)', fontsize=9.5, weight='bold', color=GREY, va='center')
box(0.5, 2.3, 1.3, 0.6, 'PV generation\n$P_t^{PV}$ (600 kW)', fc='#E3F3EA', ec=GREEN)
box(0.5, 1.45, 1.3, 0.6, 'Battery $E^B$\n1.6 MWh, 800 kW', fc='#E3ECF7', ec=BLUE)
box(0.5, 0.6, 1.3, 0.6, 'Grid import\n$P_t^{imp}\\leq P^{PCC}_{max}$', fc='#EEEEEE', ec=GREY)
box(2.2, 1.45, 1.1, 0.6, 'Bus\nbalance (20)', fc='white')
for yy in (2.6, 1.75, 0.9):
    arr(1.8, yy, 2.2, 1.75, color=GREY)
box(3.7, 2.3, 1.45, 0.6, 'IT load $P_t^{IT}$ (6)\n2,500 servers', fc='#FBE9DD', ec=ORANGE)
box(3.7, 1.45, 1.45, 0.6, 'Cooling $P_t^{cool}$ (14)\nroom $T_t^r$ (10)', fc='#FBE9DD', ec=ORANGE)
box(3.7, 0.6, 1.45, 0.6, 'Workload queue\n$B_t$, deadline 6 h (5)', fc='#FBE9DD', ec=ORANGE)
for yy in (2.6, 1.75, 0.9):
    arr(3.3, 1.75, 3.7, yy, color=GREY)
# controller
box(6.0, 3.6, 2.7, 1.2, 'Controller (Section 3)\nPPO policy $\\pi_\\theta$, or\nreference strategies B0 to B6', fc='#E3ECF7', ec=BLUE, size=9, bold=False)
box(6.0, 2.25, 2.7, 0.8, 'Safety layer (Section 3.2)\ncooling (32)(33), battery (34)', fc='#FBE3E0', ec=RED)
box(6.0, 0.9, 2.7, 0.8, 'Commands applied\n$\\phi_t$, $P_t^{B}$, $T_t^{sup}$', fc='white')
arr(7.35, 3.6, 7.35, 3.05); ax.text(7.45, 3.32, 'requested $a_t$ (25)', fontsize=8, color=GREY, va='center')
arr(7.35, 2.25, 7.35, 1.7); ax.text(7.45, 1.97, 'corrected', fontsize=8, color=GREY, va='center')
arr(6.0, 1.3, 5.15, 1.3, color=DARK)
# observation and reward feedback
arr(5.3, 3.0, 6.0, 4.2, color=BLUE)
ax.text(5.25, 3.75, 'observation $o_t$\n(Table 1)', fontsize=8, color=BLUE, ha='right', va='center')
ax.add_patch(FancyArrowPatch((2.8, 0.3), (2.8, -0.0), arrowstyle='-', color='white'))
arr(2.8, 0.3, 2.8, 0.15, color=ORANGE, style='-'); arr(2.8, 0.15, 8.75, 0.15, color=ORANGE, style='-'); arr(8.75, 0.15, 8.75, 3.6, color=ORANGE)
ax.text(5.5, 0.08, 'cost (23), emissions (24), service: reward $r_t$ (29)', fontsize=8, color=ORANGE, ha='center', va='top')
ax.text(0.45, 4.95, 'Exogenous inputs: workload arrivals, irradiance, weather, price $\\pi_t$, carbon $\\chi_t$ (Table 3)', fontsize=8.5, color=GREY, va='center')
arr(2.9, 4.8, 2.9, 3.45, color=GREY, style='-|>')
fig.savefig(O + 'fig1_architecture.png', bbox_inches='tight', facecolor='white'); plt.close(fig)

# ---------- Fig 3: the three tariff scenarios
fig, axes = plt.subplots(3, 2, figsize=(9, 6.2), dpi=300, gridspec_kw={'width_ratios': [2.6, 1], 'hspace': 0.45, 'wspace': 0.25})
for r, s in enumerate(['th', 'sg', 'ca']):
    name, f = SC[s]; d = pd.read_csv(D + f, index_col=0); d.index = pd.to_datetime(d.index, utc=True)
    a = axes[r, 0]
    if s == 'th':
        dd = d.loc['2023-03-06':'2023-03-19']; a.plot(dd.index, dd.price, color=BLUE, lw=0.8); a.set_ylim(0.07, 0.14)
    else:
        dd = d; a.plot(d.index, d.price, color=BLUE, lw=0.5)
    a.axhline(d.price.mean(), color=ORANGE, lw=1.2, ls='--')
    a.set_title(f'({chr(97+r)}) {name}', fontsize=10, loc='left', weight='bold')
    a.set_ylabel('USD/kWh', fontsize=9); a.tick_params(labelsize=8); a.spines[['top', 'right']].set_visible(False)
    a.text(0.99, 0.95, f'mean {d.price.mean():.3f}, p99/mean {d.price.quantile(.99)/d.price.mean():.1f}', transform=a.transAxes, ha='right', va='top', fontsize=8, color=DARK, bbox=dict(fc='white', ec='none', alpha=0.9))
    if s == 'th':
        a.set_xticks(pd.to_datetime(['2023-03-06', '2023-03-11', '2023-03-13', '2023-03-18'], utc=True).tz_convert('Asia/Bangkok')); a.set_xticklabels(['Mon 6 Mar', 'Sat 11', 'Mon 13', 'Sat 18'])
        a.text(0.01, 0.95, 'two weeks shown; weekends and holidays are off-peak all day', transform=a.transAxes, fontsize=7.5, color=GREY, va='top')
    else:
        a.set_xticks(pd.to_datetime(['2023-01-01', '2023-04-01', '2023-07-01', '2023-10-01', '2024-01-01'], utc=True)); a.set_xticklabels(['Jan', 'Apr', 'Jul', 'Oct', 'Jan'])
    b = axes[r, 1]
    hourly = d.groupby(d.index.tz_convert('Asia/Bangkok' if s == 'th' else ('Asia/Singapore' if s == 'sg' else 'America/Los_Angeles')).hour).price.mean()
    b.bar(hourly.index, hourly.values, color=BLUE, width=0.8); b.set_title('mean daily profile', fontsize=9, loc='left', color=GREY)
    b.set_xlabel('hour', fontsize=8); b.tick_params(labelsize=8); b.spines[['top', 'right']].set_visible(False)
    b.text(0.02, 0.97, f'carbon {d.carbon.min():.3f} to {d.carbon.max():.3f} kgCO$_2$/kWh', transform=b.transAxes, ha='left', va='top', fontsize=7, color=GREEN, bbox=dict(fc='white', ec='none', alpha=0.9))
    b.set_ylim(0, hourly.max() * 1.3)
fig.savefig(O + 'fig3_tariffs.png', bbox_inches='tight', facecolor='white'); plt.close(fig)

# ---------- Fig 4: regime map (regenerated, no overlap)
rt = pd.read_csv(D + 'regime_table.csv')
fig, (a, b) = plt.subplots(1, 2, figsize=(9, 3.8), dpi=300, gridspec_kw={'wspace': 0.3})
order = ['SG', 'TH', 'CA']; lab = {'SG': 'Singapore\nspot (USEP)', 'TH': 'Thailand\nTOU (MEA/PEA 4.2)', 'CA': 'California\nspot (CAISO)'}
x = np.arange(3); w = 0.26
for i, (col, c, l) in enumerate([('rule_vs_b0', BLUE, 'best rule'), ('ppo_vs_b0', ORANGE, 'safe PPO'), ('oracle_vs_b0', GREEN, 'perfect-foresight benchmark')]):
    v = [float(rt[rt.arm == k][col].iloc[0]) for k in order]
    bars = a.bar(x + (i - 1) * w, v, w, color=c, label=l)
    for xx, vv in zip(x + (i - 1) * w, v):
        a.text(xx, vv + (0.3 if vv > 0 else -0.3), f'{vv:+.1f}', ha='center', va='bottom' if vv > 0 else 'top', fontsize=7.5)
a.axhline(0, color='black', lw=0.8); a.set_xticks(x); a.set_xticklabels([lab[k] for k in order], fontsize=8.5)
a.set_ylabel('30-day bill vs status quo (%)', fontsize=9); a.legend(fontsize=8, frameon=False, loc='upper left'); a.set_ylim(-12.5, 10)
a.set_title('(a) What each controller saves', fontsize=10, loc='left', weight='bold'); a.spines[['top', 'right']].set_visible(False); a.tick_params(labelsize=8)
for k in order:
    r = rt[rt.arm == k].iloc[0]; c = ORANGE if r.ppo_vs_rule < 0 else BLUE
    b.scatter(r.peak_share * 100, r.price_tail, s=260, color=c, zorder=3)
    dy = {'SG': 0.0, 'TH': 0.0, 'CA': 0.0}[k]; dx = 1.4
    b.text(r.peak_share * 100 + dx, r.price_tail + dy, f'{k}\nPPO vs rule {r.ppo_vs_rule:+.1f}%', fontsize=8, va='center')
b.set_xlim(0, 32); b.set_ylim(0, 9.5); b.set_xlabel('demand charge, share of status-quo bill (%)', fontsize=9); b.set_ylabel('price tail (p99 / mean)', fontsize=9)
b.set_title('(b) Where learning pays', fontsize=10, loc='left', weight='bold'); b.spines[['top', 'right']].set_visible(False); b.tick_params(labelsize=8); b.grid(color='#E5E5E5', lw=0.5)
b.text(0.98, 0.97, 'orange: PPO beats the best rule\nblue: rule is cheaper', transform=b.transAxes, ha='right', va='top', fontsize=7.5, color=GREY)
fig.savefig(O + 'fig4_regime_map.png', bbox_inches='tight', facecolor='white'); plt.close(fig)

# ---------- Fig 6: benchmark daily saving distribution
fig, axes = plt.subplots(1, 3, figsize=(9, 2.9), dpi=300, sharey=False, gridspec_kw={'wspace': 0.3})
for a, s in zip(axes, ['th', 'sg', 'ca']):
    o = pd.read_csv(D + f'oracle_{s}.csv'); g = (o.B0_cost - o.cost) / o.B0_cost * 100
    a.hist(g, bins=np.arange(0, 36, 1.5), color=GREEN, alpha=0.85)
    a.axvline(g.median(), color=DARK, lw=1.2, ls='--'); a.text(g.median() + 0.5, a.get_ylim()[1] * 0.92, f'median {g.median():.1f}%', fontsize=8, va='top')
    a.set_title(f'({chr(97+["th","sg","ca"].index(s))}) {SC[s][0].split(" (")[0]}', fontsize=9.5, loc='left', weight='bold')
    a.set_xlabel('attainable daily saving vs B0 (%)', fontsize=8.5); a.tick_params(labelsize=8); a.spines[['top', 'right']].set_visible(False)
    a.text(0.98, 0.7, f'range {g.min():.1f} to {g.max():.1f}%\n{(g>5).sum()}/30 days above 5%', transform=a.transAxes, ha='right', va='top', fontsize=8, color=GREY)
axes[0].set_ylabel('test days', fontsize=9)
fig.savefig(O + 'fig6_benchmark_distribution.png', bbox_inches='tight', facecolor='white'); plt.close(fig)

# ---------- Fig 7: safety ablation (Singapore, reference curve)
fig, (a, b) = plt.subplots(1, 2, figsize=(9, 3.2), dpi=300, gridspec_kw={'wspace': 0.35})
ctrl = ['PPO', 'SAC', 'MPC-4h']; with_ = [-3.33, 5.76, 0.56 * 0 + np.nan]; without = [2.70, np.nan, np.nan]
# values from the matched ablation: PPO with layer -3.33, without +2.70; MPC 4.7% cheaper with layer
vals_with = {'PPO': -3.33, 'MPC-4h': 0.0}; vals_wo = {'PPO': 2.70, 'MPC-4h': 4.7}
x = np.arange(2); w = 0.35
a.bar(x - w / 2, [vals_with['PPO'], vals_with['MPC-4h']], w, color=BLUE, label='with safety layer')
a.bar(x + w / 2, [vals_wo['PPO'], vals_wo['MPC-4h']], w, color=RED, label='without safety layer')
for i, (k) in enumerate(['PPO', 'MPC-4h']):
    a.text(i - w / 2, vals_with[k] + (0.2 if vals_with[k] >= 0 else -0.2), f'{vals_with[k]:+.1f}', ha='center', va='bottom' if vals_with[k] >= 0 else 'top', fontsize=8)
    a.text(i + w / 2, vals_wo[k] + 0.2, f'{vals_wo[k]:+.1f}', ha='center', va='bottom', fontsize=8)
a.axhline(0, color='black', lw=0.8); a.set_xticks(x); a.set_xticklabels(['PPO (vs B0)', 'MPC-4h (vs with layer)'], fontsize=8.5); a.set_ylabel('cost change (%)', fontsize=9)
a.set_title('(a) Cost effect of the safety layer', fontsize=10, loc='left', weight='bold'); a.legend(fontsize=8, frameon=False); a.spines[['top', 'right']].set_visible(False); a.tick_params(labelsize=8); a.set_ylim(-4.5, 6.5)
names = ['PPO\nwith layer', 'PPO\nwithout', 'SAC\nwithout']; viol = [0.0, 2.1, 16.2]; tmax = [27.0, 30.0, 30.4]
b.bar(names, viol, color=[BLUE, RED, RED], width=0.55)
for i, (v, t) in enumerate(zip(viol, tmax)):
    b.text(i, v + 0.4, f'{v:.1f}% of steps\nmax {t:.1f} °C', ha='center', va='bottom', fontsize=8)
b.set_ylabel('thermal-limit violations (% of steps)', fontsize=9); b.set_ylim(0, 21); b.tick_params(labelsize=8); b.spines[['top', 'right']].set_visible(False)
b.set_title('(b) Violations of the 27 °C limit', fontsize=10, loc='left', weight='bold')
fig.savefig(O + 'fig7_safety_ablation.png', bbox_inches='tight', facecolor='white'); plt.close(fig)

# ---------- Fig 8: running mean of the paired PPO difference, three scenarios, 5 seeds
fig, axes = plt.subplots(1, 3, figsize=(9, 2.9), dpi=300, gridspec_kw={'wspace': 0.3})
for a, s in zip(axes, ['th', 'sg', 'ca']):
    d = pd.read_csv(D + f'raw_{s}.csv'); b0 = d[d.controller == 'B0 status-quo'].set_index('episode').cost
    ppo = d[d.controller == 'R1 PPO'].groupby('episode').cost.mean(); delta = ((ppo - b0) / b0 * 100).sort_index(); run = delta.expanding().mean(); n = np.arange(1, len(delta) + 1)
    a.bar(delta.index + 1, delta.values, color=np.where(delta.values < 0, BLUE, ORANGE), alpha=0.35, width=0.8)
    a.plot(n, run.values, color=DARK, lw=1.8); a.axhline(0, color='black', lw=0.7); a.axvline(20, color=GREY, lw=0.8, ls='--')
    a.text(29.5, run.iloc[-1], f'{run.iloc[-1]:+.2f}%', fontsize=8, ha='right', va='bottom' if run.iloc[-1] < 0 else 'top', weight='bold')
    a.set_title(f'({chr(97+["th","sg","ca"].index(s))}) {SC[s][0].split(" (")[0]}', fontsize=9.5, loc='left', weight='bold')
    a.set_xlabel('held-out days included', fontsize=8.5); a.tick_params(labelsize=8); a.spines[['top', 'right']].set_visible(False)
axes[0].set_ylabel('PPO cost vs B0 (%)', fontsize=9)
fig.savefig(O + 'fig8_convergence.png', bbox_inches='tight', facecolor='white'); plt.close(fig)
shutil.copy(D + 'figD_daily_cost.png', O + 'fig5_daily_cost.png'); shutil.copy(D + 'figE_lanta_validation.png', O + 'fig2_lanta_calibration.png')
print('ok')
