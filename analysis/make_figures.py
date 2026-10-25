"""Графики для файла гипотез (docs/figures/). Запуск из корня репозитория: python analysis/make_figures.py
Данные: рис. 5 — analysis/results/pairs_pub_per_pair.csv; рис. 1-4 — сводные числа K=10 из analysis/results/analyze_pairs_sg_k10.txt и K=20 из analyze_pairs_v1_pub.txt (вписаны вручную)."""
import os
os.makedirs("docs/figures", exist_ok=True)
import numpy as np, pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

OUT = 'docs/figures'
R = 'analysis/results'
INK, INK2, GRID, SURF = '#0b0b0b', '#52514e', '#e6e5e1', '#ffffff'
BLUE, ORANGE, VIOLET, GRAY, LGRAY = '#2a78d6', '#eb6834', '#6250d6', '#8a8985', '#d9d8d3'
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 11, 'text.color': INK, 'axes.labelcolor': INK2,
                     'xtick.color': INK2, 'ytick.color': INK2, 'axes.edgecolor': GRID, 'figure.facecolor': SURF,
                     'axes.facecolor': SURF, 'savefig.facecolor': SURF})
fmt = lambda v, d=3: f'{v:+.{d}f}'.replace('.', ',').replace('-', '−')


def clean(ax, grid='x'):
    for s in ('top', 'right', 'left'):
        ax.spines[s].set_visible(False)
    ax.tick_params(length=0)
    ax.grid(axis=grid, color=GRID, lw=0.8)
    ax.set_axisbelow(True)


# ---------- рис. 1: «десять попыток» на пальцах ----------
fig, ax = plt.subplots(figsize=(8.6, 2.9))
ax.set_xlim(-0.3, 14.2); ax.set_ylim(-0.2, 3.6); ax.axis('off')
rows = [('Стабильная структура', 0.569, BLUE, 2.0), ('Метастабильная структура', 0.472, ORANGE, 0.7)]
for name, rate, col, y in rows:
    ax.text(-0.3, y + 0.78, name, fontsize=12, color=INK, fontweight='bold', va='center')
    for i in range(10):
        ax.add_patch(Rectangle((i * 1.12, y - 0.1), 1.0, 0.7, fc=SURF, ec=LGRAY, lw=1.2))
        fill = max(0.0, min(1.0, rate * 10 - i))
        if fill > 0:
            ax.add_patch(Rectangle((i * 1.12, y - 0.1), 1.0 * fill, 0.7, fc=col, ec='none'))
    ax.text(11.5, y + 0.25, f'в среднем {rate * 10:.1f} из 10'.replace('.', ','), fontsize=12, color=INK, va='center')
ax.text(-0.3, 3.55, 'Одна цель, десять попыток генерации: закрашено — сколько в среднем совпало с целью', fontsize=11,
        color=INK2, va='top')
fig.savefig(f'{OUT}/fig1_ten_attempts.png', dpi=170, bbox_inches='tight'); plt.close(fig)

# ---------- рис. 2: чем заканчиваются пары ----------
cats = ['оба полиморфа', 'только стабильный', 'только метастабильный', 'ни один']
colors = [GRAY, BLUE, ORANGE, LGRAY]
data = {'только состав,\n20 попыток': [76, 44, 17, 35], 'состав и группа,\n10 попыток': [90, 33, 21, 28]}
fig, ax = plt.subplots(figsize=(8.6, 2.6))
for r, (lab, vals) in enumerate(data.items()):
    left = 0
    for v, c, cn in zip(vals, colors, cats):
        ax.barh(r, v - 1, left=left, color=c, height=0.55)
        ax.text(left + (v - 1) / 2, r, str(v), ha='center', va='center', fontsize=11,
                color=SURF if c in (GRAY, BLUE, ORANGE) else INK)
        left += v
ax.set_yticks(range(2)); ax.set_yticklabels(list(data), color=INK); ax.invert_yaxis()
ax.set_xlim(0, 172); ax.set_xlabel('число пар (всего 172); цвет — у какой структуры пары была хотя бы одна удача')
clean(ax, 'x')
handles = [Rectangle((0, 0), 1, 1, color=c) for c in colors]
ax.legend(handles, cats, ncol=4, loc='upper center', bbox_to_anchor=(0.5, 1.28), frameon=False, fontsize=10, labelcolor=INK2)
fig.savefig(f'{OUT}/fig2_pair_outcomes.png', dpi=170, bbox_inches='tight'); plt.close(fig)

# ---------- рис. 3: разрыв по стратам сложности ----------
strata = [('все пары (172)', 0.098, 0.002, 0.195, 0.073, -0.005, 0.146),
          ('метастабильный сложнее (60)', 0.345, 0.198, 0.487, 0.249, 0.147, 0.353),
          ('сложность одинакова (40)', 0.185, 0.075, 0.303, 0.070, -0.049, 0.185),
          ('метастабильный проще (72)', -0.157, -0.294, -0.023, -0.072, -0.196, 0.056),
          ('сопоставимые, разность ≤ 2 (90)', 0.050, -0.057, 0.148, 0.007, -0.090, 0.102)]
fig, ax = plt.subplots(figsize=(8.6, 4.0))
for i, (lab, m, lo, hi, m0, lo0, hi0) in enumerate(strata):
    ax.plot([lo0, hi0], [i + 0.17, i + 0.17], color=GRAY, lw=2, solid_capstyle='round')
    ax.plot(m0, i + 0.17, 'o', color=GRAY, ms=7, mec=SURF, mew=1.5)
    ax.plot([lo, hi], [i - 0.17, i - 0.17], color=VIOLET, lw=2, solid_capstyle='round')
    ax.plot(m, i - 0.17, 'o', color=VIOLET, ms=7, mec=SURF, mew=1.5)
    ax.text(0.585, i - 0.17, f'{fmt(m)}', va='center', fontsize=10, color=INK)
    ax.text(0.585, i + 0.17, f'{fmt(m0)}', va='center', fontsize=10, color=INK2)
ax.axvline(0, color=INK2, lw=1.2)
ax.set_yticks(range(len(strata))); ax.set_yticklabels([s[0] for s in strata], color=INK); ax.invert_yaxis()
ax.set_xlim(-0.4, 0.68)
ax.set_xlabel('← метастабильный удаётся чаще          разность доли удачных попыток (стабильный − метастабильный)          стабильный удаётся чаще →', fontsize=9)
clean(ax, 'x')
ax.plot([], [], 'o-', color=VIOLET, label='состав и группа, 10 попыток'); ax.plot([], [], 'o-', color=GRAY, label='только состав, 20 попыток')
ax.legend(loc='upper center', bbox_to_anchor=(0.45, 1.13), ncol=2, frameon=False, fontsize=10, labelcolor=INK2)
fig.savefig(f'{OUT}/fig3_gap_by_complexity.png', dpi=170, bbox_inches='tight'); plt.close(fig)

# ---------- рис. 4: наклон по энергии внутри состава ----------
slopes = [('только состав, 20 попыток\nполные контроли', -0.0140, -0.0282, 0.0002, GRAY),
          ('состав и группа, 10 попыток\nбез контролей', -0.0180, -0.0358, -0.0002, VIOLET),
          ('состав и группа, 10 попыток\nчисло свободных координат и атомов', -0.0281, -0.0451, -0.0110, VIOLET),
          ('состав и группа, 10 попыток\nполные контроли', -0.0234, -0.0362, -0.0106, VIOLET)]
fig, ax = plt.subplots(figsize=(8.6, 3.9))
ax.axvspan(-0.0216, 0.0216, color='#e8f0fb', zorder=0)
ax.text(0, -0.62, 'полоса «эффекта практически нет» (±0,0216)', ha='center', va='center', fontsize=9.5, color=INK2, bbox=dict(fc='#e8f0fb', ec='none', pad=3), zorder=5)
for i, (lab, m, lo, hi, c) in enumerate(slopes):
    ax.plot([lo, hi], [i, i], color=c, lw=2.2, solid_capstyle='round')
    ax.plot(m, i, 'o', color=c, ms=7.5, mec=SURF, mew=1.5)
    ax.text(0.034, i, fmt(m, 4), va='center', fontsize=10, color=INK)
ax.axvline(0, color=INK2, lw=1.2)
ax.set_yticks(range(len(slopes))); ax.set_yticklabels([s[0] for s in slopes], color=INK, fontsize=10)
ax.set_ylim(len(slopes) - 0.5, -1.0)
ax.set_xlim(-0.05, 0.05)
ax.set_xlabel('изменение доли удачных попыток на каждые +10 мэВ/атом (90%-й интервал)', fontsize=10)
clean(ax, 'x')
fig.savefig(f'{OUT}/fig4_slope_within_composition.png', dpi=170, bbox_inches='tight'); plt.close(fig)

# ---------- рис. 5: каждая пара — точка (только состав, 20 попыток) ----------
p = pd.read_csv(f'{R}/pairs_pub_per_pair.csv')
rng = np.random.default_rng(1)
colmap = {'оба': GRAY, 'только стабильный': BLUE, 'только метастабильный': ORANGE, 'ни один': LGRAY}
fig, ax = plt.subplots(figsize=(5.4, 5.2))
ax.plot([0, 20], [0, 20], color=INK2, lw=1.1, ls=(0, (4, 3)))
for cat, c in colmap.items():
    d = p[p.category == cat.replace('только стабильный', 'только стабильный')]
    ax.scatter(d.k_m + rng.uniform(-0.25, 0.25, len(d)), d.k_s + rng.uniform(-0.25, 0.25, len(d)), s=26, color=c,
               alpha=0.9, edgecolor=SURF, linewidth=0.7, label=cat, zorder=3)
ax.set_xlim(-0.8, 20.8); ax.set_ylim(-0.8, 20.8); ax.set_aspect('equal')
ax.set_xlabel('удачных попыток из 20: метастабильный полиморф'); ax.set_ylabel('удачных попыток из 20: стабильный')
clean(ax, 'both')
ax.spines['left'].set_visible(False)
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.14), ncol=2, frameon=False, fontsize=9.5, labelcolor=INK2)
fig.savefig(f'{OUT}/fig5_pairs_scatter.png', dpi=170, bbox_inches='tight'); plt.close(fig)
print('ok')
