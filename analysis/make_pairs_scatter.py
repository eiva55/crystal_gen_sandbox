"""Рисунок «пара — одна точка» для пар test и val (K=10): DiffCSP++ и CrystaLLM-small рядом.

Каждая точка — пара полиморфов одного состава: по горизонтали удачных попыток метастабильного из 10, по вертикали —
стабильного; цвет и форма — разность сложности шаблонов Δn_своб = n_своб(метастабильный) − n_своб(стабильный).
Входы: analysis/results/pairs_diffcsp_tv_k10_per_pair.csv, pairs_sg_tv_k10_per_pair.csv (analyze_pairs_v1.py).
Для CrystaLLM-small показаны те же пары, что у DiffCSP++ (оба полиморфа с n_своб ≥ 1).

Запуск из корня репозитория: python analysis/make_pairs_scatter.py
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

R, OUT = 'analysis/results', 'docs/figures'
INK, INK2, GRID, SURF = '#0b0b0b', '#52514e', '#e6e5e1', '#ffffff'
BLUE, ORANGE, GREEN = '#2a78d6', '#eb6834', '#1baf7a'
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 11, 'text.color': INK, 'axes.labelcolor': INK2,
                     'xtick.color': INK2, 'ytick.color': INK2, 'axes.edgecolor': GRID, 'figure.facecolor': SURF,
                     'axes.facecolor': SURF, 'savefig.facecolor': SURF})
STR = [('метастабильный сложнее (Δn > 2)', lambda d: d.dn > 2, ORANGE, '^'),
       ('сопоставимые (|Δn| ≤ 2)', lambda d: d.dn.abs() <= 2, GREEN, 's'),
       ('метастабильный проще (Δn < −2)', lambda d: d.dn < -2, BLUE, 'o')]

dc = pd.read_csv(f'{R}/pairs_diffcsp_tv_k10_per_pair.csv')
cl = pd.read_csv(f'{R}/pairs_sg_tv_k10_per_pair.csv')
cl = cl.merge(dc[['id_s', 'id_m']], on=['id_s', 'id_m'])
fig, axs = plt.subplots(1, 2, figsize=(11.2, 5.9))
for ax, d, title in ((axs[0], dc, f'DiffCSP++, полный шаблон ({len(dc)} пар)'),
                     (axs[1], cl, f'CrystaLLM-small, состав и группа ({len(cl)} пар)')):
    d = d.assign(dn=d.nf_m - d.nf_s)
    rng = np.random.default_rng(1)
    ax.plot([0, 10], [0, 10], color=INK2, lw=1.1, ls=(0, (4, 3)))
    for lab, sel, c, mk in STR:
        q = d[sel(d)]
        ax.scatter(q.k_m + rng.uniform(-0.22, 0.22, len(q)), q.k_s + rng.uniform(-0.22, 0.22, len(q)), s=30, color=c,
                   marker=mk, alpha=0.85, edgecolor=SURF, linewidth=0.7, label=f'{lab}: {len(q)}', zorder=3)
    above, below = int((d.k_s > d.k_m).sum()), int((d.k_s < d.k_m).sum())
    ax.text(0.4, 9.6, f'стабильный удачнее: {above}', fontsize=9.5, color=INK2, va='top')
    ax.text(9.6, 0.4, f'метастабильный удачнее: {below}', fontsize=9.5, color=INK2, ha='right')
    ax.set_xlim(-0.6, 10.6); ax.set_ylim(-0.6, 10.6); ax.set_aspect('equal')
    ax.set_xticks(range(0, 11, 2)); ax.set_yticks(range(0, 11, 2))
    ax.set_xlabel('удачных из 10: метастабильный полиморф'); ax.set_ylabel('удачных из 10: стабильный полиморф')
    ax.set_title(title, fontsize=11.5)
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
    ax.grid(color=GRID, lw=0.8); ax.set_axisbelow(True); ax.tick_params(length=0)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=1, frameon=False, fontsize=9, labelcolor=INK2)
fig.tight_layout()
fig.savefig(f'{OUT}/fig19_pairs_scatter_tv_k10.png', dpi=160, bbox_inches='tight')
print('ok: docs/figures/fig19_pairs_scatter_tv_k10.png')
