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


# ---------- рис. 6: дизайн сравнения пары полиморфов (гипотеза 1) ----------
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

def box(ax, x, y, w, h, text, fc=SURF, ec=GRAY, fs=9.5, bold=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.02,rounding_size=0.12', fc=fc, ec=ec, lw=1.4))
    ax.text(x + w / 2, y + h / 2, text, ha='center', va='center', fontsize=fs, color=INK, fontweight='bold' if bold else 'normal', linespacing=1.35)

def arrow(ax, x0, y0, x1, y1):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle='-|>', mutation_scale=14, color=INK2, lw=1.3))

fig, ax = plt.subplots(figsize=(10.2, 4.0))
ax.set_xlim(0, 20.4); ax.set_ylim(0, 8); ax.axis('off')
box(ax, 0.2, 3.2, 3.4, 1.6, 'Один химический\nсостав', fc='#f3f2ee', bold=True)
box(ax, 4.6, 5.0, 4.0, 2.0, 'Стабильный полиморф\n(над оболочкой\nменее 5 мэВ/атом)', ec=BLUE)
box(ax, 4.6, 1.0, 4.0, 2.0, 'Метастабильный\nполиморф\nтого же состава', ec=ORANGE)
box(ax, 9.8, 3.0, 3.6, 2.0, 'Запрос модели:\nсостав + группа\nцели', fc='#f3f2ee')
box(ax, 14.4, 3.0, 2.6, 2.0, '10 попыток\nна каждую\nструктуру')
box(ax, 17.7, 3.0, 2.5, 2.0, 'Сравнение\nс целью\n(0,2; 0,3; 5°)')
arrow(ax, 3.6, 4.3, 4.6, 5.9); arrow(ax, 3.6, 3.7, 4.6, 2.1)
arrow(ax, 8.6, 6.0, 9.8, 4.5); arrow(ax, 8.6, 2.0, 9.8, 3.5)
arrow(ax, 13.4, 4.0, 14.4, 4.0); arrow(ax, 17.0, 4.0, 17.7, 4.0)
ax.text(10.2, 0.35, 'Результат пары: доля удач стабильного минус доля удач метастабильного; усреднение по 129 составам (172 пары)',
        ha='center', fontsize=10.5, color=INK2)
fig.savefig(f'{OUT}/fig6_pair_design.png', dpi=170, bbox_inches='tight')
plt.close(fig)

# ---------- рис. 7: один шаблон, две раскладки (гипотеза 4) ----------
fig, ax = plt.subplots(figsize=(10.2, 4.3))
ax.set_xlim(0, 20.4); ax.set_ylim(0, 8.6); ax.axis('off')
box(ax, 0.2, 3.1, 5.2, 2.6, 'Шаблон R3m:\nодин набор орбит\nMg 1a, 1a; Al 1a, 3b\nO 1a, 1a, 3b, 3b', fc='#f3f2ee', bold=False)
box(ax, 7.4, 5.4, 5.6, 2.2, 'Упорядоченная раскладка\n(mp-3536, энергия над\nоболочкой 0)', ec=BLUE)
box(ax, 7.4, 1.0, 5.6, 2.2, 'Антисайтная пара Mg↔Al\n(mp-34330, энергия над\nоболочкой 0,0505 эВ/атом)', ec=ORANGE)
arrow(ax, 5.4, 4.7, 7.4, 6.5); arrow(ax, 5.4, 4.1, 7.4, 2.1)
for (y, k, col) in ((6.0, 14, BLUE), (1.6, 36, ORANGE)):
    ax.add_patch(Rectangle((14.0, y), 6.0, 0.9, fc=SURF, ec=LGRAY, lw=1.2))
    ax.add_patch(Rectangle((14.0, y), 6.0 * k / 50, 0.9, fc=col, ec='none'))
    ax.text(14.0, y - 0.55, f'{k} из 50 образцов', fontsize=11, color=INK, va='center')
ax.text(14.0, 0.2, 'Доля антисайтных: 0,72\n(доверительный интервал 0,58–0,83)', fontsize=10.5, color=INK2, va='center')
fig.savefig(f'{OUT}/fig7_spinel_template.png', dpi=170, bbox_inches='tight')
plt.close(fig)


# ===================== рисунки для отчёта пилота (рис. 8–11) =====================
# Числа вписаны вручную из таблиц docs/pilot_results_report.md.

# ---------- рис. 8: DiffCSP++, сложность шаблона и покрытие обучающей выборкой ----------
fig, axs = plt.subplots(1, 2, figsize=(10.2, 3.6), gridspec_kw={'width_ratios': [1, 1.15]})
ax = axs[0]
lab = ['0', '1', '2', '3–4', '5–8', '≥ 9']; n = [47, 49, 37, 42, 56, 69]; v = [1.00, 0.98, 1.00, 0.98, 0.86, 0.44]
ax.bar(range(6), v, color=[BLUE] * 4 + [ORANGE] * 2, width=0.62)
for i, (a, b) in enumerate(zip(v, n)):
    ax.text(i, a + 0.02, f'{a:.2f}'.replace('.', ','), ha='center', fontsize=10, color=INK)
    ax.text(i, 0.04, f'{b}', ha='center', fontsize=9, color=SURF)
ax.set_xticks(range(6)); ax.set_xticklabels(lab); ax.set_ylim(0, 1.12)
ax.set_xlabel('число свободных координат шаблона'); ax.set_ylabel('доля совпадений (станд. допуски)')
ax.set_title('а  сложность шаблона', loc='left', fontsize=11, color=INK)
clean(ax, 'y')
ax = axs[1]
lab = ['0', '1–3', '4–15', '16–60', '> 60']
te = [0.33, 0.50, 0.82, 0.95, 0.96]; tr = [0.54, 0.67, 0.77, 1.00, 1.00]
x = np.arange(5); w = 0.36
ax.bar(x - w / 2, tr, w, color=GRAY, label='цели из обучающей части')
ax.bar(x + w / 2, te, w, color=BLUE, label='цели из тестовой части')
ax.set_xticks(x); ax.set_xticklabels(lab); ax.set_ylim(0, 1.12)
ax.set_xlabel('структур того же типа в обучающей выборке'); ax.set_ylabel('доля совпадений (строгие допуски)')
ax.set_title('б  покрытие обучающей выборкой', loc='left', fontsize=11, color=INK)
ax.legend(frameon=False, fontsize=9.5, loc='upper left', labelcolor=INK2); clean(ax, 'y')
fig.tight_layout(w_pad=3.0); fig.savefig(f'{OUT}/fig8_diffcsp_complexity_coverage.png', dpi=170, bbox_inches='tight'); plt.close(fig)

# ---------- рис. 9: CrystaLLM-small, вся тестовая часть, запрос только с составом ----------
bins = ['0–5', '5–20', '20–40', '40–60', '60–80']
per_try = [0.660, 0.526, 0.475, 0.422, 0.395]; any20 = [0.827, 0.733, 0.700, 0.636, 0.601]
fig, ax = plt.subplots(figsize=(8.6, 3.4))
x = np.arange(5); w = 0.36
ax.bar(x - w / 2, any20, w, color=LGRAY, label='цель угадана хотя бы раз из 20')
ax.bar(x + w / 2, per_try, w, color=BLUE, label='доля удачных попыток')
for i in range(5):
    ax.text(x[i] - w / 2, any20[i] + 0.02, f'{any20[i]:.2f}'.replace('.', ','), ha='center', fontsize=9, color=INK2)
    ax.text(x[i] + w / 2, per_try[i] + 0.02, f'{per_try[i]:.2f}'.replace('.', ','), ha='center', fontsize=9, color=INK)
ax.set_xticks(x); ax.set_xticklabels(bins); ax.set_ylim(0, 1.0)
ax.set_xlabel('энергия цели над выпуклой оболочкой, мэВ/атом'); ax.set_ylabel('строгое совпадение')
ax.legend(frameon=False, fontsize=9.5, loc='upper right', labelcolor=INK2); clean(ax, 'y')
fig.savefig(f'{OUT}/fig9_crystallm_test_bins.png', dpi=170, bbox_inches='tight'); plt.close(fig)

# ---------- рис. 10: что модель выдаёт вместо цели ----------
cats = ['цель (строго)', 'цель (только станд.)', 'другой известный полиморф', 'новая структура того же состава', 'другой состав', 'невалидный CIF']
stab = [0.595, 0.066, 0.038, 0.205, 0.081, 0.015]; meta = [0.456, 0.065, 0.068, 0.293, 0.098, 0.021]
cols = [BLUE, '#9cc0ee', VIOLET, ORANGE, GRAY, LGRAY]
fig, ax = plt.subplots(figsize=(8.6, 2.4))
for r, vals in enumerate((stab, meta)):
    left = 0
    for v, c, cn in zip(vals, cols, cats):
        ax.barh(r, v, left=left, color=c, height=0.58, label=cn if r == 0 else None)
        if v >= 0.06:
            ax.text(left + v / 2, r, f'{v:.2f}'.replace('.', ','), ha='center', va='center', fontsize=9.5,
                    color=SURF if c in (BLUE, VIOLET, ORANGE, GRAY) else INK)
        left += v
ax.set_yticks([0, 1]); ax.set_yticklabels(['стабильные цели (129)', 'метастабильные цели (174)'], color=INK); ax.invert_yaxis()
ax.set_xlim(0, 1); ax.set_xlabel('доля попыток')
ax.legend(frameon=False, fontsize=9, ncol=3, loc='upper center', bbox_to_anchor=(0.45, -0.38), labelcolor=INK2); clean(ax, 'x')
fig.savefig(f'{OUT}/fig10_attempt_categories.png', dpi=170, bbox_inches='tight'); plt.close(fig)

# ---------- рис. 11: отдельные фазы HfO₂ и ZrO₂, две модели ----------
ph = [('mp-352 HfO₂ P2₁/c', 0, 'train', 13, 10), ('mp-2858 ZrO₂ P2₁/c', 0, 'train', 13, 10),
      ('mp-754403 ZrO₂ I4₁/amd', 15, 'test', 19, 10), ('mp-556605 ZrO₂ Pca2₁', 22, 'val', 5, 10),
      ('mp-755089 ZrO₂ Pnma', 22, 'test', 0, 0), ('mp-776532 HfO₂ P4₂/mnm', 25, 'train', 20, 10),
      ('mp-685097 HfO₂ Pca2₁', 29, 'test', 2, 9), ('mp-755759 ZrO₂ Pnma', 35, 'train', 7, 3),
      ('mp-2574 ZrO₂ P4₂/nmc', 36, 'test', 0, 8), ('mp-1018721 HfO₂ P4₂/nmc', 55, 'test', 0, 10)]
fig, ax = plt.subplots(figsize=(8.6, 4.6))
for i, (nm, e, part, c20, d10) in enumerate(ph):
    a, b = c20 / 20, d10 / 10
    ax.plot([a, b], [i - 0.14, i + 0.14], color=LGRAY, lw=2, zorder=1)
    ax.plot(a, i - 0.14, 'o', color=ORANGE, ms=8, mec=SURF, mew=1.4, zorder=3)
    ax.plot(b, i + 0.14, 's', color=VIOLET, ms=7.5, mec=SURF, mew=1.4, zorder=3)
    ax.text(1.06, i, f'{e} мэВ/атом, {part}', va='center', fontsize=9.5, color=INK2)
ax.set_yticks(range(len(ph))); ax.set_yticklabels([p[0] for p in ph], color=INK, fontsize=10); ax.invert_yaxis()
ax.set_xlim(-0.04, 1.04); ax.set_xlabel('доля строгих совпадений с целью')
ax.plot([], [], 'o', color=ORANGE, label='CrystaLLM-small: состав и группа, 20 попыток')
ax.plot([], [], 's', color=VIOLET, label='DiffCSP++: полный шаблон, 10 попыток')
ax.legend(frameon=False, fontsize=9.5, loc='upper center', bbox_to_anchor=(0.45, 1.12), ncol=2, labelcolor=INK2)
clean(ax, 'x')
fig.savefig(f'{OUT}/fig11_special_phases.png', dpi=170, bbox_inches='tight'); plt.close(fig)
print('ok pilot')
