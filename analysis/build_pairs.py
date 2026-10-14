"""Набор «пары»: все структуры test, у которых есть другая структура того же состава (полиморфы). Берётся из bench_full_meta.csv."""
import pandas as pd

m = pd.read_csv('analysis/results/bench_full_meta.csv')
cnt = m.groupby('formula').material_id.transform('size')
p = m[cnt >= 2].sort_values(['formula', 'e_above_hull']).reset_index(drop=True)
p.to_csv('analysis/results/bench_pair_meta.csv', index=False)
print(f"составов {p.formula.nunique()}, целей {len(p)}; n_free=0: {(p.n_free == 0).sum()}; по корзинам:")
print(p.bin.value_counts().sort_index())
