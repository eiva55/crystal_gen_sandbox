"""Контроль частоты прототипа: число структур train с тем же (анонимизированная формула, пр. группа)."""
import pandas as pd
from pymatgen.core import Composition

D = 'models/diffcsp_pp/data/mp_20'
meta = pd.read_csv('analysis/results/bench_v0_meta.csv')
tr = pd.read_csv(f'{D}/train.csv', usecols=['pretty_formula', 'spacegroup.number'])
anon = lambda f: Composition(f).anonymized_formula
tr['proto'] = [anon(f) + '_' + str(g) for f, g in zip(tr.pretty_formula, tr['spacegroup.number'])]
cnt = tr.proto.value_counts()
meta['proto'] = [anon(f) + '_' + str(g) for f, g in zip(meta.formula, meta.sg)]
meta['proto_train_count'] = meta.proto.map(cnt).fillna(0).astype(int)

cov = pd.read_csv('analysis/results/bench_v0_covariates.csv')
cov = cov.drop(columns=['proto_train_count'], errors='ignore').merge(
    meta[['material_id', 'proto_train_count']], on='material_id')
cov.to_csv('analysis/results/bench_v0_covariates.csv', index=False)
print(f"целей с прототипом, не встречавшимся в train: {(meta.proto_train_count == 0).sum()} из {len(meta)}")
print(meta.groupby('bin', sort=False).proto_train_count.median().rename('медиана proto_train_count'))
