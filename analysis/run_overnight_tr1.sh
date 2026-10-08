#!/usr/bin/env bash
cd /home/bebra/matgen/crystal_gen_sandbox
PY=/home/bebra/miniconda3/envs/mp_lookup/bin/python
for off in 0 12 24 36 48; do
  $PY analysis/run_bench.py --set tr1 --tag tr1_o${off} --offset $off --per-bin 12 --k 1 --batch-size 20 > analysis/results/bench_tr1_o${off}.log 2>&1
done
echo done > analysis/results/overnight_tr1.done
