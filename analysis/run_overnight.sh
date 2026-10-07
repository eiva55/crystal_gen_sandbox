#!/usr/bin/env bash
cd /home/bebra/matgen/crystal_gen_sandbox
for spec in 4:8 12:12 24:12 36:12 48:12; do
  off=${spec%%:*}; n=${spec##*:}
  /home/bebra/miniconda3/envs/mp_lookup/bin/python analysis/run_bench.py --tag v0_o${off} \
      --offset $off --per-bin $n --k 1 --batch-size 20 > analysis/results/bench_v0_o${off}.log 2>&1
done
echo done > analysis/results/overnight.done
