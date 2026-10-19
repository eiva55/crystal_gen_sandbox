#!/usr/bin/env bash
# DiffCSP++ по набору частями (возобновляется: готовые части пропускаются).
# Использование: bash analysis/run_pairs_diffcsp.sh <набор> <K> [целей в части]
#   например: nohup bash analysis/run_pairs_diffcsp.sh pairs_v1nf 3 20 > /dev/null 2>&1 &
SET=$1; K=${2:-3}; CS=${3:-20}
cd /home/bebra/matgen/crystal_gen_sandbox
PY=/home/bebra/miniconda3/envs/mp_lookup/bin/python
N=$($PY -c "import pandas as pd, math; print(math.ceil(len(pd.read_csv('analysis/results/bench_${SET}_meta.csv')) / $CS))")
LOG=outputs/bench/${SET}_k${K}_progress.log
mkdir -p outputs/bench
for ((i = 0; i < N; i++)); do
  TAG=$(printf "%s_k%s_c%02d" $SET $K $i); OUT=outputs/bench/$TAG
  [ -f $OUT/done.flag ] && continue
  echo "$(date '+%F %T') часть $i из $N: старт" >> $LOG
  $PY analysis/run_bench.py --set $SET --tag $TAG --chunk-size $CS --chunk $i --k $K --batch-size 20 > outputs/bench/$TAG.log 2>&1
  need=$($PY -c "import json; print(len(json.load(open('$OUT/order.json'))))")
  have=$(ls $OUT/*.cif 2>/dev/null | wc -l)
  if grep -q "Saving structures" outputs/bench/$TAG.log; then      # модель дошла до записи (отдельные пустые структуры допустимы)
    $PY analysis/eval_bench.py $OUT >> outputs/bench/$TAG.log 2>&1 && echo ok > $OUT/done.flag
    echo "$(date '+%F %T') часть $i: конец ($have CIF)" >> $LOG
  else
    echo "$(date '+%F %T') часть $i: CIF $have из $need — часть не завершена, остановка" >> $LOG
    exit 1
  fi
done
echo "$(date '+%F %T') все части готовы" >> $LOG
