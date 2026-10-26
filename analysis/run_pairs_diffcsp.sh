#!/usr/bin/env bash
# DiffCSP++ по набору частями (возобновляется: готовые части пропускаются).
# Использование: bash analysis/run_pairs_diffcsp.sh <набор> <K> [целей в части]
#   WSL:    nohup bash analysis/run_pairs_diffcsp.sh pairs_v1nf 3 20 > /dev/null 2>&1 &
#   сервер: CARD=0 STRIDE=2 OFFSET=0 DIFFCSP_PY=python PY=python bash analysis/run_pairs_diffcsp.sh pairs_v1nf 10 20
# Переменные окружения (все необязательны):
#   REPO_DIR    корень репозитория (по умолчанию — папка над этим скриптом)
#   PY          Python с pandas для служебных шагов и eval_bench.py (по умолчанию mp_lookup в WSL, иначе python)
#   DIFFCSP_PY  команда Python для sample.py, передаётся в run_bench.py (по умолчанию conda run -n diffcsp_pp python)
#   CARD        номер карты: ставится в CUDA_VISIBLE_DEVICES
#   STRIDE, OFFSET  брать только части с номером i, для которого i % STRIDE == OFFSET (две карты: 2/0 и 2/1)
#   BATCH       размер батча sample.py (по умолчанию 20)
#   IMAGE       тег или digest образа Docker — пишется в журнал
#   DRY=1       только подготовить targets.json и order.json, без генерации
# Журнал запусков: outputs/bench/runs_journal.tsv (одна строка на часть).
SET=$1; K=${2:-3}; CS=${3:-20}
REPO_DIR=${REPO_DIR:-$(cd "$(dirname "$0")/.." && pwd)}
cd "$REPO_DIR" || exit 1
if [ -z "$PY" ]; then
  if [ -x /home/bebra/miniconda3/envs/mp_lookup/bin/python ]; then PY=/home/bebra/miniconda3/envs/mp_lookup/bin/python; else PY=python; fi
fi
STRIDE=${STRIDE:-1}; OFFSET=${OFFSET:-0}; BATCH=${BATCH:-20}
[ -n "$CARD" ] && export CUDA_VISIBLE_DEVICES=$CARD
N=$($PY -c "import pandas as pd, math; print(math.ceil(len(pd.read_csv('analysis/results/bench_${SET}_meta.csv')) / $CS))") || exit 1
LOG=outputs/bench/${SET}_k${K}_progress.log
JOURNAL=outputs/bench/runs_journal.tsv
mkdir -p outputs/bench
[ -f $JOURNAL ] || printf 'начало\tконец\tсекунд\tхост\tкарта\tGPU\tнабор\tK\tчасть\tструктур\tCIF\tкоммит\tобраз\tseed\n' > $JOURNAL
COMMIT=$(git rev-parse --short HEAD 2>/dev/null || echo нет)
GPU=$( { [ -n "$CARD" ] && nvidia-smi --query-gpu=name --format=csv,noheader -i "$CARD" 2>/dev/null; } || echo нет)
for ((i = 0; i < N; i++)); do
  (( i % STRIDE == OFFSET )) || continue
  TAG=$(printf "%s_k%s_c%02d" $SET $K $i); OUT=outputs/bench/$TAG
  [ -f $OUT/done.flag ] && continue
  T0=$(date +%s); D0=$(date '+%F %T')
  echo "$D0 часть $i из $N: старт (карта ${CARD:-нет})" >> $LOG
  if [ -n "$DRY" ]; then
    $PY analysis/run_bench.py --set $SET --tag $TAG --chunk-size $CS --chunk $i --k $K --batch-size $BATCH --dry-run > outputs/bench/$TAG.log 2>&1
    continue
  fi
  $PY analysis/run_bench.py --set $SET --tag $TAG --chunk-size $CS --chunk $i --k $K --batch-size $BATCH > outputs/bench/$TAG.log 2>&1
  need=$($PY -c "import json; print(len(json.load(open('$OUT/order.json'))))")
  have=$(ls $OUT/*.cif 2>/dev/null | wc -l)
  if grep -q "Saving structures" outputs/bench/$TAG.log; then      # модель дошла до записи (отдельные пустые структуры допустимы)
    $PY analysis/eval_bench.py $OUT >> outputs/bench/$TAG.log 2>&1 && echo ok > $OUT/done.flag
    echo "$(date '+%F %T') часть $i: конец ($have CIF)" >> $LOG
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$D0" "$(date '+%F %T')" $(( $(date +%s) - T0 )) \
      "$(hostname)" "${CARD:-нет}" "$GPU" $SET $K $i $need $have $COMMIT "${IMAGE:-нет}" "нет (sample.py без параметра seed)" >> $JOURNAL
  else
    echo "$(date '+%F %T') часть $i: CIF $have из $need — часть не завершена, остановка" >> $LOG
    exit 1
  fi
done
echo "$(date '+%F %T') все части готовы (STRIDE=$STRIDE, OFFSET=$OFFSET)" >> $LOG
