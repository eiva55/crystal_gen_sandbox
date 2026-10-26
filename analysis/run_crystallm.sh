#!/usr/bin/env bash
# CrystaLLM по частям запросов (prompts_NN.tar.gz из crystallm_prepare.py); готовые части пропускаются.
# Использование: bash analysis/run_crystallm.sh <тег> [K]   (тег, например, pairs_v1_sg; K — число попыток на цель)
#   сервер: CARD=1 DEVICE=cuda STRIDE=2 OFFSET=1 CRYSTALLM_DIR=/work/CrystaLLM CRYSTALLM_PY=python SEED=3001 \
#           bash analysis/run_crystallm.sh pairs_val_v1_sg 10
# Переменные окружения (все необязательны):
#   REPO_DIR       корень этого репозитория (по умолчанию — папка над скриптом)
#   CRYSTALLM_DIR  папка CrystaLLM (по умолчанию /home/bebra/matgen/external/CrystaLLM)
#   CRYSTALLM_PY   Python окружения CrystaLLM (по умолчанию env crystallm в WSL)
#   DEVICE, DTYPE  устройство и тип (по умолчанию cpu и float32); CARD — номер карты для CUDA_VISIBLE_DEVICES
#   SEED           seed генерации (по умолчанию 1337); при дозапуске тех же целей нужен другой seed
#   STRIDE, OFFSET брать части с номером N, для которого N % STRIDE == OFFSET
#   IMAGE          тег или digest образа — пишется в журнал
# Журнал: outputs/bench_crystallm/runs_journal.tsv (одна строка на часть).
TAG=$1; K=${2:-1}
REPO_DIR=${REPO_DIR:-$(cd "$(dirname "$0")/.." && pwd)}
CRYSTALLM_DIR=${CRYSTALLM_DIR:-/home/bebra/matgen/external/CrystaLLM}
PY=${CRYSTALLM_PY:-/home/bebra/miniconda3/envs/crystallm/bin/python}
DEVICE=${DEVICE:-cpu}; DTYPE=${DTYPE:-float32}; SEED=${SEED:-1337}; STRIDE=${STRIDE:-1}; OFFSET=${OFFSET:-0}
[ -n "$CARD" ] && export CUDA_VISIBLE_DEVICES=$CARD
OUT=$REPO_DIR/outputs/bench_crystallm/$TAG
JOURNAL=$REPO_DIR/outputs/bench_crystallm/runs_journal.tsv
[ -f $JOURNAL ] || printf 'начало\tконец\tсекунд\tхост\tкарта\tустройство\tтег\tK\tчасть\tseed\tкоммит\tобраз\n' > $JOURNAL
COMMIT=$(git -C "$REPO_DIR" rev-parse --short HEAD 2>/dev/null || echo нет)
cd "$CRYSTALLM_DIR" || exit 1
for P in $OUT/prompts_[0-9]*.tar.gz; do
  B=$(basename $P .tar.gz); N=${B#prompts_}
  (( 10#$N % STRIDE == OFFSET )) || continue
  [ -f $OUT/post_$N.tar.gz ] && continue          # часть уже готова: пропускаем (возобновление после обрыва)
  T0=$(date +%s); D0=$(date '+%F %T')
  echo "$D0 часть $N: старт (устройство $DEVICE, карта ${CARD:-нет}, seed $SEED)" >> $OUT/progress.log
  $PY bin/generate_cifs.py --model cif_model_mp_20_b --prompts $P --out $OUT/raw_$N.tar.gz \
      --device $DEVICE --dtype $DTYPE --num-gens $K --seed $SEED > $OUT/gen_$N.log 2>&1 && \
  $PY bin/postprocess.py $OUT/raw_$N.tar.gz $OUT/post_$N.tar.gz >> $OUT/gen_$N.log 2>&1
  echo "$(date '+%F %T') часть $N: конец" >> $OUT/progress.log
  [ -f $OUT/post_$N.tar.gz ] && printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$D0" "$(date '+%F %T')" \
      $(( $(date +%s) - T0 )) "$(hostname)" "${CARD:-нет}" $DEVICE $TAG $K $N $SEED $COMMIT "${IMAGE:-нет}" >> $JOURNAL
done
ls $OUT/prompts_[0-9]*.tar.gz | while read P; do B=$(basename $P .tar.gz); [ -f $OUT/post_${B#prompts_}.tar.gz ] || exit 1; done \
  && echo done > $OUT/done.flag
