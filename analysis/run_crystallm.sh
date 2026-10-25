#!/usr/bin/env bash
# Использование: bash analysis/run_crystallm.sh <тег> [K]   (тег, например, v0_sg; K — число попыток на цель)
TAG=$1; K=${2:-1}
REPO=/home/bebra/matgen/external/CrystaLLM
PY=/home/bebra/miniconda3/envs/crystallm/bin/python
OUT=/home/bebra/matgen/crystal_gen_sandbox/outputs/bench_crystallm/$TAG
cd $REPO
for P in $OUT/prompts_[0-9]*.tar.gz; do
  B=$(basename $P .tar.gz); N=${B#prompts_}
  [ -f $OUT/post_$N.tar.gz ] && continue          # часть уже готова: пропускаем (возобновление после обрыва)
  echo "$(date '+%F %T') часть $N: старт" >> $OUT/progress.log
  $PY bin/generate_cifs.py --model cif_model_mp_20_b --prompts $P --out $OUT/raw_$N.tar.gz \
      --device cpu --dtype float32 --num-gens $K --seed ${SEED:-1337} > $OUT/gen_$N.log 2>&1 && \
  $PY bin/postprocess.py $OUT/raw_$N.tar.gz $OUT/post_$N.tar.gz >> $OUT/gen_$N.log 2>&1
  echo "$(date '+%F %T') часть $N: конец" >> $OUT/progress.log
done
echo done > $OUT/done.flag
