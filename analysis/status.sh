#!/usr/bin/env bash
cd /home/bebra/matgen/crystal_gen_sandbox
echo "сейчас: $(date '+%F %T') | WSL работает: $(uptime -p) | нагрузка (1/5/15 мин): $(cut -d' ' -f1-3 /proc/loadavg)"
echo; echo "== процессы =="
ps -eo pid,etime,time,pcpu,args | grep -E "scripts/sample.py|run_bench|run_overnight" | grep -v grep | cut -c1-110
echo; echo "== чанки tr1 (CIF записываются в конце чанка) =="
for d in outputs/bench/tr1_o*; do
  [ -d "$d" ] || continue
  n=$(ls $d/*.cif 2>/dev/null | wc -l)
  m=$(python3 -c "import json;print(len(json.load(open('$d/order.json'))))" 2>/dev/null)
  echo "$d: CIF $n из $m"
done
last=$(ls -t analysis/results/bench_tr1_o*.log 2>/dev/null | head -1)
if [ -n "$last" ]; then
  echo; echo "== лог: $last =="
  b=$(( $(tr '\r' '\n' < "$last" | grep -c "999/999 \[") / 2 ))
  echo "завершённых батчей этого чанка (по 20 структур, всего 3): $b"
  tr '\r' '\n' < "$last" | grep -E "[0-9]+/999" | tail -n 1
fi
[ -f analysis/results/overnight_tr1.done ] && echo && echo "ФИНИШ: файл overnight_tr1.done создан (проверь число CIF выше)"
