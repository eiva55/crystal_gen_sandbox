# Код для тестов гипотезы 1 (conditional crystal generation vs energy above hull)

Данные: MP-20 (train/val/test). Модели: DiffCSP++ (`mp_csp`), CrystaLLM-small (`cif_model_mp_20_b`).
Результаты анализов лежат в `analysis/results/*.txt`.

## CrystaLLM-small: готовые генерации авторов, весь test (запрос — состав, K=20)
1. `build_full_test.py` — метаданные всех 9046 тестовых структур (E above hull, группа, n_своб, частоты) → `bench_full_meta.csv`
2. `eval_published.py` — оценка готовых генераций (StructureMatcher, группа, состав) → `outputs/bench_crystallm/<set>_pub/eval.csv`
3. `nearest_train.py` — ближайшие структуры того же типа в train, счётчик n_тип
4. `analyze_pub.py` — логистическая регрессия с кластерами по цели, TOST-эквивалентность
5. `analyze_adjusted.py` — то же с контролями покрытия (n_тип) и ступень/наклон
6. `compare_pub_sg.py` — сравнение режимов запроса: с группой и без (взаимодействие энергия × режим)
7. `analyze_train_control.py`, `analyze_neighbors.py`, `coverage_table.py` — контроль train/test, соседи, таблицы покрытия

## CrystaLLM-small: собственные генерации
- `crystallm_prepare.py` — формирование запросов (состав + символ группы), `SHUFFLE=1` для случайного порядка
- `run_crystallm.sh <set> <K>` — генерация по частям (возобновляется с последней завершённой части)
- `eval_crystallm.py <set>` — оценка генераций → `outputs/bench_crystallm/<set>/eval.csv`
- `build_pairs.py` — парный набор: 889 структур 377 составов с несколькими полиморфами
- `analyze_pairs.py` — анализ внутри состава (модель с фиксированными эффектами, парная разность)

## Основной пайплайн (DiffCSP++ и остальные модели)
- `run.py`, `configs/` (Hydra), `envs/` — запуск генераций; наборы целей — `configs/csp_targets/*.json` (`bench_v0` — 300 тестовых целей, `bench_tr1` — 300 обучающих, `hfo2_pilot`, `spinel_pilot`). `aggregate_sweep_metrics.py` собирает метрики прогонов. CIF готовых прогонов лежат в `archive/*.tar.gz`.

## DiffCSP++: бенчмарк 300 test + 300 train (условие: группа, орбиты Вайкоффа, виды атомов)
- `build_benchmark_targets.py`, `build_set.py` — формирование наборов целей по корзинам energy above hull
- `add_covariates.py`, `add_proto_freq.py` — контрольные переменные (n_гр, n_ат, n_своб, частота прототипа)
- `eval_bench.py` — оценка генераций
- `analyze_bench.py`, `analyze_bench2.py`, `analyze_bench3.py` — анализ бенчмарка (последовательные версии; итоговый вывод в `results/bench_v0_analysis*.txt`)
- `bench_windows.py`, `polar_check.py`, `polar_model.py` — окно допуска симметрии и полярные цели
- `mp20_membership.py`, `nearest_train.py`, `analyze_train_control.py`, `analyze_neighbors.py`, `coverage_table.py` — принадлежность train/test, ближайшие структуры, покрытие

## Кейс HfO₂ и шпинели
- `mp20_hfo2_split.py`, `hfo2_matcher.py`, `hfo2_symprec_sweep.py`, `hfo2_windows.py`, `fetch_refs.py` (эталоны в `refs/`)
- `relax_chgnet.py`, `relax_summary.py` — релаксация CHGNet и сводка

Не включены в репозиторий (`outputs/` в .gitignore): сырые генерации (`*.cif`, `*.tar.gz`), промежуточные `eval.csv`, одноразовые `patch_*.py` (правки внесены в скрипты). Итоговые таблицы и метаданные — в `analysis/results/`.

## Данные для воспроизведения анализа
`analysis/results/eval/`: `full_pub_eval.csv.gz` (весь test, K=20; перед использованием `gunzip -k`), `v0_sg_eval.csv` (300 целей с группой в запросе, K=1). Результат парного набора будет добавлен после окончания генерации.
