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

## DiffCSP++
(допиши: скрипты бенчмарка 300 test + 300 train целей и анализ)

Не включены в репозиторий (`outputs/` в .gitignore): сырые генерации (`*.cif`, `*.tar.gz`), промежуточные `eval.csv`, одноразовые `patch_*.py` (правки внесены в скрипты). Итоговые таблицы и метаданные — в `analysis/results/`.

## Данные для воспроизведения анализа
`analysis/results/eval/`: `full_pub_eval.csv.gz` (весь test, K=20; перед использованием `gunzip -k`), `v0_sg_eval.csv` (300 целей с группой в запросе, K=1). Результат парного набора будет добавлен после окончания генерации.
