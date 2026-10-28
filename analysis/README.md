# Код для проверки гипотез: условная генерация кристаллов и энергия над выпуклой оболочкой

Данные: MP-20 (train/val/test). Модели: DiffCSP++ (`mp_csp`), CrystaLLM-small (`cif_model_mp_20_b`).
Результаты анализов лежат в `analysis/results/*.txt`.

## CrystaLLM-small: готовые генерации авторов, весь test (запрос — состав, K=20)
1. `build_full_test.py` — метаданные всех 9046 тестовых структур (энергия над выпуклой оболочкой, группа, n_своб, частоты) → `bench_full_meta.csv`
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

## Пары полиморфов одного состава (стабильный против метастабильного)
- `build_pairs_v1.py` — набор `pairs_v1` (составы test с почти стабильным, E < 5 мэВ/атом, и метастабильным полиморфом) и набор отдельных фаз `--set special_v1 --ids ...`; пишет `bench_<set>_meta.csv`, `bench_<set>nf_meta.csv` (n_free >= 1, для DiffCSP++) и условия DiffCSP++ в `configs/csp_targets/`
- `cross_match.py` — что выдала модель на каждую попытку: цель, другой известный полиморф того же состава из MP-20, новая структура, другой состав, невалидный CIF
- `analyze_pairs_v1.py` — по каждой цели «удачных попыток из K», по каждой паре стабильный/метастабильный, разность долей с бутстрэпом по составам, модель внутри состава; объединяет несколько eval.csv
- `run_pairs_diffcsp.sh <набор> <K> [целей в части]` — DiffCSP++ по набору частями с возобновлением (`run_bench.py --chunk-size/--chunk`)
- результат на готовых генерациях (запрос только с составом, K=20): `results/analyze_pairs_v1_pub.txt`, `results/pairs_pub_per_*.csv`
- `build_pairs_v1.py --split val --exclude-sets pairs_v1` — пары валидационной части (`pairs_val_v1`); `combine_sets.py --dst <имя> --src <наборы>` — объединение (`pairs_tv`, `pairs_tv_nf`); `subset_set.py` — подмножество для пробных прогонов
- `within_strata.py --label <метка> [--exclude-degenerate template_degeneracy.csv] [--only-set <набор>]` — модель внутри состава по отборам пар (основной критерий проверки 6), без пар с общим шаблоном, по частям MP-20
- `cross_substitution.py <cross.csv>` — замещение другим полиморфом (12 сравнений, поправка Холма)
- `new_structure_energy.py`, `isolated_atoms_check.py` — энергии CHGNet новых структур, проверка изолированных атомов
- план разбора DiffCSP++ на парах, зафиксированный до генерации: `docs/analysis_plan_diffcsp_pairs.md`

## Шаблоны, общие для нескольких полиморфов (гипотеза 4)
- `template_degeneracy.py` — шаблоны всех структур MP-20 (`mp20_templates.csv`), группы «состав, шаблон» с несовпадающими полиморфами и частоты раскладок в обучении (`template_degeneracy.csv/.txt`)
- `build_degeneracy_set.py --name deg_v1 --pairs ...` — набор шаблонов для генерации
- `run_spinel_n.py`, `spinel_decoration.py <папка>` — генерация шпинели частями и раскладка образцов по числу ближайших атомов кислорода

## Между составами на большом наборе (проверка 3)
- `build_set.py --set test3k --split test --per-bin 600 --seed 2` — 3000 целей тестовой части
- `analyze_test3k.py` — логистическая модель между составами для DiffCSP++ и CrystaLLM-small, эквивалентность, ступень, разность наклонов моделей → `results/test3k_between.txt`

## Запуск на GPU
- `docker/` — образ с окружениями `diffcsp_pp` и `crystallm`, порядок переноса и запуска на общем сервере; `run_pairs_diffcsp.sh` и `run_crystallm.sh` настраиваются переменными (`CARD`, `STRIDE`/`OFFSET`, `PY`, `DIFFCSP_PY`, `CRYSTALLM_DIR`, `IMAGE`) и пишут журнал `runs_journal.tsv`

## Основной пайплайн (DiffCSP++ и остальные модели)
- `run.py`, `configs/` (Hydra), `envs/` — запуск генераций; наборы целей — `configs/csp_targets/*.json` (`bench_v0` — 300 тестовых целей, `bench_tr1` — 300 обучающих, `hfo2_pilot`, `spinel_pilot`). `aggregate_sweep_metrics.py` собирает метрики прогонов. CIF готовых прогонов лежат в `archive/*.tar.gz`.

## DiffCSP++: бенчмарк 300 test + 300 train (условие: группа, орбиты Вайкоффа, виды атомов)
Скрипты читают данные из `models/diffcsp_pp/data/mp_20` и результаты прогонов из `outputs/bench/*/eval.csv` (ни то, ни другое не входит в репозиторий).
- `build_benchmark_targets.py` — стратифицированный набор из test MP-20 (по 60 целей на 5 корзин энергии над выпуклой оболочкой, seed 0)
- `build_set.py --set <имя> --split train|test --per-bin 60` — набор целей из заданной части MP-20 с ковариатами (контроль «train против test», новые наборы)
- `add_covariates.py` — число свободных координатных параметров орбит Вайкоффа (n_своб)
- `add_proto_freq.py` — частота прототипа: число структур train с той же анонимизированной формулой и группой
- `eval_bench.py <папка_прогона>` — совпадение (StructureMatcher), RMSE и сохранение группы для вывода `run_bench.py`
- `analyze_bench.py` — доли совпадений по корзинам энергии над выпуклой оболочкой и логистическая регрессия с контролями
- `analyze_bench2.py` — расширенный анализ: нетривиальные цели, страты по сложности шаблона, строгий матчер, RMSE
- `analyze_bench3.py` — полные таблицы коэффициентов и анализ недо-искажения (окна допуска symprec)
- `bench_windows.py` — окно допуска symprec, в котором эталон и образец сохраняют целевую группу
- `polar_check.py` — недо-искажение для полярных и неполярных целей
- `polar_model.py` — отличается ли успех для полярных целей при равной сложности шаблона и частоте прототипа
- `nearest_train.py`, `analyze_train_control.py`, `analyze_neighbors.py`, `coverage_table.py` — ближайшие структуры в train, контроль train/test, покрытие структурным типом

## Фазы HfO₂ и шпинель
- `mp20_membership.py` — есть ли фазы HfO₂ (mp-352, mp-685097) в MP-20 и как часто встречаются группы
- `mp20_hfo2_split.py`, `hfo2_matcher.py`, `hfo2_symprec_sweep.py`, `hfo2_windows.py` — разбиение, допуски StructureMatcher и symprec, окна допуска для HfO₂
- `fetch_refs.py` — загрузка эталонов (`analysis/refs/`: mp-352, mp-685097, mp-3536, mp-34330)
- `relax_chgnet.py`, `relax_summary.py` — релаксация CHGNet и сводка

Не включены в репозиторий (`outputs/` в .gitignore): сырые генерации (`*.cif`, `*.tar.gz`), промежуточные `eval.csv`, одноразовые `patch_*.py` (правки внесены в скрипты). Итоговые таблицы и метаданные — в `analysis/results/`.

## Данные для воспроизведения анализа
`analysis/results/eval/`: `full_pub_eval.csv.gz` (весь test, K=20; перед использованием `gunzip -k`), `v0_sg_eval.csv` (300 целей с группой в запросе, K=1). Генерации пар и наборов окна GPU 27.10.2026 в репозиторий не входят (папка `outputs_server/` на WSL); итоговые таблицы их разбора — в `analysis/results/` (список ниже).
