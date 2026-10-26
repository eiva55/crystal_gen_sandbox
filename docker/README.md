# Запуск генераций на сервере с GPU (tesla-2)

Сервер общий: перед любым запуском смотреть занятость карт (`nvidia-smi`), работать только в своём контейнере
(`eivanov_crystalgen`) и своих сессиях tmux, чужие контейнеры и процессы не трогать, общие настройки не менять.

## Что перенести на сервер

| что | откуда (WSL) | куда на сервере |
|---|---|---|
| репозиторий | `git clone https://github.com/eiva55/crystal_gen_sandbox` | `~/crystal_gen_sandbox` |
| чекпоинт DiffCSP++ `mp_csp` (файлы `*.ckpt`; в git только `hparams.yaml`) | `models/diffcsp_pp/checkpoints/pretrained_models/mp_csp/` | та же папка внутри `~/crystal_gen_sandbox` |
| чекпоинт CrystaLLM | `~/matgen/external/CrystaLLM/cif_model_mp_20_b/` | `~/ckpt/cif_model_mp_20_b/` |
| запросы CrystaLLM (`outputs/` не в git) | `outputs/bench_crystallm/pairs_val_v1_sg/prompts_*.tar.gz` | та же папка внутри `~/crystal_gen_sandbox` |

Размеры перед переносом: `du -sh models/diffcsp_pp/checkpoints/pretrained_models/mp_csp ~/matgen/external/CrystaLLM/cif_model_mp_20_b`.

## Сборка и проверка образа

```bash
cd ~/crystal_gen_sandbox
docker build -t eivanov/crystalgen:2026-10 docker/
docker image ls --digests eivanov/crystalgen          # digest записать: он идёт в журнал запусков (IMAGE)

nvidia-smi                                            # выбрать две свободные карты, например 2 и 3
docker run --rm --gpus '"device=2"' eivanov/crystalgen:2026-10 bash -c '
  /opt/conda/envs/diffcsp_pp/bin/python -c "import torch, torch_scatter, torch_geometric, spglib, pymatgen.core; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0), spglib.__version__)"
  /opt/conda/envs/crystallm/bin/python -c "import torch, crystallm; print(torch.__version__, torch.cuda.is_available())"'
```

## Запуск

Внутри контейнера выбранные карты видны под номерами 0 и 1.

```bash
docker run -d --name eivanov_crystalgen --gpus '"device=2,3"' \
  -v ~/crystal_gen_sandbox:/work/crystal_gen_sandbox \
  -v ~/ckpt/cif_model_mp_20_b:/opt/CrystaLLM/cif_model_mp_20_b \
  -e IMAGE=<digest образа> eivanov/crystalgen:2026-10
docker exec -it eivanov_crystalgen tmux new -s gen
```

В tmux (окно 0 — карта 0, окно 1 — карта 1; `Ctrl+b c` — новое окно, `Ctrl+b d` — выйти, оставив работу):

```bash
# карта 0: DiffCSP++ на тестовых, затем на валидационных парах
CARD=0 bash analysis/run_pairs_diffcsp.sh pairs_v1nf 10 20 && CARD=0 bash analysis/run_pairs_diffcsp.sh pairs_val_v1nf 10 20
# карта 1: CrystaLLM на новых составах
CARD=1 DEVICE=cuda SEED=1337 bash analysis/run_crystallm.sh pairs_val_v1_sg 10
```

Ход: `tail -3 outputs/bench/pairs_v1nf_k10_progress.log`, `cat outputs/bench/runs_journal.tsv`,
`cat outputs/bench_crystallm/pairs_val_v1_sg/progress.log`. Остановить свои задачи: в своём окне tmux `Ctrl+c`;
контейнер после окна — `docker stop eivanov_crystalgen && docker rm eivanov_crystalgen`.

## После окна

Сопоставление для отчёта повторить на WSL в окружении `mp_lookup` (той же версией pymatgen, что и прежние
результаты): команды — раздел 8 `docs/analysis_plan_diffcsp_pairs.md`. На сервер для этого ничего не нужно, кроме
скачанных папок `outputs/bench/pairs_*_k10_c??` и `outputs/bench_crystallm/pairs_val_v1_sg/post_*.tar.gz`.
