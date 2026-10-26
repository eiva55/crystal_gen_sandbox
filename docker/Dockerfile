# Образ для генерации DiffCSP++ и CrystaLLM-small на GPU.
# CUDA 11.8 приходит в колёсах PyTorch; драйвер берётся с хоста (nvidia-container-toolkit).
# Контекст сборки — папка docker/ (данные и репозиторий в образ не копируются, они монтируются при запуске).
#   docker build -t eivanov/crystalgen:2026-10 docker/
FROM condaforge/miniforge3:24.3.0-0

ENV DEBIAN_FRONTEND=noninteractive PIP_NO_CACHE_DIR=1 PYTHONUNBUFFERED=1
RUN apt-get update && apt-get install -y --no-install-recommends git tmux procps && rm -rf /var/lib/apt/lists/*
# репозиторий монтируется от другого пользователя: без этого git не прочитает коммит для журнала запусков
RUN git config --system --add safe.directory '*'

# Два окружения, как на WSL: diffcsp_pp (Python 3.9) и crystallm (Python 3.10)
RUN conda create -y -n diffcsp_pp python=3.9 && conda create -y -n crystallm python=3.10 && conda clean -afy

COPY requirements_diffcsp_pp.txt requirements_crystallm.txt /tmp/
# spglib 2.7.0 для Python 3.9 есть только в conda-forge; если сборки нет, ставится 2.6.0 из PyPI
RUN (conda install -y -n diffcsp_pp -c conda-forge "spglib=2.7.0" || conda run -n diffcsp_pp pip install "spglib==2.6.0") \
 && conda run -n diffcsp_pp pip install -r /tmp/requirements_diffcsp_pp.txt \
 && conda clean -afy
RUN conda run -n crystallm pip install -r /tmp/requirements_crystallm.txt

# CrystaLLM на том же коммите, что и на WSL; чекпоинт cif_model_mp_20_b монтируется в /opt/CrystaLLM/cif_model_mp_20_b
RUN git clone https://github.com/lantunes/CrystaLLM /opt/CrystaLLM \
 && git -C /opt/CrystaLLM checkout 8e81a86f9e064f219f09312f29ecbfc867211784 \
 && conda run -n crystallm pip install --no-deps -e /opt/CrystaLLM

# Пути для DiffCSP++ (вместо файла .env) и для скриптов репозитория; репозиторий монтируется в /work/crystal_gen_sandbox
ENV PROJECT_ROOT=/work/crystal_gen_sandbox/models/diffcsp_pp \
    HYDRA_JOBS=/work/hydra \
    WANDB_DIR=/work/wandb \
    WANDB_MODE=disabled \
    REPO_DIR=/work/crystal_gen_sandbox \
    PY=/opt/conda/envs/diffcsp_pp/bin/python \
    DIFFCSP_PY=/opt/conda/envs/diffcsp_pp/bin/python \
    CRYSTALLM_DIR=/opt/CrystaLLM \
    CRYSTALLM_PY=/opt/conda/envs/crystallm/bin/python

WORKDIR /work/crystal_gen_sandbox
CMD ["sleep", "infinity"]
