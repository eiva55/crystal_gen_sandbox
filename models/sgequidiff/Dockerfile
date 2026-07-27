FROM nvcr.io/nvidia/pyg:24.03-py3
RUN pip install uv
RUN uv pip install --system pymatgen==2023.10.11 \
    && uv pip install --system pyxtal==0.6.0 \
    && uv pip install --system matminer==0.9.2 \
    && uv pip install --system smact==2.2.1 \
    && uv pip install --system meshpy==2022.1.3 \
    && uv pip install --system p-tqdm==1.4.0 \
    && uv pip install --system hydra-core \
    && uv pip install --system wandb \
    && uv pip install --system cloudpickle==3.1.1 \
    && uv pip install --system torch-scatter==2.1.2 -f https://data.pyg.org/whl/torch-2.6.0+cu124.html \
    && uv pip install --system torchdiffeq==0.2.5 \
    && uv pip install --system numpy==1.26.4 \
    && uv pip install --system ruamel.yaml==0.17.21

RUN mkdir /home/sgequidiff
WORKDIR /home/sgequidiff
ENV PYTHONPATH=/home/sgequidiff:/home/sgequidiff/src