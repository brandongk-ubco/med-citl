ARG PYTHON_VERSION=3.12
ARG PYTORCH_VERSION=2.4
ARG CUDA_VERSION=12.1.0
ARG LIGHTNING_VERSION=2.4.0

FROM pytorchlightning/pytorch_lightning:${LIGHTNING_VERSION}-py${PYTHON_VERSION}-torch${PYTORCH_VERSION}-cuda${CUDA_VERSION}
ENV POETRY_VIRTUALENVS_CREATE=false
ENV POETRY_VIRTUALENVS_IN_PROJECT=false
ENV POETRY_CACHE_DIR="/workspaces/med-citl/.pypi_cache"
ENV DPKG_FORCE_OVERWRITE=1

RUN id -u vscode &>/dev/null || \
    (useradd -ms /bin/bash vscode && echo 'vscode ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers)

RUN apt-get clean

RUN apt-get update && \
    apt-get install --reinstall -y libpython3.12-minimal && \
    apt-get install -f -y

RUN apt remove -y python3-blinker
RUN python -m pip install --upgrade pip

RUN apt update && apt install -y parallel

COPY poetry.lock pyproject.toml .
RUN pip install poetry
RUN poetry --no-root install

ENV PATH_DATASETS=/workspaces/med-citl/datasets/
