ARG PYTHON_VERSION=3.12
ARG PYTORCH_VERSION=2.5
ARG CUDA_VERSION=12.1.1
ARG LIGHTNING_VERSION=2.5.0

FROM pytorchlightning/pytorch_lightning:${LIGHTNING_VERSION}-py${PYTHON_VERSION}-torch${PYTORCH_VERSION}-cuda${CUDA_VERSION}
ENV POETRY_VIRTUALENVS_CREATE=false
ENV POETRY_VIRTUALENVS_IN_PROJECT=false
ENV POETRY_CACHE_DIR="/workspaces/med-citl/.pypi_cache"
ENV DPKG_FORCE_OVERWRITE=1

RUN mkdir -p /tmp && \
    chmod 1777 /tmp && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* && \
    rm -rf /tmp/*

RUN id -u vscode &>/dev/null || \
    (useradd -ms /bin/bash vscode && echo 'vscode ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers)

RUN apt-get clean

RUN apt remove -y python3-blinker
RUN python -m pip install --upgrade pip

RUN pip install uv

COPY pyproject.toml uv.lock README.md ./

RUN uv sync

ENV PATH_DATASETS=/workspaces/med-citl/datasets/
ENV VIRTUAL_ENV=/workspaces/med-citl/.venv