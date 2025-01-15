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


# # Update package lists and install prerequisites
# RUN apt-get update && apt-get install -y \
#     apt-transport-https \
#     ca-certificates \
#     curl \
#     gnupg \
#     lsb-release && \
#     curl -fsSL https://download.docker.com/linux/ubuntu/gpg | gpg --dearmor -o /usr/share/keyrings/docker-archive-keyring.gpg

# RUN echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/docker-archive-keyring.gpg] https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable" > /etc/apt/sources.list.d/docker.list

# # Install Docker Engine
# RUN apt-get update && apt-get install -y \
#     docker-ce \
#     docker-ce-cli \
#     containerd.io && \
#     apt-get clean && rm -rf /var/lib/apt/lists/*

RUN id -u vscode &>/dev/null || \
    (useradd -ms /bin/bash vscode && echo 'vscode ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers)

RUN apt-get clean

RUN apt remove -y python3-blinker
RUN python -m pip install --upgrade pip

RUN apt-get update && apt-get install -y parallel

RUN pip install poetry

RUN poetry config installer.max-workers 12
RUN poetry config installer.parallel true

COPY poetry.lock pyproject.toml ./
RUN poetry --no-root install

COPY requirements.txt .

RUN pip install --no-deps -r requirements.txt 
RUN python -m pip install --force-reinstall numpy scikit-learn

ENV PATH_DATASETS=/workspaces/med-citl/datasets/
