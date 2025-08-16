FROM python:3.12-slim

WORKDIR /srv

# Install uv
RUN apt-get update && apt-get install -y curl ca-certificates && rm -rf /var/lib/apt/lists/*
RUN curl -LsSf https://astral.sh/uv/install.sh | sh -s -- 
ENV PATH="/root/.local/bin:${PATH}"

# Install base dependencies (cached layer)
COPY services/jupyter/base-requirements.txt /srv/base-requirements.txt
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --system -r /srv/base-requirements.txt

# Install project-specific dependencies
COPY services/jupyter/requirements.txt /srv/requirements.txt
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --system -r /srv/requirements.txt
