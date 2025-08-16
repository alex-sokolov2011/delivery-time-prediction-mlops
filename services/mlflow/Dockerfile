FROM python:3.12

WORKDIR /srv

# Install uv
RUN apt-get update && apt-get install -y curl ca-certificates && rm -rf /var/lib/apt/lists/*
RUN curl -LsSf https://astral.sh/uv/install.sh | sh -s -- 
ENV PATH="/root/.local/bin:${PATH}"

# Copy deps early to leverage layer cache
COPY services/mlflow/requirements.txt /srv/requirements.txt

# Install dependencies
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --system -r /srv/requirements.txt

COPY docker-entrypoint.sh /usr/local/bin/docker-entrypoint.sh
RUN chmod +x /usr/local/bin/docker-entrypoint.sh

ENTRYPOINT ["/usr/local/bin/docker-entrypoint.sh"]