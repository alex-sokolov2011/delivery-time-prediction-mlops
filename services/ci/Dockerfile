FROM python:3.12

WORKDIR /srv

# Install uv
RUN apt-get update && apt-get install -y curl ca-certificates && rm -rf /var/lib/apt/lists/*
RUN curl -LsSf https://astral.sh/uv/install.sh | sh -s -- 
ENV PATH="/root/.local/bin:${PATH}"

# Install dependencies
COPY services/production/requirements.txt  /srv/requirements.txt
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --system -r /srv/requirements.txt

COPY data_store/prod_model.cbm /srv/data/prod_model.cbm
COPY src/ /srv/src/

CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8090", "--reload"]