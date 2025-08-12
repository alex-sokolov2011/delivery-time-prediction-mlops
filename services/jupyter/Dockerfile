FROM python:3.12-slim

WORKDIR /srv

# Install system dependencies
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
    build-essential \
 && apt-get clean \
 && rm -rf /var/lib/apt/lists/*

# Install uv for fast Python package installation
RUN pip install --no-cache-dir uv

# Install base dependencies (cached layer)
COPY base-requirements.txt /srv/base-requirements.txt
RUN uv pip install --system --no-cache -r /srv/base-requirements.txt

# Install project-specific dependencies
COPY requirements.txt /srv/requirements.txt
RUN uv pip install --system --no-cache -r /srv/requirements.txt
