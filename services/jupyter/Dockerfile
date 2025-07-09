FROM python:3.11

COPY requirements.txt /srv/requirements.txt
WORKDIR /srv
RUN python -m pip install --no-cache-dir -r requirements.txt && \
    python -m pip install statsmodels
