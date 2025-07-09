FROM python:3.12

COPY requirements.txt /srv/requirements.txt
WORKDIR /srv
RUN python -m pip install --no-cache-dir -r requirements.txt