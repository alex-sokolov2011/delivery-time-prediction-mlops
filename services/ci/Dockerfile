FROM python:3.12

COPY services/production/requirements.txt /srv/requirements.txt
COPY src/ /srv/src/

WORKDIR /srv
RUN python -m pip install --no-cache-dir -r requirements.txt

CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8090", "--reload"]