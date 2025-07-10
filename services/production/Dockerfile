FROM python:3.12

COPY services/production/requirements.txt /srv/requirements.txt
COPY data_store/prod_model.cbm /srv/data/prod_model.cbm
COPY src/ /srv/src/

WORKDIR /srv
RUN python -m pip install --upgrade pip && python -m pip install --no-cache-dir -r requirements.txt

CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8090", "--reload"]