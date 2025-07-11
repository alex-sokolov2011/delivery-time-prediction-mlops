FROM python:3.12-slim

WORKDIR /srv

COPY services/ci/requirements.txt ./requirements.txt
RUN python -m pip install --upgrade pip && python -m pip install --no-cache-dir -r requirements.txt

COPY . /srv

CMD ["pytest", "src/tests"]