FROM python:3.10-slim AS builder

WORKDIR /build

COPY ./requirements.txt .

RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

FROM python:3.10-slim

WORKDIR /app

COPY --from=builder /install /usr/local

COPY week02.py .

CMD ["python", "week02.py"]