FROM python:3.10-slim AS builder

WORKDIR /build

COPY ./requirements.txt .

RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

FROM python:3.10-slim

COPY --from=builder /install /usr/local

COPY week01.py .
CMD ["python", "week01.py"]