# One image runs the API, the UI and the ingestion job (the command differs, see docker-compose.yml).
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/models \
    TOKENIZERS_PARALLELISM=false

WORKDIR /app

# CPU-only PyTorch first: the default build drags in GPU libraries we cannot use and gigabytes of disk.
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY src ./src
COPY ui ./ui

# Run as a normal user. /models is created first so the named volume inherits its ownership.
RUN useradd --create-home app && mkdir -p /models /app/data && chown -R app /models /app/data
USER app

EXPOSE 8000 8501
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
