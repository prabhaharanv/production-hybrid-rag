## ---------- Stage 1: builder ----------
FROM python:3.12-slim-bookworm AS builder

WORKDIR /build

RUN apt-get update && apt-get upgrade -y && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir --prefix=/install -r requirements.txt

## ---------- Stage 2: runtime ----------
FROM python:3.12-slim-bookworm

# Apply latest security patches, install runtime deps for PyMuPDF/PDF handling, upgrade pip
RUN apt-get update && apt-get upgrade -y && \
    apt-get install -y --no-install-recommends libmupdf-dev libfreetype6 libharfbuzz0b libjbig2dec0 libjpeg62-turbo libopenjp2-7 && \
    rm -rf /var/lib/apt/lists/* && \
    pip install --no-cache-dir --upgrade pip

# Create non-root user
RUN groupadd --gid 1000 appuser \
    && useradd --uid 1000 --gid appuser --shell /bin/bash --create-home appuser

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /install /usr/local

COPY --chown=appuser:appuser . .

USER appuser

EXPOSE 8000

CMD ["uvicorn", "app.api:app", "--host", "0.0.0.0", "--port", "8000"]
