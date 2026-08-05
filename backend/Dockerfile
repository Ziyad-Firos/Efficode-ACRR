# ACRR backend.
#
# Two things were wrong with the previous version of this file and both
# meant the image could never actually start:
#   1. It ran `uvicorn app.main:app`. uvicorn is an ASGI server; this is a
#      Flask (WSGI) app, so it cannot be served that way. gunicorn is correct.
#   2. It pinned python:3.11-slim. The project targets 3.12 — that mismatch
#      is what caused the earlier scikit-learn wheel problems.
FROM python:3.12-slim

WORKDIR /app

# Install dependencies first so this layer caches across source changes.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Train the complexity model at build time so it is baked into the image
# and the first request does not pay for it.
RUN python -c "from app.ml.complexity_predictor import train_model; train_model()"

EXPOSE 8000
ENV PORT=8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:8000/health').status==200 else 1)"

CMD ["sh", "-c", "gunicorn --bind 0.0.0.0:${PORT} --workers 1 --threads 4 --timeout 60 --preload app.main:app"]
