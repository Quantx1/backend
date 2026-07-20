FROM python:3.12-slim

# Build toolchain only — needed to compile pyqlib's Cython extensions.
# (The old image also built the TA-Lib C library from source, which cost
# several minutes per build. Nothing imports it: requirements.txt uses the
# pure-Python `ta` package, not the TA-Lib bindings.)
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ make && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# requirements.txt carries --extra-index-url for the CPU torch wheels.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt && \
    # Pin last so nothing downgrades it; transformers 4.57+ requires >=0.34.0.
    pip install --no-cache-dir --force-reinstall --no-deps 'huggingface-hub>=0.34.0,<1.0'

COPY backend/ backend/
COPY ml/ ml/
COPY artifacts/ artifacts/
COPY data/ data/
# Not imported as a package (no __init__.py) — these are operational scripts
# run inside the container: qlib NSE ingestion and the train_* entrypoints
# that backend/ai/qlib/ and model_registry.py tell operators to invoke.
COPY scripts/ scripts/

# Honour the platform-injected $PORT; fall back to 8000 for local runs.
ENV PORT=8000
EXPOSE 8000

CMD ["sh", "-c", "uvicorn backend.api.app:app --host 0.0.0.0 --port ${PORT:-8000}"]
