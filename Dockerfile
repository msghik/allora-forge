FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    ALLORA_DATA_DIR=/data \
    ALLORA_MODELS_DIR=/models

# libgomp1 is required by LightGBM at runtime.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY forge ./forge

# Optional: stamp the build's git sha into model metadata.
ARG GIT_SHA=unknown
ENV ALLORA_GIT_SHA=$GIT_SHA

EXPOSE 8000
# Default command runs the inference server; compose overrides it for the trainer.
CMD ["uvicorn", "forge.server:app", "--host", "0.0.0.0", "--port", "8000"]
