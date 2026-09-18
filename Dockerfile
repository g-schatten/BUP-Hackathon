FROM python:3.12-slim

WORKDIR /app

# System deps: scipy/numpy wheels are manylinux, no compiler needed normally,
# but keep curl for the container's own healthcheck.
RUN apt-get update && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app

EXPOSE 8000

# GET /health must respond within 60s of container start (Participant Guide
# Section 08). uvicorn binds 0.0.0.0 so the judge harness can reach it.
HEALTHCHECK --interval=15s --timeout=5s --start-period=20s --retries=5 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
