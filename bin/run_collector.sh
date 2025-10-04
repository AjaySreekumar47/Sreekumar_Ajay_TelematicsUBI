#!/usr/bin/env bash
set -euo pipefail

# Run Collector API (FastAPI)
uvicorn src.ingestion.collector:app --host 0.0.0.0 --port 8000 --reload
