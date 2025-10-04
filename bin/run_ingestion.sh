#!/usr/bin/env bash
set -euo pipefail
python -m src.ingestion.simulator \
  --config config/simulator_full.yaml \
  --schema docs/schemas/telematics_event_v4.avsc \
  --batch 250
python bin/gen_data_dictionary.py \
  --schema docs/schemas/telematics_event_v4.avsc \
  --out docs/data_dictionary_v4.md
