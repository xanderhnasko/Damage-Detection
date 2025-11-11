#!/usr/bin/env bash
# Sync training outputs from local machine to Google Cloud Storage
# Uploads everything from ~/project/outputs/ to GCS bucket for backup
set -euo pipefail

OUT_LOCAL="${HOME}/project/outputs"
OUT_BUCKET="gs://229_project_bucket/outputs"

echo "[INFO] Syncing ${OUT_LOCAL} -> ${OUT_BUCKET}"
gcloud storage rsync --recursive "${OUT_LOCAL}" "${OUT_BUCKET}"
echo "[OK] Outputs synced."