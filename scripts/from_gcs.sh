#!/usr/bin/env bash
# Sync training data from Google Cloud Storage to local machine.
# Downloads images and labels from GCS bucket to ~/project/data/train/train/
set -euo pipefail

# Source 
SRC="gs://229_project_bucket/data/train/train"
# Destination in cache
DST="${HOME}/project/data/train/train"

mkdir -p "${DST}"
echo "[INFO] Syncing ${SRC} -> ${DST}"
gcloud storage rsync --recursive "${SRC}" "${DST}"

echo "[OK] Data available under: ${DST}"
ls -lah "${DST}"
