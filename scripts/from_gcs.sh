#!/usr/bin/env bash
# download training data from GCS to local machine.
set -euo pipefail

SRC="gs://229_project_bucket/data/train/train"
DST="${HOME}/project/data/train/train"

mkdir -p "${DST}"
echo "[INFO] Syncing ${SRC} -> ${DST}"
gcloud storage rsync --recursive "${SRC}" "${DST}"

echo "[OK] Data available under: ${DST}"
ls -lah "${DST}"
