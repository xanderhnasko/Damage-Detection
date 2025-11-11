#!/usr/bin/env bash
set -euo pipefail
BUCKET_ROOT="${1:?usage: ./list_gcs.sh gs://bucket/path}"

mkdir -p gcs_lists
# two flat lists of images and labels
gcloud storage ls --recursive "${BUCKET_ROOT}/images/**" > gcs_lists/images.all.txt
gcloud storage ls --recursive "${BUCKET_ROOT}/labels/**" > gcs_lists/labels.all.txt

echo "Wrote gcs_lists/images.all.txt and gcs_lists/labels.all.txt"

