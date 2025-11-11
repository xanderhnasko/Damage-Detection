PY=python

IMAGES_ROOT=$(HOME)/project/data/train/train/images
LABELS_ROOT=$(HOME)/project/data/train/train/labels
MANIFEST=$(HOME)/project/data/train/train/manifest.csv
OUTDIR=$(HOME)/project/outputs/$(shell date +%F_%H%M%S)

setup:
	pip install -r requirements.txt || true

fetch-data:
	bash scripts/from_gcs.sh

manifest:
	$(PY) src/make_manifest.py --images_root $(IMAGES_ROOT) --labels_root $(LABELS_ROOT) --out_csv $(MANIFEST) --pad 16

train-post:
	$(PY) src/train_resnet18.py --manifest $(MANIFEST) --out_dir $(OUTDIR) --epochs 10

sync-outputs:
	bash scripts/output_to_gcs.sh
