PY=python3

IMAGES_ROOT=$(HOME)/project/data/train/train/images
LABELS_ROOT=$(HOME)/project/data/train/train/labels
LOCAL_IMAGES=/mnt/disks/localssd/local_images
MANIFEST=$(HOME)/project/data/train/train/manifest.csv
OUTDIR=$(HOME)/project/outputs/$(shell date +%F_%H%M%S)
YOLO_DATA=$(CURDIR)/data/yolo_buildings
YOLO_DET_CSV=$(HOME)/project/outputs/yolo/detections.csv
YOLO_CONFIG?=configs/hurricanes.yaml
YOLO_SPLIT?=test
YOLO_DET_OUT?=$(YOLO_DET_CSV)
YOLO_MANIFEST?=$(MANIFEST)

setup:
	pip install -r requirements.txt || true

fetch-data:
	bash scripts/from_gcs.sh

copy-images:
	sudo mkdir -p $(LOCAL_IMAGES)
	sudo cp -r $(IMAGES_ROOT)/* $(LOCAL_IMAGES)/

manifest:
	$(PY) scripts/make_manifests.py --images_root $(IMAGES_ROOT) --labels_root $(LABELS_ROOT) --out_csv $(MANIFEST) --local_root $(LOCAL_IMAGES) --pad 16

train:
	TORCHVISION_USE_LIBJPEG_TURBO=1 $(PY) src/train_resnet.py --manifest $(MANIFEST) --out_dir $(OUTDIR) --epochs 10

sync-outputs:
	bash scripts/output_to_gcs.sh

train-hurricanes:
	TORCHVISION_USE_LIBJPEG_TURBO=1 $(PY) src/train_resnet.py --manifest $(MANIFEST) --out_dir $(OUTDIR) --epochs 20 --config configs/hurricanes.yaml

yolo-data:
	$(PY) scripts/make_yolo_dataset.py --images_root $(LOCAL_IMAGES) --labels_root $(LABELS_ROOT) --out_dir $(YOLO_DATA) --config $(YOLO_CONFIG) --pad 8

yolo-train:
	$(PY) src/train_yolo.py --data configs/yolo_building.yaml --project $(HOME)/project/outputs/yolo

yolo-detect:
	@if [ -z "$(WEIGHTS)" ]; then echo "Set WEIGHTS=/path/to/yolo/weights.pt"; exit 1; fi
	$(PY) src/yolo_detect.py --weights $(WEIGHTS) --data_root $(YOLO_DATA) --split $(YOLO_SPLIT) --out_csv $(YOLO_DET_OUT) --manifest_gt $(YOLO_MANIFEST)
