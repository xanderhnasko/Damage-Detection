PY=python3

IMAGES_ROOT=$(HOME)/project/data/train/train/images
LABELS_ROOT=$(HOME)/project/data/train/train/labels
LOCAL_IMAGES=/mnt/disks/localssd/local_images
MANIFEST=$(HOME)/project/data/train/train/manifest.csv
OUTDIR=$(HOME)/project/outputs/$(shell date +%F_%H%M%S)
YOLO_DATA=$(CURDIR)/data/yolo_buildings
YOLO_CONFIG?=configs/full_ds.yaml
YOLO_SPLIT?=test
YOLO_DET_OUT?=$(if $(WEIGHTS),$(dir $(WEIGHTS))detect_$(YOLO_SPLIT).csv,$(HOME)/project/outputs/yolo/detect_$(YOLO_SPLIT).csv)
YOLO_BUCKET?=gs://229_project_bucket/outputs

RESNET_TRAIN_MANIFEST?=$(HOME)/project/outputs/yolo/detect_train.csv
RESNET_VAL_MANIFEST?=$(HOME)/project/outputs/yolo/detect_val.csv
RESNET_TEST_MANIFEST?=$(HOME)/project/outputs/yolo/detect_test.csv
RESNET_OUTDIR?=$(OUTDIR)

setup:
	pip install -r requirements.txt || true

fetch-data:
	bash scripts/from_gcs.sh

copy-images:
	sudo mkdir -p $(LOCAL_IMAGES)
	sudo cp -r $(IMAGES_ROOT)/* $(LOCAL_IMAGES)/

manifest:
	$(PY) scripts/make_manifests.py --images_root $(IMAGES_ROOT) --labels_root $(LABELS_ROOT) --out_csv $(MANIFEST) --local_root $(LOCAL_IMAGES) --pad 16

yolo-data:
	$(PY) scripts/make_yolo_dataset.py --images_root $(LOCAL_IMAGES) --labels_root $(LABELS_ROOT) --out_dir $(YOLO_DATA) --config $(YOLO_CONFIG) --pad 8

yolo-train:
	$(PY) src/train_yolo.py --data configs/yolo_building.yaml --project $(HOME)/project/outputs/yolo

yolo-detect:
	@if [ -z "$(WEIGHTS)" ]; then echo "Set WEIGHTS=/path/to/yolo/weights.pt"; exit 1; fi
	$(PY) src/yolo_detect.py --weights $(WEIGHTS) --data_root $(YOLO_DATA) --split $(YOLO_SPLIT) --out_csv $(YOLO_DET_OUT) --manifest_gt $(MANIFEST) --keep_unmatched

yolo-detect-test:
	@if [ -z "$(WEIGHTS)" ]; then echo "Set WEIGHTS=/path/to/yolo/weights.pt"; exit 1; fi
	$(PY) src/yolo_detect.py --weights $(WEIGHTS) --data_root $(YOLO_DATA) --split test --out_csv $(YOLO_DET_OUT) --manifest_gt $(MANIFEST) --keep_unmatched

yolo-summary:
	@if [ -z "$(DET_CSV)" ]; then echo "Set DET_CSV=/path/to/detect.csv (e.g., from yolo-detect-test)"; exit 1; fi
	$(PY) scripts/summarize_detections.py --det_csv $(DET_CSV) --manifest $(MANIFEST)

resnet:
	TORCHVISION_USE_LIBJPEG_TURBO=1 $(PY) src/train_resnet.py --train_manifest $(RESNET_TRAIN_MANIFEST) --val_manifest $(RESNET_VAL_MANIFEST) --test_manifest $(RESNET_TEST_MANIFEST) --out_dir $(RESNET_OUTDIR) --background_label 4

sync-outputs:
	gsutil -m cp -r $(HOME)/project/outputs $(YOLO_BUCKET)/
