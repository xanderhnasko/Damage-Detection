# Baseline Model

In GCC VM:

``` bash
make setup
make fetch-data
make copy-images
make manifest
make train-hurricanes
```

Data gets synced from GCS to `~/project/data/train/train/`. The manifest CSV is created from the label JSON files, extracting bounding boxes for each damage annotation. Training outputs go to timestamped directories under `~/project/outputs/`.