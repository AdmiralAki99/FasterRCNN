# Faster R-CNN — TensorFlow/Keras Research Implementation

A from-scratch implementation of **Faster R-CNN** using TensorFlow/Keras, with a modular design supporting custom datasets, advanced training utilities, and evaluation metrics.  
This repository emphasizes **readability**, **reproducibility**, and **research extensibility**.

---

## Overview

This project implements the **Faster R-CNN** object detection architecture with a **VGG-16 backbone** and modular TensorFlow/Keras components for:
- Region Proposal Network (RPN)
- RoI Pooling/Align and Classification Heads
- Custom training loop and TensorBoard logging
- Flexible validation and mAP computation

This implementation is ideal for:
- Research and experimentation
- Custom dataset adaptation
- Reproducible benchmarking and ablation studies

---

## Key Features

- **Pure TensorFlow/Keras implementation** — no external detection frameworks
- **VGG-16 backbone** without fully connected layers
- **RPN and RoI head** as modular `tf.keras.Model` components
- **Custom Pascal VOC 2012 loader** with TFDS
- **CLI utilities** for progress and validation
- **Learning rate scheduler** (warmup + hold + cosine decay)
- **Automatic metric evaluation:** mAP (VOC), Precision, Recall, F1
- **Exportable validation summaries** (CSV + JSON)

---

## Process Diagrams

This is the overall pipeline for the Faster RCNN model
<p align="center"> <img src="Flowchart-Overall Pipeline.png" alt="Overall Process Diagram"/></p>

This is the training pipeline for the Faster RCNN model

<p align="center"> <img src="Flowchart-Training Pipeline.png" alt="Training Process Diagram"/></p>

This is the inference pipeline for the Faster RCNN model

<p align="center"> <img src="Flowchart-Inference Pipeline.png" alt="Inference Process Diagram"/></p>
---

## Repository Structure

```
.
├── model.py # Backbone (VGG16), RPN, RoI Head, FasterRCNN class
├── util.py # Geometry, IoU, anchor generation, losses, metrics
├── dataset.py # Custom Pascal VOC loader & TFDS preprocessing
├── scheduler.py # Step-based LR scheduler with warmup/decay
├── cli.py # CLI utilities (progress bar, banners)
├── validate.py # Validation + mAP computation
├── requirements.txt # Python dependencies
└── README.md

```


---

## Model Overview

### 1. `VGG_16_NFCL`
Feature extractor backbone using VGG-16 (no fully connected layers).

### 2. `RegionProposalNetwork`
Generates anchors, predicts objectness scores, and bounding box deltas.

### 3. `RoIHead`
Performs classification and bounding box regression per RoI.

### 4. `FasterRCNN`
Combines backbone, RPN, and RoI Head into an end-to-end detector with:
- Custom training step logic
- Integrated validation pipeline
- Optional debug output (tensor shapes, IoUs)

---

## Installation

```bash
git clone https://github.com/<your-username>/FasterRCNN-Research.git
cd FasterRCNN-Research
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Dataset: Pascal VOC 2012
Custom loader built on **TensorFlow Datasets** with preprocessing for Faster R-CNN.

```
from dataset import load_pascal_voc_dataset

train_ds, val_ds, meta = load_pascal_voc_dataset(
    batch_size=2,
    val_pct=0.1,
    seed=1337,
    prefer_official=True
)

```
Each dataset element contains:

```
{
  "image": [B,H,W,3],        # float32
  "gt_boxes": [B,N,4],       # [xmin, ymin, xmax, ymax]
  "gt_labels": [B,N]         # int (1..num_classes)
}
```

## Training Example
```
from model import FasterRCNN
from dataset import load_pascal_voc_dataset
from scheduler import CustomStepBasedScheduler

model = FasterRCNN(num_classes=20)
train_ds, val_ds, meta = load_pascal_voc_dataset(batch_size=2)

scheduler = CustomStepBasedScheduler(
    base_learning_rate=1e-3,
    steps_per_epoch=meta["steps_per_epoch"],
    total_epochs=20,
    warmup_steps=500,
    hold_steps=500,
    minimum_learning_rate=1e-5
)
```
## Validation
Use the flexible validation script:

```
python validate.py \
  --model-module model \
  --model-fn FasterRCNN \
  --dataset-module dataset \
  --dataset-fn load_pascal_voc_dataset \
  --split val \
  --batch-size 2 \
  --run-dir logs/faster_rcnn/20251021-000629 \
  --iou-thr 0.5 \
  --score-thr 0.05 \
  --export-dir ./validation_out
```
Outputs:
* `report.json` — summary (mAP, class APs)
* `per_class_prf1.csv` — precision, recall, F1 per class
* IoU statistics and inference timings (printed to console)

## Example Validation Output

```
==== Validation Summary ====
Images evaluated:    4952
IoU threshold:       0.50
Score threshold:     0.05
mAP (VOC @ 0.50):    0.7324
Avg inference/batch: 0.1120 sec   |   Throughput: 8.9 img/s
Top-5 classes by F1:
  class 15: F1=0.874  P=0.921  R=0.835  support=287
  class 7 : F1=0.845  P=0.888  R=0.806  support=421
  class 12: F1=0.832  P=0.852  R=0.813  support=273
  class 3 : F1=0.816  P=0.805  R=0.828  support=183
  class 20: F1=0.802  P=0.851  R=0.758  support=112

```

## Custom Learning Rate Scheduler
Implements warmup, hold, and cosine decay phases.

```
from scheduler import CustomStepBasedScheduler

lr = CustomStepBasedScheduler(
    base_learning_rate=1e-3,
    steps_per_epoch=1000,
    total_epochs=20,
    warmup_steps=500,
    hold_steps=500,
    minimum_learning_rate=1e-5
)
```

## Visualizing Dataset
```
from dataset import load_pascal_voc_dataset, show_dataset_examples
_, val_ds, _ = load_pascal_voc_dataset(batch_size=2)
show_dataset_examples(val_ds.unbatch().take(5))
```
## Detection Results

Below are sample detections produced by the trained Faster R-CNN model.

<p align="center"> <img src="detection_sample1.png" width="45%" alt="Detection Example 1"/></p>

## Metrics

| Metric                      | Description                              |
| --------------------------- | ---------------------------------------- |
| **mAP (VOC)**               | Mean Average Precision @ IoU ≥ 0.5       |
| **Precision / Recall / F1** | Per-class metrics                        |
| **IoU Probes**              | Median/90th percentile IoU overlap check |
| **Throughput**              | Images/sec during validation             |

## Notes

* Modular design: swap backbones or heads easily
* All components are TensorFlow-native (`tf.keras.layers`, `tf.data`)
* Debug-friendly: tensor shapes and IoU checks logged
* Scripts support both ad-hoc validation and reproducible runs

## License

MIT License © 2025 Akhilesh Warty
Feel free to use and modify with proper attribution.
