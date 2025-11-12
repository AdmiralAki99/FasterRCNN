
#!/usr/bin/env python3
"""
validate.py — Flexible validator for Faster R-CNN–style models.

Usage (example):
  python validate.py \
    --model-module model \
    --model-fn FasterRCNN \
    --run-dir runs/exp_001 \
    --weights-subdir weights \
    --dataset-module dataset \
    --dataset-fn load_pascal_voc_dataset \
    --dataset-kwargs '{"val_pct": 0.25, "seed": 2025, "prefer_official": false}' \
    --split val \
    --batch-size 3 \
    --iou-thr 0.5 \
    --score-thr 0.05 \
    --max-batches 0 \
    --export-dir ./validation_out

Assumptions:
- Your model factory (function or class) can be called as fn(**kwargs) where kwargs come from --model-kwargs.
- Your dataset loader returns (train_ds, val_ds, meta) or (train_ds, val_ds) and accepts kwargs from --dataset-kwargs.
- Each dataset batch yields dicts with:
    {
      "image":     [B,H,W,3] float32,
      "gt_boxes":  ragged or [B,Ni,4] in xyxy,
      "gt_labels": ragged or [B,Ni]    ints in 1..C
    }
- Model(images, training=False) returns dict with keys "boxes","scores","labels" of shapes [B,M,4], [B,M], [B,M].
"""

import os
import sys
import time
import json
import argparse
import importlib
from typing import Tuple, Dict, Any, List
from glob import glob
from pathlib import Path
import re

import numpy as np
import tensorflow as tf

# Try to import project util.py if present
UTIL = None
try:
    import util as UTIL
except Exception:
    UTIL = None
    print("[validate] util.py not found or import failed — fallbacks will be used where needed.", file=sys.stderr)

# ----------------------------
# Utilities & fallbacks
# ----------------------------

def _iou_xyxy_single(box: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    x1 = np.maximum(box[0], boxes[:,0])
    y1 = np.maximum(box[1], boxes[:,1])
    x2 = np.minimum(box[2], boxes[:,2])
    y2 = np.minimum(box[3], boxes[:,3])
    inter = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
    a1 = np.maximum(0.0, (box[2]-box[0])) * np.maximum(0.0, (box[3]-box[1]))
    a2 = np.maximum(0.0, (boxes[:,2]-boxes[:,0])) * np.maximum(0.0, (boxes[:,3]-boxes[:,1]))
    union = a1 + a2 - inter + 1e-9
    return inter / union

def _compute_map_voc_fallback(det_boxes, det_scores, det_labels,
                              gt_boxes, gt_labels,
                              num_classes, iou_thr=0.5, use_11pt=False):
    B = len(det_boxes)
    C = num_classes
    ap_per_class = np.full(C, np.nan, dtype=np.float32)
    for c in range(1, C+1):
        gts = []
        for b in range(B):
            gmask = (np.asarray(gt_labels[b]) == c)
            gtb = np.asarray(gt_boxes[b], dtype=float)[gmask]
            gts.append({"boxes": gtb, "matched": np.zeros(len(gtb), dtype=bool)})
        npos = int(sum(len(g["boxes"]) for g in gts))
        dets = []
        for b in range(B):
            dmask = (np.asarray(det_labels[b]) == c)
            for i in np.where(dmask)[0]:
                dets.append((float(det_scores[b][i]), b, np.asarray(det_boxes[b][i], dtype=float)))
        if npos == 0 and len(dets) == 0:
            ap_per_class[c-1] = np.nan
            continue
        if len(dets) == 0:
            ap_per_class[c-1] = 0.0
            continue
        dets.sort(key=lambda t: -t[0])
        tp = np.zeros(len(dets)); fp = np.zeros(len(dets))
        for i, (_, b, box) in enumerate(dets):
            gt_boxes_b = gts[b]["boxes"]
            if gt_boxes_b.shape[0] == 0:
                fp[i] = 1.0; continue
            ious = _iou_xyxy_single(box, gt_boxes_b)
            j = int(np.argmax(ious))
            if ious[j] >= iou_thr and not gts[b]["matched"][j]:
                tp[i] = 1.0; gts[b]["matched"][j] = True
            else:
                fp[i] = 1.0
        tp_c = np.cumsum(tp); fp_c = np.cumsum(fp)
        rec = tp_c / (npos + 1e-9)
        prec = tp_c / (tp_c + fp_c + 1e-9)

        if use_11pt:
            ap = 0.0
            for t in np.linspace(0,1,11):
                p = np.max(prec[rec >= t]) if np.any(rec >= t) else 0.0
                ap += p / 11.0
        else:
            mrec = np.concatenate(([0.0], rec, [1.0]))
            mpre = np.concatenate(([0.0], prec, [0.0]))
            for i in range(mpre.size-1, 0, -1):
                mpre[i-1] = np.maximum(mpre[i-1], mpre[i])
            idx = np.where(mrec[1:] != mrec[:-1])[0]
            ap = np.sum((mrec[idx+1] - mrec[idx]) * mpre[idx+1])
        ap_per_class[c-1] = ap
    mAP = np.nanmean(ap_per_class) if np.any(~np.isnan(ap_per_class)) else 0.0
    return float(mAP), ap_per_class

def _evaluation_prf1_report_fallback(det_boxes_b, det_scores_b, det_labels_b,
                                     gt_boxes_b, gt_labels_b, num_classes,
                                     iou_thr=0.5, score_thr=0.0):
    B = len(det_boxes_b)
    report = {c: {"TP":0,"FP":0,"FN":0,"precision":0.0,"recall":0.0,"f1":0.0,"support":0} for c in range(1, num_classes+1)}
    gts = []
    for b in range(B):
        pool = {}
        for c in range(1, num_classes+1):
            mask = (np.asarray(gt_labels_b[b]) == c)
            g = np.asarray(gt_boxes_b[b], dtype=float)[mask]
            pool[c] = {"boxes": g, "matched": np.zeros(len(g), dtype=bool)}
        gts.append(pool)
    for c in range(1, num_classes+1):
        dets = []
        for b in range(B):
            m = (np.asarray(det_labels_b[b]) == c) & (np.asarray(det_scores_b[b]) >= score_thr)
            idxs = np.where(m)[0]
            for i in idxs:
                dets.append((float(det_scores_b[b][i]), b, np.asarray(det_boxes_b[b][i], dtype=float)))
        dets.sort(key=lambda t: -t[0])
        tp=fp=0
        support = sum(len(gts[b][c]["boxes"]) for b in range(B))
        for score, b, box in dets:
            gt_boxes = gts[b][c]["boxes"]
            if len(gt_boxes) == 0:
                fp += 1; continue
            ious = _iou_xyxy_single(box, gt_boxes)
            j = int(np.argmax(ious))
            if ious[j] >= iou_thr and not gts[b][c]["matched"][j]:
                tp += 1; gts[b][c]["matched"][j] = True
            else:
                fp += 1
        fn = sum(int((~gts[b][c]["matched"]).sum()) for b in range(B))
        prec = tp/(tp+fp+1e-9); rec = tp/(support+1e-9); f1=(2*prec*rec)/(prec+rec+1e-9)
        report[c] = {"TP":tp,"FP":fp,"FN":fn,"precision":float(prec),"recall":float(rec),"f1":float(f1),"support":int(support)}
    return report

# Prefer util.py if present
compute_map_voc = getattr(UTIL, "compute_map_voc", _compute_map_voc_fallback)
evaluation_prf1_report = getattr(UTIL, "evaluation_prf1_report", _evaluation_prf1_report_fallback)

# ----------------------------
# Model / dataset adaptors
# ----------------------------

def dynamic_import(module_name: str):
    try:
        return importlib.import_module(module_name)
    except Exception as e:
        print(f"[validate] Failed to import module '{module_name}': {e}", file=sys.stderr)
        sys.exit(2)

def get_model(model_module: str, model_fn: str, model_kwargs_json: str):
    mm = dynamic_import(model_module)
    fn = getattr(mm, model_fn, None)
    if fn is None:
        print(f"[validate] '{model_fn}' not found in module '{model_module}'.", file=sys.stderr)
        sys.exit(2)
    try:
        kwargs = json.loads(model_kwargs_json) if model_kwargs_json else {}
        if not isinstance(kwargs, dict):
            raise ValueError("--model-kwargs must be a JSON object")
    except Exception as e:
        print(f"[validate] Could not parse --model-kwargs: {e}", file=sys.stderr)
        sys.exit(2)
    model = fn(**kwargs)
    return model

def get_dataset(dataset_module: str, dataset_fn: str, split: str, batch_size: int, dataset_kwargs_json: str):
    dm = dynamic_import(dataset_module)
    loader = getattr(dm, dataset_fn, None)
    if loader is None:
        print(f"[validate] '{dataset_fn}' not found in module '{dataset_module}'.", file=sys.stderr)
        sys.exit(2)

    try:
        dkwargs = json.loads(dataset_kwargs_json) if dataset_kwargs_json else {}
        if not isinstance(dkwargs, dict):
            raise ValueError("--dataset-kwargs must be a JSON object")
    except Exception as e:
        print(f"[validate] Could not parse --dataset-kwargs: {e}", file=sys.stderr)
        sys.exit(2)
    dkwargs["batch_size"] = batch_size
    if "return_meta" in loader.__code__.co_varnames:
        dkwargs.setdefault("return_meta", True)

    ds_objects = loader(**dkwargs)

    if isinstance(ds_objects, tuple) and len(ds_objects) == 3:
        train_ds, val_ds, meta = ds_objects
    elif isinstance(ds_objects, tuple) and len(ds_objects) == 2:
        train_ds, val_ds = ds_objects
        meta = {}
    else:
        raise ValueError("Dataset loader must return (train_ds, val_ds[, meta]).")

    if split.lower() in ("val", "valid", "validation"):
        ds = val_ds
    elif split.lower() in ("test",):
        ds = val_ds
    else:
        ds = train_ds
    return ds, meta

def map_model_outputs(outputs: Dict[str, tf.Tensor]) -> Tuple[np.ndarray,np.ndarray,np.ndarray]:
    boxes = outputs.get("Boxes")
    scores = outputs.get("Scores")
    labels = outputs.get("Classes")
    if boxes is None or scores is None or labels is None:
        raise KeyError("Model outputs must include keys 'Boxes', 'Scores', and 'Classes'.")
    return boxes.numpy(), scores.numpy(), labels.numpy()

def to_numpy_list(x):
    if isinstance(x, tf.RaggedTensor):
        return [np.asarray(t) for t in x.to_list()]
    if tf.is_tensor(x):
        x = x.numpy()
    x = np.asarray(x, dtype=object)
    if x.dtype != object and x.ndim >= 2:
        return [x[i] for i in range(x.shape[0])]
    return list(x)

# ----------------------------
# Weights auto-discovery
# ----------------------------

def _find_latest_weights(run_dir: str, subdir: str = "weights") -> str:
    base = Path(run_dir) / subdir
    if not base.exists():
        raise FileNotFoundError(f"No such directory: {base}")
    pattern = str(base / "weights_*.weights.h5")
    candidates = glob(pattern)
    if not candidates:
        raise FileNotFoundError(f"No weights matching {pattern}")
    def _key(p):
        name = Path(p).name
        m = re.match(r"weights_(\\d{8}-\\d{6})\\.weights\\.h5$", name)
        if m:
            ts = m.group(1)
            return ts.replace("-", "")
        return Path(p).stat().st_mtime
    candidates.sort(key=_key, reverse=True)
    return candidates[0]

# ----------------------------
# Main eval loop
# ----------------------------

def run_validation(args):
    os.makedirs(args.export_dir, exist_ok=True)

    # Resolve weights if not provided
    if not args.weights:
        if not args.run_dir:
            print("[validate] Either --weights or --run-dir must be provided.", file=sys.stderr)
            sys.exit(2)
        try:
            args.weights = _find_latest_weights(args.run_dir, args.weights_subdir)
            print(f"[validate] Auto-selected latest weights: {args.weights}")
        except Exception as e:
            print(f"[validate] Failed to find latest weights in run dir: {e}", file=sys.stderr)
            sys.exit(2)

    print("[validate] Building model...")
    model = get_model(args.model_module, args.model_fn, args.model_kwargs)
    print("[validate] Loading weights:", args.weights)
    model.load_weights(args.weights)

    print("[validate] Loading dataset...")
    dataset, meta = get_dataset(args.dataset_module, args.dataset_fn, args.split, args.batch_size, args.dataset_kwargs)

    DET_BOXES: List[np.ndarray] = []
    DET_SCORES: List[np.ndarray] = []
    DET_LABELS: List[np.ndarray] = []
    GT_BOXES:  List[np.ndarray] = []
    GT_LABELS: List[np.ndarray] = []

    n_images = 0
    inf_times = []

    print("[validate] Running inference...")
    for step, batch in enumerate(dataset):
                    
        if args.max_batches and step >= args.max_batches:
            break
        images = tf.convert_to_tensor(batch["image"])
        GT_BOXES.extend(to_numpy_list(batch["gt_boxes"]))
        GT_LABELS.extend(to_numpy_list(batch["gt_labels"]))

        t0 = time.time()
        outputs = model(images, training=False)
        dt = time.time() - t0
        inf_times.append(float(dt))

        boxes_b, scores_b, labels_b = map_model_outputs(outputs)
        
        boxes_b = boxes_b[..., [1, 0, 3, 2]]
        
        if step == 0:
            _first_scores_min = float(scores_b.min()) if scores_b.size else 0.0
            _first_scores_max = float(scores_b.max()) if scores_b.size else 0.0
            _first_labels_min = int(labels_b.min()) if labels_b.size else -999
            _first_labels_max = int(labels_b.max()) if labels_b.size else -999
        
        if labels_b.size and labels_b.min() == 0:
            labels_b = labels_b + 1
        
        if boxes_b.size:
            max_coord = float(np.max(boxes_b))
            if max_coord <= 2.0:  # likely normalized
                H = int(images.shape[1]); W = int(images.shape[2])
                boxes_b[..., 0] *= W  # x1
                boxes_b[..., 2] *= W  # x2
                boxes_b[..., 1] *= H  # y1
                boxes_b[..., 3] *= H  # y2
        
        for b in range(boxes_b.shape[0]):
            keep = scores_b[b] >= args.score_thr
            DET_BOXES.append(boxes_b[b][keep])
            DET_SCORES.append(scores_b[b][keep])
            DET_LABELS.append(labels_b[b][keep].astype(np.int32))

        n_images += int(images.shape[0])
        if (step+1) % max(1, args.log_every) == 0:
            print(f"[validate] Processed batches: {step+1}, images: {n_images}")

    if "num_classes" in meta:
        num_classes = int(meta["num_classes"])
    else:
        max_gt = max((int(np.max(g)) if len(g)>0 else 0) for g in GT_LABELS) if len(GT_LABELS)>0 else 0
        max_dt = max((int(np.max(d)) if len(d)>0 else 0) for d in DET_LABELS) if len(DET_LABELS)>0 else 0
        num_classes = max(max_gt, max_dt)
        if num_classes <= 0:
            print("[validate] Could not infer num_classes; defaulting to 1.", file=sys.stderr)
            num_classes = 1
            
    if n_images == 0:
        print("[validate] ERROR: Evaluated 0 images. The chosen split seems empty. "
          "Check --split and --dataset-kwargs (e.g., val_pct).", file=sys.stderr)
        sys.exit(3)
        
    # Testing checking
    
    import random
    random.seed(0)
    sample_idx = random.sample(range(len(DET_BOXES)), k=min(50, len(DET_BOXES)))
    overlaps = []
    for i in sample_idx:
        db = DET_BOXES[i]
        gb = GT_BOXES[i]
        if len(db) == 0 or len(gb) == 0:
            overlaps.append(0.0); continue
        # take up to 50 biggest-score dets if scores available
        try:
            order = np.argsort(-DET_SCORES[i])[:50]
        except Exception:
            order = np.arange(min(50, len(db)))
        db = db[order]
        # best IoU for any det vs any GT (class-agnostic)
        from itertools import product
        maxiou = 0.0
        for d in db:
            # vectorized IoU vs all GT
            x1 = np.maximum(d[0], gb[:,0]); y1 = np.maximum(d[1], gb[:,1])
            x2 = np.minimum(d[2], gb[:,2]); y2 = np.minimum(d[3], gb[:,3])
            inter = np.maximum(0.0, x2-x1) * np.maximum(0.0, y2-y1)
            ad = np.maximum(0.0, d[2]-d[0]) * np.maximum(0.0, d[3]-d[1])
            ag = np.maximum(0.0, gb[:,2]-gb[:,0]) * np.maximum(0.0, gb[:,3]-gb[:,1])
            iou = inter / (ad + ag - inter + 1e-9)
            maxiou = max(maxiou, float(iou.max()))
        overlaps.append(maxiou)
    print("[probe] class-agnostic max IoU (50 imgs):",
      f"median={np.median(overlaps):.3f}  p90={np.quantile(overlaps,0.9):.3f}  max={np.max(overlaps):.3f}")
    
    total_dt = sum(int(len(x)) for x in DET_BOXES)
    total_gt = sum(int(len(x)) for x in GT_BOXES)
    print(f"[probe] totals  det={total_dt}  gt={total_gt}")

    # look at raw model output of the first batch we saw (pre-filter)
    # store a few first-batch values during the loop (see probe lines below)
    try:
        print("[probe] first-batch raw:",
            f"labels_range=({int(_first_labels_min)},{int(_first_labels_max)})",
            f"scores_range=({float(_first_scores_min):.3f},{float(_first_scores_max):.3f})")
    except NameError:
        pass

    print("[validate] Computing metrics...")
    if UTIL is not None and hasattr(UTIL, "compute_map_voc"):
        mAP, ap_per_class = UTIL.compute_map_voc(DET_BOXES, DET_SCORES, DET_LABELS, GT_BOXES, GT_LABELS,
                                                 num_classes=num_classes, iou_thr=args.iou_thr, use_11pt=False)
    else:
        mAP, ap_per_class = compute_map_voc(DET_BOXES, DET_SCORES, DET_LABELS, GT_BOXES, GT_LABELS,
                                            num_classes=num_classes, iou_thr=args.iou_thr, use_11pt=False)

    if UTIL is not None and hasattr(UTIL, "evaluation_prf1_report"):
        prf1 = UTIL.evaluation_prf1_report(DET_BOXES, DET_SCORES, DET_LABELS, GT_BOXES, GT_LABELS,
                                           num_classes=num_classes, iou_thr=args.iou_thr, score_thr=args.score_thr)
    else:
        prf1 = evaluation_prf1_report(DET_BOXES, DET_SCORES, DET_LABELS, GT_BOXES, GT_LABELS,
                                      num_classes=num_classes, iou_thr=args.iou_thr, score_thr=args.score_thr)

    avg_inf = float(np.mean(inf_times)) if len(inf_times) else 0.0
    ips = (n_images / sum(inf_times)) if sum(inf_times) > 0 else 0.0

    report = {
        "n_images": n_images,
        "num_classes": num_classes,
        "iou_thr": args.iou_thr,
        "score_thr": args.score_thr,
        "mAP": float(mAP),
        "ap_per_class": [None if (np.isnan(x) or x is None) else float(x) for x in ap_per_class],
        "per_class": prf1,
        "avg_inference_sec_per_batch": avg_inf,
        "throughput_images_per_sec": float(ips),
    }
    out_json = os.path.join(args.export_dir, "report.json")
    with open(out_json, "w") as f:
        json.dump(report, f, indent=2)
    print(f"[validate] Wrote {out_json}")

    import csv
    out_csv = os.path.join(args.export_dir, "per_class_prf1.csv")
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["class_id","precision","recall","f1","support","TP","FP","FN","AP"])
        for c in range(1, num_classes+1):
            row = prf1.get(c, {})
            ap = ap_per_class[c-1] if c-1 < len(ap_per_class) else np.nan
            w.writerow([c,
                        row.get("precision",0.0),
                        row.get("recall",0.0),
                        row.get("f1",0.0),
                        row.get("support",0),
                        row.get("TP",0),
                        row.get("FP",0),
                        row.get("FN",0),
                        ("" if np.isnan(ap) else float(ap))])
    print(f"[validate] Wrote {out_csv}")

    print("\n==== Validation Summary ====")
    print(f"Images evaluated:    {n_images}")
    print(f"IoU threshold:       {args.iou_thr}")
    print(f"Score threshold:     {args.score_thr}")
    print(f"mAP (VOC @ {args.iou_thr:.2f}): {mAP:.4f}")
    print(f"Avg inference/batch: {avg_inf:.4f} sec   |   Throughput: {ips:.2f} img/s")
    print("Top-5 classes by F1:")
    scores = []
    for c in range(1, num_classes+1):
        f1 = prf1.get(c,{}).get("f1", 0.0)
        scores.append((f1, c))
    scores.sort(reverse=True)
    for f1, c in scores[:5]:
        prec = prf1.get(c,{}).get("precision", 0.0)
        rec  = prf1.get(c,{}).get("recall", 0.0)
        sup  = prf1.get(c,{}).get("support", 0)
        print(f"  class {c:>2}: F1={f1:.3f}  P={prec:.3f}  R={rec:.3f}  support={sup}")

def parse_args():
    p = argparse.ArgumentParser(description="Validation script for Faster R-CNN–style models.")
    p.add_argument("--model-module", type=str, required=True, help="Python module path with model factory/class, e.g., 'model'")
    p.add_argument("--model-fn", type=str, default="build_model", help="Factory/class to instantiate the model (default: build_model)")
    p.add_argument("--model-kwargs", type=str, default="{}", help="JSON dict of kwargs passed to model factory/class")
    p.add_argument("--weights", type=str, default="", help="Path to model weights (.h5 or checkpoint). If empty, will try --run-dir")
    p.add_argument("--run-dir", type=str, default="", help="Experiment run directory that contains the weights subdir")
    p.add_argument("--weights-subdir", type=str, default="weights", help="Subdirectory name that contains timestamped weights")
    p.add_argument("--dataset-module", type=str, required=True, help="Python module path with dataset loader, e.g., 'dataset'")
    p.add_argument("--dataset-fn", type=str, default="load_pascal_voc_dataset", help="Loader returning (train_ds, val_ds[, meta])")
    p.add_argument("--dataset-kwargs", type=str, default="{}", help="JSON dict of kwargs passed to dataset loader")
    p.add_argument("--split", type=str, default="val", choices=["train","val","test"], help="Which split to validate on")
    p.add_argument("--batch-size", type=int, default=2, help="Batch size for evaluation")
    p.add_argument("--max-batches", type=int, default=0, help="Limit number of batches (0=all)")
    p.add_argument("--iou-thr", type=float, default=0.5, help="IoU threshold for matching")
    p.add_argument("--score-thr", type=float, default=0.05, help="Score threshold for counting detections")
    p.add_argument("--export-dir", type=str, default="./validation_out", help="Directory to save reports")
    p.add_argument("--log-every", type=int, default=50, help="Log every N batches")
    return p.parse_args()

if __name__ == "__main__":
    args = parse_args()
    run_validation(args)
