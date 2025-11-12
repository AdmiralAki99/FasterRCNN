import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # Suppress TensorFlow logging (1: errors, 2: warnings, 3: info)
os.environ["GLOG_minloglevel"] = "2"
os.environ["ABSL_LOG_SEVERITY_THRESHOLD"] = "3"
os.environ["TF_DISABLE_XLA"] = "1"

import argparse
import datetime
from pathlib import Path
import tensorflow as tf, sys
from tensorboard import program
import webbrowser
import matplotlib.pyplot as plt
import numpy as np
from util import compute_map_voc, evaluation_prf1_report

tf.get_logger().setLevel("ERROR") # Further suppress TF logging

gpus = tf.config.list_physical_devices('GPU')
for _g in gpus:
    try:
        tf.config.experimental.set_memory_growth(_g, True)
    except Exception:
        pass


from dataset import *
from helpers import make_run_dir, get_viz_path, _StderrFilter
from cli import *
from model import *
from scheduler import *

try:
    sys.stderr = _StderrFilter(r"(gpu_timer\.cc:114|Skipping the delay kernel)")
except Exception:
    pass

RUN_STAMP = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
BASE_LOG_DIR = Path("./logs/faster_rcnn")  
TENSORBOARD_LOG_DIR = BASE_LOG_DIR / RUN_STAMP  
TENSORBOARD_LOG_DIR.mkdir(parents=True, exist_ok=True)
DEBUG_IMAGE_ROOT = Path("./Debug Images")
CKPT_DIR = Path(TENSORBOARD_LOG_DIR) / "checkpoints"
CKPT_DIR.mkdir(parents=True, exist_ok=True)


def start_tensorboard(logdir, port=6006, host="127.0.0.1", open_browser=True):
    tb = program.TensorBoard()
    tb.configure(argv=[None, "--logdir", str(logdir), "--port", str(port), "--host", str(host)])
    url = tb.launch()
    print(f"TensorBoard started at: {url}")
    if open_browser:
        try:
            webbrowser.open(url)
        except Exception:
            pass
    return url

def train_step(batch,trainable_backbone: bool = True):
    with tf.GradientTape() as tape:
        outputs = model(
            batch['image'],
            gt_boxes=batch['gt_boxes'],
            gt_labels=batch['gt_labels'],
            training=True,
            debug=False
        )
        loss = outputs['total_loss']
        
    # Getting all the variables to be trained
    trainable_variables = []
    if trainable_backbone:
        trainable_variables += backbone_vars
    
    # Adding RPN and ROI head variables since they are always trainable
    trainable_variables += rpn_vars
    trainable_variables += roi_vars
        
    # Getting all the gradients    
    grads = tape.gradient(loss, trainable_variables)
    
    # Applying gradients to the respective optimizers
    head = 0
    if trainable_backbone:
        grad_backbone = grads[head:head+len(backbone_vars)]
        head += len(backbone_vars)
    else:
        grad_backbone = []
        
    grad_rpn = grads[head:head+len(rpn_vars)]
    head += len(rpn_vars)
    
    grad_roi = grads[head:head+len(roi_vars)]
    
    if trainable_backbone:
        optimizer_backbone.apply_gradients(zip(grad_backbone, backbone_vars))
        
    optimizer_rpn.apply_gradients(zip(grad_rpn, rpn_vars))
    optimizer_roi.apply_gradients(zip(grad_roi, roi_vars))
    
    return outputs

def save_final_weights(model, run_dir, subdir="weights"):
    """
    Saves weights as HDF5 using a timestamped filename:
      <run_dir>/<subdir>/weights_YYYYmmdd-HHMMSS.weights.h5
    """
    out_dir = Path(run_dir) / subdir
    out_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    weights_path = out_dir / f"weights_{timestamp}.weights.h5"

    if not getattr(model, "built", False):
        try:
            model.build(input_shape=(None, None, None, 3))  
        except Exception:
            pass  

    model.save_weights(str(weights_path))
    print(f"[✓] Saved model weights to: {weights_path}")
    return str(weights_path)

def make_epoch_dataset(ds, steps_per_epoch, skip=0):
    """
    Returns a dataset view for one epoch, skipping `skip` steps if resuming mid-epoch.
    If your dataset doesn't support .skip() efficiently, the training loop still guards with a Python-side fast-forward.
    """
    if skip > 0:
        try:
            ds = ds.skip(int(skip))
        except Exception:
            pass
    return ds

def latest_run_with_ckpt(base: Path) -> Path | None:
    """Return newest subdir under base that contains a 'checkpoints' folder with files."""
    if not base.exists():
        return None
    candidates = []
    for p in base.iterdir():
        if p.is_dir() and (p / "checkpoints").exists():
            if any((p / "checkpoints").iterdir()):
                candidates.append(p)
    if not candidates:
        return None
    # newest by mtime
    candidates.sort(key=lambda d: d.stat().st_mtime, reverse=True)
    return candidates[0]

def _to_numpy_list(x):
    # Accept ragged or dense tensors/lists and return [np.ndarray(...)] per image
    if isinstance(x, tf.RaggedTensor):
        return [np.asarray(t) for t in x.to_list()]
    if tf.is_tensor(x):
        x = x.numpy()
    x = np.asarray(x, dtype=object)
    if x.dtype != object and x.ndim >= 2:
        return [x[i] for i in range(x.shape[0])]
    return list(x)

def _map_model_outputs_to_metrics(res, images):
    """
    Your model returns normalized [y1,x1,y2,x2] and labels, scores per image.
    Convert to pixel [x1,y1,x2,y2] so they match GT format.
    """
    boxes_yxyx = res['Boxes']      # (B, M, 4) normalized yxyx
    scores     = res['Scores']     # (B, M)
    labels     = res['Classes']    # (B, M) expected 1..C (no background)

    # Ensure NumPy for easy per-image handling
    if tf.is_tensor(boxes_yxyx): boxes_yxyx = boxes_yxyx.numpy()
    if tf.is_tensor(scores):     scores     = scores.numpy()
    if tf.is_tensor(labels):     labels     = labels.numpy()

    B = boxes_yxyx.shape[0]
    det_boxes  = []
    det_scores = []
    det_labels = []

    for b in range(B):
        H = int(images.shape[1]); W = int(images.shape[2])
        y1, x1, y2, x2 = [boxes_yxyx[b][..., i] for i in range(4)]
        # scale from [0,1] to pixels
        x1p = x1 * W; x2p = x2 * W
        y1p = y1 * H; y2p = y2 * H
        # swap to xyxy
        boxes_xyxy = np.stack([x1p, y1p, x2p, y2p], axis=-1)

        # sanity: enforce x1<=x2,y1<=y2
        x1f = np.minimum(boxes_xyxy[...,0], boxes_xyxy[...,2])
        y1f = np.minimum(boxes_xyxy[...,1], boxes_xyxy[...,3])
        x2f = np.maximum(boxes_xyxy[...,0], boxes_xyxy[...,2])
        y2f = np.maximum(boxes_xyxy[...,1], boxes_xyxy[...,3])
        boxes_xyxy = np.stack([x1f,y1f,x2f,y2f], axis=-1)

        det_boxes.append(boxes_xyxy.astype(np.float32))
        det_scores.append(scores[b].astype(np.float32))
        det_labels.append(labels[b].astype(np.int32))  # expected 1..C

    return det_boxes, det_scores, det_labels

def _forward_inference(images):
    # Your plotting call uses model.call(..., training=False, debug=True),
    # which yields 'Boxes'/'Scores'/'Classes' in normalized yxyx (per your plot). We mirror that.
    return model.call(images=images, training=False, debug=False)

def run_validation_short(dataset, max_batches=200, iou_thr=0.5, score_thr=0.05, log_every=50):
    """
    Iterate a few batches, compute VOC mAP@IoU and per-class PR/F1.
    Returns (mAP, report_dict, throughput_imgs_per_sec, n_images).
    """
    DET_BOXES, DET_SCORES, DET_LABELS = [], [], []
    GT_BOXES,  GT_LABELS              = [], []
    n_images = 0
    inf_times = []

    for step, batch in enumerate(dataset):
        if max_batches and step >= max_batches:
            break

        images = tf.convert_to_tensor(batch['image'])
        # collect GT as pixel xyxy (your GT is non-normalized in the plot) :contentReference[oaicite:2]{index=2}
        GT_BOXES.extend(_to_numpy_list(batch['gt_boxes']))
        GT_LABELS.extend(_to_numpy_list(batch['gt_labels']))

        t0 = tf.timestamp()
        res = _forward_inference(images)
        dt = float(tf.timestamp() - t0)
        inf_times.append(dt)

        # map model outputs to metric format
        boxes_b, scores_b, labels_b = _map_model_outputs_to_metrics(res, images)

        # filter on score
        for b in range(len(boxes_b)):
            keep = scores_b[b] >= score_thr
            DET_BOXES.append(boxes_b[b][keep])
            DET_SCORES.append(scores_b[b][keep])
            DET_LABELS.append(labels_b[b][keep])

        n_images += int(images.shape[0])
        if (step+1) % max(1, log_every) == 0:
            print(f"[val] processed {step+1} batches, images={n_images}")

    if n_images == 0:
        return 0.0, {}, 0.0, 0

    # infer class count from data (or keep your own meta if you track it)
    max_gt = max((int(np.max(g)) if len(g) else 0) for g in GT_LABELS) if GT_LABELS else 0
    max_dt = max((int(np.max(d)) if len(d) else 0) for d in DET_LABELS) if DET_LABELS else 0
    num_classes = max(max_gt, max_dt) if max(max_gt, max_dt) > 0 else 1

    # metrics (use your util.py)
    mAP, ap_per_class = compute_map_voc(
        DET_BOXES, DET_SCORES, DET_LABELS, GT_BOXES, GT_LABELS,
        num_classes=num_classes, iou_thr=iou_thr, use_11pt=False
    )
    prf1 = evaluation_prf1_report(
        DET_BOXES, DET_SCORES, DET_LABELS, GT_BOXES, GT_LABELS,
        num_classes=num_classes, iou_thr=iou_thr, score_thr=score_thr
    )

    ips = (n_images / sum(inf_times)) if sum(inf_times) > 0 else 0.0
    return float(mAP), prf1, float(ips), int(n_images)

def to_numpy_list(x):
    if isinstance(x, tf.RaggedTensor):
        return [np.asarray(t) for t in x.to_list()]
    if tf.is_tensor(x):
        x = x.numpy()
    x = np.asarray(x, dtype=object)
    if x.dtype != object and x.ndim >= 2:
        return [x[i] for i in range(x.shape[0])]
    return list(x)

# Map detections: normalized y1,x1,y2,x2 -> pixel xyxy per image
def map_det_to_xyxy_pixels(res, images, *, label_shift_if_zero_based=True):
    boxes = res["Boxes"]; scores = res["Scores"]; labels = res["Classes"]
    if tf.is_tensor(boxes):  boxes  = boxes.numpy()
    if tf.is_tensor(scores): scores = scores.numpy()
    if tf.is_tensor(labels): labels = labels.numpy()

    B = boxes.shape[0]
    out_boxes, out_scores, out_labels = [], [], []
    for b in range(B):
        H = int(images.shape[1]); W = int(images.shape[2])
        y1, x1, y2, x2 = [boxes[b][..., i] for i in range(4)]
        # these are normalized in your debug path → scale to pixels
        x1p = x1 * W; x2p = x2 * W
        y1p = y1 * H; y2p = y2 * H
        # enforce xyxy
        x1f = np.minimum(x1p, x2p); y1f = np.minimum(y1p, y2p)
        x2f = np.maximum(x1p, x2p); y2f = np.maximum(y1p, y2p)
        out_boxes.append(np.stack([x1f, y1f, x2f, y2f], axis=-1).astype(np.float32))
        out_scores.append(scores[b].astype(np.float32))
        out_labels.append(labels[b].astype(np.int32))

    if label_shift_if_zero_based and len(out_labels) and out_labels[0].size and out_labels[0].min() == 0:
        out_labels = [x + 1 for x in out_labels]
    return out_boxes, out_scores, out_labels

# Map proposals: auto-detect normalized vs pixels, output pixel xyxy per image
def map_props_to_xyxy_pixels(res, images):
    # Try a few common keys in case your dict uses another name
    props = None
    for k in ("Proposals", "RPN_Proposals", "NMS_Proposals"):
        if k in res:
            props = res[k]
            break
    if props is None:
        return []

    if tf.is_tensor(props):
        props = props.numpy()
    B = props.shape[0]
    out = []
    for b in range(B):
        H = int(images.shape[1]); W = int(images.shape[2])
        y1, x1, y2, x2 = [props[b][..., i] for i in range(4)]
        # AUTO-DETECT scale: if max <= 1.01 → normalized → scale; else assume pixels
        maxv = float(np.nanmax(props[b])) if props[b].size else 0.0
        if maxv <= 1.01:
            x1p = x1 * W; x2p = x2 * W
            y1p = y1 * H; y2p = y2 * H
        else:
            # already pixels
            x1p, y1p, x2p, y2p = x1, y1, x2, y2
        x1f = np.minimum(x1p, x2p); y1f = np.minimum(y1p, y2p)
        x2f = np.maximum(x1p, x2p); y2f = np.maximum(y1p, y2p)
        out.append(np.stack([x1f, y1f, x2f, y2f], axis=-1).astype(np.float32))
    return out

# Use YOUR TF recall on a B-sized batch (proposals/dt & gt are lists of per-image arrays)
def batch_recall_tf(proposals_xyxy_list, gt_boxes_xyxy_list, gt_labels_list, *, k=100, iou_thr=0.5):
    # Clip to top-K per image (to mirror "recall@K")
    props_k = [p[:min(k, len(p))] if len(p) else p for p in proposals_xyxy_list]
    # Pad to a batch tensor (B, P, 4) and (B, G, 4); pad labels to (B, G)
    B = len(props_k)
    Pmax = max((len(p) for p in props_k), default=0)
    Gmax = max((len(g) for g in gt_boxes_xyxy_list), default=0)

    if Pmax == 0 or Gmax == 0:
        return 0.0

    def pad_boxes(lst, maxn):
        out = []
        for arr in lst:
            a = np.zeros((maxn, 4), dtype=np.float32)
            if len(arr):
                a[:len(arr)] = arr
            out.append(a)
        return np.stack(out, axis=0)  # (B, maxn, 4)

    def pad_labels(lst, maxn):
        out = []
        for arr in lst:
            a = np.zeros((maxn,), dtype=np.int32)
            if len(arr):
                a[:len(arr)] = arr
            out.append(a)
        return np.stack(out, axis=0)  # (B, maxn)

    Pbat = pad_boxes(props_k, Pmax)
    Gbat = pad_boxes(gt_boxes_xyxy_list, Gmax)
    Lbat = pad_labels(gt_labels_list, Gmax)

    # Call YOUR TF recall util, get global recall
    _, global_recall = util_rpn_recall_post_nms(Pbat, Gbat, Lbat, iou_thresh=iou_thr)
    try:
        return float(global_recall.numpy())
    except Exception:
        return float(global_recall)

def run_validation_with_recall(
    model,
    dataset,
    tensorboard_writer=None,
    global_step=None,
    summary_tag_prefix="val",
    max_batches=0,
    iou_thr=0.5,
    score_thr=0.05,
    recall_k=100,
    print_every=100,
    label_shift_if_zero_based=True,
):
    DET_BOXES, DET_SCORES, DET_LABELS = [], [], []
    GT_BOXES,  GT_LABELS              = [], []
    # Keep all proposals for a final “global” recall too
    ALL_PROPS                          = []

    n_images = 0
    times = []

    iou_tag = str(iou_thr).replace(".", "p")
    tb_tag_recall = f"{summary_tag_prefix}/rpn_recall_at_{recall_k}_iou_{iou_tag}"

    for step, batch in enumerate(dataset):
        if max_batches and step >= max_batches:
            break

        images = tf.convert_to_tensor(batch["image"])
        gt_boxes_b  = to_numpy_list(batch["gt_boxes"])
        gt_labels_b = to_numpy_list(batch["gt_labels"])
        GT_BOXES.extend(gt_boxes_b)
        GT_LABELS.extend(gt_labels_b)

        # EAGER inference (matches your debug path)
        t0 = tf.timestamp()
        res = model(images=images, training=False, debug=True)
        dt = float(tf.timestamp() - t0)
        times.append(dt)

        # Map detections & proposals
        boxes_b, scores_b, labels_b = map_det_to_xyxy_pixels(res, images, label_shift_if_zero_based=label_shift_if_zero_based)
        props_b = map_props_to_xyxy_pixels(res, images)

        # Accumulate dets (apply score filter)
        for b in range(len(boxes_b)):
            keep = scores_b[b] >= score_thr
            DET_BOXES.append(boxes_b[b][keep])
            DET_SCORES.append(scores_b[b][keep])
            DET_LABELS.append(labels_b[b][keep])

        # Accumulate proposals
        if props_b:
            ALL_PROPS.extend(props_b)

        # Per-batch recall (THIS is the value you’re used to seeing)
        if (step + 1) % max(1, print_every) == 0 and props_b:
            r_batch = batch_recall_tf(props_b, gt_boxes_b, gt_labels_b, k=recall_k, iou_thr=iou_thr)
            print(f"[val] step {step+1:>5} | imgs {n_images:>5} | RPN recall@{recall_k} IoU={iou_thr:.2f} (batch) : {r_batch:.3f}")
            if tensorboard_writer is not None and global_step is not None:
                with tensorboard_writer.as_default():
                    tf.summary.scalar(tb_tag_recall, r_batch, step=global_step + (step + 1))

        n_images += int(images.shape[0])

    if n_images == 0:
        print("[val] No images evaluated; check your split.")
        if tensorboard_writer is not None and global_step is not None:
            with tensorboard_writer.as_default():
                tf.summary.scalar(f"{summary_tag_prefix}/mAP_iou_{iou_tag}", 0.0, step=global_step)
                tf.summary.scalar(f"{summary_tag_prefix}/throughput_img_per_sec", 0.0, step=global_step)
                tf.summary.scalar(tb_tag_recall, 0.0, step=global_step)
                tf.summary.scalar(f"{summary_tag_prefix}/images_evaluated", 0, step=global_step)
        return 0.0, {}, 0.0, 0.0, 0

    # Final/global recall over ALL seen props & GTs
    r_global = batch_recall_tf(ALL_PROPS, GT_BOXES, GT_LABELS, k=recall_k, iou_thr=iou_thr)

    # mAP + PRF1
    max_gt = max((int(np.max(g)) if len(g) else 0) for g in GT_LABELS) if GT_LABELS else 0
    max_dt = max((int(np.max(d)) if len(d) else 0) for d in DET_LABELS) if DET_LABELS else 0
    num_classes = max(max_gt, max_dt) if max(max_gt, max_dt) > 0 else 1

    mAP, _ = compute_map_voc(
        DET_BOXES, DET_SCORES, DET_LABELS, GT_BOXES, GT_LABELS,
        num_classes=num_classes, iou_thr=iou_thr, use_11pt=False
    )
    prf1 = evaluation_prf1_report(
        DET_BOXES, DET_SCORES, DET_LABELS, GT_BOXES, GT_LABELS,
        num_classes=num_classes, iou_thr=iou_thr, score_thr=score_thr
    )

    ips = (n_images / sum(times)) if sum(times) > 0 else 0.0
    print(f"[val] DONE | images={n_images}  mAP@{iou_thr:.2f}={mAP:.4f}  RPN recall@{recall_k} (global)={r_global:.3f}  throughput={ips:.2f} img/s")

    if tensorboard_writer is not None and global_step is not None:
        with tensorboard_writer.as_default():
            tf.summary.scalar(f"{summary_tag_prefix}/mAP_iou_{iou_tag}", float(mAP), step=global_step)
            tf.summary.scalar(f"{summary_tag_prefix}/throughput_img_per_sec", float(ips), step=global_step)
            tf.summary.scalar(tb_tag_recall, float(r_global), step=global_step)
            tf.summary.scalar(f"{summary_tag_prefix}/images_evaluated", int(n_images), step=global_step)

    return float(mAP), prf1, float(r_global), float(ips), int(n_images)

def has_any(name, toks):
    name = name.lower()
    return any(tok in name for tok in toks)

def set_trainable(epoch):
    # Epochs 0–1: freeze backbone, train heads only
    if epoch < 2:
        if hasattr(model, "backbone"):
            model.backbone.trainable = False
        for v in backbone_vars: v._trainable = False
    else:
        if hasattr(model, "backbone"):
            model.backbone.trainable = True
        for v in backbone_vars: v._trainable = True

def current_lr(opt, step=None):
    """Return the scalar learning rate for an optimizer, regardless of type."""
    lr = opt.learning_rate
    # If it's a schedule (CosineDecay, Piecewise, etc.), call it with a step
    if isinstance(lr, tf.keras.optimizers.schedules.LearningRateSchedule):
        s = step if step is not None else int(opt.iterations.numpy())
        return tf.cast(lr(s), tf.float32)
    # If it's a keras/TF variable or tensor, just return its value
    if tf.is_tensor(lr):
        return tf.cast(lr, tf.float32)
    # If it's a python float
    return tf.convert_to_tensor(lr, dtype=tf.float32)

if __name__ == "__main__":
    
    print("Python:", sys.executable)
    print("TF version:", tf.__version__)
    print("GPUs:", tf.config.list_physical_devices('GPU'))
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", type=str, default="", help="Run directory to use (if empty, create new timestamped)")
    parser.add_argument("--tb", action="store_true", help="Launch TensorBoard")
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoints in --run_dir (or latest if not given)")
    parser.add_argument("--port", type=int, default=6006)
    parser.add_argument("--epochs", type=int, default=15, help="Total epochs")
    parser.add_argument("--val_every", type=int, default=1, help="Validate every N epochs (0=off)")
    parser.add_argument("--val_batches", type=int, default=200, help="Max batches to evaluate per validation pass (0=all)")
    parser.add_argument("--val_iou", type=float, default=0.5, help="IoU threshold for validation mAP/F1")
    parser.add_argument("--val_score_thr", type=float, default=0.05, help="Score threshold for validation metrics")
    parser.add_argument("--save_best", action="store_true", help="Save a checkpoint whenever val mAP improves")
    parser.add_argument("--val_pct", type=float, default=0.0, help="Percentage of training data to use for validation (default 0.0)")
    parser.add_argument("--batch_size", type=int, default=3, help="Batch size for training and validation")
    args = parser.parse_args()
    
    if args.resume:
        if args.run_dir:
            RUN_DIR = Path(args.run_dir)
            if not RUN_DIR.exists():
                raise FileNotFoundError(f"--resume specified but run_dir does not exist: {RUN_DIR}")
        else:
            RUN_DIR = latest_run_with_ckpt(BASE_LOG_DIR)
            if RUN_DIR is None:
                raise FileNotFoundError("No previous run with checkpoints found under ./logs/faster_rcnn. "
                                        "Provide --run_dir or start without --resume.")
    else:
        RUN_STAMP = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        RUN_DIR = BASE_LOG_DIR / RUN_STAMP
        RUN_DIR.mkdir(parents=True, exist_ok=True)

    
    num_epochs = args.epochs

    if args.tb:
        start_tensorboard(BASE_LOG_DIR, port=args.port)

    model = FasterRCNN()
    _ = model(tf.zeros([1, 800, 800, 3]), training=False)  # Build the model
    
    if hasattr(model, "backbone"):
        backbone_vars = model.backbone.trainable_variables
    else:
        backbone_vars = [v for v in model.trainable_variables if has_any(v.name, ["backbone", "resnet", "vgg"])]

    if hasattr(model, "rpn"):
        rpn_vars = model.rpn.trainable_variables
    else:
        rpn_vars = [v for v in model.trainable_variables if has_any(v.name, ["rpn", "proposal", "anchor"])]

    # RoI head (classifier + bbox regressor)
    if hasattr(model, "roi_head"):
        roi_vars = model.roi_head.trainable_variables
    else:
        roi_vars = [v for v in model.trainable_variables if has_any(v.name, [
            "roi_head", "box_head", "cls_head", "classifier", "class_logits", "bbox_pred", "bbox_reg"
        ])]
    
    train_ds, val_ds, meta = load_pascal_voc_dataset(
        batch_size=args.batch_size, val_pct=args.val_pct, seed=2025, prefer_official=False, return_meta=True
    )

    rpn_scheduler = CustomStepBasedScheduler(
        base_learning_rate=1e-3,
        steps_per_epoch=meta['steps_per_epoch'],
        total_epochs=num_epochs,
        warmup_steps=meta['steps_per_epoch'],
        hold_steps=900,
        minimum_learning_rate=1e-6
    )
    
    roi_sched  = CustomStepBasedScheduler(
        base_learning_rate=1e-3,
        steps_per_epoch=meta['steps_per_epoch'],
        total_epochs=num_epochs,
        warmup_steps=meta['steps_per_epoch'],
        hold_steps=900,
        minimum_learning_rate=1e-6
    )
    
    backbone_sched  = CustomStepBasedScheduler(
        base_learning_rate=1e-4,
        steps_per_epoch=meta['steps_per_epoch'],
        total_epochs=num_epochs,
        warmup_steps=meta['steps_per_epoch'],
        hold_steps=900,
        minimum_learning_rate=1e-6
    )
    
    optimizer_rpn = tf.keras.optimizers.Adam(learning_rate=rpn_scheduler)
    optimizer_roi = tf.keras.optimizers.Adam(learning_rate=roi_sched)
    optimizer_backbone = tf.keras.optimizers.Adam(learning_rate=backbone_sched)
    
    steps_per_epoch = meta['steps_per_epoch']
    log_file_path = str(TENSORBOARD_LOG_DIR) + f"/training_log_{RUN_STAMP}.txt"
    debug_folder_path = make_run_dir(DEBUG_IMAGE_ROOT)
    
    CKPT_DIR = RUN_DIR / "checkpoints"
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    
    tensorboard_writer = tf.summary.create_file_writer(str(RUN_DIR))
    
    ckpt = tf.train.Checkpoint(
        epoch=tf.Variable(0, dtype=tf.int64),           
        step_in_epoch=tf.Variable(0, dtype=tf.int64),   
        global_step=tf.Variable(0, dtype=tf.int64),     
        optimizer_backbone=optimizer_backbone,
        optoptimizer_rpn_rpn=optimizer_rpn,
        optimizer_roi=optimizer_roi,
        model=model
    )
    manager = tf.train.CheckpointManager(ckpt, str(CKPT_DIR), max_to_keep=5)
    
    if manager.latest_checkpoint:
        ckpt.restore(manager.latest_checkpoint).expect_partial()
        print(f"[✓] Restored from {manager.latest_checkpoint} "
              f"(epoch={int(ckpt.epoch.numpy())}, "
              f"step_in_epoch={int(ckpt.step_in_epoch.numpy())}, "
              f"global_step={int(ckpt.global_step.numpy())})")
    else:
        print("[ℹ] No checkpoint found, starting fresh.")
    

    
    banner(f"Run: {RUN_STAMP} | LogDir {str(TENSORBOARD_LOG_DIR)} | TF {tf.__version__} | GPUs: {tf.config.list_physical_devices('GPU')}")

    start_epoch = int(ckpt.epoch.numpy())
    start_step_in_epoch = int(ckpt.step_in_epoch.numpy())
    
    with open(log_file_path, "w") as log_file:
        for epoch in range(start_epoch,num_epochs):
            
            set_trainable(epoch) # Checking which layers to train
            
            banner(f"Epoch {epoch+1}/{num_epochs}")
            bar = CLIProgressBar(total=steps_per_epoch, bar_len=34)
            epoch_header = f"\nEpoch {epoch + 1}/{num_epochs}"
            log_file.write(epoch_header+"\n")
            
            ds_epoch = make_epoch_dataset(train_ds, steps_per_epoch,skip=start_step_in_epoch if epoch == start_epoch else 0)
            
            already_done = start_step_in_epoch if epoch == start_epoch else 0
            
            if already_done > 0:
                bar.start()
                bar.update(already_done - 1, resumed=True)
            else:
                bar.start()
                
            if epoch != start_epoch or start_step_in_epoch == 0:
                ckpt.step_in_epoch.assign(0)

            for i, batch in enumerate(ds_epoch):
                
                if epoch == start_epoch and start_step_in_epoch > 0 and i < 1:
                    i_offset = start_step_in_epoch
                else:
                    i_offset = 0
                    
                step_in_epoch = i + i_offset
                if step_in_epoch >= steps_per_epoch:
                    break
                
                train_backbone_flag = (epoch >= 2)
                
                outputs = train_step(batch, trainable_backbone=train_backbone_flag)
                
                ckpt.step_in_epoch.assign(step_in_epoch + 1)
                ckpt.global_step.assign_add(1)
                step = int(ckpt.global_step.numpy())
                
                bar.update(
                    step_in_epoch,
                    loss=outputs['total_loss'],
                    rpn=outputs['rpn_loss'],
                    roi_cls=outputs['roi_cls_loss'],
                    roi_box=outputs['roi_reg_loss'],
                    rpn_rec=outputs['rpn_recall_global'],
                    pos=outputs['roi_sampler_pos_frac_global'],
                )

                if step_in_epoch % 10 == 0:
                    msg = (
                        f"Step {i:03d} | "
                        f"Total Loss: {outputs['total_loss']:.4f} | "
                        f"RPN Objectness Loss: {outputs['objectness_loss']:.4f}, "
                        f"RPN Regression Loss: {outputs['bbox_regression_loss']:.4f}, "
                        f"RPN Loss: {outputs['rpn_loss']:.4f}, "
                        f"ROI Cls: {outputs['roi_cls_loss']:.4f}, "
                        f"RoI Box: {outputs['roi_reg_loss']:.4f}, "
                        f"RoI Head Regressor Gain: {outputs['roi_head_regressor_gain']:.4f}, "
                        f"RPN Recall: {outputs['rpn_recall_global']:.4f}, "
                        f"RoI Sampler Positive Ratio: {outputs['roi_sampler_pos_frac_global']:.4f}, "
                    )
                    log_file.write(msg + "\n")
                    log_file.flush()

                if step_in_epoch % 100 == 0:
                    img = batch['image'][0]
                    viz_path = get_viz_path(debug_folder_path, epoch, step_in_epoch)
                    res = model.call(images=img[tf.newaxis, ...], training=False, debug=True)
                    fig, ax = inference_plot_predicted_bounding_boxes(
                        img,
                        res['Boxes'][0],
                        res['Scores'][0],
                        res['Classes'][0],
                        class_names=pascal_labels,
                        order='yxyx',
                        is_normalized=True,
                        gt_boxes=batch['gt_boxes'][0],
                        gt_classes=batch['gt_labels'][0],
                        gt_is_normalized=False,
                        show_legend=True,
                        debug=False
                    )
                    fig.savefig(viz_path)
                    plt.close(fig)
                    
                    mAP, prf1, rpn_rec, ips, n_eval = run_validation_with_recall(model,val_ds,tensorboard_writer=tensorboard_writer,global_step=int(ckpt.global_step.numpy()),summary_tag_prefix="val", max_batches=200,iou_thr=0.5,score_thr=0.05,recall_k=100,print_every=100)
                    print(f"[val] Interim | images={n_eval}  mAP@0.5={mAP:.4f}  RPN recall@100= {rpn_rec:.3f}  throughput={ips:.2f} img/s")
                    log_file.write(f"[val] Interim | images={n_eval}  mAP@0.5={mAP:.4f}  RPN recall@100= {rpn_rec:.3f}  throughput={ips:.2f} img/s" + "\n")
                    log_file.flush()

                with tensorboard_writer.as_default():
                    tf.summary.scalar("train/Total Loss", outputs['total_loss'], step=step)
                    tf.summary.scalar("train/RPN Loss", outputs['rpn_loss'], step=step)
                    tf.summary.scalar("train/RoI Cls Loss", outputs['roi_cls_loss'], step=step)
                    tf.summary.scalar("train/RoI Box Loss", outputs['roi_reg_loss'], step=step)
                    tf.summary.scalar("train/RPN Objectness Loss", outputs['objectness_loss'], step=step)
                    tf.summary.scalar("train/RPN Regression Loss", outputs['bbox_regression_loss'], step=step)
                    tf.summary.scalar("train/RoI Head Regressor Gain", outputs['roi_head_regressor_gain'], step=step)
                    tf.summary.scalar("train/RPN Recall", outputs['rpn_recall_global'], step=step)
                    tf.summary.scalar("train/RoI Sampler Positive Ratio", outputs['roi_sampler_pos_frac_global'], step=step)
                    tf.summary.scalar("lr/rpn", current_lr(optimizer_rpn,step), step=step)
                    tf.summary.scalar("lr/roi",  current_lr(optimizer_roi,step), step=step)
                    tf.summary.scalar("lr/backbone",  current_lr(optimizer_backbone,step), step=step)
                    
            bar.end()
            epoch_end_msg = f"--- End of Epoch {epoch + 1} ---"
            log_file.write(epoch_end_msg + "\n")
            
            ckpt.epoch.assign(epoch + 1)
            ckpt.step_in_epoch.assign(0)
            save_path = manager.save()
            print(f"[✓] Saved checkpoint to: {save_path}")
            
            start_step_in_epoch = 0
            
            if args.val_every and ((epoch + 1) % args.val_every == 0):
            # Prefer val_ds if you actually have one; else fall back to a train slice.
                eval_ds = val_ds if meta.get('val_size', 0) else train_ds
                print(f"[val] Running validation on {'val' if eval_ds is val_ds else 'train'} split "
                f"(max {args.val_batches} batches) ...")
                mAP, prf1, ips, n_eval = run_validation_short(
                    eval_ds,
                    max_batches=args.val_batches,
                    iou_thr=args.val_iou,
                    score_thr=args.val_score_thr,
                    log_every=50
                )
                print(f"[val] images={n_eval}  mAP@{args.val_iou:.2f}={mAP:.4f}  throughput={ips:.2f} img/s")
                
                log_file.write(f"[val] images={n_eval}  mAP@{args.val_iou:.2f}={mAP:.4f}  throughput={ips:.2f} img/s" + "\n")
                log_file.flush()
                
                with tensorboard_writer.as_default():
                    tf.summary.scalar("val/mAP", mAP, step=int(ckpt.global_step.numpy()))
                    tf.summary.scalar("val/throughput_img_per_sec", ips, step=int(ckpt.global_step.numpy()))
                    
                if args.save_best:
                    best_file = RUN_DIR / "best_map.txt"
                    try:
                        prev = float(best_file.read_text().strip())
                    except Exception:
                        prev = -1.0
                    if mAP > prev:
                        best_file.write_text(str(mAP))
                        best_path = str(CKPT_DIR / f"best_map_epoch{epoch+1:03d}")
                        tf.train.Checkpoint(optimizer_backbone = optimizer_backbone, optimizer_rpn = optimizer_rpn, optimizer_roi = optimizer_roi, model=model).save(best_path)
                        print(f"[✓] New best mAP {mAP:.4f} — saved {best_path}")

    # Log to TensorBoard
    with tensorboard_writer.as_default():
        tf.summary.scalar("val/mAP", mAP, step=int(ckpt.global_step.numpy()))
        tf.summary.scalar("val/throughput_img_per_sec", ips, step=int(ckpt.global_step.numpy()))

    # Save best mAP checkpoint if requested
    if args.save_best:
        best_file = RUN_DIR / "best_map.txt"
        try:
            prev = float(best_file.read_text().strip())
        except Exception:
            prev = -1.0
        if mAP > prev:
            best_file.write_text(str(mAP))
            best_path = str(CKPT_DIR / f"best_map_epoch{epoch+1:03d}")
            tf.train.Checkpoint(optimizer_backbone = optimizer_backbone, optimizer_rpn = optimizer_rpn, optimizer_roi = optimizer_roi, model=model).save(best_path)
            print(f"[✓] New best mAP {mAP:.4f} — saved {best_path}")
            
    final_weights_path = save_final_weights(model, RUN_DIR)
    print(f"[✓] Training complete. Final weights saved at: {final_weights_path}")
    