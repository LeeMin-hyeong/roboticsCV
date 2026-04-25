import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from ultralytics import YOLO


ROOT = Path(__file__).resolve().parent
VOC_ROOT = ROOT / "voc3"
OUT_DIR = ROOT / "outputs"
FIG_DIR = OUT_DIR / "figures"

CLASSES = ["person", "car", "dog"]
CLS2IDX = {c: i for i, c in enumerate(CLASSES)}

# Keep the same evaluation-like threshold as previous analysis code
CONF_THR = 0.15
IOU_THR = 0.5

BASELINE_PT = ROOT / "ablation_runs" / "cv1_baseline" / "weights" / "best.pt"
OPTIMIZED_PT = ROOT / "opt_runs" / "opt_11" / "weights" / "best.pt"
MANIFEST_OPT = OUT_DIR / "optimization_manifest_cv1.json"


def iou_xyxy(a, b):
    xi = max(a[0], b[0])
    yi = max(a[1], b[1])
    xa = min(a[2], b[2])
    ya = min(a[3], b[3])
    inter = max(0.0, xa - xi) * max(0.0, ya - yi)
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter + 1e-9
    return inter / ua


def load_gt_for_image(img_path: Path):
    img = Image.open(img_path)
    w, h = img.size
    label_path = Path(str(img_path).replace("\\images\\", "\\labels\\").replace("/images/", "/labels/")).with_suffix(".txt")
    gts = []
    if not label_path.exists():
        return gts
    for ln in label_path.read_text(encoding="utf-8").splitlines():
        ln = ln.strip()
        if not ln:
            continue
        p = ln.split()
        ci = int(p[0])
        cx, cy, bw, bh = [float(v) for v in p[1:]]
        x1 = (cx - bw / 2.0) * w
        y1 = (cy - bh / 2.0) * h
        x2 = (cx + bw / 2.0) * w
        y2 = (cy + bh / 2.0) * h
        gts.append({"cls": ci, "box": [x1, y1, x2, y2]})
    return gts


def preds_for_image(model: YOLO, img_path: Path):
    r = model(str(img_path), conf=CONF_THR, verbose=False)[0]
    preds = []
    for b in r.boxes:
        cls_name = model.names[int(b.cls[0].item())]
        if cls_name not in CLS2IDX:
            continue
        preds.append(
            {
                "cls": CLS2IDX[cls_name],
                "box": b.xyxy[0].cpu().numpy().tolist(),
                "conf": float(b.conf[0].item()),
            }
        )
    return preds


def confusion_for_model(weights_path: Path):
    model = YOLO(str(weights_path))
    cm = np.zeros((len(CLASSES), len(CLASSES)), dtype=np.int64)
    matched_pairs = 0
    unmatched_gt = 0
    unmatched_pred = 0

    val_images = sorted((VOC_ROOT / "images" / "val").glob("*.jpg"))
    for img_path in val_images:
        gts = load_gt_for_image(img_path)
        preds = preds_for_image(model, img_path)

        if not gts and not preds:
            continue

        pairs = []
        for gi, gt in enumerate(gts):
            for pi, pr in enumerate(preds):
                v = iou_xyxy(gt["box"], pr["box"])
                if v >= IOU_THR:
                    pairs.append((v, gi, pi))

        pairs.sort(key=lambda x: x[0], reverse=True)
        used_g = set()
        used_p = set()
        for _, gi, pi in pairs:
            if gi in used_g or pi in used_p:
                continue
            used_g.add(gi)
            used_p.add(pi)
            gt_cls = gts[gi]["cls"]
            pr_cls = preds[pi]["cls"]
            cm[gt_cls, pr_cls] += 1
            matched_pairs += 1

        unmatched_gt += max(0, len(gts) - len(used_g))
        unmatched_pred += max(0, len(preds) - len(used_p))

    row_sums = cm.sum(axis=1, keepdims=True)
    cm_norm = np.zeros_like(cm, dtype=np.float64)
    np.divide(cm, np.maximum(row_sums, 1), out=cm_norm, where=row_sums > 0)

    meta = {
        "weights": str(weights_path.resolve()),
        "matched_pairs": int(matched_pairs),
        "unmatched_gt": int(unmatched_gt),
        "unmatched_pred": int(unmatched_pred),
        "conf_thr": CONF_THR,
        "iou_thr": IOU_THR,
    }
    return cm, cm_norm, meta


def draw_cm(ax, cm, title, normalized=False):
    im = ax.imshow(cm, cmap="Blues", vmin=0.0 if normalized else None, vmax=1.0 if normalized else None)
    ax.set_title(title, fontsize=11, weight="bold")
    ax.set_xticks(range(len(CLASSES)))
    ax.set_yticks(range(len(CLASSES)))
    ax.set_xticklabels(CLASSES, rotation=20, ha="right")
    ax.set_yticklabels(CLASSES)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Ground Truth")

    for i in range(len(CLASSES)):
        for j in range(len(CLASSES)):
            txt = f"{cm[i, j]:.2f}" if normalized else f"{int(cm[i, j])}"
            color = "white" if cm[i, j] > (0.5 if normalized else (cm.max() * 0.45 if cm.max() > 0 else 0)) else "black"
            ax.text(j, i, txt, ha="center", va="center", color=color, fontsize=9)
    return im


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    if MANIFEST_OPT.exists():
        try:
            info = json.loads(MANIFEST_OPT.read_text(encoding="utf-8"))
            best_pt = Path(info.get("best_pt", ""))
            if best_pt.exists():
                global OPTIMIZED_PT
                OPTIMIZED_PT = best_pt
        except Exception:
            pass

    if not BASELINE_PT.exists():
        raise FileNotFoundError(f"Baseline weights not found: {BASELINE_PT}")
    if not OPTIMIZED_PT.exists():
        raise FileNotFoundError(f"Optimized weights not found: {OPTIMIZED_PT}")

    base_cm, base_norm, base_meta = confusion_for_model(BASELINE_PT)
    opt_cm, opt_norm, opt_meta = confusion_for_model(OPTIMIZED_PT)

    fig1, axes1 = plt.subplots(1, 2, figsize=(13, 5))
    draw_cm(axes1[0], base_cm, "Baseline (Counts)", normalized=False)
    draw_cm(axes1[1], opt_cm, "Optimized (Counts)", normalized=False)
    fig1.suptitle("3-Class Confusion Matrix Comparison (Counts)", fontsize=13, weight="bold")
    fig1.tight_layout(rect=(0, 0, 1, 0.96))
    p_counts = FIG_DIR / "confusion_compare_counts_cv1.png"
    fig1.savefig(p_counts, dpi=170)
    plt.close(fig1)

    fig2, axes2 = plt.subplots(1, 2, figsize=(13, 5))
    draw_cm(axes2[0], base_norm, "Baseline (Row-Normalized)", normalized=True)
    draw_cm(axes2[1], opt_norm, "Optimized (Row-Normalized)", normalized=True)
    fig2.suptitle("3-Class Confusion Matrix Comparison (Normalized)", fontsize=13, weight="bold")
    fig2.tight_layout(rect=(0, 0, 1, 0.96))
    p_norm = FIG_DIR / "confusion_compare_normalized_cv1.png"
    fig2.savefig(p_norm, dpi=170)
    plt.close(fig2)

    report = {
        "counts_png": str(p_counts.resolve()),
        "normalized_png": str(p_norm.resolve()),
        "baseline_meta": base_meta,
        "optimized_meta": opt_meta,
        "baseline_cm_counts": base_cm.tolist(),
        "optimized_cm_counts": opt_cm.tolist(),
    }
    out_json = OUT_DIR / "confusion_compare_cv1.json"
    out_json.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
