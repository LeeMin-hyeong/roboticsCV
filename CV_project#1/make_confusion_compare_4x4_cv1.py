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

CLASSES_3 = ["person", "car", "dog"]
BG = "background"
CLASSES_4 = CLASSES_3 + [BG]
N3 = len(CLASSES_3)
N4 = len(CLASSES_4)
BG_IDX = N3
CLS2IDX = {c: i for i, c in enumerate(CLASSES_3)}

CONF_THR = 0.15
IOU_THR = 0.5

BASELINE_PT = ROOT / "ablation_runs" / "cv1_baseline" / "weights" / "best.pt"
MANIFEST_OPT = OUT_DIR / "optimization_manifest_cv1.json"
OPTIMIZED_PT = ROOT / "opt_runs" / "opt_11" / "weights" / "best.pt"


def iou_xyxy(a, b):
    xi = max(a[0], b[0])
    yi = max(a[1], b[1])
    xa = min(a[2], b[2])
    ya = min(a[3], b[3])
    inter = max(0.0, xa - xi) * max(0.0, ya - yi)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter + 1e-9
    return inter / union


def load_gt(img_path: Path):
    img = Image.open(img_path)
    w, h = img.size
    label_path = Path(str(img_path).replace("\\images\\", "\\labels\\").replace("/images/", "/labels/")).with_suffix(".txt")
    out = []
    if not label_path.exists():
        return out
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
        out.append({"cls": ci, "box": [x1, y1, x2, y2]})
    return out


def load_pred(model: YOLO, img_path: Path):
    r = model(str(img_path), conf=CONF_THR, verbose=False)[0]
    out = []
    for b in r.boxes:
        cls_name = model.names[int(b.cls[0].item())]
        if cls_name not in CLS2IDX:
            continue
        out.append(
            {
                "cls": CLS2IDX[cls_name],
                "box": b.xyxy[0].cpu().numpy().tolist(),
                "conf": float(b.conf[0].item()),
            }
        )
    return out


def build_cm_4x4(weights_path: Path):
    model = YOLO(str(weights_path))
    cm = np.zeros((N4, N4), dtype=np.int64)

    val_images = sorted((VOC_ROOT / "images" / "val").glob("*.jpg"))
    for img_path in val_images:
        gts = load_gt(img_path)
        preds = load_pred(model, img_path)

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

        # FN: GT missed -> predicted background
        for gi, gt in enumerate(gts):
            if gi not in used_g:
                cm[gt["cls"], BG_IDX] += 1

        # FP: background predicted as class
        for pi, pr in enumerate(preds):
            if pi not in used_p:
                cm[BG_IDX, pr["cls"]] += 1

    cm_norm = np.zeros_like(cm, dtype=np.float64)
    row_sums = cm.sum(axis=1, keepdims=True)
    np.divide(cm, np.maximum(row_sums, 1), out=cm_norm, where=row_sums > 0)

    meta = {
        "weights": str(weights_path.resolve()),
        "conf_thr": CONF_THR,
        "iou_thr": IOU_THR,
    }
    return cm, cm_norm, meta


def draw_cm(ax, cm, title, normalized):
    im = ax.imshow(cm, cmap="Blues", vmin=0.0 if normalized else None, vmax=1.0 if normalized else None)
    ax.set_title(title, fontsize=11, weight="bold")
    ax.set_xticks(range(N4))
    ax.set_yticks(range(N4))
    ax.set_xticklabels(CLASSES_4, rotation=20, ha="right")
    ax.set_yticklabels(CLASSES_4)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Ground Truth")

    vmax = float(cm.max()) if cm.size else 0.0
    for i in range(N4):
        for j in range(N4):
            txt = f"{cm[i, j]:.2f}" if normalized else str(int(cm[i, j]))
            thr = 0.5 if normalized else (0.45 * vmax if vmax > 0 else 0.0)
            color = "white" if cm[i, j] > thr else "black"
            ax.text(j, i, txt, ha="center", va="center", fontsize=9, color=color)
    return im


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    if MANIFEST_OPT.exists():
        try:
            m = json.loads(MANIFEST_OPT.read_text(encoding="utf-8"))
            p = Path(m.get("best_pt", ""))
            if p.exists():
                global OPTIMIZED_PT
                OPTIMIZED_PT = p
        except Exception:
            pass

    if not BASELINE_PT.exists():
        raise FileNotFoundError(f"Baseline weights not found: {BASELINE_PT}")
    if not OPTIMIZED_PT.exists():
        raise FileNotFoundError(f"Optimized weights not found: {OPTIMIZED_PT}")

    base_cm, base_norm, base_meta = build_cm_4x4(BASELINE_PT)
    opt_cm, opt_norm, opt_meta = build_cm_4x4(OPTIMIZED_PT)

    fig1, ax1 = plt.subplots(1, 2, figsize=(14, 5.5))
    draw_cm(ax1[0], base_cm, "Baseline 4x4 (Counts)", normalized=False)
    draw_cm(ax1[1], opt_cm, "Optimized 4x4 (Counts)", normalized=False)
    fig1.suptitle("4x4 Confusion Matrix with Background (Counts)", fontsize=13, weight="bold")
    fig1.tight_layout(rect=(0, 0, 1, 0.95))
    p_counts = FIG_DIR / "confusion_compare_4x4_counts_cv1.png"
    fig1.savefig(p_counts, dpi=170)
    plt.close(fig1)

    fig2, ax2 = plt.subplots(1, 2, figsize=(14, 5.5))
    draw_cm(ax2[0], base_norm, "Baseline 4x4 (Row-Normalized)", normalized=True)
    draw_cm(ax2[1], opt_norm, "Optimized 4x4 (Row-Normalized)", normalized=True)
    fig2.suptitle("4x4 Confusion Matrix with Background (Normalized)", fontsize=13, weight="bold")
    fig2.tight_layout(rect=(0, 0, 1, 0.95))
    p_norm = FIG_DIR / "confusion_compare_4x4_normalized_cv1.png"
    fig2.savefig(p_norm, dpi=170)
    plt.close(fig2)

    out = {
        "classes": CLASSES_4,
        "counts_png": str(p_counts.resolve()),
        "normalized_png": str(p_norm.resolve()),
        "baseline_meta": base_meta,
        "optimized_meta": opt_meta,
        "baseline_cm_counts": base_cm.tolist(),
        "optimized_cm_counts": opt_cm.tolist(),
    }
    out_json = OUT_DIR / "confusion_compare_4x4_cv1.json"
    out_json.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(out, ensure_ascii=False))


if __name__ == "__main__":
    main()
