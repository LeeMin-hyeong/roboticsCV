from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from ultralytics import YOLO

ROOT = Path(r"C:\Users\lmhst\git\roboticsCV\CV_project#1")
VOC_ROOT = ROOT / "voc3"
OUT = ROOT / "outputs"
FIG = OUT / "figures"

WEIGHTS = ROOT / "opt_aug_large_runs" / "lg_02_best_prev" / "weights" / "best.pt"

CLASSES = ["person", "car", "dog", "background"]
N = 4
BG_IDX = 3
CLASS_TO_IDX = {"person": 0, "car": 1, "dog": 2}

MATCH_IOU = 0.5
BASE_CFG = {"name": "op_base", "conf": 0.35, "nms_iou": 0.45, "max_det": 20, "person_thr": 0.35}
TUNE_CFG = {"name": "op_person_thr_040", "conf": 0.35, "nms_iou": 0.45, "max_det": 20, "person_thr": 0.40}

OUT_JSON = OUT / "confusion_operating_personthr_cv1.json"
FIG_COUNTS = FIG / "confusion_operating_personthr_counts_cv1.png"
FIG_NORM = FIG / "confusion_operating_personthr_normalized_cv1.png"


def iou_xyxy(a, b):
    xi = max(a[0], b[0]); yi = max(a[1], b[1]); xa = min(a[2], b[2]); ya = min(a[3], b[3])
    inter = max(0.0, xa - xi) * max(0.0, ya - yi)
    union = max(1e-9, (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter)
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


def load_preds(model: YOLO, img_path: Path, cfg: dict):
    r = model.predict(source=str(img_path), conf=cfg["conf"], iou=cfg["nms_iou"], max_det=cfg["max_det"], verbose=False, save=False, device=0)[0]
    out = []
    for b in r.boxes:
        cls_name = model.names[int(b.cls[0].item())]
        if cls_name not in CLASS_TO_IDX:
            continue
        conf = float(b.conf[0].item())
        cidx = CLASS_TO_IDX[cls_name]
        if cidx == 0 and conf < cfg["person_thr"]:
            continue
        out.append({
            "cls": cidx,
            "box": b.xyxy[0].cpu().numpy().tolist(),
            "conf": conf,
        })
    out.sort(key=lambda x: x["conf"], reverse=True)
    return out[: cfg["max_det"]]


def build_cm(model: YOLO, cfg: dict):
    cm = np.zeros((N, N), dtype=np.int64)
    val_images = sorted((VOC_ROOT / "images" / "val").glob("*.jpg"))
    for img_path in val_images:
        gts = load_gt(img_path)
        preds = load_preds(model, img_path, cfg)

        pairs = []
        for gi, gt in enumerate(gts):
            for pi, pr in enumerate(preds):
                v = iou_xyxy(gt["box"], pr["box"])
                if v >= MATCH_IOU:
                    pairs.append((v, gi, pi))
        pairs.sort(key=lambda x: x[0], reverse=True)

        used_g = set(); used_p = set()
        for _v, gi, pi in pairs:
            if gi in used_g or pi in used_p:
                continue
            used_g.add(gi); used_p.add(pi)
            cm[gts[gi]["cls"], preds[pi]["cls"]] += 1

        for gi, gt in enumerate(gts):
            if gi not in used_g:
                cm[gt["cls"], BG_IDX] += 1

        for pi, pr in enumerate(preds):
            if pi not in used_p:
                cm[BG_IDX, pr["cls"]] += 1

    norm = np.zeros_like(cm, dtype=np.float64)
    rowsum = cm.sum(axis=1, keepdims=True)
    np.divide(cm, np.maximum(rowsum, 1), out=norm, where=rowsum > 0)
    return cm, norm


def draw(ax, cm, title: str, normalized: bool):
    im = ax.imshow(cm, cmap="Blues", vmin=0.0 if normalized else None, vmax=1.0 if normalized else None)
    ax.set_title(title, fontsize=11, weight="bold")
    ax.set_xticks(range(N)); ax.set_yticks(range(N))
    ax.set_xticklabels(CLASSES, rotation=20, ha="right")
    ax.set_yticklabels(CLASSES)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Ground Truth")
    vmax = float(cm.max()) if cm.size else 0.0
    for i in range(N):
        for j in range(N):
            txt = f"{cm[i,j]:.2f}" if normalized else str(int(cm[i,j]))
            thr = 0.5 if normalized else (0.45 * vmax if vmax > 0 else 0.0)
            ax.text(j, i, txt, ha="center", va="center", fontsize=9, color=("white" if cm[i,j] > thr else "black"))
    return im


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    if not WEIGHTS.exists():
        raise FileNotFoundError(WEIGHTS)

    model = YOLO(str(WEIGHTS))
    base_cm, base_norm = build_cm(model, BASE_CFG)
    tune_cm, tune_norm = build_cm(model, TUNE_CFG)

    fig1, ax1 = plt.subplots(1, 2, figsize=(14, 5.5))
    draw(ax1[0], base_cm, "Current Operating (person_thr=0.35)", normalized=False)
    draw(ax1[1], tune_cm, "Tuned Operating (person_thr=0.40)", normalized=False)
    fig1.suptitle("4x4 Confusion with Background (Counts)", fontsize=13, weight="bold")
    fig1.tight_layout(rect=(0, 0, 1, 0.95))
    fig1.savefig(FIG_COUNTS, dpi=170)
    plt.close(fig1)

    fig2, ax2 = plt.subplots(1, 2, figsize=(14, 5.5))
    draw(ax2[0], base_norm, "Current Operating (person_thr=0.35)", normalized=True)
    draw(ax2[1], tune_norm, "Tuned Operating (person_thr=0.40)", normalized=True)
    fig2.suptitle("4x4 Confusion with Background (Row-Normalized)", fontsize=13, weight="bold")
    fig2.tight_layout(rect=(0, 0, 1, 0.95))
    fig2.savefig(FIG_NORM, dpi=170)
    plt.close(fig2)

    out = {
        "classes": CLASSES,
        "weights": str(WEIGHTS.resolve()),
        "match_iou": MATCH_IOU,
        "baseline_cfg": BASE_CFG,
        "tuned_cfg": TUNE_CFG,
        "baseline_cm_counts": base_cm.tolist(),
        "tuned_cm_counts": tune_cm.tolist(),
        "baseline_bg_to_person_fp": int(base_cm[BG_IDX, 0]),
        "tuned_bg_to_person_fp": int(tune_cm[BG_IDX, 0]),
        "counts_png": str(FIG_COUNTS.resolve()),
        "normalized_png": str(FIG_NORM.resolve()),
    }
    OUT_JSON.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(out, ensure_ascii=False))


if __name__ == "__main__":
    main()
