from __future__ import annotations

import json
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import yaml
from ultralytics import YOLO


ROOT = Path(r"C:\Users\lmhst\git\roboticsCV\CV_project#1")
OUT = ROOT / "outputs"
FIG = OUT / "figures"
MET = OUT / "metrics"
REP = OUT / "reports"

DATA = ROOT / "voc3" / "data.yaml"
BEFORE_PT = "yolov8n.pt"
AFTER_PT = ROOT / "last_squeeze_runs" / "squeeze_refine1" / "weights" / "best.pt"

CLASSES_3 = ["person", "car", "dog"]
CLASSES_4 = CLASSES_3 + ["bg/other"]
CLS2IDX = {c: i for i, c in enumerate(CLASSES_3)}
BG_IDX = 3

BEFORE_CONF = 0.15
AFTER_CONF = 0.35
NMS_IOU = 0.7
MAX_DET = 300
MATCH_IOU = 0.5

JSON_OUT = MET / "final_vs_coco_on_baseline_voc3_conf035_cv1.json"
FIG_METRIC = FIG / "final_vs_coco_on_baseline_voc3_metrics_cv1.png"
FIG_CM_COUNT = FIG / "final_vs_coco_on_baseline_voc3_confusion_counts_cv1.png"
FIG_CM_NORM = FIG / "final_vs_coco_on_baseline_voc3_confusion_normalized_cv1.png"
REPORT_MD = REP / "final_vs_coco_on_baseline_voc3_conf035_report_cv1.md"


def iou_xyxy(a: np.ndarray, b: np.ndarray) -> float:
    x1 = max(float(a[0]), float(b[0]))
    y1 = max(float(a[1]), float(b[1]))
    x2 = min(float(a[2]), float(b[2]))
    y2 = min(float(a[3]), float(b[3]))
    iw = max(0.0, x2 - x1)
    ih = max(0.0, y2 - y1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    aa = max(0.0, float(a[2] - a[0])) * max(0.0, float(a[3] - a[1]))
    bb = max(0.0, float(b[2] - b[0])) * max(0.0, float(b[3] - b[1]))
    uu = aa + bb - inter
    return float(inter / uu) if uu > 0 else 0.0


def safe_div(a: float, b: float) -> float:
    return float(a / b) if b > 0 else 0.0


def load_paths_from_yaml() -> tuple[list[Path], Path]:
    cfg = yaml.safe_load(DATA.read_text(encoding="utf-8"))
    val = Path(cfg["val"])
    if not val.is_absolute():
        val = (DATA.parent / val).resolve()
    labels = val.parent.parent / "labels" / "val"
    images = sorted([p for p in val.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}])
    return images, labels


def load_gt_one(img_path: Path, labels_dir: Path) -> list[dict]:
    im = cv2.imread(str(img_path))
    if im is None:
        return []
    h, w = im.shape[:2]
    lp = labels_dir / f"{img_path.stem}.txt"
    out = []
    if not lp.exists():
        return out
    for ln in lp.read_text(encoding="utf-8").splitlines():
        ln = ln.strip()
        if not ln:
            continue
        p = ln.split()
        cls = int(float(p[0]))
        if cls not in {0, 1, 2}:
            continue
        cx, cy, bw, bh = [float(v) for v in p[1:5]]
        out.append(
            {
                "cls": cls,
                "box": np.array(
                    [
                        (cx - bw / 2.0) * w,
                        (cy - bh / 2.0) * h,
                        (cx + bw / 2.0) * w,
                        (cy + bh / 2.0) * h,
                    ],
                    dtype=np.float32,
                ),
            }
        )
    return out


def predict_one(model: YOLO, img_path: Path, conf_thr: float) -> list[dict]:
    r = model.predict(
        source=str(img_path),
        conf=float(conf_thr),
        iou=NMS_IOU,
        max_det=MAX_DET,
        imgsz=640,
        verbose=False,
        save=False,
        device=0,
    )[0]
    out = []
    if r.boxes is None or len(r.boxes) == 0:
        return out
    cls = r.boxes.cls.cpu().numpy().astype(int)
    conf = r.boxes.conf.cpu().numpy().astype(float)
    box = r.boxes.xyxy.cpu().numpy().astype(np.float32)
    for c, s, b in zip(cls, conf, box):
        cname = model.names[int(c)]
        if cname not in CLS2IDX:
            continue
        out.append({"cls": CLS2IDX[cname], "conf": float(s), "box": b})
    out.sort(key=lambda x: x["conf"], reverse=True)
    return out


def build_cm_and_stats(weights: str | Path, images: list[Path], labels_dir: Path, conf_thr: float) -> tuple[np.ndarray, dict]:
    model = YOLO(str(weights))
    cm = np.zeros((4, 4), dtype=np.int64)

    tp = fp = fn = 0
    tp_c = np.zeros(3, dtype=np.int64)
    fp_c = np.zeros(3, dtype=np.int64)
    fn_c = np.zeros(3, dtype=np.int64)

    for ip in images:
        gts = load_gt_one(ip, labels_dir)
        prs = predict_one(model, ip, conf_thr=conf_thr)

        pairs = []
        for gi, gt in enumerate(gts):
            for pi, pr in enumerate(prs):
                v = iou_xyxy(gt["box"], pr["box"])
                if v >= MATCH_IOU:
                    pairs.append((v, gi, pi))
        pairs.sort(key=lambda x: x[0], reverse=True)

        used_g = set()
        used_p = set()
        for _v, gi, pi in pairs:
            if gi in used_g or pi in used_p:
                continue
            used_g.add(gi)
            used_p.add(pi)
            gc = gts[gi]["cls"]
            pc = prs[pi]["cls"]
            cm[gc, pc] += 1
            if gc == pc:
                tp += 1
                tp_c[gc] += 1
            else:
                fp += 1
                fn += 1
                fp_c[pc] += 1
                fn_c[gc] += 1

        for gi, gt in enumerate(gts):
            if gi not in used_g:
                cm[gt["cls"], BG_IDX] += 1
                fn += 1
                fn_c[gt["cls"]] += 1

        for pi, pr in enumerate(prs):
            if pi not in used_p:
                cm[BG_IDX, pr["cls"]] += 1
                fp += 1
                fp_c[pr["cls"]] += 1

    precision = safe_div(tp, tp + fp)
    recall = safe_div(tp, tp + fn)
    f1 = safe_div(2 * precision * recall, precision + recall)

    cls_metrics = {}
    for i, name in enumerate(CLASSES_3):
        p = safe_div(int(tp_c[i]), int(tp_c[i] + fp_c[i]))
        r = safe_div(int(tp_c[i]), int(tp_c[i] + fn_c[i]))
        ff = safe_div(2 * p * r, p + r)
        cls_metrics[name] = {
            "precision": float(p),
            "recall": float(r),
            "f1": float(ff),
            "tp": int(tp_c[i]),
            "fp": int(fp_c[i]),
            "fn": int(fn_c[i]),
        }

    stats = {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "class_metrics": cls_metrics,
    }
    return cm, stats


def row_norm(cm: np.ndarray) -> np.ndarray:
    out = np.zeros_like(cm, dtype=np.float64)
    rs = cm.sum(axis=1, keepdims=True)
    np.divide(cm, np.maximum(rs, 1), out=out, where=rs > 0)
    return out


def draw_cm(ax, cm: np.ndarray, title: str, normalized: bool):
    vmin, vmax = (0.0, 1.0) if normalized else (None, None)
    ax.imshow(cm, cmap="Blues", vmin=vmin, vmax=vmax)
    ax.set_title(title, fontsize=11, weight="bold")
    ax.set_xticks(range(4))
    ax.set_yticks(range(4))
    ax.set_xticklabels(CLASSES_4, rotation=20, ha="right")
    ax.set_yticklabels(CLASSES_4)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    mx = float(cm.max()) if cm.size else 0.0
    for i in range(4):
        for j in range(4):
            txt = f"{cm[i, j]:.2f}" if normalized else str(int(cm[i, j]))
            thr = 0.5 if normalized else (0.45 * mx if mx > 0 else 0.0)
            ax.text(j, i, txt, ha="center", va="center", fontsize=9, color=("white" if cm[i, j] > thr else "black"))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    MET.mkdir(parents=True, exist_ok=True)
    REP.mkdir(parents=True, exist_ok=True)

    if not DATA.exists():
        raise FileNotFoundError(f"missing data yaml: {DATA}")
    if not AFTER_PT.exists():
        raise FileNotFoundError(f"missing final weights: {AFTER_PT}")

    images, labels_dir = load_paths_from_yaml()

    # mAP metrics under same voc3 protocol
    before_val = YOLO(BEFORE_PT).val(data=str(DATA), imgsz=640, verbose=False, device=0)
    after_val = YOLO(str(AFTER_PT)).val(data=str(DATA), imgsz=640, verbose=False, device=0)

    before_map = {"map50": float(before_val.box.map50), "map50_95": float(before_val.box.map)}
    after_map = {"map50": float(after_val.box.map50), "map50_95": float(after_val.box.map)}

    # confusion + f1 stats at fixed operating point
    before_cm, before_stats = build_cm_and_stats(BEFORE_PT, images, labels_dir, conf_thr=BEFORE_CONF)
    after_cm, after_stats = build_cm_and_stats(AFTER_PT, images, labels_dir, conf_thr=AFTER_CONF)
    before_norm = row_norm(before_cm)
    after_norm = row_norm(after_cm)

    # metric bar
    fig, ax = plt.subplots(figsize=(8, 5), dpi=160)
    x = np.arange(2)
    labs = ["mAP50", "mAP50-95"]
    b = [before_map["map50"], before_map["map50_95"]]
    a = [after_map["map50"], after_map["map50_95"]]
    ax.bar(x - 0.18, b, 0.35, label="Before (COCO)", color="#94a3b8")
    ax.bar(x + 0.18, a, 0.35, label="After (Final)", color="#22c55e")
    ax.set_xticks(x)
    ax.set_xticklabels(labs)
    ax.set_ylim(0, 1)
    ax.set_title("Before vs After on voc3/val")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG_METRIC)
    plt.close(fig)

    # confusion figures
    fig1, ax1 = plt.subplots(1, 2, figsize=(14, 5.5), dpi=170)
    draw_cm(ax1[0], before_cm, f"Before (counts, conf={BEFORE_CONF:.2f})", normalized=False)
    draw_cm(ax1[1], after_cm, f"After (counts, conf={AFTER_CONF:.2f})", normalized=False)
    fig1.suptitle("4x4 Confusion Matrix with Background (voc3/val)", fontsize=13, weight="bold")
    fig1.tight_layout(rect=(0, 0, 1, 0.95))
    fig1.savefig(FIG_CM_COUNT)
    plt.close(fig1)

    fig2, ax2 = plt.subplots(1, 2, figsize=(14, 5.5), dpi=170)
    draw_cm(ax2[0], before_norm, f"Before (normalized, conf={BEFORE_CONF:.2f})", normalized=True)
    draw_cm(ax2[1], after_norm, f"After (normalized, conf={AFTER_CONF:.2f})", normalized=True)
    fig2.suptitle("4x4 Confusion Matrix Row-Normalized (voc3/val)", fontsize=13, weight="bold")
    fig2.tight_layout(rect=(0, 0, 1, 0.95))
    fig2.savefig(FIG_CM_NORM)
    plt.close(fig2)

    out = {
        "dataset": str(DATA),
        "val_images": len(images),
        "operating_point": {
            "before_conf": BEFORE_CONF,
            "after_conf": AFTER_CONF,
            "nms_iou": NMS_IOU,
            "max_det": MAX_DET,
            "match_iou": MATCH_IOU,
        },
        "before": {
            "weights": BEFORE_PT,
            "map": before_map,
            "stats": before_stats,
            "cm_counts": before_cm.tolist(),
        },
        "after": {
            "weights": str(AFTER_PT),
            "map": after_map,
            "stats": after_stats,
            "cm_counts": after_cm.tolist(),
        },
        "delta_after_minus_before": {
            "map50": after_map["map50"] - before_map["map50"],
            "map50_95": after_map["map50_95"] - before_map["map50_95"],
            "f1": after_stats["f1"] - before_stats["f1"],
            "precision": after_stats["precision"] - before_stats["precision"],
            "recall": after_stats["recall"] - before_stats["recall"],
            "tp": after_stats["tp"] - before_stats["tp"],
            "fp": after_stats["fp"] - before_stats["fp"],
            "fn": after_stats["fn"] - before_stats["fn"],
        },
        "figures": {
            "metrics": str(FIG_METRIC),
            "cm_counts": str(FIG_CM_COUNT),
            "cm_normalized": str(FIG_CM_NORM),
        },
    }
    JSON_OUT.write_text(json.dumps(out, indent=2), encoding="utf-8")

    md = [
        "# Final vs COCO on Baseline voc3 (conf=0.35)",
        "",
        f"- dataset: `{DATA}` (val images={len(images)})",
        f"- before: `{BEFORE_PT}`",
        f"- after: `{AFTER_PT}`",
        f"- operating: before_conf={BEFORE_CONF}, after_conf={AFTER_CONF}, nms_iou={NMS_IOU}, max_det={MAX_DET}, match_iou={MATCH_IOU}",
        "",
        "## mAP",
        f"- before: mAP50={before_map['map50']:.4f}, mAP50-95={before_map['map50_95']:.4f}",
        f"- after: mAP50={after_map['map50']:.4f}, mAP50-95={after_map['map50_95']:.4f}",
        "",
        "## F1 (from same confusion protocol)",
        f"- before: P={before_stats['precision']:.4f}, R={before_stats['recall']:.4f}, F1={before_stats['f1']:.4f}, TP={before_stats['tp']}, FP={before_stats['fp']}, FN={before_stats['fn']}",
        f"- after: P={after_stats['precision']:.4f}, R={after_stats['recall']:.4f}, F1={after_stats['f1']:.4f}, TP={after_stats['tp']}, FP={after_stats['fp']}, FN={after_stats['fn']}",
        "",
        "![metrics](C:/Users/lmhst/git/roboticsCV/CV_project#1/outputs/figures/final_vs_coco_on_baseline_voc3_metrics_cv1.png)",
        "",
        "![cm_counts](C:/Users/lmhst/git/roboticsCV/CV_project#1/outputs/figures/final_vs_coco_on_baseline_voc3_confusion_counts_cv1.png)",
        "",
        "![cm_norm](C:/Users/lmhst/git/roboticsCV/CV_project#1/outputs/figures/final_vs_coco_on_baseline_voc3_confusion_normalized_cv1.png)",
    ]
    REPORT_MD.write_text("\n".join(md), encoding="utf-8")

    print(f"[saved] {JSON_OUT}")
    print(f"[saved] {REPORT_MD}")
    print(f"[saved] {FIG_METRIC}")
    print(f"[saved] {FIG_CM_COUNT}")
    print(f"[saved] {FIG_CM_NORM}")
    print(
        "[summary]",
        json.dumps(
            {
                "before_map50": before_map["map50"],
                "after_map50": after_map["map50"],
                "before_map50_95": before_map["map50_95"],
                "after_map50_95": after_map["map50_95"],
                "before_f1": before_stats["f1"],
                "after_f1": after_stats["f1"],
            },
            ensure_ascii=False,
        ),
    )


if __name__ == "__main__":
    main()
