
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs"
FIG = OUT / "figures"

MODEL = ROOT / "opt_aug_large_runs" / "lg_02_best_prev" / "weights" / "best.pt"
DATA_YAML = ROOT / "voc3" / "data.yaml"

CONF_FIXED = 0.35
MATCH_IOU = 0.5  # GT-Pred matching IoU threshold
RECALL_FLOOR = 0.55
MAX_DET_GRID = [20, 50, 100, 150, 300]
NMS_IOU_GRID = [0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75]

CSV_PATH = OUT / "lg02_operating_tune2_sweep_cv1.csv"
REPORT_MD = OUT / "lg02_operating_tune2_report_cv1.md"
MANIFEST = OUT / "lg02_operating_tune2_manifest_cv1.json"
FIG_F1 = FIG / "lg02_tune2_f1_heatmap_cv1.png"
FIG_FP = FIG / "lg02_tune2_fp_heatmap_cv1.png"


def ensure_dirs() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)


def xywhn_to_xyxy(xywhn: np.ndarray, w: int, h: int) -> np.ndarray:
    x, y, bw, bh = xywhn
    x1 = (x - bw / 2.0) * w
    y1 = (y - bh / 2.0) * h
    x2 = (x + bw / 2.0) * w
    y2 = (y + bh / 2.0) * h
    return np.array([x1, y1, x2, y2], dtype=np.float32)


def iou_xyxy(a: np.ndarray, b: np.ndarray) -> float:
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    inter_w = max(0.0, x2 - x1)
    inter_h = max(0.0, y2 - y1)
    inter = inter_w * inter_h
    if inter <= 0:
        return 0.0
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - inter
    if union <= 0:
        return 0.0
    return float(inter / union)


@dataclass
class Box:
    cls: int
    xyxy: np.ndarray


def load_data_paths(data_yaml: Path) -> tuple[Path, Path, list[str]]:
    data = yaml.safe_load(data_yaml.read_text(encoding="utf-8"))
    val_images = Path(data["val"])
    if not val_images.is_absolute():
        val_images = (data_yaml.parent / val_images).resolve()
    names = data.get("names", [])
    if isinstance(names, dict):
        names = [names[i] for i in sorted(names.keys())]
    labels_dir = val_images.parent.parent / "labels" / "val"
    return val_images, labels_dir, names


def load_gt_for_image(img_path: Path, labels_dir: Path) -> list[Box]:
    label_path = labels_dir / f"{img_path.stem}.txt"
    if not label_path.exists():
        return []

    import cv2

    im = cv2.imread(str(img_path))
    if im is None:
        raise RuntimeError(f"Failed to read image: {img_path}")
    h, w = im.shape[:2]

    boxes: list[Box] = []
    for line in label_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        p = line.split()
        cls = int(float(p[0]))
        xywhn = np.array([float(v) for v in p[1:5]], dtype=np.float32)
        boxes.append(Box(cls=cls, xyxy=xywhn_to_xyxy(xywhn, w, h)))
    return boxes


def extract_pred_for_result(res) -> list[Box]:
    if res.boxes is None or len(res.boxes) == 0:
        return []
    cls = res.boxes.cls.cpu().numpy().astype(int)
    xyxy = res.boxes.xyxy.cpu().numpy().astype(np.float32)
    return [Box(cls=int(c), xyxy=b) for c, b in zip(cls, xyxy)]


def match_image(gt: list[Box], pred: list[Box], ncls: int) -> dict:
    tp = 0
    fp = 0
    fn = 0

    tp_c = np.zeros(ncls, dtype=np.int64)
    fp_c = np.zeros(ncls, dtype=np.int64)
    fn_c = np.zeros(ncls, dtype=np.int64)

    used_pred = set()

    for gi, g in enumerate(gt):
        best_j = -1
        best_iou = 0.0
        for pj, p in enumerate(pred):
            if pj in used_pred:
                continue
            if p.cls != g.cls:
                continue
            iou = iou_xyxy(g.xyxy, p.xyxy)
            if iou >= MATCH_IOU and iou > best_iou:
                best_iou = iou
                best_j = pj
        if best_j >= 0:
            used_pred.add(best_j)
            tp += 1
            tp_c[g.cls] += 1
        else:
            fn += 1
            fn_c[g.cls] += 1

    for pj, p in enumerate(pred):
        if pj not in used_pred:
            fp += 1
            fp_c[p.cls] += 1

    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tp_c": tp_c,
        "fp_c": fp_c,
        "fn_c": fn_c,
    }


def safe_div(a: float, b: float) -> float:
    return float(a / b) if b > 0 else 0.0


def evaluate_combo(model: YOLO, val_images: Path, labels_dir: Path, ncls: int, nms_iou: float, max_det: int) -> dict:
    image_paths = sorted([p for p in val_images.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}])

    preds = model.predict(
        source=[str(p) for p in image_paths],
        conf=CONF_FIXED,
        iou=nms_iou,
        max_det=max_det,
        imgsz=640,
        verbose=False,
        save=False,
        device=0,
    )

    tp = fp = fn = 0
    tp_c = np.zeros(ncls, dtype=np.int64)
    fp_c = np.zeros(ncls, dtype=np.int64)
    fn_c = np.zeros(ncls, dtype=np.int64)

    for img_path, res in zip(image_paths, preds):
        gt = load_gt_for_image(img_path, labels_dir)
        pred = extract_pred_for_result(res)
        m = match_image(gt, pred, ncls)
        tp += m["tp"]
        fp += m["fp"]
        fn += m["fn"]
        tp_c += m["tp_c"]
        fp_c += m["fp_c"]
        fn_c += m["fn_c"]

    precision = safe_div(tp, tp + fp)
    recall = safe_div(tp, tp + fn)
    f1 = safe_div(2 * precision * recall, precision + recall)

    prec_c = [safe_div(int(tp_c[i]), int(tp_c[i] + fp_c[i])) for i in range(ncls)]
    rec_c = [safe_div(int(tp_c[i]), int(tp_c[i] + fn_c[i])) for i in range(ncls)]
    f1_c = [safe_div(2 * prec_c[i] * rec_c[i], prec_c[i] + rec_c[i]) for i in range(ncls)]

    return {
        "conf": CONF_FIXED,
        "nms_iou": nms_iou,
        "max_det": max_det,
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "precision_macro": float(np.mean(prec_c)),
        "recall_macro": float(np.mean(rec_c)),
        "f1_macro": float(np.mean(f1_c)),
    }


def choose_recommendation(df: pd.DataFrame) -> pd.Series:
    eligible = df[df["recall"] >= RECALL_FLOOR].copy()
    if len(eligible) == 0:
        return df.sort_values(["f1", "precision", "fp"], ascending=[False, False, True]).iloc[0]
    return eligible.sort_values(["fp", "f1", "fn"], ascending=[True, False, True]).iloc[0]


def make_heatmap(df: pd.DataFrame, value_col: str, title: str, out_path: Path, cmap: str) -> None:
    p = df.pivot(index="nms_iou", columns="max_det", values=value_col).sort_index().sort_index(axis=1)
    fig, ax = plt.subplots(figsize=(8, 5), dpi=150)
    im = ax.imshow(p.values, aspect="auto", cmap=cmap)

    ax.set_xticks(np.arange(p.shape[1]))
    ax.set_yticks(np.arange(p.shape[0]))
    ax.set_xticklabels([str(c) for c in p.columns])
    ax.set_yticklabels([f"{idx:.2f}" for idx in p.index])
    ax.set_xlabel("max_det")
    ax.set_ylabel("nms_iou")
    ax.set_title(title)

    for i in range(p.shape[0]):
        for j in range(p.shape[1]):
            v = p.values[i, j]
            ax.text(j, i, f"{v:.3f}" if isinstance(v, (float, np.floating)) else str(v), ha="center", va="center", fontsize=8)

    fig.colorbar(im, ax=ax, shrink=0.85)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def main() -> None:
    ensure_dirs()

    if not MODEL.exists():
        raise FileNotFoundError(f"Model not found: {MODEL}")
    if not DATA_YAML.exists():
        raise FileNotFoundError(f"Data yaml not found: {DATA_YAML}")

    val_images, labels_dir, names = load_data_paths(DATA_YAML)
    ncls = len(names)
    if ncls <= 0:
        raise RuntimeError("Could not infer class names from data.yaml")

    model = YOLO(str(MODEL))

    rows: list[dict] = []
    for nms_iou in NMS_IOU_GRID:
        for max_det in MAX_DET_GRID:
            print(f"[run] conf={CONF_FIXED:.2f}, nms_iou={nms_iou:.2f}, max_det={max_det}")
            rows.append(evaluate_combo(model, val_images, labels_dir, ncls, nms_iou, max_det))

    df = pd.DataFrame(rows)
    df.to_csv(CSV_PATH, index=False, encoding="utf-8-sig")

    best = choose_recommendation(df)

    make_heatmap(df, "f1", "F1 Heatmap (conf=0.35)", FIG_F1, "viridis")
    make_heatmap(df, "fp", "FP Heatmap (lower is better, conf=0.35)", FIG_FP, "magma_r")

    report_lines = [
        "# lg_02_best_prev Operating Tuning (2nd Stage)",
        "",
        f"- model: `{MODEL}`",
        f"- eval set: `voc3 val`",
        f"- fixed conf: **{CONF_FIXED:.2f}**",
        f"- sweep: `nms_iou in {NMS_IOU_GRID}`, `max_det in {MAX_DET_GRID}`",
        f"- selection rule: min FP under recall>={RECALL_FLOOR:.2f} (tie-break: higher F1, then lower FN)",
        "",
        "## Recommended Operating Point",
        "",
        f"- nms_iou: **{best['nms_iou']:.2f}**",
        f"- max_det: **{int(best['max_det'])}**",
        f"- TP={int(best['tp'])}, FP={int(best['fp'])}, FN={int(best['fn'])}",
        f"- Precision={best['precision']:.4f}, Recall={best['recall']:.4f}, F1={best['f1']:.4f}",
        f"- Macro Precision={best['precision_macro']:.4f}, Macro Recall={best['recall_macro']:.4f}, Macro F1={best['f1_macro']:.4f}",
        "",
        "![f1_heatmap](C:/Users/lmhst/git/roboticsCV/CV_project#1/outputs/figures/lg02_tune2_f1_heatmap_cv1.png)",
        "",
        "![fp_heatmap](C:/Users/lmhst/git/roboticsCV/CV_project#1/outputs/figures/lg02_tune2_fp_heatmap_cv1.png)",
    ]
    REPORT_MD.write_text("\n".join(report_lines), encoding="utf-8")

    manifest = {
        "csv": str(CSV_PATH),
        "report_md": str(REPORT_MD),
        "fig_f1": str(FIG_F1),
        "fig_fp": str(FIG_FP),
        "fixed_conf": CONF_FIXED,
        "selection_rule": f"min FP under recall>={RECALL_FLOOR}",
        "recommended": {
            "nms_iou": float(best["nms_iou"]),
            "max_det": int(best["max_det"]),
            "tp": int(best["tp"]),
            "fp": int(best["fp"]),
            "fn": int(best["fn"]),
            "precision": float(best["precision"]),
            "recall": float(best["recall"]),
            "f1": float(best["f1"]),
            "precision_macro": float(best["precision_macro"]),
            "recall_macro": float(best["recall_macro"]),
            "f1_macro": float(best["f1_macro"]),
        },
    }
    MANIFEST.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"[saved] {CSV_PATH}")
    print(f"[saved] {REPORT_MD}")
    print(f"[saved] {MANIFEST}")
    print(f"[best] nms_iou={best['nms_iou']:.2f}, max_det={int(best['max_det'])}, P={best['precision']:.4f}, R={best['recall']:.4f}, F1={best['f1']:.4f}, FP={int(best['fp'])}")


if __name__ == "__main__":
    main()
