from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from ultralytics import YOLO

ROOT = Path(r"C:\Users\lmhst\git\roboticsCV\CV_project#1")
OUT = ROOT / "outputs"
FIG = OUT / "figures"

MODEL = ROOT / "opt_aug_large_runs" / "lg_02_best_prev" / "weights" / "best.pt"
DATA_YAML = ROOT / "voc3" / "data.yaml"

GLOBAL_CONF = 0.35
NMS_IOU = 0.45
MAX_DET = 20
MATCH_IOU = 0.5

PERSON_THR_GRID = [0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65]
MIN_AREA_GRID = [0.0, 0.0005, 0.0010, 0.0015, 0.0020]
AR_RANGE_GRID = [
    (0.10, 2.50),
    (0.15, 2.00),
    (0.20, 1.80),
    (0.25, 1.50),
    (0.30, 1.30),
]

THR_CSV = OUT / "person_threshold_sweep_cv1.csv"
FILT_CSV = OUT / "person_shape_filter_sweep_cv1.csv"
REPORT_MD = OUT / "person_fp_control_report_cv1.md"
MANIFEST = OUT / "person_fp_control_manifest_cv1.json"
FIG_THR = FIG / "person_threshold_tradeoff_cv1.png"
FIG_FILT = FIG / "person_filter_bgfp_heatmap_cv1.png"


@dataclass
class GTBox:
    cls: int
    xyxy: np.ndarray


@dataclass
class PredBox:
    cls: int
    conf: float
    xyxy: np.ndarray


def ensure_dirs() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)


def safe_div(a: float, b: float) -> float:
    return float(a / b) if b > 0 else 0.0


def xywhn_to_xyxy(xywhn: np.ndarray, w: int, h: int) -> np.ndarray:
    x, y, bw, bh = xywhn
    return np.array([(x - bw / 2.0) * w, (y - bh / 2.0) * h, (x + bw / 2.0) * w, (y + bh / 2.0) * h], dtype=np.float32)


def iou_xyxy(a: np.ndarray, b: np.ndarray) -> float:
    x1 = max(a[0], b[0]); y1 = max(a[1], b[1]); x2 = min(a[2], b[2]); y2 = min(a[3], b[3])
    iw = max(0.0, x2 - x1); ih = max(0.0, y2 - y1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    aa = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    bb = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    uu = aa + bb - inter
    return float(inter / uu) if uu > 0 else 0.0


def load_data(data_yaml: Path):
    data = yaml.safe_load(data_yaml.read_text(encoding="utf-8"))
    val_images = Path(data["val"])
    if not val_images.is_absolute():
        val_images = (data_yaml.parent / val_images).resolve()
    labels_dir = val_images.parent.parent / "labels" / "val"
    names = data.get("names", [])
    if isinstance(names, dict):
        names = [names[i] for i in sorted(names.keys())]
    return val_images, labels_dir, names


def load_gt_for_image(img_path: Path, labels_dir: Path) -> list[GTBox]:
    import cv2

    label_path = labels_dir / f"{img_path.stem}.txt"
    if not label_path.exists():
        return []
    im = cv2.imread(str(img_path))
    if im is None:
        raise RuntimeError(f"Failed to read image: {img_path}")
    h, w = im.shape[:2]

    gts: list[GTBox] = []
    for line in label_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        p = line.split()
        cls = int(float(p[0]))
        xywhn = np.array([float(v) for v in p[1:5]], dtype=np.float32)
        gts.append(GTBox(cls=cls, xyxy=xywhn_to_xyxy(xywhn, w, h)))
    return gts


def get_image_hw(img_path: Path) -> tuple[int, int]:
    import cv2

    im = cv2.imread(str(img_path))
    if im is None:
        raise RuntimeError(f"Failed to read image: {img_path}")
    h, w = im.shape[:2]
    return h, w


def extract_preds(res) -> list[PredBox]:
    if res.boxes is None or len(res.boxes) == 0:
        return []
    cls = res.boxes.cls.cpu().numpy().astype(int)
    conf = res.boxes.conf.cpu().numpy().astype(float)
    xyxy = res.boxes.xyxy.cpu().numpy().astype(np.float32)
    return [PredBox(cls=int(c), conf=float(cf), xyxy=b) for c, cf, b in zip(cls, conf, xyxy)]


def post_filter_preds(preds: list[PredBox], person_idx: int, person_thr: float, min_area_frac: float, ar_min: float, ar_max: float, img_h: int, img_w: int) -> list[PredBox]:
    out: list[PredBox] = []
    img_area = float(img_h * img_w)

    for p in preds:
        if p.cls == person_idx:
            if p.conf < person_thr:
                continue
            w = max(0.0, float(p.xyxy[2] - p.xyxy[0]))
            h = max(0.0, float(p.xyxy[3] - p.xyxy[1]))
            area_frac = (w * h) / img_area if img_area > 0 else 0.0
            ar = (w / h) if h > 1e-9 else 999.0
            if area_frac < min_area_frac:
                continue
            if ar < ar_min or ar > ar_max:
                continue
            out.append(p)
        else:
            if p.conf >= GLOBAL_CONF:
                out.append(p)

    out.sort(key=lambda x: x.conf, reverse=True)
    return out[:MAX_DET]


def evaluate_combo(image_paths: list[Path], labels_dir: Path, person_idx: int, raw_preds_by_image: dict[str, list[PredBox]], person_thr: float, min_area_frac: float, ar_min: float, ar_max: float) -> dict:
    tp = fp = fn = 0
    person_tp = person_fp = person_fn = 0

    for img_path in image_paths:
        gts = load_gt_for_image(img_path, labels_dir)
        h, w = get_image_hw(img_path)
        preds = post_filter_preds(raw_preds_by_image[img_path.name], person_idx, person_thr, min_area_frac, ar_min, ar_max, h, w)

        used = set()
        for g in gts:
            best_j = -1
            best_iou = 0.0
            for j, p in enumerate(preds):
                if j in used or p.cls != g.cls:
                    continue
                iou = iou_xyxy(g.xyxy, p.xyxy)
                if iou >= MATCH_IOU and iou > best_iou:
                    best_iou = iou
                    best_j = j
            if best_j >= 0:
                used.add(best_j)
                tp += 1
                if g.cls == person_idx:
                    person_tp += 1
            else:
                fn += 1
                if g.cls == person_idx:
                    person_fn += 1

        for j, p in enumerate(preds):
            if j not in used:
                fp += 1
                if p.cls == person_idx:
                    person_fp += 1  # background->person FP

    precision = safe_div(tp, tp + fp)
    recall = safe_div(tp, tp + fn)
    f1 = safe_div(2 * precision * recall, precision + recall)

    person_precision = safe_div(person_tp, person_tp + person_fp)
    person_recall = safe_div(person_tp, person_tp + person_fn)
    person_f1 = safe_div(2 * person_precision * person_recall, person_precision + person_recall)

    return {
        "person_thr": person_thr,
        "min_area_frac": min_area_frac,
        "ar_min": ar_min,
        "ar_max": ar_max,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "person_tp": person_tp,
        "person_fp_bg": person_fp,
        "person_fn": person_fn,
        "person_precision": person_precision,
        "person_recall": person_recall,
        "person_f1": person_f1,
    }


def select_best_threshold(df: pd.DataFrame, baseline_person_recall: float) -> pd.Series:
    # prioritize lowest background->person FP with recall guardrail
    floor = baseline_person_recall - 0.03
    eligible = df[df["person_recall"] >= floor].copy()
    if len(eligible) == 0:
        eligible = df.copy()
    return eligible.sort_values(["person_fp_bg", "person_f1", "f1"], ascending=[True, False, False]).iloc[0]


def select_best_filter(df: pd.DataFrame, baseline_person_recall: float) -> pd.Series:
    floor = baseline_person_recall - 0.03
    eligible = df[df["person_recall"] >= floor].copy()
    if len(eligible) == 0:
        eligible = df.copy()
    return eligible.sort_values(["person_fp_bg", "person_f1", "f1"], ascending=[True, False, False]).iloc[0]


def plot_threshold_tradeoff(df: pd.DataFrame, out_path: Path) -> None:
    d = df.sort_values("person_thr")
    fig, ax1 = plt.subplots(figsize=(8, 5), dpi=150)
    ax1.plot(d["person_thr"], d["person_fp_bg"], marker="o", color="#d62728", label="background->person FP")
    ax1.set_xlabel("person threshold")
    ax1.set_ylabel("background->person FP", color="#d62728")
    ax1.tick_params(axis="y", labelcolor="#d62728")

    ax2 = ax1.twinx()
    ax2.plot(d["person_thr"], d["person_recall"], marker="s", color="#1f77b4", label="person recall")
    ax2.set_ylabel("person recall", color="#1f77b4")
    ax2.tick_params(axis="y", labelcolor="#1f77b4")
    ax2.set_ylim(0, 1.05)

    ax1.set_title("Person Threshold Sweep: FP vs Recall")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_filter_heatmap(df: pd.DataFrame, ar_label: str, out_path: Path) -> None:
    sub = df[df["ar_label"] == ar_label].copy()
    p = sub.pivot(index="min_area_frac", columns="person_thr", values="person_fp_bg").sort_index().sort_index(axis=1)
    fig, ax = plt.subplots(figsize=(8, 5), dpi=150)
    im = ax.imshow(p.values, aspect="auto", cmap="magma_r")
    ax.set_xticks(np.arange(p.shape[1]))
    ax.set_yticks(np.arange(p.shape[0]))
    ax.set_xticklabels([f"{x:.2f}" for x in p.columns])
    ax.set_yticklabels([f"{x:.4f}" for x in p.index])
    ax.set_xlabel("person threshold")
    ax.set_ylabel("min_area_frac")
    ax.set_title(f"background->person FP Heatmap (AR={ar_label})")

    for i in range(p.shape[0]):
        for j in range(p.shape[1]):
            ax.text(j, i, f"{int(p.values[i, j])}", ha="center", va="center", fontsize=8)

    fig.colorbar(im, ax=ax, shrink=0.85)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def main() -> None:
    ensure_dirs()
    if not MODEL.exists():
        raise FileNotFoundError(MODEL)

    val_images, labels_dir, names = load_data(DATA_YAML)
    if "person" not in names:
        raise RuntimeError(f"'person' class not found in names={names}")
    person_idx = names.index("person")

    image_paths = sorted([p for p in val_images.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}])

    model = YOLO(str(MODEL))
    preds = model.predict(source=[str(p) for p in image_paths], conf=GLOBAL_CONF, iou=NMS_IOU, max_det=300, imgsz=640, verbose=False, save=False, device=0)

    raw_preds_by_image: dict[str, list[PredBox]] = {}
    for p, r in zip(image_paths, preds):
        raw_preds_by_image[p.name] = extract_preds(r)

    # A) threshold-only sweep (no shape filter)
    thr_rows = []
    for t in PERSON_THR_GRID:
        row = evaluate_combo(image_paths, labels_dir, person_idx, raw_preds_by_image, t, 0.0, 0.0, 999.0)
        thr_rows.append(row)
        print(f"[thr] t={t:.2f} -> person_fp_bg={row['person_fp_bg']}, person_recall={row['person_recall']:.4f}, person_f1={row['person_f1']:.4f}")

    thr_df = pd.DataFrame(thr_rows)
    thr_df.to_csv(THR_CSV, index=False, encoding="utf-8-sig")

    baseline_row = thr_df.loc[np.isclose(thr_df["person_thr"], 0.35)].iloc[0]
    best_thr_row = select_best_threshold(thr_df, float(baseline_row["person_recall"]))

    # B) shape/size filter sweep around thresholds + AR/min area
    filt_rows = []
    for t in PERSON_THR_GRID:
        for min_area in MIN_AREA_GRID:
            for ar_min, ar_max in AR_RANGE_GRID:
                row = evaluate_combo(image_paths, labels_dir, person_idx, raw_preds_by_image, t, min_area, ar_min, ar_max)
                row["ar_label"] = f"[{ar_min:.2f},{ar_max:.2f}]"
                filt_rows.append(row)

    filt_df = pd.DataFrame(filt_rows)
    filt_df.to_csv(FILT_CSV, index=False, encoding="utf-8-sig")
    best_filter_row = select_best_filter(filt_df, float(baseline_row["person_recall"]))

    plot_threshold_tradeoff(thr_df, FIG_THR)
    # heatmap for best AR band in top candidates
    best_ar_label = f"[{best_filter_row['ar_min']:.2f},{best_filter_row['ar_max']:.2f}]"
    plot_filter_heatmap(filt_df, best_ar_label, FIG_FILT)

    report_lines = [
        "# person FP 제어 실험 (Threshold + Shape/Size Filter)",
        "",
        f"- model: `{MODEL}`",
        f"- fixed operating: conf={GLOBAL_CONF}, nms_iou={NMS_IOU}, max_det={MAX_DET}",
        "- objective: background->person FP 최소화 (person recall guardrail: baseline-0.03)",
        "",
        "## A) person threshold sweep 결과",
        f"- baseline(t=0.35): person_fp_bg={int(baseline_row['person_fp_bg'])}, person_recall={baseline_row['person_recall']:.4f}, person_f1={baseline_row['person_f1']:.4f}",
        f"- best threshold: t={best_thr_row['person_thr']:.2f}, person_fp_bg={int(best_thr_row['person_fp_bg'])}, person_recall={best_thr_row['person_recall']:.4f}, person_f1={best_thr_row['person_f1']:.4f}",
        "",
        "![threshold_tradeoff](C:/Users/lmhst/git/roboticsCV/CV_project#1/outputs/figures/person_threshold_tradeoff_cv1.png)",
        "",
        "## B) person shape/size filter sweep 결과",
        f"- best filter setting: person_thr={best_filter_row['person_thr']:.2f}, min_area_frac={best_filter_row['min_area_frac']:.4f}, ar=[{best_filter_row['ar_min']:.2f},{best_filter_row['ar_max']:.2f}]",
        f"- metrics: person_fp_bg={int(best_filter_row['person_fp_bg'])}, person_recall={best_filter_row['person_recall']:.4f}, person_f1={best_filter_row['person_f1']:.4f}, overall_f1={best_filter_row['f1']:.4f}",
        "",
        f"![filter_heatmap](C:/Users/lmhst/git/roboticsCV/CV_project#1/outputs/figures/person_filter_bgfp_heatmap_cv1.png)",
    ]
    REPORT_MD.write_text("\n".join(report_lines), encoding="utf-8")

    manifest = {
        "threshold_csv": str(THR_CSV),
        "filter_csv": str(FILT_CSV),
        "report_md": str(REPORT_MD),
        "fig_threshold": str(FIG_THR),
        "fig_filter": str(FIG_FILT),
        "baseline": {
            "person_thr": float(baseline_row["person_thr"]),
            "person_fp_bg": int(baseline_row["person_fp_bg"]),
            "person_recall": float(baseline_row["person_recall"]),
            "person_f1": float(baseline_row["person_f1"]),
        },
        "best_threshold": {
            "person_thr": float(best_thr_row["person_thr"]),
            "person_fp_bg": int(best_thr_row["person_fp_bg"]),
            "person_recall": float(best_thr_row["person_recall"]),
            "person_f1": float(best_thr_row["person_f1"]),
        },
        "best_filter": {
            "person_thr": float(best_filter_row["person_thr"]),
            "min_area_frac": float(best_filter_row["min_area_frac"]),
            "ar_min": float(best_filter_row["ar_min"]),
            "ar_max": float(best_filter_row["ar_max"]),
            "person_fp_bg": int(best_filter_row["person_fp_bg"]),
            "person_recall": float(best_filter_row["person_recall"]),
            "person_f1": float(best_filter_row["person_f1"]),
            "overall_f1": float(best_filter_row["f1"]),
        },
    }
    MANIFEST.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"[saved] {THR_CSV}")
    print(f"[saved] {FILT_CSV}")
    print(f"[saved] {REPORT_MD}")
    print(f"[saved] {MANIFEST}")
    print(f"[best-thr] t={best_thr_row['person_thr']:.2f}, person_fp_bg={int(best_thr_row['person_fp_bg'])}, person_recall={best_thr_row['person_recall']:.4f}")
    print(f"[best-filter] t={best_filter_row['person_thr']:.2f}, min_area={best_filter_row['min_area_frac']:.4f}, ar=[{best_filter_row['ar_min']:.2f},{best_filter_row['ar_max']:.2f}], person_fp_bg={int(best_filter_row['person_fp_bg'])}, person_recall={best_filter_row['person_recall']:.4f}")


if __name__ == "__main__":
    main()
