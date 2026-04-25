from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy.stats import wilcoxon, ttest_rel, binomtest
from ultralytics import YOLO

ROOT = Path(r"C:\Users\lmhst\git\roboticsCV\CV_project#1")
OUT = ROOT / "outputs"

MODEL = ROOT / "opt_aug_large_runs" / "lg_02_best_prev" / "weights" / "best.pt"
DATA_YAML = ROOT / "voc3" / "data.yaml"
MATCH_IOU = 0.5

BASE = {"conf": 0.35, "nms_iou": 0.70, "max_det": 300}
TUNED = {"conf": 0.35, "nms_iou": 0.45, "max_det": 20}

IMG_CSV = OUT / "lg02_operating_tune2_significance_per_image_cv1.csv"
SUMMARY_JSON = OUT / "lg02_operating_tune2_significance_manifest_cv1.json"
REPORT_MD = OUT / "lg02_operating_tune2_significance_report_cv1.md"

np.random.seed(42)


def safe_div(a: float, b: float) -> float:
    return float(a / b) if b > 0 else 0.0


def xywhn_to_xyxy(xywhn: np.ndarray, w: int, h: int) -> np.ndarray:
    x, y, bw, bh = xywhn
    return np.array([(x - bw / 2.0) * w, (y - bh / 2.0) * h, (x + bw / 2.0) * w, (y + bh / 2.0) * h], dtype=np.float32)


def iou_xyxy(a: np.ndarray, b: np.ndarray) -> float:
    x1 = max(a[0], b[0]); y1 = max(a[1], b[1]); x2 = min(a[2], b[2]); y2 = min(a[3], b[3])
    iw = max(0.0, x2 - x1); ih = max(0.0, y2 - y1)
    inter = iw * ih
    if inter <= 0: return 0.0
    aa = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    bb = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    uu = aa + bb - inter
    return float(inter / uu) if uu > 0 else 0.0


@dataclass
class Box:
    cls: int
    xyxy: np.ndarray


def load_data_paths(data_yaml: Path):
    data = yaml.safe_load(data_yaml.read_text(encoding="utf-8"))
    val_images = Path(data["val"])
    if not val_images.is_absolute():
        val_images = (data_yaml.parent / val_images).resolve()
    labels_dir = val_images.parent.parent / "labels" / "val"
    return val_images, labels_dir


def load_gt(img_path: Path, labels_dir: Path):
    import cv2
    label_path = labels_dir / f"{img_path.stem}.txt"
    if not label_path.exists():
        return []
    im = cv2.imread(str(img_path))
    if im is None:
        raise RuntimeError(f"failed to read {img_path}")
    h, w = im.shape[:2]
    out = []
    for line in label_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        p = line.split()
        out.append(Box(cls=int(float(p[0])), xyxy=xywhn_to_xyxy(np.array([float(v) for v in p[1:5]], dtype=np.float32), w, h)))
    return out


def preds_from_result(res):
    if res.boxes is None or len(res.boxes) == 0:
        return []
    cls = res.boxes.cls.cpu().numpy().astype(int)
    xyxy = res.boxes.xyxy.cpu().numpy().astype(np.float32)
    return [Box(cls=int(c), xyxy=b) for c, b in zip(cls, xyxy)]


def match_counts(gt, pred):
    used = set(); tp = 0; fn = 0; fp = 0
    for g in gt:
        best_j = -1; best_iou = 0.0
        for j, p in enumerate(pred):
            if j in used or p.cls != g.cls:
                continue
            iou = iou_xyxy(g.xyxy, p.xyxy)
            if iou >= MATCH_IOU and iou > best_iou:
                best_iou = iou; best_j = j
        if best_j >= 0:
            used.add(best_j); tp += 1
        else:
            fn += 1
    for j, _ in enumerate(pred):
        if j not in used:
            fp += 1
    return tp, fp, fn


def infer_counts(model, image_paths, labels_dir, conf, nms_iou, max_det):
    preds = model.predict(source=[str(p) for p in image_paths], conf=conf, iou=nms_iou, max_det=max_det, imgsz=640, verbose=False, save=False, device=0)
    rows = []
    for p, r in zip(image_paths, preds):
        gt = load_gt(p, labels_dir)
        pr = preds_from_result(r)
        tp, fp, fn = match_counts(gt, pr)
        prec = safe_div(tp, tp + fp); rec = safe_div(tp, tp + fn); f1 = safe_div(2 * prec * rec, prec + rec)
        rows.append({"image": p.name, "tp": tp, "fp": fp, "fn": fn, "precision": prec, "recall": rec, "f1": f1})
    return pd.DataFrame(rows)


def global_metrics(df):
    tp = int(df.tp.sum()); fp = int(df.fp.sum()); fn = int(df.fn.sum())
    p = safe_div(tp, tp + fp); r = safe_div(tp, tp + fn); f1 = safe_div(2 * p * r, p + r)
    return {"tp": tp, "fp": fp, "fn": fn, "precision": p, "recall": r, "f1": f1}


def bootstrap_delta(base_df, tuned_df, n_boot=20000):
    n = len(base_df); idx = np.arange(n)
    deltas = {"precision": [], "recall": [], "f1": [], "fp": []}
    for _ in range(n_boot):
        s = np.random.choice(idx, size=n, replace=True)
        gb = global_metrics(base_df.iloc[s]); gt = global_metrics(tuned_df.iloc[s])
        deltas["precision"].append(gt["precision"] - gb["precision"])
        deltas["recall"].append(gt["recall"] - gb["recall"])
        deltas["f1"].append(gt["f1"] - gb["f1"])
        deltas["fp"].append(gt["fp"] - gb["fp"])
    out = {}
    for k, arr in deltas.items():
        a = np.array(arr)
        out[k] = {
            "mean": float(np.mean(a)),
            "ci95_low": float(np.percentile(a, 2.5)),
            "ci95_high": float(np.percentile(a, 97.5)),
            "p_two_sided": float(min(1.0, 2 * min(np.mean(a <= 0), np.mean(a >= 0)))),
        }
    return out


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    val_images, labels_dir = load_data_paths(DATA_YAML)
    image_paths = sorted([p for p in val_images.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}])
    model = YOLO(str(MODEL))

    base_df = infer_counts(model, image_paths, labels_dir, **BASE)
    tuned_df = infer_counts(model, image_paths, labels_dir, **TUNED)

    merged = base_df.merge(tuned_df, on="image", suffixes=("_base", "_tuned"))
    for m in ["precision", "recall", "f1", "tp", "fp", "fn"]:
        merged[f"delta_{m}"] = merged[f"{m}_tuned"] - merged[f"{m}_base"]
    merged.to_csv(IMG_CSV, index=False, encoding="utf-8-sig")

    base_global = global_metrics(base_df); tuned_global = global_metrics(tuned_df)

    tests = {}
    for m in ["precision", "recall", "f1", "fp"]:
        d = merged[f"delta_{m}"].astype(float).values
        tt = ttest_rel(merged[f"{m}_tuned"], merged[f"{m}_base"])
        nz = d[np.abs(d) > 1e-12]
        p_w = float(wilcoxon(nz, alternative="two-sided").pvalue) if len(nz) else 1.0
        tests[m] = {"mean_delta": float(np.mean(d)), "median_delta": float(np.median(d)), "ttest_p": float(tt.pvalue), "wilcoxon_p": p_w, "n_nonzero": int((np.abs(d) > 1e-12).sum())}

    d_f1 = merged.delta_f1.values
    n_pos = int((d_f1 > 0).sum()); n_neg = int((d_f1 < 0).sum()); n_eff = n_pos + n_neg
    sign_p = float(binomtest(k=n_pos, n=n_eff, p=0.5, alternative="two-sided").pvalue) if n_eff > 0 else 1.0

    boot = bootstrap_delta(base_df, tuned_df, n_boot=20000)

    summary = {
        "model": str(MODEL), "baseline_setting": BASE, "tuned_setting": TUNED,
        "base_global": base_global, "tuned_global": tuned_global,
        "global_delta": {"precision": tuned_global["precision"] - base_global["precision"], "recall": tuned_global["recall"] - base_global["recall"], "f1": tuned_global["f1"] - base_global["f1"], "fp": tuned_global["fp"] - base_global["fp"]},
        "paired_tests_image_level": tests,
        "sign_test_f1": {"n_pos": n_pos, "n_neg": n_neg, "n_effective": n_eff, "p_value": sign_p},
        "bootstrap_global_delta": boot,
        "n_images": len(merged),
    }
    SUMMARY_JSON.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    lines = [
        "# lg_02 운영 2차 튜닝 유의성 검정", "",
        f"- baseline: conf={BASE['conf']}, nms_iou={BASE['nms_iou']}, max_det={BASE['max_det']}",
        f"- tuned: conf={TUNED['conf']}, nms_iou={TUNED['nms_iou']}, max_det={TUNED['max_det']}",
        f"- images: {len(merged)}", "",
        "## Global Metrics", "",
        f"- baseline: P={base_global['precision']:.4f}, R={base_global['recall']:.4f}, F1={base_global['f1']:.4f}, FP={base_global['fp']}",
        f"- tuned: P={tuned_global['precision']:.4f}, R={tuned_global['recall']:.4f}, F1={tuned_global['f1']:.4f}, FP={tuned_global['fp']}",
        f"- delta: dP={summary['global_delta']['precision']:+.4f}, dR={summary['global_delta']['recall']:+.4f}, dF1={summary['global_delta']['f1']:+.4f}, dFP={summary['global_delta']['fp']:+.0f}", "",
        "## Paired Tests (image-level)", "",
    ]
    for m in ["precision", "recall", "f1", "fp"]:
        t = tests[m]
        lines.append(f"- {m}: mean_delta={t['mean_delta']:+.4f}, t-test p={t['ttest_p']:.4g}, Wilcoxon p={t['wilcoxon_p']:.4g}, nonzero={t['n_nonzero']}")
    lines += [f"- sign-test(f1 improvement): +={n_pos}, -={n_neg}, p={sign_p:.4g}", "", "## Bootstrap (global delta, 20k resamples)", ""]
    for k in ["precision", "recall", "f1", "fp"]:
        b = boot[k]
        lines.append(f"- {k}: mean={b['mean']:+.4f}, 95%CI=[{b['ci95_low']:+.4f}, {b['ci95_high']:+.4f}], p~{b['p_two_sided']:.4g}")
    REPORT_MD.write_text("\n".join(lines), encoding="utf-8")

    print("[saved]", IMG_CSV)
    print("[saved]", SUMMARY_JSON)
    print("[saved]", REPORT_MD)
    print("[result]", json.dumps({"global_delta": summary["global_delta"], "f1_wilcoxon_p": tests["f1"]["wilcoxon_p"], "fp_wilcoxon_p": tests["fp"]["wilcoxon_p"], "f1_boot_p": boot["f1"]["p_two_sided"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
