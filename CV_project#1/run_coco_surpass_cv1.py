from __future__ import annotations

import itertools
import json
import math
import shutil
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import yaml
from scipy.stats import ttest_1samp, wilcoxon
from ultralytics import YOLO

ROOT = Path(r"C:\Users\lmhst\git\roboticsCV\CV_project#1")
OUT = ROOT / "outputs"
RUNS = ROOT / "coco_surpass_runs"

DATA_EVAL = ROOT / "voc3" / "data.yaml"

# Train sets to try
DATASETS = {
    "voc3": ROOT / "voc3" / "data.yaml",
    "voc3_large": ROOT / "voc3_large" / "data.yaml",
}

MATCH_IOU = 0.5
RAW_CONF = 0.05

# Operating sweep space (same for COCO and candidates)
NMS_GRID = [0.45, 0.60]
MAXDET_GRID = [50, 100]
TP_GRID = [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40]
TC_GRID = [0.10, 0.15, 0.20, 0.25, 0.30, 0.35]
TD_GRID = [0.10, 0.15, 0.20, 0.25, 0.30, 0.35]

# Gentle fine-tune candidates to avoid catastrophic forgetting
CANDIDATES = [
    {
        "tag": "n_gentle_1",
        "base_model": "yolov8n.pt",
        "train_data": "voc3_large",
        "epochs": 8,
        "batch": 16,
        "imgsz": 640,
        "lr0": 0.0003,
        "freeze": 10,
        "mosaic": 0.2,
        "mixup": 0.0,
        "close_mosaic": 2,
    },
    {
        "tag": "n_gentle_2",
        "base_model": "yolov8n.pt",
        "train_data": "voc3_large",
        "epochs": 12,
        "batch": 16,
        "imgsz": 640,
        "lr0": 0.0002,
        "freeze": 12,
        "mosaic": 0.3,
        "mixup": 0.0,
        "close_mosaic": 2,
    },
    {
        "tag": "n_gentle_3",
        "base_model": "yolov8n.pt",
        "train_data": "voc3_large",
        "epochs": 16,
        "batch": 16,
        "imgsz": 640,
        "lr0": 0.00015,
        "freeze": 14,
        "mosaic": 0.15,
        "mixup": 0.0,
        "close_mosaic": 2,
    },
    {
        "tag": "n_voc3_focus",
        "base_model": "yolov8n.pt",
        "train_data": "voc3",
        "epochs": 10,
        "batch": 16,
        "imgsz": 640,
        "lr0": 0.00025,
        "freeze": 8,
        "mosaic": 0.2,
        "mixup": 0.0,
        "close_mosaic": 2,
    },
]

SEARCH_CSV = OUT / "coco_surpass_search_cv1.csv"
SEARCH_REPORT = OUT / "coco_surpass_search_report_cv1.md"
SEARCH_MANIFEST = OUT / "coco_surpass_search_manifest_cv1.json"

SEED_REPEAT_CSV = OUT / "coco_surpass_seedrepeat_cv1.csv"
SEED_REPEAT_REPORT = OUT / "coco_surpass_seedrepeat_report_cv1.md"
SEED_REPEAT_MANIFEST = OUT / "coco_surpass_seedrepeat_manifest_cv1.json"

SEEDS = [101, 202, 303, 404, 505]


@dataclass
class EvalData:
    names: list[str]
    imgs: list[Path]
    labels_dir: Path


def safe_div(a: float, b: float) -> float:
    return float(a / b) if b > 0 else 0.0


def iou_xyxy(a, b) -> float:
    x1 = max(a[0], b[0]); y1 = max(a[1], b[1]); x2 = min(a[2], b[2]); y2 = min(a[3], b[3])
    iw = max(0.0, x2 - x1); ih = max(0.0, y2 - y1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    aa = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    bb = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    uu = aa + bb - inter
    return float(inter / uu) if uu > 0 else 0.0


def load_eval_data() -> EvalData:
    cfg = yaml.safe_load(DATA_EVAL.read_text(encoding="utf-8"))
    names = cfg["names"]
    if isinstance(names, dict):
        names = [names[i] for i in sorted(names.keys())]
    val = Path(cfg["val"])
    if not val.is_absolute():
        val = (DATA_EVAL.parent / val).resolve()
    labels = val.parent.parent / "labels" / "val"
    imgs = sorted([p for p in val.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}])
    return EvalData(names=names, imgs=imgs, labels_dir=labels)


def load_gt(img: Path, labels_dir: Path):
    im = cv2.imread(str(img))
    h, w = im.shape[:2]
    lp = labels_dir / f"{img.stem}.txt"
    out = []
    if lp.exists():
        for ln in lp.read_text(encoding="utf-8").splitlines():
            ln = ln.strip()
            if not ln:
                continue
            p = ln.split()
            c = int(float(p[0]))
            x, y, bw, bh = [float(v) for v in p[1:5]]
            out.append((c, [(x - bw / 2) * w, (y - bh / 2) * h, (x + bw / 2) * w, (y + bh / 2) * h]))
    return out


def cache_preds(model_ref: str, data: EvalData, nms_iou: float, max_det: int):
    m = YOLO(model_ref)
    name2idx = {n: i for i, n in enumerate(data.names)}
    rs = m.predict(
        source=[str(p) for p in data.imgs],
        conf=RAW_CONF,
        iou=nms_iou,
        max_det=max_det,
        imgsz=640,
        verbose=False,
        save=False,
        device=0,
    )
    out = {}
    for p, r in zip(data.imgs, rs):
        arr = []
        if r.boxes is not None and len(r.boxes) > 0:
            cls = r.boxes.cls.cpu().numpy().astype(int)
            conf = r.boxes.conf.cpu().numpy().astype(float)
            box = r.boxes.xyxy.cpu().numpy().astype(float)
            for c, s, b in zip(cls, conf, box):
                cname = m.names[int(c)]
                if cname not in name2idx:
                    continue
                arr.append((name2idx[cname], float(s), b.tolist()))
        out[p.name] = arr
    return out


def eval_with_thr(data: EvalData, cache: dict, thr: dict[str, float], max_det: int):
    names = data.names
    tp = fp = fn = 0
    tp_c = np.zeros(len(names), dtype=np.int64)
    fp_c = np.zeros(len(names), dtype=np.int64)
    fn_c = np.zeros(len(names), dtype=np.int64)

    for img in data.imgs:
        gts = load_gt(img, data.labels_dir)
        prs = []
        for c, s, b in cache[img.name]:
            if s >= thr[names[c]]:
                prs.append((c, s, b))
        prs.sort(key=lambda x: x[1], reverse=True)
        prs = prs[:max_det]

        used = set()
        for gc, gb in gts:
            bj = -1
            bi = 0.0
            for j, (pc, _ps, pb) in enumerate(prs):
                if j in used or pc != gc:
                    continue
                v = iou_xyxy(gb, pb)
                if v >= MATCH_IOU and v > bi:
                    bi = v
                    bj = j
            if bj >= 0:
                used.add(bj)
                tp += 1
                tp_c[gc] += 1
            else:
                fn += 1
                fn_c[gc] += 1

        for j, (pc, _ps, _pb) in enumerate(prs):
            if j not in used:
                fp += 1
                fp_c[pc] += 1

    p = safe_div(tp, tp + fp)
    r = safe_div(tp, tp + fn)
    f1 = safe_div(2 * p * r, p + r)

    fms = []
    for i in range(3):
        pp = safe_div(int(tp_c[i]), int(tp_c[i] + fp_c[i]))
        rr = safe_div(int(tp_c[i]), int(tp_c[i] + fn_c[i]))
        ff = safe_div(2 * pp * rr, pp + rr)
        fms.append(ff)

    return {
        "tp": int(tp), "fp": int(fp), "fn": int(fn),
        "precision": p, "recall": r, "f1": f1,
        "f1_macro": float(np.mean(fms)),
    }


def sweep_best(data: EvalData, model_ref: str):
    best = None
    best_detail = None
    for nms in NMS_GRID:
        for md in MAXDET_GRID:
            cache = cache_preds(model_ref, data, nms_iou=nms, max_det=300)
            for tp in TP_GRID:
                for tc in TC_GRID:
                    for td in TD_GRID:
                        thr = {"person": tp, "car": tc, "dog": td}
                        m = eval_with_thr(data, cache, thr, max_det=md)
                        row = {
                            "nms_iou": nms,
                            "max_det": md,
                            "thr_person": tp,
                            "thr_car": tc,
                            "thr_dog": td,
                            **m,
                        }
                        if best is None or row["f1"] > best["f1"] or (abs(row["f1"] - best["f1"]) < 1e-12 and row["fp"] < best["fp"]):
                            best = row
                            best_detail = row.copy()
    return best_detail


def train_from_scratch_like(cfg: dict, seed: int = 42, suffix: str = "") -> Path:
    tag = cfg["tag"] + suffix
    exp = RUNS / tag
    if exp.exists():
        shutil.rmtree(exp)

    model = YOLO(cfg["base_model"])
    model.train(
        data=str(DATASETS[cfg["train_data"]]),
        model=cfg["base_model"],
        epochs=int(cfg["epochs"]),
        imgsz=int(cfg["imgsz"]),
        batch=int(cfg["batch"]),
        lr0=float(cfg["lr0"]),
        freeze=int(cfg["freeze"]),
        mosaic=float(cfg["mosaic"]),
        mixup=float(cfg["mixup"]),
        close_mosaic=int(cfg["close_mosaic"]),
        project=str(RUNS),
        name=tag,
        exist_ok=True,
        verbose=False,
        workers=0,
        seed=int(seed),
        deterministic=True,
        resume=False,
    )
    best = exp / "weights" / "best.pt"
    if not best.exists():
        best = exp / "weights" / "last.pt"
    return best


def search_phase(data: EvalData):
    rows = []

    coco_best = sweep_best(data, "yolov8n.pt")

    for cfg in CANDIDATES:
        print("[train-search]", cfg["tag"])
        best_pt = train_from_scratch_like(cfg, seed=42)
        fin_best = sweep_best(data, str(best_pt))
        row = {
            "tag": cfg["tag"],
            "best_pt": str(best_pt),
            "base_model": cfg["base_model"],
            "train_data": cfg["train_data"],
            "epochs": cfg["epochs"],
            "lr0": cfg["lr0"],
            "freeze": cfg["freeze"],
            "mosaic": cfg["mosaic"],
            "mixup": cfg["mixup"],
            "best_f1": fin_best["f1"],
            "best_tp": fin_best["tp"],
            "best_fp": fin_best["fp"],
            "best_nms_iou": fin_best["nms_iou"],
            "best_max_det": fin_best["max_det"],
            "best_thr_person": fin_best["thr_person"],
            "best_thr_car": fin_best["thr_car"],
            "best_thr_dog": fin_best["thr_dog"],
            "delta_f1_vs_coco_best": fin_best["f1"] - coco_best["f1"],
            "delta_tp_vs_coco_best": fin_best["tp"] - coco_best["tp"],
            "delta_fp_vs_coco_best": fin_best["fp"] - coco_best["fp"],
        }
        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(SEARCH_CSV, index=False, encoding="utf-8-sig")

    best_idx = int(df["best_f1"].idxmax())
    best_row = df.iloc[best_idx].to_dict()

    lines = [
        "# COCO Surpass Search",
        "",
        f"- COCO best: F1={coco_best['f1']:.4f}, TP={coco_best['tp']}, FP={coco_best['fp']}, nms_iou={coco_best['nms_iou']}, max_det={coco_best['max_det']}, thr=({coco_best['thr_person']:.2f},{coco_best['thr_car']:.2f},{coco_best['thr_dog']:.2f})",
        "",
        "## Candidate Results",
    ]
    for r in rows:
        lines.append(
            f"- {r['tag']}: F1={r['best_f1']:.4f}, TP={int(r['best_tp'])}, FP={int(r['best_fp'])}, dF1_vs_COCO={r['delta_f1_vs_coco_best']:+.4f}"
        )
    lines += [
        "",
        f"## Best Candidate: {best_row['tag']}",
        f"- F1={best_row['best_f1']:.4f}, TP={int(best_row['best_tp'])}, FP={int(best_row['best_fp'])}, dF1_vs_COCO={best_row['delta_f1_vs_coco_best']:+.4f}",
    ]
    SEARCH_REPORT.write_text("\n".join(lines), encoding="utf-8")

    manifest = {
        "search_csv": str(SEARCH_CSV),
        "search_report": str(SEARCH_REPORT),
        "coco_best": coco_best,
        "best_candidate": best_row,
    }
    SEARCH_MANIFEST.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    return coco_best, best_row


def seed_repeat_phase(data: EvalData, coco_best: dict, best_cfg_row: dict):
    # reconstruct candidate cfg
    cfg = next(c for c in CANDIDATES if c["tag"] == best_cfg_row["tag"])

    rows = []
    deltas = []

    for s in SEEDS:
        print("[seed-repeat]", cfg["tag"], "seed", s)
        best_pt = train_from_scratch_like(cfg, seed=s, suffix=f"_seed{s}")
        fin_best = sweep_best(data, str(best_pt))

        delta = fin_best["f1"] - coco_best["f1"]
        deltas.append(delta)
        rows.append(
            {
                "seed": s,
                "best_pt": str(best_pt),
                "fin_f1": fin_best["f1"],
                "fin_tp": fin_best["tp"],
                "fin_fp": fin_best["fp"],
                "fin_nms_iou": fin_best["nms_iou"],
                "fin_max_det": fin_best["max_det"],
                "fin_thr_person": fin_best["thr_person"],
                "fin_thr_car": fin_best["thr_car"],
                "fin_thr_dog": fin_best["thr_dog"],
                "coco_f1": coco_best["f1"],
                "delta_f1_vs_coco": delta,
            }
        )

    df = pd.DataFrame(rows)
    df.to_csv(SEED_REPEAT_CSV, index=False, encoding="utf-8-sig")

    # one-sample tests on delta > 0
    arr = np.array(deltas, dtype=float)
    t = ttest_1samp(arr, popmean=0.0, alternative="greater")
    nz = arr[np.abs(arr) > 1e-12]
    w_p = float(wilcoxon(nz, alternative="greater").pvalue) if len(nz) > 0 else 1.0

    res = {
        "n": int(len(arr)),
        "mean_delta_f1": float(np.mean(arr)),
        "std_delta_f1": float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0,
        "ttest_p_greater": float(t.pvalue),
        "wilcoxon_p_greater": w_p,
        "significant_at_0_05": bool((float(t.pvalue) < 0.05) and (np.mean(arr) > 0)),
    }

    lines = [
        "# COCO Surpass Seed Repeat",
        "",
        f"- candidate: {cfg['tag']}",
        f"- COCO best F1 reference: {coco_best['f1']:.4f}",
        f"- seeds: {SEEDS}",
        "",
        "## Result",
        f"- mean delta F1 (finetuned - coco): {res['mean_delta_f1']:+.4f}",
        f"- one-sample t-test (H1: delta>0) p={res['ttest_p_greater']:.4g}",
        f"- Wilcoxon signed-rank (H1: delta>0) p={res['wilcoxon_p_greater']:.4g}",
        f"- significant@0.05: {res['significant_at_0_05']}",
    ]
    SEED_REPEAT_REPORT.write_text("\n".join(lines), encoding="utf-8")

    manifest = {
        "seed_csv": str(SEED_REPEAT_CSV),
        "seed_report": str(SEED_REPEAT_REPORT),
        "coco_best": coco_best,
        "candidate": cfg,
        "result": res,
    }
    SEED_REPEAT_MANIFEST.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    return res


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    RUNS.mkdir(parents=True, exist_ok=True)

    data = load_eval_data()

    coco_best, best_row = search_phase(data)
    print("[search-best]", best_row["tag"], best_row["best_f1"], "dF1", best_row["delta_f1_vs_coco_best"])

    res = seed_repeat_phase(data, coco_best, best_row)
    print("[seed-repeat]", res)


if __name__ == "__main__":
    main()
