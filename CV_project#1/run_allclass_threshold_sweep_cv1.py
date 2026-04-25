from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from ultralytics import YOLO
import cv2

ROOT = Path(r"C:\Users\lmhst\git\roboticsCV\CV_project#1")
OUT = ROOT / "outputs"
FIG = OUT / "figures"

MODEL = ROOT / "opt_aug_large_runs" / "lg_02_best_prev" / "weights" / "best.pt"
DATA = ROOT / "voc3" / "data.yaml"

NMS_IOU = 0.45
MAX_DET = 20
MATCH_IOU = 0.5

# Raw prediction floor for caching candidates
RAW_CONF = 0.20

# Per-class threshold grid
GRID = {
    "person": [0.36, 0.38, 0.40, 0.42],
    "car":    [0.30, 0.35, 0.40],
    "dog":    [0.30, 0.35, 0.40],
}

BASE = {"person": 0.40, "car": 0.35, "dog": 0.35}

CSV = OUT / "allclass_threshold_sweep_cv1.csv"
REPORT = OUT / "allclass_threshold_sweep_report_cv1.md"
MANIFEST = OUT / "allclass_threshold_sweep_manifest_cv1.json"
FIG_PARETO = FIG / "allclass_threshold_pareto_cv1.png"


@dataclass
class GT:
    cls: int
    box: np.ndarray


@dataclass
class PR:
    cls: int
    conf: float
    box: np.ndarray


def iou(a: np.ndarray, b: np.ndarray) -> float:
    x1 = max(a[0], b[0]); y1 = max(a[1], b[1]); x2 = min(a[2], b[2]); y2 = min(a[3], b[3])
    iw = max(0.0, x2 - x1); ih = max(0.0, y2 - y1)
    inter = iw * ih
    if inter <= 0: return 0.0
    aa = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    bb = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    uu = aa + bb - inter
    return float(inter / uu) if uu > 0 else 0.0


def safe_div(a: float, b: float) -> float:
    return float(a / b) if b > 0 else 0.0


def load_cfg():
    cfg = yaml.safe_load(DATA.read_text(encoding="utf-8"))
    names = cfg["names"]
    if isinstance(names, dict):
        names = [names[i] for i in sorted(names.keys())]
    val = Path(cfg["val"])
    if not val.is_absolute():
        val = (DATA.parent / val).resolve()
    labels = val.parent.parent / "labels" / "val"
    return val, labels, names


def load_gt(img: Path, labels_dir: Path) -> list[GT]:
    im = cv2.imread(str(img))
    if im is None:
        raise RuntimeError(f"cannot read {img}")
    h, w = im.shape[:2]
    lp = labels_dir / f"{img.stem}.txt"
    out: list[GT] = []
    if not lp.exists():
        return out
    for ln in lp.read_text(encoding="utf-8").splitlines():
        ln = ln.strip()
        if not ln:
            continue
        p = ln.split()
        c = int(float(p[0]))
        x, y, bw, bh = [float(v) for v in p[1:5]]
        out.append(GT(c, np.array([(x - bw / 2) * w, (y - bh / 2) * h, (x + bw / 2) * w, (y + bh / 2) * h], dtype=np.float32)))
    return out


def extract_pr(res) -> list[PR]:
    out: list[PR] = []
    if res.boxes is None or len(res.boxes) == 0:
        return out
    cls = res.boxes.cls.cpu().numpy().astype(int)
    conf = res.boxes.conf.cpu().numpy().astype(float)
    box = res.boxes.xyxy.cpu().numpy().astype(np.float32)
    for c, s, b in zip(cls, conf, box):
        out.append(PR(int(c), float(s), b))
    return out


def postfilter(preds: list[PR], cls_names: list[str], thr_map: dict[str, float]) -> list[PR]:
    out = []
    for p in preds:
        name = cls_names[p.cls]
        t = thr_map.get(name, 0.35)
        if p.conf >= t:
            out.append(p)
    out.sort(key=lambda x: x.conf, reverse=True)
    return out[:MAX_DET]


def eval_combo(images: list[Path], labels_dir: Path, cls_names: list[str], raw: dict[str, list[PR]], thr_map: dict[str, float]):
    ncls = len(cls_names)
    tp = fp = fn = 0
    tp_c = np.zeros(ncls, dtype=np.int64)
    fp_c = np.zeros(ncls, dtype=np.int64)
    fn_c = np.zeros(ncls, dtype=np.int64)

    for im in images:
        gts = load_gt(im, labels_dir)
        prs = postfilter(raw[im.name], cls_names, thr_map)
        used = set()
        for g in gts:
            bj = -1
            bi = 0.0
            for j, p in enumerate(prs):
                if j in used or p.cls != g.cls:
                    continue
                v = iou(g.box, p.box)
                if v >= MATCH_IOU and v > bi:
                    bi = v
                    bj = j
            if bj >= 0:
                used.add(bj)
                tp += 1
                tp_c[g.cls] += 1
            else:
                fn += 1
                fn_c[g.cls] += 1
        for j, p in enumerate(prs):
            if j not in used:
                fp += 1
                fp_c[p.cls] += 1

    p = safe_div(tp, tp + fp)
    r = safe_div(tp, tp + fn)
    f1 = safe_div(2 * p * r, p + r)

    rec = [safe_div(int(tp_c[i]), int(tp_c[i] + fn_c[i])) for i in range(ncls)]

    row = {
        "thr_person": thr_map["person"],
        "thr_car": thr_map["car"],
        "thr_dog": thr_map["dog"],
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "precision": p,
        "recall": r,
        "f1": f1,
        "rec_person": rec[0],
        "rec_car": rec[1],
        "rec_dog": rec[2],
        "fp_person": int(fp_c[0]),
        "fp_car": int(fp_c[1]),
        "fp_dog": int(fp_c[2]),
    }
    return row


def plot_pareto(df: pd.DataFrame, base_row: pd.Series):
    fig, ax = plt.subplots(figsize=(8, 5), dpi=150)
    ax.scatter(df["fp"], df["tp"], c=df["f1"], cmap="viridis", s=55, alpha=0.85)
    ax.scatter([base_row["fp"]], [base_row["tp"]], c="red", s=90, marker="x", label="current")
    ax.set_xlabel("FP")
    ax.set_ylabel("TP")
    ax.set_title("All-Class Threshold Sweep: TP-FP Tradeoff")
    ax.legend(loc="lower right")
    cb = plt.colorbar(ax.collections[0], ax=ax)
    cb.set_label("F1")
    fig.tight_layout()
    fig.savefig(FIG_PARETO)
    plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)

    val, labels, names = load_cfg()
    images = sorted([p for p in val.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}])

    model = YOLO(str(MODEL))
    raw_preds = model.predict(source=[str(p) for p in images], conf=RAW_CONF, iou=NMS_IOU, max_det=300, imgsz=640, verbose=False, save=False, device=0)
    raw = {p.name: extract_pr(r) for p, r in zip(images, raw_preds)}

    rows = []
    for tp in GRID["person"]:
        for tc in GRID["car"]:
            for td in GRID["dog"]:
                thr = {"person": tp, "car": tc, "dog": td}
                row = eval_combo(images, labels, names, raw, thr)
                rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(CSV, index=False, encoding="utf-8-sig")

    base = df[(np.isclose(df.thr_person, BASE["person"])) & (np.isclose(df.thr_car, BASE["car"])) & (np.isclose(df.thr_dog, BASE["dog"]))].iloc[0]

    # Best with minimal FP increase and TP gain preference
    cand = df[(df["tp"] >= base["tp"]) & (df["fp"] <= base["fp"] + 1)].copy()
    if len(cand) == 0:
        cand = df.copy()
    best = cand.sort_values(["tp", "fp", "f1"], ascending=[False, True, False]).iloc[0]

    # pure TP-max candidate
    tpmax = df.sort_values(["tp", "fp", "f1"], ascending=[False, True, False]).iloc[0]

    plot_pareto(df, base)

    lines = [
        "# All-Class Threshold Sweep (person/car/dog)",
        "",
        f"- model: `{MODEL}`",
        f"- fixed: nms_iou={NMS_IOU}, max_det={MAX_DET}",
        f"- raw cache conf={RAW_CONF}",
        "",
        "## Current setting",
        f"- thr(person,car,dog)=({base['thr_person']:.2f},{base['thr_car']:.2f},{base['thr_dog']:.2f})",
        f"- TP={int(base['tp'])}, FP={int(base['fp'])}, FN={int(base['fn'])}, F1={base['f1']:.4f}",
        "",
        "## Recommended (TP up with small FP cost)",
        f"- thr(person,car,dog)=({best['thr_person']:.2f},{best['thr_car']:.2f},{best['thr_dog']:.2f})",
        f"- TP={int(best['tp'])}, FP={int(best['fp'])}, FN={int(best['fn'])}, F1={best['f1']:.4f}",
        f"- delta vs current: dTP={int(best['tp']-base['tp'])}, dFP={int(best['fp']-base['fp'])}, dF1={best['f1']-base['f1']:+.4f}",
        "",
        "## TP-Max candidate",
        f"- thr(person,car,dog)=({tpmax['thr_person']:.2f},{tpmax['thr_car']:.2f},{tpmax['thr_dog']:.2f})",
        f"- TP={int(tpmax['tp'])}, FP={int(tpmax['fp'])}, FN={int(tpmax['fn'])}, F1={tpmax['f1']:.4f}",
        f"- delta vs current: dTP={int(tpmax['tp']-base['tp'])}, dFP={int(tpmax['fp']-base['fp'])}, dF1={tpmax['f1']-base['f1']:+.4f}",
        "",
        "![pareto](C:/Users/lmhst/git/roboticsCV/CV_project#1/outputs/figures/allclass_threshold_pareto_cv1.png)",
    ]
    REPORT.write_text("\n".join(lines), encoding="utf-8")

    manifest = {
        "csv": str(CSV),
        "report": str(REPORT),
        "figure": str(FIG_PARETO),
        "base": {k: (float(base[k]) if isinstance(base[k], (np.floating, float)) else int(base[k])) for k in ["thr_person","thr_car","thr_dog","tp","fp","fn","f1"]},
        "best": {k: (float(best[k]) if isinstance(best[k], (np.floating, float)) else int(best[k])) for k in ["thr_person","thr_car","thr_dog","tp","fp","fn","f1"]},
        "tpmax": {k: (float(tpmax[k]) if isinstance(tpmax[k], (np.floating, float)) else int(tpmax[k])) for k in ["thr_person","thr_car","thr_dog","tp","fp","fn","f1"]},
    }
    MANIFEST.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"[saved] {CSV}")
    print(f"[saved] {REPORT}")
    print(f"[saved] {MANIFEST}")
    print(f"[base] TP={int(base['tp'])} FP={int(base['fp'])} F1={base['f1']:.4f}")
    print(f"[best] TP={int(best['tp'])} FP={int(best['fp'])} F1={best['f1']:.4f} thr=({best['thr_person']:.2f},{best['thr_car']:.2f},{best['thr_dog']:.2f})")
    print(f"[tpmax] TP={int(tpmax['tp'])} FP={int(tpmax['fp'])} F1={tpmax['f1']:.4f} thr=({tpmax['thr_person']:.2f},{tpmax['thr_car']:.2f},{tpmax['thr_dog']:.2f})")


if __name__ == "__main__":
    main()
