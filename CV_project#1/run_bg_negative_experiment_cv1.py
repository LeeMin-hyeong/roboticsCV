import csv
import json
import random
import shutil
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torchvision
import yaml
from PIL import Image
from ultralytics import YOLO


ROOT = Path(__file__).resolve().parent
DATA_ROOT = ROOT / "data"
VOC3_ROOT = ROOT / "voc3"
OUT_DIR = ROOT / "outputs"
FIG_DIR = OUT_DIR / "figures"
RUNS_DIR = ROOT / "bg_neg_runs"
DS_ROOT = ROOT / "datasets_bgneg"

TARGET = ["person", "car", "dog"]
CLS2IDX = {c: i for i, c in enumerate(TARGET)}

NEG_RATIOS = [0.3, 0.6, 1.0]
SEEDS = [111, 222, 333]

# Use previously selected strong config as fixed training recipe
TRAIN_CFG = {
    "epochs": 16,
    "imgsz": 640,
    "batch": 12,
    "lr0": 0.0009,
    "freeze": 5,
    "mosaic": 0.6,
    "mixup": 0.1,
    "close_mosaic": 3,
}

EVAL_DATA = VOC3_ROOT / "data.yaml"  # keep baseline eval split fixed
BASELINE_CSV = OUT_DIR / "ablation_train_summary_cv1.csv"

CONF_THR = 0.15
IOU_THR = 0.5


def parse_voc(ann):
    objs = ann["annotation"]["object"]
    return [objs] if isinstance(objs, dict) else objs


def has_target(objs):
    return any(o["name"] in CLS2IDX for o in objs)


def gather_negative_filenames():
    neg = {"train": [], "val": []}
    voc_tr = torchvision.datasets.VOCDetection(root=str(DATA_ROOT), year="2007", image_set="trainval", download=True)
    voc_te = torchvision.datasets.VOCDetection(root=str(DATA_ROOT), year="2007", image_set="test", download=True)

    for sp, ds in [("train", voc_tr), ("val", voc_te)]:
        for i in range(len(ds)):
            _, t = ds[i]
            objs = parse_voc(t)
            if has_target(objs):
                continue
            fn = t["annotation"]["filename"]
            src = DATA_ROOT / "VOCdevkit" / "VOC2007" / "JPEGImages" / fn
            if src.exists():
                neg[sp].append(fn)
    return neg


def build_dataset_with_negatives(ratio: float, neg_pool: dict):
    tag = f"r{int(ratio * 100):02d}"
    ds_dir = DS_ROOT / f"voc3_neg_{tag}"
    if ds_dir.exists():
        shutil.rmtree(ds_dir, ignore_errors=True)

    # Start from exact positive-only dataset (fixed baseline split)
    for sp in ["train", "val"]:
        (ds_dir / "images" / sp).mkdir(parents=True, exist_ok=True)
        (ds_dir / "labels" / sp).mkdir(parents=True, exist_ok=True)
        for ip in sorted((VOC3_ROOT / "images" / sp).glob("*.jpg")):
            lp = VOC3_ROOT / "labels" / sp / (ip.stem + ".txt")
            shutil.copy2(ip, ds_dir / "images" / sp / ip.name)
            shutil.copy2(lp, ds_dir / "labels" / sp / lp.name)

    # Add negatives with empty label files
    meta = {"ratio": ratio}
    for sp in ["train", "val"]:
        n_pos = len(list((VOC3_ROOT / "images" / sp).glob("*.jpg")))
        n_neg = int(round(n_pos * ratio))
        rng = random.Random(1000 + int(ratio * 1000) + (0 if sp == "train" else 99))
        pool = list(neg_pool[sp])
        rng.shuffle(pool)
        chosen = pool[: min(n_neg, len(pool))]
        added = 0
        for fn in chosen:
            src = DATA_ROOT / "VOCdevkit" / "VOC2007" / "JPEGImages" / fn
            if not src.exists():
                continue
            dst_img = ds_dir / "images" / sp / f"neg_{fn}"
            dst_lb = ds_dir / "labels" / sp / f"neg_{Path(fn).stem}.txt"
            if dst_img.exists():
                continue
            shutil.copy2(src, dst_img)
            dst_lb.write_text("", encoding="utf-8")
            added += 1
        meta[f"{sp}_pos"] = n_pos
        meta[f"{sp}_neg_added"] = added

    data_yaml = {
        "path": str(ds_dir.resolve()),
        "train": "images/train",
        "val": "images/val",
        "nc": 3,
        "names": TARGET,
    }
    with open(ds_dir / "data.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(data_yaml, f, sort_keys=False)
    return ds_dir, meta


def iou_xyxy(a, b):
    xi = max(a[0], b[0])
    yi = max(a[1], b[1])
    xa = min(a[2], b[2])
    ya = min(a[3], b[3])
    inter = max(0.0, xa - xi) * max(0.0, ya - yi)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter + 1e-9
    return inter / union


def confusion_bg_counts(weights_path: Path):
    model = YOLO(str(weights_path))
    fp_bg_to_cls = 0
    fn_cls_to_bg = 0
    tp = 0
    val_imgs = sorted((VOC3_ROOT / "images" / "val").glob("*.jpg"))

    for img_path in val_imgs:
        img = Image.open(img_path)
        w, h = img.size
        lp = VOC3_ROOT / "labels" / "val" / (img_path.stem + ".txt")
        gts = []
        for ln in lp.read_text(encoding="utf-8").splitlines():
            ln = ln.strip()
            if not ln:
                continue
            p = ln.split()
            ci = int(p[0])
            cx, cy, bw, bh = [float(v) for v in p[1:]]
            gts.append(
                {
                    "cls": ci,
                    "box": [(cx - bw / 2) * w, (cy - bh / 2) * h, (cx + bw / 2) * w, (cy + bh / 2) * h],
                }
            )

        preds = []
        r = model(str(img_path), conf=CONF_THR, verbose=False)[0]
        for b in r.boxes:
            name = model.names[int(b.cls[0].item())]
            if name not in CLS2IDX:
                continue
            preds.append({"cls": CLS2IDX[name], "box": b.xyxy[0].cpu().numpy().tolist(), "conf": float(b.conf[0].item())})

        pairs = []
        for gi, gt in enumerate(gts):
            for pi, pr in enumerate(preds):
                if gt["cls"] != pr["cls"]:
                    continue
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
            tp += 1

        fn_cls_to_bg += max(0, len(gts) - len(used_g))
        fp_bg_to_cls += max(0, len(preds) - len(used_p))

    return {"tp": tp, "fn_cls_to_bg": fn_cls_to_bg, "fp_bg_to_cls": fp_bg_to_cls}


def train_one(ratio: float, seed: int, ds_yaml: Path):
    tag = f"bgneg_r{int(ratio*100):02d}_s{seed}"
    exp = RUNS_DIR / tag
    if exp.exists():
        shutil.rmtree(exp, ignore_errors=True)

    model = YOLO("yolov8n.pt")
    model.train(
        data=str(ds_yaml),
        project=str(RUNS_DIR),
        name=tag,
        exist_ok=True,
        verbose=False,
        workers=0,
        seed=seed,
        deterministic=True,
        resume=False,
        **TRAIN_CFG,
    )
    best = exp / "weights" / "best.pt"

    # evaluate on fixed baseline split
    m = YOLO(str(best)).val(data=str(EVAL_DATA), imgsz=int(TRAIN_CFG["imgsz"]), verbose=False, workers=0)
    bg = confusion_bg_counts(best)
    row = {
        "ratio": ratio,
        "seed": seed,
        "map50": float(m.box.map50),
        "map50_95": float(m.box.map),
        "precision": float(m.box.mp),
        "recall": float(m.box.mr),
        "tp": int(bg["tp"]),
        "fn_cls_to_bg": int(bg["fn_cls_to_bg"]),
        "fp_bg_to_cls": int(bg["fp_bg_to_cls"]),
        "best_pt": str(best.resolve()),
        "train_data": str(ds_yaml.resolve()),
    }
    print(
        f"[ratio={ratio:.1f} seed={seed}] map50={row['map50']:.4f} map50_95={row['map50_95']:.4f} "
        f"P={row['precision']:.4f} R={row['recall']:.4f} FPbg={row['fp_bg_to_cls']} FNbg={row['fn_cls_to_bg']}"
    )
    return row


def baseline_metrics():
    rows = list(csv.DictReader(open(BASELINE_CSV, encoding="utf-8")))
    b = next(r for r in rows if r["variant"] == "baseline")
    return {
        "map50": float(b["map50"]),
        "map50_95": float(b["map50_95"]),
        "precision": float(b["precision"]),
        "recall": float(b["recall"]),
    }


def summarize(rows):
    ratios = sorted({float(r["ratio"]) for r in rows})
    out = []
    for rr in ratios:
        rs = [r for r in rows if float(r["ratio"]) == rr]
        item = {"ratio": rr, "n": len(rs)}
        for k in ["map50", "map50_95", "precision", "recall", "tp", "fn_cls_to_bg", "fp_bg_to_cls"]:
            vals = np.array([float(r[k]) for r in rs], dtype=float)
            item[f"{k}_mean"] = float(np.mean(vals))
            item[f"{k}_std"] = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
        out.append(item)
    return out


def make_figures(sum_rows, base):
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    x = np.arange(len(sum_rows))
    labels = [f"{r['ratio']:.1f}" for r in sum_rows]

    # Performance
    plt.figure(figsize=(9, 5))
    map50 = [r["map50_mean"] for r in sum_rows]
    map95 = [r["map50_95_mean"] for r in sum_rows]
    plt.plot(x, map50, marker="o", label="mAP50")
    plt.plot(x, map95, marker="o", label="mAP50-95")
    plt.axhline(base["map50"], color="#3b82f6", linestyle="--", linewidth=1, alpha=0.5, label="baseline mAP50")
    plt.axhline(base["map50_95"], color="#f59e0b", linestyle="--", linewidth=1, alpha=0.5, label="baseline mAP50-95")
    plt.xticks(x, labels)
    plt.xlabel("negative ratio (neg/pos)")
    plt.ylabel("score")
    plt.title("Performance vs Negative Ratio (mean over seeds)")
    plt.legend()
    plt.tight_layout()
    p1 = FIG_DIR / "bgneg_performance_vs_ratio_cv1.png"
    plt.savefig(p1, dpi=180)
    plt.close()

    # FP/FN trend
    plt.figure(figsize=(9, 5))
    fp = [r["fp_bg_to_cls_mean"] for r in sum_rows]
    fn = [r["fn_cls_to_bg_mean"] for r in sum_rows]
    plt.plot(x, fp, marker="o", label="FP (background->class)")
    plt.plot(x, fn, marker="o", label="FN (class->background)")
    plt.xticks(x, labels)
    plt.xlabel("negative ratio (neg/pos)")
    plt.ylabel("count on voc3 val")
    plt.title("Background-related Error Trend vs Negative Ratio")
    plt.legend()
    plt.tight_layout()
    p2 = FIG_DIR / "bgneg_fp_fn_vs_ratio_cv1.png"
    plt.savefig(p2, dpi=180)
    plt.close()
    return p1, p2


def main():
    t0 = time.time()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    DS_ROOT.mkdir(parents=True, exist_ok=True)

    if not VOC3_ROOT.exists():
        raise FileNotFoundError(f"missing baseline dataset: {VOC3_ROOT}")
    if not EVAL_DATA.exists():
        raise FileNotFoundError(f"missing baseline eval yaml: {EVAL_DATA}")

    neg_pool = gather_negative_filenames()
    all_rows = []
    ds_meta = []

    for ratio in NEG_RATIOS:
        ds_dir, meta = build_dataset_with_negatives(ratio, neg_pool)
        ds_meta.append({"dataset": str(ds_dir.resolve()), **meta})
        ds_yaml = ds_dir / "data.yaml"
        for seed in SEEDS:
            all_rows.append(train_one(ratio, seed, ds_yaml))

    # Save raw rows
    raw_csv = OUT_DIR / "bgneg_experiment_raw_cv1.csv"
    keys = list(all_rows[0].keys())
    with open(raw_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(all_rows)

    # Save summary
    summary = summarize(all_rows)
    sum_csv = OUT_DIR / "bgneg_experiment_summary_cv1.csv"
    sum_keys = list(summary[0].keys())
    with open(sum_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=sum_keys)
        w.writeheader()
        w.writerows(summary)

    base = baseline_metrics()
    fig1, fig2 = make_figures(summary, base)

    # pick best ratio by map50_95 mean first, then map50 mean
    best = sorted(summary, key=lambda r: (r["map50_95_mean"], r["map50_mean"]), reverse=True)[0]

    rep = []
    rep.append("# Background Negative Image Experiment Report (CV1)")
    rep.append("")
    rep.append("- Goal: improve background discrimination by adding target-absent images")
    rep.append("- Baseline dataset fixed: voc3 (positive-only)")
    rep.append("- Training recipe fixed (from best large): epochs=16, imgsz=640, batch=12, lr0=0.0009, freeze=5, mosaic=0.6, mixup=0.1, close_mosaic=3")
    rep.append(f"- Ratios tested: {NEG_RATIOS}, seeds: {SEEDS}")
    rep.append("")
    rep.append("## Dataset Construction")
    for m in ds_meta:
        rep.append(
            f"- ratio={m['ratio']:.1f}: train pos={m['train_pos']} neg={m['train_neg_added']}, "
            f"val pos={m['val_pos']} neg={m['val_neg_added']}"
        )
    rep.append("")
    rep.append("## Plots")
    rep.append(f"![performance_vs_ratio]({fig1.as_posix()})")
    rep.append("")
    rep.append(f"![fp_fn_vs_ratio]({fig2.as_posix()})")
    rep.append("")
    rep.append("## Best Ratio by mAP50-95 mean")
    rep.append(
        f"- ratio={best['ratio']:.1f}, mAP50={best['map50_mean']:.4f}±{best['map50_std']:.4f}, "
        f"mAP50-95={best['map50_95_mean']:.4f}±{best['map50_95_std']:.4f}"
    )
    rep.append(
        f"- FP(bg->cls)={best['fp_bg_to_cls_mean']:.2f}, FN(cls->bg)={best['fn_cls_to_bg_mean']:.2f}"
    )
    rep.append("")
    rep.append("## Baseline reference")
    rep.append(
        f"- baseline mAP50={base['map50']:.4f}, mAP50-95={base['map50_95']:.4f}, "
        f"P={base['precision']:.4f}, R={base['recall']:.4f}"
    )

    report_md = OUT_DIR / "bgneg_experiment_report_cv1.md"
    report_md.write_text("\n".join(rep), encoding="utf-8")

    manifest = {
        "raw_csv": str(raw_csv.resolve()),
        "summary_csv": str(sum_csv.resolve()),
        "report_md": str(report_md.resolve()),
        "fig_performance": str(fig1.resolve()),
        "fig_fp_fn": str(fig2.resolve()),
        "best_ratio": best["ratio"],
        "best_map50": best["map50_mean"],
        "best_map50_95": best["map50_95_mean"],
        "elapsed_sec": time.time() - t0,
    }
    man = OUT_DIR / "bgneg_experiment_manifest_cv1.json"
    man.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False))


if __name__ == "__main__":
    main()
