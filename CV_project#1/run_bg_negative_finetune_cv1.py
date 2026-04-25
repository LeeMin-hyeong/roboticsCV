import csv
import json
import shutil
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from ultralytics import YOLO


ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT / "outputs"
FIG_DIR = OUT_DIR / "figures"
RUNS_DIR = ROOT / "bg_neg_finetune_runs"

DS_YAML = ROOT / "datasets_bgneg_low" / "voc3_neg_r10" / "data.yaml"  # ratio=0.1 fixed
EVAL_YAML = ROOT / "voc3" / "data.yaml"  # fixed baseline eval
BASELINE_CSV = OUT_DIR / "ablation_train_summary_cv1.csv"

SEEDS = [111, 222, 333]

# Fine-tuning around previous setting: reduce mosaic/mixup
CONFIGS = [
    {"tag": "ft_ref_m06_x01", "epochs": 16, "imgsz": 640, "batch": 12, "lr0": 0.0009, "freeze": 5, "mosaic": 0.6, "mixup": 0.10, "close_mosaic": 3},
    {"tag": "ft_m05_x05", "epochs": 16, "imgsz": 640, "batch": 12, "lr0": 0.0009, "freeze": 5, "mosaic": 0.5, "mixup": 0.05, "close_mosaic": 3},
    {"tag": "ft_m04_x05", "epochs": 16, "imgsz": 640, "batch": 12, "lr0": 0.0009, "freeze": 5, "mosaic": 0.4, "mixup": 0.05, "close_mosaic": 3},
    {"tag": "ft_m03_x00", "epochs": 16, "imgsz": 640, "batch": 12, "lr0": 0.0009, "freeze": 5, "mosaic": 0.3, "mixup": 0.00, "close_mosaic": 2},
    {"tag": "ft_m05_x00", "epochs": 16, "imgsz": 640, "batch": 12, "lr0": 0.0009, "freeze": 5, "mosaic": 0.5, "mixup": 0.00, "close_mosaic": 2},
    {"tag": "ft_m04_x00_lr1e3", "epochs": 16, "imgsz": 640, "batch": 12, "lr0": 0.0010, "freeze": 5, "mosaic": 0.4, "mixup": 0.00, "close_mosaic": 2},
]


def baseline():
    rows = list(csv.DictReader(open(BASELINE_CSV, encoding="utf-8")))
    b = next(r for r in rows if r["variant"] == "baseline")
    return {
        "map50": float(b["map50"]),
        "map50_95": float(b["map50_95"]),
        "precision": float(b["precision"]),
        "recall": float(b["recall"]),
    }


def train_eval(cfg: dict, seed: int):
    tag = cfg["tag"]
    run_name = f"{tag}_s{seed}"
    exp = RUNS_DIR / run_name
    if exp.exists():
        shutil.rmtree(exp, ignore_errors=True)

    kw = dict(cfg)
    tag = kw.pop("tag")
    epochs = int(kw.pop("epochs"))
    imgsz = int(kw.pop("imgsz"))
    batch = int(kw.pop("batch"))
    lr0 = float(kw.pop("lr0"))
    freeze = int(kw.pop("freeze"))

    model = YOLO("yolov8n.pt")
    model.train(
        data=str(DS_YAML),
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        lr0=lr0,
        freeze=freeze,
        project=str(RUNS_DIR),
        name=run_name,
        exist_ok=True,
        verbose=False,
        workers=0,
        deterministic=True,
        seed=seed,
        resume=False,
        **kw,
    )

    best = exp / "weights" / "best.pt"
    m = YOLO(str(best)).val(data=str(EVAL_YAML), imgsz=imgsz, verbose=False, workers=0)
    row = {
        "tag": cfg["tag"],
        "seed": seed,
        "epochs": epochs,
        "imgsz": imgsz,
        "batch": batch,
        "lr0": lr0,
        "freeze": freeze,
        "mosaic": float(cfg["mosaic"]),
        "mixup": float(cfg["mixup"]),
        "close_mosaic": int(cfg["close_mosaic"]),
        "map50": float(m.box.map50),
        "map50_95": float(m.box.map),
        "precision": float(m.box.mp),
        "recall": float(m.box.mr),
        "best_pt": str(best.resolve()),
        "train_data": str(DS_YAML.resolve()),
        "eval_data": str(EVAL_YAML.resolve()),
    }
    print(
        f"[{cfg['tag']} seed={seed}] map50={row['map50']:.4f} map50_95={row['map50_95']:.4f} "
        f"P={row['precision']:.4f} R={row['recall']:.4f}"
    )
    return row


def summarize(rows):
    out = []
    tags = sorted({r["tag"] for r in rows})
    for t in tags:
        rs = [r for r in rows if r["tag"] == t]
        item = {
            "tag": t,
            "n": len(rs),
            "mosaic": rs[0]["mosaic"],
            "mixup": rs[0]["mixup"],
            "close_mosaic": rs[0]["close_mosaic"],
            "lr0": rs[0]["lr0"],
        }
        for k in ["map50", "map50_95", "precision", "recall"]:
            vals = np.array([float(r[k]) for r in rs], dtype=float)
            item[f"{k}_mean"] = float(np.mean(vals))
            item[f"{k}_std"] = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
        out.append(item)
    return out


def make_figures(summary_rows, base):
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    rs = sorted(summary_rows, key=lambda r: (r["map50_95_mean"], r["map50_mean"]), reverse=True)
    tags = [r["tag"] for r in rs]
    x = np.arange(len(tags))

    plt.figure(figsize=(11, 5))
    plt.bar(x - 0.18, [r["map50_mean"] for r in rs], width=0.36, label="mAP50")
    plt.bar(x + 0.18, [r["map50_95_mean"] for r in rs], width=0.36, label="mAP50-95")
    plt.axhline(base["map50"], color="#3b82f6", linestyle="--", linewidth=1, alpha=0.6, label="baseline mAP50")
    plt.axhline(base["map50_95"], color="#f59e0b", linestyle="--", linewidth=1, alpha=0.6, label="baseline mAP50-95")
    plt.xticks(x, tags, rotation=25, ha="right")
    plt.ylabel("score (mean over seeds)")
    plt.title("Negative Ratio 0.1: Fine-tune Augmentation")
    plt.legend()
    plt.tight_layout()
    p1 = FIG_DIR / "bgneg_finetune_performance_cv1.png"
    plt.savefig(p1, dpi=180)
    plt.close()

    best = rs[0]
    cats = ["mAP50", "mAP50-95", "Precision", "Recall"]
    bvals = [base["map50"], base["map50_95"], base["precision"], base["recall"]]
    ovals = [best["map50_mean"], best["map50_95_mean"], best["precision_mean"], best["recall_mean"]]
    xx = np.arange(len(cats))
    plt.figure(figsize=(8.5, 5))
    plt.bar(xx - 0.18, bvals, width=0.36, label="baseline", color="#94a3b8")
    plt.bar(xx + 0.18, ovals, width=0.36, label=f"best {best['tag']}", color="#16a34a")
    plt.xticks(xx, cats)
    plt.ylim(0, 1)
    plt.title("Baseline vs Best Fine-tuned Negative-Ratio Model")
    plt.legend()
    plt.tight_layout()
    p2 = FIG_DIR / "bgneg_finetune_baseline_compare_cv1.png"
    plt.savefig(p2, dpi=180)
    plt.close()
    return p1, p2


def main():
    t0 = time.time()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    if not DS_YAML.exists():
        raise FileNotFoundError(f"dataset yaml missing: {DS_YAML}")

    rows = []
    for cfg in CONFIGS:
        for s in SEEDS:
            rows.append(train_eval(cfg, s))

    raw_csv = OUT_DIR / "bgneg_finetune_raw_cv1.csv"
    with open(raw_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    summary_rows = summarize(rows)
    summary_csv = OUT_DIR / "bgneg_finetune_summary_cv1.csv"
    with open(summary_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        w.writeheader()
        w.writerows(summary_rows)

    base = baseline()
    fig1, fig2 = make_figures(summary_rows, base)
    best = sorted(summary_rows, key=lambda r: (r["map50_95_mean"], r["map50_mean"]), reverse=True)[0]

    rep = []
    rep.append("# Background-Negative Fine-tuning Report (ratio=0.1)")
    rep.append("")
    rep.append("- dataset: voc3 + negatives ratio 0.1")
    rep.append("- seeds: [111,222,333]")
    rep.append("- goal: improve background discrimination while preserving mAP50-95")
    rep.append("")
    rep.append(f"![performance]({fig1.as_posix()})")
    rep.append("")
    rep.append(f"![baseline_compare]({fig2.as_posix()})")
    rep.append("")
    rep.append(
        f"- best: {best['tag']} (mosaic={best['mosaic']}, mixup={best['mixup']}, close_mosaic={best['close_mosaic']}, lr0={best['lr0']})"
    )
    rep.append(
        f"- best metrics(mean): mAP50={best['map50_mean']:.4f}±{best['map50_std']:.4f}, "
        f"mAP50-95={best['map50_95_mean']:.4f}±{best['map50_95_std']:.4f}, "
        f"P={best['precision_mean']:.4f}, R={best['recall_mean']:.4f}"
    )
    rep.append(
        f"- baseline: mAP50={base['map50']:.4f}, mAP50-95={base['map50_95']:.4f}, "
        f"P={base['precision']:.4f}, R={base['recall']:.4f}"
    )

    report_md = OUT_DIR / "bgneg_finetune_report_cv1.md"
    report_md.write_text("\n".join(rep), encoding="utf-8")

    manifest = {
        "raw_csv": str(raw_csv.resolve()),
        "summary_csv": str(summary_csv.resolve()),
        "report_md": str(report_md.resolve()),
        "fig_performance": str(fig1.resolve()),
        "fig_compare": str(fig2.resolve()),
        "best_tag": best["tag"],
        "best_map50": best["map50_mean"],
        "best_map50_95": best["map50_95_mean"],
        "elapsed_sec": time.time() - t0,
    }
    man = OUT_DIR / "bgneg_finetune_manifest_cv1.json"
    man.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False))


if __name__ == "__main__":
    main()
