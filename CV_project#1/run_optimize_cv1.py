import csv
import json
import time
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from ultralytics import YOLO


ROOT = Path(__file__).resolve().parent
VOC_YAML = ROOT / "voc3" / "data.yaml"
OUT_DIR = ROOT / "outputs"
RUNS_DIR = ROOT / "opt_runs"
FIG_DIR = OUT_DIR / "figures"
ABLATION_CSV = OUT_DIR / "ablation_train_summary_cv1.csv"


BASELINE_CFG = {
    "epochs": 10,
    "imgsz": 640,
    "batch": 16,
    "lr0": 0.001,
    "freeze": 5,
}


# Keep fixed constraints requested earlier:
# 1) TARGET classes fixed
# 3) data selection logic fixed
# 4) model fixed to yolov8n.pt
CANDIDATES = [
    {"tag": "opt_01", "epochs": 14, "imgsz": 640, "batch": 12, "lr0": 0.0010, "freeze": 5},
    {"tag": "opt_02", "epochs": 16, "imgsz": 640, "batch": 12, "lr0": 0.0010, "freeze": 5},
    {"tag": "opt_03", "epochs": 18, "imgsz": 640, "batch": 12, "lr0": 0.0010, "freeze": 5},
    {"tag": "opt_04", "epochs": 20, "imgsz": 640, "batch": 12, "lr0": 0.0010, "freeze": 5},
    {"tag": "opt_05", "epochs": 16, "imgsz": 640, "batch": 10, "lr0": 0.0010, "freeze": 5},
    {"tag": "opt_06", "epochs": 18, "imgsz": 640, "batch": 10, "lr0": 0.0010, "freeze": 5},
    {"tag": "opt_07", "epochs": 16, "imgsz": 640, "batch": 12, "lr0": 0.0009, "freeze": 5},
    {"tag": "opt_08", "epochs": 16, "imgsz": 640, "batch": 12, "lr0": 0.0011, "freeze": 5},
    {"tag": "opt_09", "epochs": 16, "imgsz": 640, "batch": 12, "lr0": 0.0010, "freeze": 4},
    {"tag": "opt_10", "epochs": 16, "imgsz": 640, "batch": 12, "lr0": 0.0010, "freeze": 6},
    {"tag": "opt_11", "epochs": 16, "imgsz": 704, "batch": 12, "lr0": 0.0010, "freeze": 5},
    {"tag": "opt_12", "epochs": 20, "imgsz": 640, "batch": 10, "lr0": 0.0009, "freeze": 5},
    {"tag": "opt_13", "epochs": 20, "imgsz": 640, "batch": 12, "lr0": 0.0009, "freeze": 5},
    {"tag": "opt_14", "epochs": 24, "imgsz": 640, "batch": 12, "lr0": 0.0010, "freeze": 5},
    {"tag": "opt_15", "epochs": 12, "imgsz": 640, "batch": 12, "lr0": 0.0010, "freeze": 5},
    {"tag": "opt_16", "epochs": 16, "imgsz": 640, "batch": 12, "lr0": 0.0010, "freeze": 5, "cos_lr": True, "lrf": 0.2},
    {"tag": "opt_17", "epochs": 16, "imgsz": 640, "batch": 12, "lr0": 0.0010, "freeze": 5, "close_mosaic": 3},
    {"tag": "opt_18", "epochs": 14, "imgsz": 704, "batch": 12, "lr0": 0.0010, "freeze": 5},
    {"tag": "opt_19", "epochs": 18, "imgsz": 704, "batch": 12, "lr0": 0.0010, "freeze": 5},
]
STRICT_FROM_BASE = True


def load_ablation_rows():
    rows = []
    if not ABLATION_CSV.exists():
        return rows
    with open(ABLATION_CSV, "r", encoding="utf-8") as f:
        rd = csv.DictReader(f)
        for r in rd:
            rows.append(r)
    return rows


def load_baseline_metrics():
    rows = load_ablation_rows()
    if rows:
        base = next((r for r in rows if r["variant"] == "baseline"), None)
        if base:
            return {
                "map50": float(base["map50"]),
                "map50_95": float(base["map50_95"]),
                "precision": float(base["precision"]),
                "recall": float(base["recall"]),
                "source": "ablation_train_summary_cv1.csv",
            }

    # fallback: evaluate baseline directly
    model = YOLO("yolov8n.pt")
    model.train(
        data=str(VOC_YAML),
        project=str(RUNS_DIR),
        name="baseline_ref",
        exist_ok=True,
        verbose=False,
        seed=42,
        deterministic=True,
        workers=0,
        **BASELINE_CFG,
    )
    best = RUNS_DIR / "baseline_ref" / "weights" / "best.pt"
    m = YOLO(str(best)).val(data=str(VOC_YAML), imgsz=BASELINE_CFG["imgsz"], verbose=False, workers=0)
    return {
        "map50": float(m.box.map50),
        "map50_95": float(m.box.map),
        "precision": float(m.box.mp),
        "recall": float(m.box.mr),
        "source": "baseline_ref (recomputed)",
    }


def run_candidate(cfg):
    tag = cfg["tag"]
    exp_dir = RUNS_DIR / tag
    best_pt = exp_dir / "weights" / "best.pt"

    if STRICT_FROM_BASE and exp_dir.exists():
        shutil.rmtree(exp_dir, ignore_errors=True)

    print(f"[train] {tag}: {cfg}")
    model = YOLO("yolov8n.pt")
    train_kwargs = dict(cfg)
    tag = train_kwargs.pop("tag")
    model.train(
        data=str(VOC_YAML),
        epochs=int(train_kwargs.pop("epochs")),
        imgsz=int(train_kwargs.pop("imgsz")),
        batch=int(train_kwargs.pop("batch")),
        lr0=float(train_kwargs.pop("lr0")),
        freeze=int(train_kwargs.pop("freeze")),
        project=str(RUNS_DIR),
        name=tag,
        exist_ok=True,
        verbose=False,
        seed=42,
        deterministic=True,
        workers=0,
        resume=False,
        **train_kwargs,
    )

    m = YOLO(str(best_pt)).val(data=str(VOC_YAML), imgsz=int(cfg["imgsz"]), verbose=False, workers=0)
    row = {
        **cfg,
        "map50": float(m.box.map50),
        "map50_95": float(m.box.map),
        "precision": float(m.box.mp),
        "recall": float(m.box.mr),
        "best_pt": str(best_pt.resolve()),
        "init_model": "yolov8n.pt",
    }
    print(
        f"[result] {tag} mAP50={row['map50']:.4f} mAP50-95={row['map50_95']:.4f} "
        f"P={row['precision']:.4f} R={row['recall']:.4f}"
    )
    return row


def save_csv(rows, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    keyset = set()
    for r in rows:
        keyset.update(r.keys())
    preferred = [
        "tag",
        "epochs",
        "imgsz",
        "batch",
        "lr0",
        "freeze",
        "cos_lr",
        "lrf",
        "close_mosaic",
        "map50",
        "map50_95",
        "precision",
        "recall",
        "best_pt",
    ]
    keys = [k for k in preferred if k in keyset] + sorted([k for k in keyset if k not in preferred])
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def make_figures(rows, baseline):
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    sorted_rows = sorted(rows, key=lambda x: (x["map50"], x["map50_95"]), reverse=True)
    top = sorted_rows[:10]
    labels = [r["tag"] for r in top]
    vals = [r["map50"] for r in top]

    plt.figure(figsize=(11, 5))
    plt.bar(range(len(top)), vals, color="#2563eb")
    plt.axhline(baseline["map50"], color="#ef4444", linestyle="--", linewidth=1.2, label="baseline mAP50")
    plt.xticks(range(len(top)), labels, rotation=35, ha="right")
    plt.ylabel("mAP@50")
    plt.title("Top candidate leaderboard")
    plt.legend()
    plt.tight_layout()
    p1 = FIG_DIR / "opt_leaderboard_map50_cv1.png"
    plt.savefig(p1, dpi=160)
    plt.close()

    best = sorted_rows[0]
    categories = ["mAP50", "mAP50-95", "Precision", "Recall"]
    bvals = [baseline["map50"], baseline["map50_95"], baseline["precision"], baseline["recall"]]
    ovals = [best["map50"], best["map50_95"], best["precision"], best["recall"]]

    x = np.arange(len(categories))
    w = 0.36
    plt.figure(figsize=(8.5, 5))
    plt.bar(x - w / 2, bvals, width=w, color="#94a3b8", label="baseline")
    plt.bar(x + w / 2, ovals, width=w, color="#16a34a", label="optimized")
    plt.xticks(x, categories)
    plt.ylim(0, 1)
    plt.title(f"Baseline vs Optimized ({best['tag']})")
    plt.legend()
    plt.tight_layout()
    p2 = FIG_DIR / "baseline_vs_optimized_cv1.png"
    plt.savefig(p2, dpi=160)
    plt.close()

    return p1, p2


def write_report(rows, baseline, leaderboard_fig, compare_fig, out_path):
    rows_sorted = sorted(rows, key=lambda x: (x["map50"], x["map50_95"]), reverse=True)
    best = rows_sorted[0]
    prev_best = None
    ablation_rows = load_ablation_rows()
    if ablation_rows:
        prev_best = max(ablation_rows, key=lambda x: float(x["map50"]))

    d_map50 = best["map50"] - baseline["map50"]
    d_map95 = best["map50_95"] - baseline["map50_95"]
    d_p = best["precision"] - baseline["precision"]
    d_r = best["recall"] - baseline["recall"]

    lines = []
    lines.append("# CV_project#1 Performance Optimization Report")
    lines.append("")
    lines.append("## Goal")
    lines.append("- Maximize fine-tuning performance while keeping class set, dataset logic, and model fixed.")
    lines.append("")
    lines.append("## Baseline (from previous run)")
    lines.append(
        f"- mAP@50={baseline['map50']:.4f}, mAP@50:95={baseline['map50_95']:.4f}, "
        f"P={baseline['precision']:.4f}, R={baseline['recall']:.4f} ({baseline['source']})"
    )
    if prev_best:
        lines.append(
            f"- Previous best (ablation): {prev_best['variant']} -> mAP@50={float(prev_best['map50']):.4f}, "
            f"mAP@50:95={float(prev_best['map50_95']):.4f}"
        )
    lines.append("")
    lines.append("## Optimized Best")
    lines.append(f"- Best tag: {best['tag']}")
    lines.append(
        f"- Config: epochs={best['epochs']}, imgsz={best['imgsz']}, batch={best['batch']}, "
        f"lr0={best['lr0']}, freeze={best['freeze']}"
    )
    lines.append(
        f"- Metrics: mAP@50={best['map50']:.4f}, mAP@50:95={best['map50_95']:.4f}, "
        f"P={best['precision']:.4f}, R={best['recall']:.4f}"
    )
    lines.append("- Delta vs baseline:")
    lines.append(f"  map50: {d_map50:+.4f}")
    lines.append(f"  map50_95: {d_map95:+.4f}")
    lines.append(f"  precision: {d_p:+.4f}")
    lines.append(f"  recall: {d_r:+.4f}")
    lines.append("")
    lines.append("## Plots")
    lines.append(f"![leaderboard]({leaderboard_fig.as_posix()})")
    lines.append("")
    lines.append(f"![baseline_vs_optimized]({compare_fig.as_posix()})")
    lines.append("")
    lines.append("## Top 5 Candidates")
    for i, r in enumerate(rows_sorted[:5], start=1):
        lines.append(
            f"- {i}. {r['tag']}: mAP50={r['map50']:.4f}, mAP50-95={r['map50_95']:.4f}, "
            f"P={r['precision']:.4f}, R={r['recall']:.4f}, "
            f"cfg=(e{r['epochs']}, s{r['imgsz']}, b{r['batch']}, lr{r['lr0']}, f{r['freeze']})"
        )
    lines.append("")
    lines.append("## Best Weights")
    lines.append(f"- {best['best_pt']}")

    out_path.write_text("\n".join(lines), encoding="utf-8")


def main():
    t0 = time.time()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    if not VOC_YAML.exists():
        raise FileNotFoundError(f"Dataset yaml not found: {VOC_YAML}")

    baseline = load_baseline_metrics()
    rows = []
    for cfg in CANDIDATES:
        try:
            rows.append(run_candidate(cfg))
        except Exception as e:
            print(f"[error] {cfg['tag']} failed: {repr(e)}")

    if not rows:
        raise RuntimeError("No candidate run completed.")

    summary_csv = OUT_DIR / "optimization_summary_cv1.csv"
    save_csv(rows, summary_csv)

    p1, p2 = make_figures(rows, baseline)
    report_path = OUT_DIR / "optimization_report_cv1.md"
    write_report(rows, baseline, p1, p2, report_path)

    best = sorted(rows, key=lambda x: (x["map50"], x["map50_95"]), reverse=True)[0]
    manifest = {
        "summary_csv": str(summary_csv.resolve()),
        "report_md": str(report_path.resolve()),
        "leaderboard_fig": str(p1.resolve()),
        "compare_fig": str(p2.resolve()),
        "best_tag": best["tag"],
        "best_map50": best["map50"],
        "best_map50_95": best["map50_95"],
        "best_pt": best["best_pt"],
        "elapsed_sec": time.time() - t0,
    }
    (OUT_DIR / "optimization_manifest_cv1.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print("[done]", json.dumps(manifest, ensure_ascii=False))


if __name__ == "__main__":
    main()
