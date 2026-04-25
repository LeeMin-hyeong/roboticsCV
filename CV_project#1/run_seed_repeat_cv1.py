import csv
import json
import math
import shutil
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from ultralytics import YOLO


ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT / "outputs"
FIG_DIR = OUT_DIR / "figures"
RUNS_DIR = ROOT / "repeat_runs"
VOC_YAML = ROOT / "voc3" / "data.yaml"

SEEDS = [101, 202, 303, 404, 505]

BASELINE_CFG = {
    "name": "baseline",
    "epochs": 10,
    "imgsz": 640,
    "batch": 16,
    "lr0": 0.001,
    "freeze": 5,
}

OPT_CFG = {
    "name": "optimized",
    "epochs": 16,
    "imgsz": 704,
    "batch": 12,
    "lr0": 0.001,
    "freeze": 5,
}


def train_eval(tag: str, seed: int, cfg: dict) -> dict:
    exp_name = f"{tag}_seed{seed}"
    exp_dir = RUNS_DIR / exp_name
    if exp_dir.exists():
        shutil.rmtree(exp_dir, ignore_errors=True)

    model = YOLO("yolov8n.pt")
    model.train(
        data=str(VOC_YAML),
        epochs=int(cfg["epochs"]),
        imgsz=int(cfg["imgsz"]),
        batch=int(cfg["batch"]),
        lr0=float(cfg["lr0"]),
        freeze=int(cfg["freeze"]),
        project=str(RUNS_DIR),
        name=exp_name,
        exist_ok=True,
        verbose=False,
        workers=0,
        resume=False,
        deterministic=True,
        seed=int(seed),
    )

    best_pt = exp_dir / "weights" / "best.pt"
    m = YOLO(str(best_pt)).val(
        data=str(VOC_YAML),
        imgsz=int(cfg["imgsz"]),
        verbose=False,
        workers=0,
    )

    row = {
        "group": tag,
        "seed": seed,
        "epochs": cfg["epochs"],
        "imgsz": cfg["imgsz"],
        "batch": cfg["batch"],
        "lr0": cfg["lr0"],
        "freeze": cfg["freeze"],
        "map50": float(m.box.map50),
        "map50_95": float(m.box.map),
        "precision": float(m.box.mp),
        "recall": float(m.box.mr),
        "best_pt": str(best_pt.resolve()),
        "init_model": "yolov8n.pt",
    }
    print(
        f"[{tag}] seed={seed} map50={row['map50']:.4f} "
        f"map50_95={row['map50_95']:.4f} P={row['precision']:.4f} R={row['recall']:.4f}"
    )
    return row


def save_csv(rows: list, path: Path):
    keys = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def run_level_ols(rows: list, metric: str):
    y = np.array([float(r[metric]) for r in rows], dtype=float)
    g = np.array([1.0 if r["group"] == "optimized" else 0.0 for r in rows], dtype=float)
    X = np.column_stack([np.ones(len(y)), g])
    beta = np.linalg.inv(X.T @ X) @ (X.T @ y)
    resid = y - X @ beta
    n, p = X.shape
    sigma2 = float((resid @ resid) / (n - p))
    vcov = sigma2 * np.linalg.inv(X.T @ X)
    se = np.sqrt(np.diag(vcov))
    tval = float(beta[1] / se[1])
    pval = float(2 * (1 - stats.t.cdf(abs(tval), df=n - p)))
    return {
        "coef": float(beta[1]),
        "t": tval,
        "p": pval,
        "n": int(n),
    }


def paired_by_seed(rows: list, metric: str):
    base = {int(r["seed"]): float(r[metric]) for r in rows if r["group"] == "baseline"}
    opt = {int(r["seed"]): float(r[metric]) for r in rows if r["group"] == "optimized"}
    seeds = sorted(set(base.keys()) & set(opt.keys()))
    d = np.array([opt[s] - base[s] for s in seeds], dtype=float)
    n = len(d)
    mean_d = float(np.mean(d))
    sd = float(np.std(d, ddof=1)) if n > 1 else 0.0
    se = sd / math.sqrt(n) if n > 0 else float("nan")
    tval = float(mean_d / se) if se > 0 else float("inf")
    pval = float(2 * (1 - stats.t.cdf(abs(tval), df=n - 1))) if n > 1 else float("nan")
    return {
        "seeds": seeds,
        "mean_delta": mean_d,
        "t": tval,
        "p": pval,
        "n": n,
    }


def plot_metric(rows: list, metric: str, out_path: Path):
    base = [float(r[metric]) for r in rows if r["group"] == "baseline"]
    opt = [float(r[metric]) for r in rows if r["group"] == "optimized"]
    seeds = [int(r["seed"]) for r in rows if r["group"] == "baseline"]
    seeds.sort()
    bmap = {int(r["seed"]): float(r[metric]) for r in rows if r["group"] == "baseline"}
    omap = {int(r["seed"]): float(r[metric]) for r in rows if r["group"] == "optimized"}

    xs = np.arange(len(seeds))
    plt.figure(figsize=(8, 4.8))
    plt.plot(xs, [bmap[s] for s in seeds], marker="o", label="baseline", color="#64748b")
    plt.plot(xs, [omap[s] for s in seeds], marker="o", label="optimized", color="#16a34a")
    for i, s in enumerate(seeds):
        plt.plot([i, i], [bmap[s], omap[s]], color="#cbd5e1", linewidth=1)
    plt.xticks(xs, [str(s) for s in seeds])
    plt.xlabel("seed")
    plt.ylabel(metric)
    plt.title(f"{metric} by seed (paired)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=170)
    plt.close()


def main():
    t0 = time.time()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    RUNS_DIR.mkdir(parents=True, exist_ok=True)

    if not VOC_YAML.exists():
        raise FileNotFoundError(f"Dataset yaml missing: {VOC_YAML}")

    rows = []
    for s in SEEDS:
        rows.append(train_eval("baseline", s, BASELINE_CFG))
    for s in SEEDS:
        rows.append(train_eval("optimized", s, OPT_CFG))

    csv_path = OUT_DIR / "seed_repeat_summary_cv1.csv"
    save_csv(rows, csv_path)

    ols_map50 = run_level_ols(rows, "map50")
    ols_map95 = run_level_ols(rows, "map50_95")
    pair_map50 = paired_by_seed(rows, "map50")
    pair_map95 = paired_by_seed(rows, "map50_95")

    p1 = FIG_DIR / "seed_repeat_map50_cv1.png"
    p2 = FIG_DIR / "seed_repeat_map50_95_cv1.png"
    plot_metric(rows, "map50", p1)
    plot_metric(rows, "map50_95", p2)

    base_map50 = [float(r["map50"]) for r in rows if r["group"] == "baseline"]
    opt_map50 = [float(r["map50"]) for r in rows if r["group"] == "optimized"]
    base_map95 = [float(r["map50_95"]) for r in rows if r["group"] == "baseline"]
    opt_map95 = [float(r["map50_95"]) for r in rows if r["group"] == "optimized"]

    report_lines = []
    report_lines.append("# Seed Repeat Significance Report (CV1)")
    report_lines.append("")
    report_lines.append("## Setup")
    report_lines.append(f"- seeds: {SEEDS}")
    report_lines.append("- baseline config: epochs=10, imgsz=640, batch=16, lr0=0.001, freeze=5")
    report_lines.append("- optimized config: epochs=16, imgsz=704, batch=12, lr0=0.001, freeze=5")
    report_lines.append("- all runs initialized from yolov8n.pt (no resume)")
    report_lines.append("")
    report_lines.append("## Mean Metrics")
    report_lines.append(
        f"- map50: baseline={np.mean(base_map50):.4f}, optimized={np.mean(opt_map50):.4f}, "
        f"delta={np.mean(opt_map50)-np.mean(base_map50):+.4f}"
    )
    report_lines.append(
        f"- map50_95: baseline={np.mean(base_map95):.4f}, optimized={np.mean(opt_map95):.4f}, "
        f"delta={np.mean(opt_map95)-np.mean(base_map95):+.4f}"
    )
    report_lines.append("")
    report_lines.append("## OLS p-values (run-level, same as previous style)")
    report_lines.append(
        f"- map50: coef={ols_map50['coef']:+.4f}, p={ols_map50['p']:.6g}, n={ols_map50['n']}"
    )
    report_lines.append(
        f"- map50_95: coef={ols_map95['coef']:+.4f}, p={ols_map95['p']:.6g}, n={ols_map95['n']}"
    )
    report_lines.append("")
    report_lines.append("## Paired-by-seed p-values")
    report_lines.append(
        f"- map50: mean_delta={pair_map50['mean_delta']:+.4f}, p={pair_map50['p']:.6g}, n={pair_map50['n']}"
    )
    report_lines.append(
        f"- map50_95: mean_delta={pair_map95['mean_delta']:+.4f}, p={pair_map95['p']:.6g}, n={pair_map95['n']}"
    )
    report_lines.append("")
    report_lines.append("## Plots")
    report_lines.append(f"![map50 paired]({p1.as_posix()})")
    report_lines.append("")
    report_lines.append(f"![map50_95 paired]({p2.as_posix()})")

    report_path = OUT_DIR / "seed_repeat_report_cv1.md"
    report_path.write_text("\n".join(report_lines), encoding="utf-8")

    payload = {
        "summary_csv": str(csv_path.resolve()),
        "report_md": str(report_path.resolve()),
        "fig_map50": str(p1.resolve()),
        "fig_map50_95": str(p2.resolve()),
        "ols_map50": ols_map50,
        "ols_map50_95": ols_map95,
        "paired_map50": pair_map50,
        "paired_map50_95": pair_map95,
        "elapsed_sec": time.time() - t0,
    }
    (OUT_DIR / "seed_repeat_manifest_cv1.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(payload, ensure_ascii=False))


if __name__ == "__main__":
    main()
