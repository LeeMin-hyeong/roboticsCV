import csv
import json
import math
import shutil
import time
from pathlib import Path

import numpy as np
from scipy import stats
from ultralytics import YOLO


ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT / "outputs"
RUNS_DIR = ROOT / "repeat_runs_bgneg_finetune"

BASE_TRAIN = ROOT / "voc3" / "data.yaml"
OPT_TRAIN = ROOT / "datasets_bgneg_low" / "voc3_neg_r10" / "data.yaml"
EVAL_DATA = ROOT / "voc3" / "data.yaml"

SEEDS = [101, 202, 303, 404, 505, 606, 707, 808, 909, 1001]

BASELINE_CFG = {
    "epochs": 10,
    "imgsz": 640,
    "batch": 16,
    "lr0": 0.001,
    "freeze": 5,
}

# from bg-negative finetune best: ft_m05_x00
OPT_CFG = {
    "epochs": 16,
    "imgsz": 640,
    "batch": 12,
    "lr0": 0.0009,
    "freeze": 5,
    "mosaic": 0.5,
    "mixup": 0.0,
    "close_mosaic": 2,
}


def train_eval(group: str, seed: int, cfg: dict, train_data: Path, eval_data: Path):
    run_name = f"{group}_seed{seed}"
    exp = RUNS_DIR / run_name
    if exp.exists():
        shutil.rmtree(exp, ignore_errors=True)

    kw = dict(cfg)
    epochs = int(kw.pop("epochs"))
    imgsz = int(kw.pop("imgsz"))
    batch = int(kw.pop("batch"))
    lr0 = float(kw.pop("lr0"))
    freeze = int(kw.pop("freeze"))

    model = YOLO("yolov8n.pt")
    model.train(
        data=str(train_data),
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
    m = YOLO(str(best)).val(data=str(eval_data), imgsz=imgsz, verbose=False, workers=0)

    row = {
        "group": group,
        "seed": seed,
        "train_data": str(train_data.resolve()),
        "eval_data": str(eval_data.resolve()),
        "epochs": epochs,
        "imgsz": imgsz,
        "batch": batch,
        "lr0": lr0,
        "freeze": freeze,
        "map50": float(m.box.map50),
        "map50_95": float(m.box.map),
        "precision": float(m.box.mp),
        "recall": float(m.box.mr),
        "best_pt": str(best.resolve()),
        "init_model": "yolov8n.pt",
    }
    for k, v in kw.items():
        row[k] = v
    print(
        f"[{group}] seed={seed} map50={row['map50']:.4f} map50_95={row['map50_95']:.4f} "
        f"P={row['precision']:.4f} R={row['recall']:.4f}"
    )
    return row


def save_csv(rows, path: Path):
    keyset = set()
    for r in rows:
        keyset.update(r.keys())
    pref = [
        "group", "seed", "train_data", "eval_data",
        "epochs", "imgsz", "batch", "lr0", "freeze",
        "mosaic", "mixup", "close_mosaic",
        "map50", "map50_95", "precision", "recall",
        "best_pt", "init_model",
    ]
    cols = [k for k in pref if k in keyset] + sorted([k for k in keyset if k not in pref])
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)


def ols(rows, metric: str):
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
    return {"coef": float(beta[1]), "t": tval, "p": pval, "n": int(n)}


def paired(rows, metric: str):
    b = {int(r["seed"]): float(r[metric]) for r in rows if r["group"] == "baseline"}
    o = {int(r["seed"]): float(r[metric]) for r in rows if r["group"] == "optimized"}
    seeds = sorted(set(b.keys()) & set(o.keys()))
    d = np.array([o[s] - b[s] for s in seeds], dtype=float)
    n = len(d)
    md = float(np.mean(d))
    sd = float(np.std(d, ddof=1)) if n > 1 else 0.0
    se = sd / math.sqrt(n) if n > 0 else float("nan")
    tval = float(md / se) if se > 0 else float("inf")
    pval = float(2 * (1 - stats.t.cdf(abs(tval), df=n - 1))) if n > 1 else float("nan")
    return {"mean_delta": md, "t": tval, "p": pval, "n": n, "seeds": seeds}


def main():
    t0 = time.time()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    if not BASE_TRAIN.exists():
        raise FileNotFoundError(f"missing baseline train yaml: {BASE_TRAIN}")
    if not OPT_TRAIN.exists():
        raise FileNotFoundError(f"missing optimized train yaml: {OPT_TRAIN}")
    if not EVAL_DATA.exists():
        raise FileNotFoundError(f"missing eval yaml: {EVAL_DATA}")

    rows = []
    for s in SEEDS:
        rows.append(train_eval("baseline", s, BASELINE_CFG, BASE_TRAIN, EVAL_DATA))
    for s in SEEDS:
        rows.append(train_eval("optimized", s, OPT_CFG, OPT_TRAIN, EVAL_DATA))

    csv_path = OUT_DIR / "seed_repeat_bgneg_finetune_summary_cv1.csv"
    save_csv(rows, csv_path)

    ols50 = ols(rows, "map50")
    ols95 = ols(rows, "map50_95")
    p50 = paired(rows, "map50")
    p95 = paired(rows, "map50_95")

    b50 = float(np.mean([float(r["map50"]) for r in rows if r["group"] == "baseline"]))
    o50 = float(np.mean([float(r["map50"]) for r in rows if r["group"] == "optimized"]))
    b95 = float(np.mean([float(r["map50_95"]) for r in rows if r["group"] == "baseline"]))
    o95 = float(np.mean([float(r["map50_95"]) for r in rows if r["group"] == "optimized"]))

    rep = []
    rep.append("# Significance Report: bg-negative finetune (CV1)")
    rep.append("")
    rep.append("- optimized config: mosaic=0.5, mixup=0.0, close_mosaic=2, epochs=16, imgsz=640, batch=12, lr0=0.0009, freeze=5")
    rep.append(f"- seeds: {SEEDS}")
    rep.append(f"- baseline train: {BASE_TRAIN.name}, optimized train: {OPT_TRAIN.name}, eval: {EVAL_DATA.name}")
    rep.append(f"- mean map50: baseline={b50:.4f}, optimized={o50:.4f}, delta={o50-b50:+.4f}")
    rep.append(f"- mean map50_95: baseline={b95:.4f}, optimized={o95:.4f}, delta={o95-b95:+.4f}")
    rep.append("")
    rep.append("## OLS")
    rep.append(f"- map50: coef={ols50['coef']:+.4f}, p={ols50['p']:.6g}, n={ols50['n']}")
    rep.append(f"- map50_95: coef={ols95['coef']:+.4f}, p={ols95['p']:.6g}, n={ols95['n']}")
    rep.append("## Paired-by-seed")
    rep.append(f"- map50: mean_delta={p50['mean_delta']:+.4f}, p={p50['p']:.6g}, n={p50['n']}")
    rep.append(f"- map50_95: mean_delta={p95['mean_delta']:+.4f}, p={p95['p']:.6g}, n={p95['n']}")
    report_md = OUT_DIR / "seed_repeat_bgneg_finetune_report_cv1.md"
    report_md.write_text("\n".join(rep), encoding="utf-8")

    manifest = {
        "summary_csv": str(csv_path.resolve()),
        "report_md": str(report_md.resolve()),
        "ols_map50": ols50,
        "ols_map50_95": ols95,
        "paired_map50": p50,
        "paired_map50_95": p95,
        "base_mean_map50": b50,
        "opt_mean_map50": o50,
        "base_mean_map50_95": b95,
        "opt_mean_map50_95": o95,
        "elapsed_sec": time.time() - t0,
    }
    man = OUT_DIR / "seed_repeat_bgneg_finetune_manifest_cv1.json"
    man.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False))


if __name__ == "__main__":
    main()
