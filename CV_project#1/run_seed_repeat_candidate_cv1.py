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
RUNS_DIR = ROOT / "repeat_runs_candidate"

BASE_DATA = ROOT / "voc3" / "data.yaml"
OPT_TRAIN_DATA = ROOT / "voc3" / "data.yaml"
EVAL_DATA = ROOT / "voc3" / "data.yaml"

SEEDS = [111, 222, 333, 444, 555]

BASELINE_CFG = {
    "epochs": 10,
    "imgsz": 640,
    "batch": 16,
    "lr0": 0.001,
    "freeze": 5,
}

# current best from augmentation search
OPT_CFG = {
    "epochs": 16,
    "imgsz": 640,
    "batch": 12,
    "lr0": 0.0009,
    "freeze": 5,
    "mosaic": 0.6,
    "mixup": 0.1,
    "close_mosaic": 3,
}


def train_eval(tag: str, seed: int, cfg: dict, train_data: Path, eval_data: Path) -> dict:
    name = f"{tag}_seed{seed}"
    exp_dir = RUNS_DIR / name
    if exp_dir.exists():
        shutil.rmtree(exp_dir, ignore_errors=True)

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
        name=name,
        exist_ok=True,
        verbose=False,
        workers=0,
        deterministic=True,
        seed=seed,
        resume=False,
        **kw,
    )

    best_pt = exp_dir / "weights" / "best.pt"
    m = YOLO(str(best_pt)).val(data=str(eval_data), imgsz=imgsz, verbose=False, workers=0)
    row = {
        "group": tag,
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
        "best_pt": str(best_pt.resolve()),
        "init_model": "yolov8n.pt",
    }
    for k, v in kw.items():
        row[k] = v
    print(
        f"[{tag}] seed={seed} map50={row['map50']:.4f} map50_95={row['map50_95']:.4f} "
        f"P={row['precision']:.4f} R={row['recall']:.4f}"
    )
    return row


def save_csv(rows, path: Path):
    keyset = set()
    for r in rows:
        keyset.update(r.keys())
    preferred = [
        "group", "seed", "train_data", "eval_data",
        "epochs", "imgsz", "batch", "lr0", "freeze",
        "mosaic", "mixup", "copy_paste", "close_mosaic",
        "hsv_h", "hsv_s", "hsv_v", "translate", "scale", "degrees", "shear",
        "cos_lr", "lrf",
        "map50", "map50_95", "precision", "recall",
        "best_pt", "init_model",
    ]
    cols = [k for k in preferred if k in keyset] + sorted([k for k in keyset if k not in preferred])
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

    if not BASE_DATA.exists():
        raise FileNotFoundError(f"missing dataset: {BASE_DATA}")
    if not OPT_TRAIN_DATA.exists():
        raise FileNotFoundError(f"missing dataset: {OPT_TRAIN_DATA}")
    if not EVAL_DATA.exists():
        raise FileNotFoundError(f"missing dataset: {EVAL_DATA}")

    rows = []
    for s in SEEDS:
        rows.append(train_eval("baseline", s, BASELINE_CFG, BASE_DATA, EVAL_DATA))
    for s in SEEDS:
        rows.append(train_eval("optimized", s, OPT_CFG, OPT_TRAIN_DATA, EVAL_DATA))

    csv_path = OUT_DIR / "seed_repeat_candidate_summary_cv1.csv"
    save_csv(rows, csv_path)

    ols50 = ols(rows, "map50")
    ols95 = ols(rows, "map50_95")
    pair50 = paired(rows, "map50")
    pair95 = paired(rows, "map50_95")

    base50 = np.mean([float(r["map50"]) for r in rows if r["group"] == "baseline"])
    opt50 = np.mean([float(r["map50"]) for r in rows if r["group"] == "optimized"])
    base95 = np.mean([float(r["map50_95"]) for r in rows if r["group"] == "baseline"])
    opt95 = np.mean([float(r["map50_95"]) for r in rows if r["group"] == "optimized"])

    report = []
    report.append("# Candidate Significance Report (CV1)")
    report.append("")
    report.append(f"- seeds: {SEEDS}")
    report.append(f"- baseline train/eval: {BASE_DATA.name}")
    report.append(f"- optimized train data: {OPT_TRAIN_DATA.name}, eval data: {EVAL_DATA.name}")
    report.append(f"- mean map50: baseline={base50:.4f}, optimized={opt50:.4f}, delta={opt50-base50:+.4f}")
    report.append(f"- mean map50_95: baseline={base95:.4f}, optimized={opt95:.4f}, delta={opt95-base95:+.4f}")
    report.append("")
    report.append("## OLS")
    report.append(f"- map50: coef={ols50['coef']:+.4f}, p={ols50['p']:.6g}, n={ols50['n']}")
    report.append(f"- map50_95: coef={ols95['coef']:+.4f}, p={ols95['p']:.6g}, n={ols95['n']}")
    report.append("## Paired-by-seed")
    report.append(f"- map50: mean_delta={pair50['mean_delta']:+.4f}, p={pair50['p']:.6g}, n={pair50['n']}")
    report.append(f"- map50_95: mean_delta={pair95['mean_delta']:+.4f}, p={pair95['p']:.6g}, n={pair95['n']}")
    report_path = OUT_DIR / "seed_repeat_candidate_report_cv1.md"
    report_path.write_text("\n".join(report), encoding="utf-8")

    manifest = {
        "summary_csv": str(csv_path.resolve()),
        "report_md": str(report_path.resolve()),
        "ols_map50": ols50,
        "ols_map50_95": ols95,
        "paired_map50": pair50,
        "paired_map50_95": pair95,
        "base_mean_map50": float(base50),
        "opt_mean_map50": float(opt50),
        "base_mean_map50_95": float(base95),
        "opt_mean_map50_95": float(opt95),
        "elapsed_sec": time.time() - t0,
    }
    man_path = OUT_DIR / "seed_repeat_candidate_manifest_cv1.json"
    man_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False))


if __name__ == "__main__":
    main()
