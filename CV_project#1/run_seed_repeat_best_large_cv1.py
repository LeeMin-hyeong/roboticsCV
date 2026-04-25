import csv
import json
import math
import os
import shutil
import time
from pathlib import Path

import numpy as np
from scipy import stats
from ultralytics import YOLO


ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT / "outputs"
RUNS_DIR = ROOT / "repeat_runs_best_large"

BASE_DATA = ROOT / "voc3" / "data.yaml"  # baseline must stay fixed
EVAL_DATA = ROOT / "voc3" / "data.yaml"  # fair comparison on original eval split
MANIFEST = OUT_DIR / "optimization_aug_large_manifest_cv1.json"
SUMMARY = OUT_DIR / "optimization_aug_large_summary_cv1.csv"

DEFAULT_SEEDS = [111, 222, 333, 444, 555]

BASELINE_CFG = {
    "epochs": 10,
    "imgsz": 640,
    "batch": 16,
    "lr0": 0.001,
    "freeze": 5,
}


def parse_value(v: str):
    if v is None or v == "":
        return None
    if v in ("True", "False"):
        return v == "True"
    try:
        if "." in v:
            return float(v)
        return int(v)
    except Exception:
        return v


def load_best_cfg():
    if not MANIFEST.exists():
        raise FileNotFoundError(f"missing manifest: {MANIFEST}")
    if not SUMMARY.exists():
        raise FileNotFoundError(f"missing summary: {SUMMARY}")

    man = json.loads(MANIFEST.read_text(encoding="utf-8"))
    best_tag = man["best_tag"]
    opt_train_data = Path(man["opt_train_data"])
    if not opt_train_data.exists():
        raise FileNotFoundError(f"missing opt_train_data: {opt_train_data}")

    rows = list(csv.DictReader(open(SUMMARY, encoding="utf-8")))
    row = next((r for r in rows if r["tag"] == best_tag), None)
    if row is None:
        raise RuntimeError(f"best tag {best_tag} not found in summary")

    # keep only train hyperparameters/augment options
    keys = [
        "epochs", "imgsz", "batch", "lr0", "freeze",
        "mosaic", "mixup", "copy_paste", "close_mosaic",
        "hsv_h", "hsv_s", "hsv_v", "translate", "scale", "degrees", "shear",
        "cos_lr", "lrf",
    ]
    cfg = {}
    for k in keys:
        if k in row:
            v = parse_value(row[k])
            if v is not None:
                cfg[k] = v
    return best_tag, cfg, opt_train_data


def train_eval(tag: str, seed: int, cfg: dict, train_data: Path, eval_data: Path):
    name = f"{tag}_seed{seed}"
    exp = RUNS_DIR / name
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
        name=name,
        exist_ok=True,
        verbose=False,
        workers=0,
        deterministic=True,
        seed=seed,
        resume=False,
        **kw,
    )

    best_pt = exp / "weights" / "best.pt"
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
    pref = [
        "group", "seed", "train_data", "eval_data",
        "epochs", "imgsz", "batch", "lr0", "freeze",
        "mosaic", "mixup", "copy_paste", "close_mosaic",
        "hsv_h", "hsv_s", "hsv_v", "translate", "scale", "degrees", "shear",
        "cos_lr", "lrf",
        "map50", "map50_95", "precision", "recall",
        "best_pt", "init_model",
    ]
    cols = [k for k in pref if k in keyset] + sorted([k for k in keyset if k not in pref])
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)


def ols(rows, metric):
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


def paired(rows, metric):
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
        raise FileNotFoundError(f"missing baseline data: {BASE_DATA}")
    if not EVAL_DATA.exists():
        raise FileNotFoundError(f"missing eval data: {EVAL_DATA}")

    best_tag, opt_cfg, opt_train_data = load_best_cfg()
    seed_env = os.environ.get("CV1_SEEDS", "").strip()
    if seed_env:
        seeds = [int(x.strip()) for x in seed_env.split(",") if x.strip()]
    else:
        seeds = list(DEFAULT_SEEDS)
    print(f"[best_large] tag={best_tag} cfg={opt_cfg}")
    print(f"[seeds] {seeds}")

    rows = []
    for s in seeds:
        rows.append(train_eval("baseline", s, BASELINE_CFG, BASE_DATA, EVAL_DATA))
    for s in seeds:
        rows.append(train_eval("optimized", s, opt_cfg, opt_train_data, EVAL_DATA))

    csv_path = OUT_DIR / "seed_repeat_best_large_summary_cv1.csv"
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
    rep.append("# Best-Large Candidate Significance Report (CV1)")
    rep.append("")
    rep.append(f"- best tag: {best_tag}")
    rep.append(f"- seeds: {seeds}")
    rep.append(f"- baseline train/eval: {BASE_DATA.name}")
    rep.append(f"- optimized train: {opt_train_data.name}, eval: {EVAL_DATA.name}")
    rep.append(f"- mean map50: baseline={b50:.4f}, optimized={o50:.4f}, delta={o50-b50:+.4f}")
    rep.append(f"- mean map50_95: baseline={b95:.4f}, optimized={o95:.4f}, delta={o95-b95:+.4f}")
    rep.append("")
    rep.append("## OLS")
    rep.append(f"- map50: coef={ols50['coef']:+.4f}, p={ols50['p']:.6g}, n={ols50['n']}")
    rep.append(f"- map50_95: coef={ols95['coef']:+.4f}, p={ols95['p']:.6g}, n={ols95['n']}")
    rep.append("## Paired-by-seed")
    rep.append(f"- map50: mean_delta={p50['mean_delta']:+.4f}, p={p50['p']:.6g}, n={p50['n']}")
    rep.append(f"- map50_95: mean_delta={p95['mean_delta']:+.4f}, p={p95['p']:.6g}, n={p95['n']}")
    rep_path = OUT_DIR / "seed_repeat_best_large_report_cv1.md"
    rep_path.write_text("\n".join(rep), encoding="utf-8")

    manifest = {
        "best_tag": best_tag,
        "summary_csv": str(csv_path.resolve()),
        "report_md": str(rep_path.resolve()),
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
    man = OUT_DIR / "seed_repeat_best_large_manifest_cv1.json"
    man.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False))


if __name__ == "__main__":
    main()
