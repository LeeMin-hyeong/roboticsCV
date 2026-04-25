import csv
import json
import shutil
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from ultralytics import YOLO


ROOT = Path(__file__).resolve().parent
VOC_YAML = ROOT / "voc3" / "data.yaml"
OUT_DIR = ROOT / "outputs"
RUNS_DIR = ROOT / "opt_aug_runs"
FIG_DIR = OUT_DIR / "figures"
ABLATION_CSV = OUT_DIR / "ablation_train_summary_cv1.csv"

# baseline used for fair comparison
BASELINE = {
    "epochs": 10,
    "imgsz": 640,
    "batch": 16,
    "lr0": 0.001,
    "freeze": 5,
}

# objective for ranking candidates (favor both strict quality and easy-detection quality)
W_MAP95 = 0.6
W_MAP50 = 0.4

# from-scratch only
STRICT_FROM_BASE = True

# Candidate set includes augmentation strategies + hyperparameters
CANDIDATES = [
    {"tag": "aug_01_ref", "epochs": 10, "imgsz": 640, "batch": 16, "lr0": 0.001, "freeze": 5},
    {"tag": "aug_02_mild_color", "epochs": 14, "imgsz": 640, "batch": 12, "lr0": 0.001, "freeze": 5, "hsv_h": 0.010, "hsv_s": 0.50, "hsv_v": 0.30},
    {"tag": "aug_03_geo_light", "epochs": 14, "imgsz": 640, "batch": 12, "lr0": 0.001, "freeze": 5, "translate": 0.06, "scale": 0.35, "degrees": 2.0, "shear": 1.0},
    {"tag": "aug_04_mosaic_mixup", "epochs": 16, "imgsz": 640, "batch": 12, "lr0": 0.001, "freeze": 5, "mosaic": 0.7, "mixup": 0.10, "close_mosaic": 3},
    {"tag": "aug_05_copy_paste", "epochs": 16, "imgsz": 640, "batch": 12, "lr0": 0.001, "freeze": 5, "copy_paste": 0.10, "mosaic": 0.7, "mixup": 0.05, "close_mosaic": 3},
    {"tag": "aug_06_conservative", "epochs": 16, "imgsz": 640, "batch": 12, "lr0": 0.001, "freeze": 5, "mosaic": 0.3, "mixup": 0.00, "copy_paste": 0.00, "scale": 0.25, "translate": 0.05},
    {"tag": "aug_07_highres_mild", "epochs": 16, "imgsz": 704, "batch": 12, "lr0": 0.001, "freeze": 5, "mosaic": 0.5, "mixup": 0.05, "close_mosaic": 4},
    {"tag": "aug_08_highres_conservative", "epochs": 14, "imgsz": 704, "batch": 12, "lr0": 0.001, "freeze": 5, "mosaic": 0.3, "mixup": 0.00, "translate": 0.05, "scale": 0.30},
    {"tag": "aug_09_cosine", "epochs": 16, "imgsz": 640, "batch": 12, "lr0": 0.001, "freeze": 5, "cos_lr": True, "lrf": 0.2, "mosaic": 0.6, "close_mosaic": 3},
    {"tag": "aug_10_low_lr_aug", "epochs": 16, "imgsz": 640, "batch": 12, "lr0": 0.0009, "freeze": 5, "mosaic": 0.6, "mixup": 0.10, "close_mosaic": 3},
    {"tag": "aug_11_freeze6_aug", "epochs": 16, "imgsz": 640, "batch": 12, "lr0": 0.001, "freeze": 6, "mosaic": 0.5, "mixup": 0.05, "close_mosaic": 3},
    {"tag": "aug_12_freeze4_aug", "epochs": 16, "imgsz": 640, "batch": 12, "lr0": 0.001, "freeze": 4, "mosaic": 0.6, "mixup": 0.10, "close_mosaic": 3},
]


def load_baseline():
    if not ABLATION_CSV.exists():
        raise FileNotFoundError(f"missing baseline csv: {ABLATION_CSV}")
    with open(ABLATION_CSV, "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    b = next((r for r in rows if r["variant"] == "baseline"), None)
    if b is None:
        raise RuntimeError("baseline row not found in ablation csv")
    return {
        "map50": float(b["map50"]),
        "map50_95": float(b["map50_95"]),
        "precision": float(b["precision"]),
        "recall": float(b["recall"]),
    }


def score_row(r: dict) -> float:
    return W_MAP95 * float(r["map50_95"]) + W_MAP50 * float(r["map50"])


def run_candidate(cfg: dict) -> dict:
    tag = cfg["tag"]
    exp_dir = RUNS_DIR / tag
    if STRICT_FROM_BASE and exp_dir.exists():
        shutil.rmtree(exp_dir, ignore_errors=True)

    train_kwargs = dict(cfg)
    tag = train_kwargs.pop("tag")
    epochs = int(train_kwargs.pop("epochs"))
    imgsz = int(train_kwargs.pop("imgsz"))
    batch = int(train_kwargs.pop("batch"))
    lr0 = float(train_kwargs.pop("lr0"))
    freeze = int(train_kwargs.pop("freeze"))

    print(f"[train] {tag}: epochs={epochs}, imgsz={imgsz}, batch={batch}, lr0={lr0}, freeze={freeze}, extras={train_kwargs}")

    model = YOLO("yolov8n.pt")
    model.train(
        data=str(VOC_YAML),
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        lr0=lr0,
        freeze=freeze,
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

    best_pt = exp_dir / "weights" / "best.pt"
    m = YOLO(str(best_pt)).val(data=str(VOC_YAML), imgsz=imgsz, verbose=False, workers=0)

    row = {
        "tag": tag,
        "epochs": epochs,
        "imgsz": imgsz,
        "batch": batch,
        "lr0": lr0,
        "freeze": freeze,
        "map50": float(m.box.map50),
        "map50_95": float(m.box.map),
        "precision": float(m.box.mp),
        "recall": float(m.box.mr),
        "score": 0.0,  # computed below
        "best_pt": str(best_pt.resolve()),
        "init_model": "yolov8n.pt",
    }
    for k, v in train_kwargs.items():
        row[k] = v
    row["score"] = score_row(row)
    print(f"[result] {tag}: map50={row['map50']:.4f}, map50_95={row['map50_95']:.4f}, score={row['score']:.4f}")
    return row


def save_csv(rows, path: Path):
    keyset = set()
    for r in rows:
        keyset.update(r.keys())
    preferred = [
        "tag", "epochs", "imgsz", "batch", "lr0", "freeze",
        "hsv_h", "hsv_s", "hsv_v",
        "translate", "scale", "degrees", "shear",
        "mosaic", "mixup", "copy_paste", "close_mosaic",
        "cos_lr", "lrf",
        "map50", "map50_95", "precision", "recall", "score",
        "best_pt", "init_model",
    ]
    cols = [k for k in preferred if k in keyset] + sorted([k for k in keyset if k not in preferred])
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)


def make_figures(rows, baseline):
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    rs = sorted(rows, key=lambda x: x["score"], reverse=True)
    top = rs[:10]

    # score leaderboard
    plt.figure(figsize=(12, 5))
    plt.bar(range(len(top)), [r["score"] for r in top], color="#2563eb")
    plt.xticks(range(len(top)), [r["tag"] for r in top], rotation=35, ha="right")
    plt.ylabel(f"score = {W_MAP95:.1f}*map50_95 + {W_MAP50:.1f}*map50")
    plt.title("Augmentation + Hyperparameter Search Leaderboard")
    plt.tight_layout()
    p1 = FIG_DIR / "opt_aug_leaderboard_cv1.png"
    plt.savefig(p1, dpi=170)
    plt.close()

    # best vs baseline
    best = rs[0]
    cats = ["mAP50", "mAP50-95", "Precision", "Recall"]
    bvals = [baseline["map50"], baseline["map50_95"], baseline["precision"], baseline["recall"]]
    ovals = [best["map50"], best["map50_95"], best["precision"], best["recall"]]
    x = np.arange(len(cats))
    w = 0.36
    plt.figure(figsize=(8.5, 5))
    plt.bar(x - w / 2, bvals, width=w, label="baseline", color="#94a3b8")
    plt.bar(x + w / 2, ovals, width=w, label="best_aug", color="#16a34a")
    plt.xticks(x, cats)
    plt.ylim(0, 1)
    plt.title(f"Baseline vs Best-Aug ({best['tag']})")
    plt.legend()
    plt.tight_layout()
    p2 = FIG_DIR / "opt_aug_baseline_compare_cv1.png"
    plt.savefig(p2, dpi=170)
    plt.close()
    return p1, p2


def write_report(rows, baseline, fig1: Path, fig2: Path, path: Path):
    rs = sorted(rows, key=lambda x: x["score"], reverse=True)
    best = rs[0]

    d50 = best["map50"] - baseline["map50"]
    d95 = best["map50_95"] - baseline["map50_95"]
    dp = best["precision"] - baseline["precision"]
    dr = best["recall"] - baseline["recall"]

    lines = []
    lines.append("# CV1 Augmentation-Aware Optimization Report")
    lines.append("")
    lines.append("## Constraints")
    lines.append("- Same dataset split as baseline")
    lines.append("- Same model family: yolov8n.pt")
    lines.append("- Every run starts from base pretrained model (no resume)")
    lines.append("")
    lines.append("## Baseline")
    lines.append(
        f"- mAP50={baseline['map50']:.4f}, mAP50-95={baseline['map50_95']:.4f}, "
        f"P={baseline['precision']:.4f}, R={baseline['recall']:.4f}"
    )
    lines.append("")
    lines.append("## Best Candidate")
    lines.append(f"- tag: {best['tag']}")
    lines.append(
        f"- config: epochs={best['epochs']}, imgsz={best['imgsz']}, batch={best['batch']}, "
        f"lr0={best['lr0']}, freeze={best['freeze']}"
    )
    extra_keys = ["hsv_h", "hsv_s", "hsv_v", "translate", "scale", "degrees", "shear", "mosaic", "mixup", "copy_paste", "close_mosaic", "cos_lr", "lrf"]
    extras = [f"{k}={best[k]}" for k in extra_keys if k in best and best[k] not in ("", None)]
    lines.append(f"- augmentation: {', '.join(extras) if extras else '(default)'}")
    lines.append(
        f"- metrics: mAP50={best['map50']:.4f}, mAP50-95={best['map50_95']:.4f}, "
        f"P={best['precision']:.4f}, R={best['recall']:.4f}, score={best['score']:.4f}"
    )
    lines.append("- delta vs baseline:")
    lines.append(f"  map50: {d50:+.4f}")
    lines.append(f"  map50_95: {d95:+.4f}")
    lines.append(f"  precision: {dp:+.4f}")
    lines.append(f"  recall: {dr:+.4f}")
    lines.append("")
    lines.append("## Plots")
    lines.append(f"![leaderboard]({fig1.as_posix()})")
    lines.append("")
    lines.append(f"![baseline_vs_best_aug]({fig2.as_posix()})")
    lines.append("")
    lines.append("## Top 5")
    for i, r in enumerate(rs[:5], start=1):
        lines.append(
            f"- {i}. {r['tag']}: map50={r['map50']:.4f}, map50_95={r['map50_95']:.4f}, "
            f"P={r['precision']:.4f}, R={r['recall']:.4f}, score={r['score']:.4f}"
        )
    lines.append("")
    lines.append("## Best Weights")
    lines.append(f"- {best['best_pt']}")

    path.write_text("\n".join(lines), encoding="utf-8")


def main():
    t0 = time.time()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    if not VOC_YAML.exists():
        raise FileNotFoundError(f"dataset yaml missing: {VOC_YAML}")

    baseline = load_baseline()
    rows = []
    for cfg in CANDIDATES:
        try:
            rows.append(run_candidate(cfg))
        except Exception as e:
            print(f"[error] {cfg['tag']}: {repr(e)}")

    if not rows:
        raise RuntimeError("No candidate finished.")

    csv_path = OUT_DIR / "optimization_aug_summary_cv1.csv"
    save_csv(rows, csv_path)

    fig1, fig2 = make_figures(rows, baseline)
    report_path = OUT_DIR / "optimization_aug_report_cv1.md"
    write_report(rows, baseline, fig1, fig2, report_path)

    best = sorted(rows, key=lambda x: x["score"], reverse=True)[0]
    payload = {
        "summary_csv": str(csv_path.resolve()),
        "report_md": str(report_path.resolve()),
        "fig_leaderboard": str(fig1.resolve()),
        "fig_compare": str(fig2.resolve()),
        "best_tag": best["tag"],
        "best_score": best["score"],
        "best_map50": best["map50"],
        "best_map50_95": best["map50_95"],
        "best_pt": best["best_pt"],
        "elapsed_sec": time.time() - t0,
    }
    manifest = OUT_DIR / "optimization_aug_manifest_cv1.json"
    manifest.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False))


if __name__ == "__main__":
    main()
