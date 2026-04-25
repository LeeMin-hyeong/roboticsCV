import csv
import json
import shutil
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torchvision
import yaml
from ultralytics import YOLO


ROOT = Path(__file__).resolve().parent
DATA_ROOT = ROOT / "data"
VOC_BASE = ROOT / "voc3"
VOC_LARGE = ROOT / "voc3_large"
OUT_DIR = ROOT / "outputs"
FIG_DIR = OUT_DIR / "figures"
RUNS_DIR = ROOT / "opt_aug_large_runs"
ABLATION_CSV = OUT_DIR / "ablation_train_summary_cv1.csv"

TARGET = ["person", "car", "dog"]
CLS2IDX = {c: i for i, c in enumerate(TARGET)}

# baseline dataset must remain unchanged
EVAL_DATA = VOC_BASE / "data.yaml"

# enlarged dataset only for optimization process
MAX_TR_LARGE = 300
MAX_VA_LARGE = 100

W_MAP95 = 0.7
W_MAP50 = 0.3

CANDIDATES = [
    {"tag": "lg_01_ref_like", "epochs": 16, "imgsz": 640, "batch": 12, "lr0": 0.0009, "freeze": 5},
    {"tag": "lg_02_best_prev", "epochs": 16, "imgsz": 640, "batch": 12, "lr0": 0.0009, "freeze": 5, "mosaic": 0.6, "mixup": 0.1, "close_mosaic": 3},
    {"tag": "lg_03_more_mixup", "epochs": 16, "imgsz": 640, "batch": 12, "lr0": 0.0009, "freeze": 5, "mosaic": 0.6, "mixup": 0.2, "close_mosaic": 3},
    {"tag": "lg_04_copy_paste", "epochs": 16, "imgsz": 640, "batch": 12, "lr0": 0.0009, "freeze": 5, "mosaic": 0.6, "mixup": 0.1, "copy_paste": 0.1, "close_mosaic": 3},
    {"tag": "lg_05_lr_up", "epochs": 16, "imgsz": 640, "batch": 12, "lr0": 0.0011, "freeze": 5, "mosaic": 0.6, "mixup": 0.1, "close_mosaic": 3},
    {"tag": "lg_06_highres", "epochs": 16, "imgsz": 704, "batch": 12, "lr0": 0.0009, "freeze": 5, "mosaic": 0.5, "mixup": 0.1, "close_mosaic": 4},
    {"tag": "lg_07_20ep", "epochs": 20, "imgsz": 640, "batch": 12, "lr0": 0.0009, "freeze": 5, "mosaic": 0.6, "mixup": 0.1, "close_mosaic": 4},
    {"tag": "lg_08_cosine", "epochs": 16, "imgsz": 640, "batch": 12, "lr0": 0.0009, "freeze": 5, "mosaic": 0.6, "mixup": 0.1, "close_mosaic": 3, "cos_lr": True, "lrf": 0.2},
]


def parse_voc(ann):
    objs = ann["annotation"]["object"]
    return [objs] if isinstance(objs, dict) else objs


def to_yolo(objs, iw, ih):
    lines = []
    for o in objs:
        c = o["name"]
        if c not in CLS2IDX:
            continue
        bb = o["bndbox"]
        x1 = max(0, float(bb["xmin"]))
        y1 = max(0, float(bb["ymin"]))
        x2 = min(iw, float(bb["xmax"]))
        y2 = min(ih, float(bb["ymax"]))
        if x2 <= x1 or y2 <= y1:
            continue
        lines.append(
            f"{CLS2IDX[c]} {(x1 + x2) / 2 / iw:.6f} {(y1 + y2) / 2 / ih:.6f} {(x2 - x1) / iw:.6f} {(y2 - y1) / ih:.6f}"
        )
    return "\n".join(lines)


def prepare_large_dataset():
    if (VOC_LARGE / "data.yaml").exists():
        print(f"[data] reuse {VOC_LARGE}")
        return

    print("[data] preparing enlarged voc3_large dataset...")
    voc_tr = torchvision.datasets.VOCDetection(root=str(DATA_ROOT), year="2007", image_set="trainval", download=True)
    voc_te = torchvision.datasets.VOCDetection(root=str(DATA_ROOT), year="2007", image_set="test", download=True)

    for sp, ds, lim in [("train", voc_tr, MAX_TR_LARGE), ("val", voc_te, MAX_VA_LARGE)]:
        (VOC_LARGE / "images" / sp).mkdir(parents=True, exist_ok=True)
        (VOC_LARGE / "labels" / sp).mkdir(parents=True, exist_ok=True)
        cnt = 0
        for i in range(len(ds)):
            if cnt >= lim:
                break
            _, t = ds[i]
            objs = parse_voc(t)
            if not any(o["name"] in TARGET for o in objs):
                continue
            fn = t["annotation"]["filename"]
            w = int(t["annotation"]["size"]["width"])
            h = int(t["annotation"]["size"]["height"])
            src = DATA_ROOT / "VOCdevkit" / "VOC2007" / "JPEGImages" / fn
            if not src.exists():
                continue
            lb = to_yolo(objs, w, h)
            if not lb.strip():
                continue
            shutil.copy2(src, VOC_LARGE / "images" / sp / fn)
            (VOC_LARGE / "labels" / sp / fn.replace(".jpg", ".txt")).write_text(lb, encoding="utf-8")
            cnt += 1

    data_yaml = {
        "path": str(VOC_LARGE.resolve()),
        "train": "images/train",
        "val": "images/val",
        "nc": 3,
        "names": TARGET,
    }
    with open(VOC_LARGE / "data.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(data_yaml, f, sort_keys=False)
    print(f"[data] done. train={MAX_TR_LARGE}, val={MAX_VA_LARGE}")


def load_baseline():
    with open(ABLATION_CSV, "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    b = next((r for r in rows if r["variant"] == "baseline"), None)
    if b is None:
        raise RuntimeError("baseline row not found")
    return {
        "map50": float(b["map50"]),
        "map50_95": float(b["map50_95"]),
        "precision": float(b["precision"]),
        "recall": float(b["recall"]),
    }


def score(r):
    return W_MAP95 * float(r["map50_95"]) + W_MAP50 * float(r["map50"])


def run_candidate(cfg):
    tag = cfg["tag"]
    exp = RUNS_DIR / tag
    if exp.exists():
        shutil.rmtree(exp, ignore_errors=True)

    kw = dict(cfg)
    tag = kw.pop("tag")
    epochs = int(kw.pop("epochs"))
    imgsz = int(kw.pop("imgsz"))
    batch = int(kw.pop("batch"))
    lr0 = float(kw.pop("lr0"))
    freeze = int(kw.pop("freeze"))

    print(f"[train] {tag} extras={kw}")
    model = YOLO("yolov8n.pt")
    model.train(
        data=str(VOC_LARGE / "data.yaml"),
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
        **kw,
    )
    best_pt = exp / "weights" / "best.pt"

    # Fair comparison: evaluate on the original baseline dataset
    m = YOLO(str(best_pt)).val(data=str(EVAL_DATA), imgsz=imgsz, verbose=False, workers=0)
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
        "score": 0.0,
        "best_pt": str(best_pt.resolve()),
        "train_data": str((VOC_LARGE / "data.yaml").resolve()),
        "eval_data": str(EVAL_DATA.resolve()),
        "init_model": "yolov8n.pt",
    }
    for k, v in kw.items():
        row[k] = v
    row["score"] = score(row)
    print(f"[result] {tag}: map50={row['map50']:.4f}, map50_95={row['map50_95']:.4f}, score={row['score']:.4f}")
    return row


def save_csv(rows, path: Path):
    keyset = set()
    for r in rows:
        keyset.update(r.keys())
    pref = [
        "tag", "epochs", "imgsz", "batch", "lr0", "freeze",
        "mosaic", "mixup", "copy_paste", "close_mosaic",
        "cos_lr", "lrf",
        "map50", "map50_95", "precision", "recall", "score",
        "train_data", "eval_data", "best_pt", "init_model",
    ]
    cols = [k for k in pref if k in keyset] + sorted([k for k in keyset if k not in pref])
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)


def make_figures(rows, baseline):
    rs = sorted(rows, key=lambda x: x["score"], reverse=True)
    top = rs[:8]
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(10, 4.8))
    plt.bar(range(len(top)), [r["score"] for r in top], color="#2563eb")
    plt.xticks(range(len(top)), [r["tag"] for r in top], rotation=35, ha="right")
    plt.ylabel(f"score={W_MAP95:.1f}*map50_95 + {W_MAP50:.1f}*map50")
    plt.title("Large-data optimization leaderboard")
    plt.tight_layout()
    p1 = FIG_DIR / "opt_aug_large_leaderboard_cv1.png"
    plt.savefig(p1, dpi=170)
    plt.close()

    best = rs[0]
    cats = ["mAP50", "mAP50-95", "Precision", "Recall"]
    b = [baseline["map50"], baseline["map50_95"], baseline["precision"], baseline["recall"]]
    o = [best["map50"], best["map50_95"], best["precision"], best["recall"]]
    x = np.arange(len(cats))
    w = 0.36
    plt.figure(figsize=(8.5, 5))
    plt.bar(x - w / 2, b, width=w, label="baseline", color="#94a3b8")
    plt.bar(x + w / 2, o, width=w, label="best_large_opt", color="#16a34a")
    plt.xticks(x, cats)
    plt.ylim(0, 1)
    plt.title(f"Baseline vs Best-Large ({best['tag']})")
    plt.legend()
    plt.tight_layout()
    p2 = FIG_DIR / "opt_aug_large_compare_cv1.png"
    plt.savefig(p2, dpi=170)
    plt.close()
    return p1, p2


def main():
    t0 = time.time()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    prepare_large_dataset()
    baseline = load_baseline()

    rows = []
    for cfg in CANDIDATES:
        try:
            rows.append(run_candidate(cfg))
        except Exception as e:
            print(f"[error] {cfg['tag']}: {repr(e)}")

    if not rows:
        raise RuntimeError("no run succeeded")

    csv_path = OUT_DIR / "optimization_aug_large_summary_cv1.csv"
    save_csv(rows, csv_path)
    fig1, fig2 = make_figures(rows, baseline)

    best = sorted(rows, key=lambda x: x["score"], reverse=True)[0]
    report = []
    report.append("# CV1 Large-Data Optimization Report")
    report.append("")
    report.append("- baseline dataset unchanged (voc3)")
    report.append("- optimization train dataset expanded (voc3_large)")
    report.append("- evaluation kept on baseline dataset (voc3) for fair comparison")
    report.append("")
    report.append(
        f"- baseline: map50={baseline['map50']:.4f}, map50_95={baseline['map50_95']:.4f}, "
        f"P={baseline['precision']:.4f}, R={baseline['recall']:.4f}"
    )
    report.append(
        f"- best_large: {best['tag']} map50={best['map50']:.4f}, map50_95={best['map50_95']:.4f}, "
        f"P={best['precision']:.4f}, R={best['recall']:.4f}, score={best['score']:.4f}"
    )
    report.append(f"- delta map50={best['map50']-baseline['map50']:+.4f}, map50_95={best['map50_95']-baseline['map50_95']:+.4f}")
    report.append("")
    report.append(f"![leaderboard]({fig1.as_posix()})")
    report.append("")
    report.append(f"![compare]({fig2.as_posix()})")
    rep_path = OUT_DIR / "optimization_aug_large_report_cv1.md"
    rep_path.write_text("\n".join(report), encoding="utf-8")

    manifest = {
        "summary_csv": str(csv_path.resolve()),
        "report_md": str(rep_path.resolve()),
        "fig_leaderboard": str(fig1.resolve()),
        "fig_compare": str(fig2.resolve()),
        "best_tag": best["tag"],
        "best_map50": best["map50"],
        "best_map50_95": best["map50_95"],
        "best_score": best["score"],
        "best_pt": best["best_pt"],
        "opt_train_data": str((VOC_LARGE / "data.yaml").resolve()),
        "eval_data": str(EVAL_DATA.resolve()),
        "elapsed_sec": time.time() - t0,
    }
    man_path = OUT_DIR / "optimization_aug_large_manifest_cv1.json"
    man_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False))


if __name__ == "__main__":
    main()
