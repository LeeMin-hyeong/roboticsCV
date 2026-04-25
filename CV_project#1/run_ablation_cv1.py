import os
import csv
import glob
import json
import time
import shutil
import random
from pathlib import Path

import yaml
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torchvision
from ultralytics import YOLO


# Fixed by user request:
# 1) TARGET (classes)
# 3) Data selection logic
# 4) Model architecture/weights (yolov8n.pt)
TARGET = ["person", "car", "dog"]
CLS2IDX = {c: i for i, c in enumerate(TARGET)}

MAX_TR = 150
MAX_VA = 50

BASELINE = {
    "epochs": 10,
    "imgsz": 640,
    "batch": 16,
    "lr0": 0.001,
    "freeze": 5,
}

# Small perturbations around baseline
ABLATION_GRID = {
    "epochs": [8, 14],
    "imgsz": [576, 704],
    "batch": [12, 20],
    "lr0": [0.0007, 0.0013],
    "freeze": [4, 6],
}

CONF_GRID = [0.05, 0.15, 0.25, 0.35]
IOU_GRID = [0.4, 0.5, 0.6]

SEED = 42
ROOT = Path(__file__).resolve().parent
DATA_ROOT = ROOT / "data"
VOC3_ROOT = ROOT / "voc3"
OUT_DIR = ROOT / "outputs"
RUNS_DIR = ROOT / "ablation_runs"
FIG_DIR = OUT_DIR / "figures"
STRICT_FROM_BASE = True


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


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


def prepare_dataset() -> None:
    if (VOC3_ROOT / "data.yaml").exists():
        print(f"[data] Reusing existing dataset at {VOC3_ROOT}")
        return

    print("[data] Downloading VOC 2007 and creating voc3 dataset...")
    voc_tr = torchvision.datasets.VOCDetection(
        root=str(DATA_ROOT), year="2007", image_set="trainval", download=True
    )
    voc_te = torchvision.datasets.VOCDetection(
        root=str(DATA_ROOT), year="2007", image_set="test", download=True
    )

    for sp, ds, lim in [("train", voc_tr, MAX_TR), ("val", voc_te, MAX_VA)]:
        (VOC3_ROOT / "images" / sp).mkdir(parents=True, exist_ok=True)
        (VOC3_ROOT / "labels" / sp).mkdir(parents=True, exist_ok=True)
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
            shutil.copy2(src, VOC3_ROOT / "images" / sp / fn)
            with open(VOC3_ROOT / "labels" / sp / fn.replace(".jpg", ".txt"), "w", encoding="utf-8") as f:
                f.write(lb)
            cnt += 1

    with open(VOC3_ROOT / "data.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(
            {
                "path": str(VOC3_ROOT.resolve()),
                "train": "images/train",
                "val": "images/val",
                "nc": 3,
                "names": TARGET,
            },
            f,
            sort_keys=False,
        )
    print(f"[data] Done. train={MAX_TR}, val={MAX_VA}, classes={TARGET}")


def get_best_weight_path(exp_dir: Path) -> Path:
    cands = list(exp_dir.rglob("best.pt"))
    if not cands:
        raise FileNotFoundError(f"best.pt not found under {exp_dir}")
    cands.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return cands[0]


def run_train_eval(tag: str, cfg: dict) -> dict:
    exp_name = f"cv1_{tag}"
    print(f"[run] {exp_name} -> {cfg}")
    exp_dir = RUNS_DIR / exp_name
    if STRICT_FROM_BASE and exp_dir.exists():
        shutil.rmtree(exp_dir, ignore_errors=True)

    # Always train from base pretrained checkpoint, never from previous best.pt
    model = YOLO("yolov8n.pt")
    model.train(
        data=str(VOC3_ROOT / "data.yaml"),
        epochs=int(cfg["epochs"]),
        imgsz=int(cfg["imgsz"]),
        batch=int(cfg["batch"]),
        lr0=float(cfg["lr0"]),
        freeze=int(cfg["freeze"]),
        project=str(RUNS_DIR),
        name=exp_name,
        exist_ok=True,
        verbose=False,
        seed=SEED,
        deterministic=True,
        workers=0,
        resume=False,
    )
    best_pt = get_best_weight_path(exp_dir)
    m = YOLO(str(best_pt)).val(
        data=str(VOC3_ROOT / "data.yaml"),
        imgsz=int(cfg["imgsz"]),
        verbose=False,
        workers=0,
    )

    row = {
        "run": exp_name,
        "variant": tag,
        "epochs": int(cfg["epochs"]),
        "imgsz": int(cfg["imgsz"]),
        "batch": int(cfg["batch"]),
        "lr0": float(cfg["lr0"]),
        "freeze": int(cfg["freeze"]),
        "map50": float(m.box.map50),
        "map50_95": float(m.box.map),
        "precision": float(m.box.mp),
        "recall": float(m.box.mr),
        "best_pt": str(best_pt.resolve()),
        "init_model": "yolov8n.pt",
    }
    print(
        f"[result] {exp_name}: mAP50={row['map50']:.4f}, mAP50-95={row['map50_95']:.4f}, "
        f"P={row['precision']:.4f}, R={row['recall']:.4f}"
    )
    return row


def class_eval(mdl: YOLO, conf=0.15, iou_thr=0.5) -> dict:
    stats = {c: {"tp": 0, "fp": 0, "fn": 0} for c in TARGET}
    coco_map = {n: n for n in mdl.names.values() if n in CLS2IDX}

    def iou(b1, b2):
        xi, yi = max(b1[0], b2[0]), max(b1[1], b2[1])
        xa, ya = min(b1[2], b2[2]), min(b1[3], b2[3])
        inter = max(0, xa - xi) * max(0, ya - yi)
        union = (
            (b1[2] - b1[0]) * (b1[3] - b1[1])
            + (b2[2] - b2[0]) * (b2[3] - b2[1])
            - inter
            + 1e-6
        )
        return inter / union

    for ip in sorted(glob.glob(str(VOC3_ROOT / "images" / "val" / "*.jpg"))):
        img = Image.open(ip)
        iw, ih = img.size
        gts = []
        lp = ip.replace("\\images\\", "\\labels\\").replace("/images/", "/labels/").replace(".jpg", ".txt")
        if os.path.exists(lp):
            with open(lp, "r", encoding="utf-8") as f:
                lines = [x.strip() for x in f.readlines() if x.strip()]
            for ln in lines:
                p = ln.split()
                ci = int(p[0])
                cx, cy, bw, bh = [float(v) for v in p[1:]]
                gts.append(
                    {
                        "cls": TARGET[ci],
                        "box": [
                            (cx - bw / 2) * iw,
                            (cy - bh / 2) * ih,
                            (cx + bw / 2) * iw,
                            (cy + bh / 2) * ih,
                        ],
                        "m": False,
                    }
                )

        preds = []
        r = mdl(ip, conf=conf, verbose=False)[0]
        for b in r.boxes:
            cn = mdl.names[int(b.cls[0].item())]
            if cn not in coco_map:
                continue
            preds.append(
                {
                    "cls": coco_map[cn],
                    "box": b.xyxy[0].cpu().numpy().tolist(),
                    "conf": float(b.conf[0].item()),
                }
            )

        for pr in sorted(preds, key=lambda x: -x["conf"]):
            best_i, best_v = None, 0.0
            for gt in gts:
                if gt["m"] or gt["cls"] != pr["cls"]:
                    continue
                v = iou(pr["box"], gt["box"])
                if v > best_v:
                    best_v = v
                    best_i = gt
            if best_v >= iou_thr and best_i is not None:
                best_i["m"] = True
                stats[pr["cls"]]["tp"] += 1
            else:
                stats[pr["cls"]]["fp"] += 1

        for gt in gts:
            if not gt["m"]:
                stats[gt["cls"]]["fn"] += 1
    return stats


def compute_prf(stats: dict) -> dict:
    p_all, r_all, f_all = [], [], []
    for c in TARGET:
        tp = stats[c]["tp"]
        fp = stats[c]["fp"]
        fn = stats[c]["fn"]
        p = tp / (tp + fp + 1e-9)
        r = tp / (tp + fn + 1e-9)
        f = 2 * p * r / (p + r + 1e-9)
        p_all.append(p)
        r_all.append(r)
        f_all.append(f)
    return {
        "precision_macro": float(np.mean(p_all)),
        "recall_macro": float(np.mean(r_all)),
        "f1_macro": float(np.mean(f_all)),
    }


def write_csv(rows: list, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def make_figures(train_rows: list, conf_rows: list, iou_rows: list) -> dict:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    baseline = next(r for r in train_rows if r["variant"] == "baseline")
    fig_paths = {}

    # 1) mAP50 change by variant
    labels = [r["variant"] for r in train_rows]
    vals = [r["map50"] for r in train_rows]
    colors = ["#22c55e" if x == "baseline" else "#3b82f6" for x in labels]
    plt.figure(figsize=(12, 5))
    plt.bar(range(len(labels)), vals, color=colors)
    plt.axhline(baseline["map50"], color="#ef4444", linestyle="--", linewidth=1.2, label="baseline mAP50")
    plt.xticks(range(len(labels)), labels, rotation=40, ha="right")
    plt.ylabel("mAP@50")
    plt.title("Ablation runs: mAP@50 by variant")
    plt.legend()
    plt.tight_layout()
    p1 = FIG_DIR / "map50_by_variant.png"
    plt.savefig(p1, dpi=160)
    plt.close()
    fig_paths["map50_by_variant"] = p1

    # 2) Per-variable delta chart
    var_order = ["epochs", "imgsz", "batch", "lr0", "freeze"]
    deltas = []
    var_labels = []
    for v in var_order:
        rows = [r for r in train_rows if r["variant"].startswith(f"{v}=")]
        rows_sorted = sorted(rows, key=lambda x: x[v])
        for r in rows_sorted:
            deltas.append(r["map50"] - baseline["map50"])
            var_labels.append(f"{v}={r[v]}")
    plt.figure(figsize=(12, 5))
    c = ["#16a34a" if d >= 0 else "#dc2626" for d in deltas]
    plt.bar(range(len(deltas)), deltas, color=c)
    plt.axhline(0, color="black", linewidth=1)
    plt.xticks(range(len(var_labels)), var_labels, rotation=45, ha="right")
    plt.ylabel("Delta mAP@50 vs baseline")
    plt.title("Sensitivity by variable (single-factor change)")
    plt.tight_layout()
    p2 = FIG_DIR / "delta_map50_by_variable.png"
    plt.savefig(p2, dpi=160)
    plt.close()
    fig_paths["delta_map50_by_variable"] = p2

    # 3) conf / iou threshold sensitivity
    plt.figure(figsize=(10, 4.5))
    xs = [r["conf"] for r in conf_rows]
    plt.plot(xs, [r["precision_macro"] for r in conf_rows], marker="o", label="Precision")
    plt.plot(xs, [r["recall_macro"] for r in conf_rows], marker="o", label="Recall")
    plt.plot(xs, [r["f1_macro"] for r in conf_rows], marker="o", label="F1")
    plt.xlabel("conf threshold")
    plt.ylabel("macro score")
    plt.title("Effect of confidence threshold (fixed model)")
    plt.ylim(0, 1)
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    p3 = FIG_DIR / "conf_sensitivity.png"
    plt.savefig(p3, dpi=160)
    plt.close()
    fig_paths["conf_sensitivity"] = p3

    plt.figure(figsize=(8, 4.5))
    xs = [r["iou_thr"] for r in iou_rows]
    plt.plot(xs, [r["precision_macro"] for r in iou_rows], marker="o", label="Precision")
    plt.plot(xs, [r["recall_macro"] for r in iou_rows], marker="o", label="Recall")
    plt.plot(xs, [r["f1_macro"] for r in iou_rows], marker="o", label="F1")
    plt.xlabel("IoU threshold")
    plt.ylabel("macro score")
    plt.title("Effect of IoU matching threshold (fixed model)")
    plt.ylim(0, 1)
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    p4 = FIG_DIR / "iou_sensitivity.png"
    plt.savefig(p4, dpi=160)
    plt.close()
    fig_paths["iou_sensitivity"] = p4

    return fig_paths


def summarize_impacts(train_rows: list, baseline_map50: float) -> list:
    impacts = []
    for v in ["epochs", "imgsz", "batch", "lr0", "freeze"]:
        rows = [r for r in train_rows if r["variant"].startswith(f"{v}=")]
        rows = sorted(rows, key=lambda x: x[v])
        best = max(rows, key=lambda x: x["map50"])
        worst = min(rows, key=lambda x: x["map50"])
        impacts.append(
            {
                "variable": v,
                "best_value": best[v],
                "best_map50": best["map50"],
                "best_delta_vs_baseline": best["map50"] - baseline_map50,
                "worst_value": worst[v],
                "worst_map50": worst["map50"],
                "worst_delta_vs_baseline": worst["map50"] - baseline_map50,
                "range_map50": max(r["map50"] for r in rows) - min(r["map50"] for r in rows),
            }
        )
    impacts.sort(key=lambda x: x["range_map50"], reverse=True)
    return impacts


def write_report(train_rows: list, conf_rows: list, iou_rows: list, fig_paths: dict) -> None:
    baseline = next(r for r in train_rows if r["variant"] == "baseline")
    impacts = summarize_impacts(train_rows, baseline["map50"])

    md = []
    md.append("# CV_project#1 Variable Impact Report")
    md.append("")
    md.append("## Fixed Conditions")
    md.append("- Fixed #1: TARGET classes unchanged = person/car/dog")
    md.append("- Fixed #3: Data selection logic unchanged (same as original notebook)")
    md.append("- Fixed #4: Model unchanged = YOLOv8n (`yolov8n.pt`)")
    md.append(f"- Fixed sample size: train/val = {MAX_TR}/{MAX_VA}")
    md.append("")
    md.append("## Baseline")
    md.append(
        f"- epochs={BASELINE['epochs']}, imgsz={BASELINE['imgsz']}, batch={BASELINE['batch']}, "
        f"lr0={BASELINE['lr0']}, freeze={BASELINE['freeze']}"
    )
    md.append(
        f"- Result: mAP@50={baseline['map50']:.4f}, mAP@50:95={baseline['map50_95']:.4f}, "
        f"Precision={baseline['precision']:.4f}, Recall={baseline['recall']:.4f}"
    )
    md.append("")
    md.append("## Overall Plots")
    md.append(f"![mAP by variant]({fig_paths['map50_by_variant'].as_posix()})")
    md.append("")
    md.append(f"![Delta mAP by variable]({fig_paths['delta_map50_by_variable'].as_posix()})")
    md.append("")
    md.append("## Threshold Effects (conf/iou)")
    md.append(f"![conf sensitivity]({fig_paths['conf_sensitivity'].as_posix()})")
    md.append("")
    md.append(f"![iou sensitivity]({fig_paths['iou_sensitivity'].as_posix()})")
    md.append("")
    md.append("## Variable-wise Summary (mAP@50)")
    for it in impacts:
        md.append(
            f"- {it['variable']}: best {it['best_value']} (delta {it['best_delta_vs_baseline']:+.4f}), "
            f"worst {it['worst_value']} (delta {it['worst_delta_vs_baseline']:+.4f}), "
            f"range={it['range_map50']:.4f}"
        )
    md.append("")
    md.append("## Interpretation")
    md.append("- epochs: Too low can underfit; higher can improve but must check overfitting.")
    md.append("- imgsz: Larger size may help small objects but increases computation.")
    md.append("- batch: Smaller can be noisier; larger can hurt generalization.")
    md.append("- lr0: One of the most sensitive knobs; too low slows convergence, too high can destabilize.")
    md.append("- freeze: More frozen layers are stable but can limit domain adaptation.")
    md.append("- conf: Higher conf tends to increase precision and decrease recall.")
    md.append("- iou_thr: Higher IoU threshold makes TP matching stricter, often lowering recall/F1.")
    md.append("")
    md.append("## Next Tuning Steps")
    md.append("- Step 1: Narrow search only around directions that improved mAP.")
    md.append("- Step 2: Jointly optimize interacting knobs (lr0-epochs, batch-lr0) with a 2D grid.")
    md.append("- Step 3: Repeat top 2-3 configs with 3 runs each for mean/variance stability.")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    report_path = OUT_DIR / "ablation_report_cv1.md"
    report_path.write_text("\n".join(md), encoding="utf-8")


def main():
    t0 = time.time()
    set_seed(SEED)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    RUNS_DIR.mkdir(parents=True, exist_ok=True)

    prepare_dataset()

    train_rows = []
    train_rows.append(run_train_eval("baseline", dict(BASELINE)))

    for var, values in ABLATION_GRID.items():
        for v in values:
            cfg = dict(BASELINE)
            cfg[var] = v
            train_rows.append(run_train_eval(f"{var}={v}", cfg))

    base_row = next(r for r in train_rows if r["variant"] == "baseline")
    best_model = YOLO(base_row["best_pt"])

    conf_rows = []
    for c in CONF_GRID:
        prf = compute_prf(class_eval(best_model, conf=c, iou_thr=0.5))
        conf_rows.append({"conf": c, **prf})

    iou_rows = []
    for iou in IOU_GRID:
        prf = compute_prf(class_eval(best_model, conf=0.15, iou_thr=iou))
        iou_rows.append({"iou_thr": iou, **prf})

    write_csv(train_rows, OUT_DIR / "ablation_train_summary_cv1.csv")
    write_csv(conf_rows, OUT_DIR / "ablation_conf_sensitivity_cv1.csv")
    write_csv(iou_rows, OUT_DIR / "ablation_iou_sensitivity_cv1.csv")

    fig_paths = make_figures(train_rows, conf_rows, iou_rows)
    write_report(train_rows, conf_rows, iou_rows, fig_paths)

    payload = {
        "train_summary": str((OUT_DIR / "ablation_train_summary_cv1.csv").resolve()),
        "conf_summary": str((OUT_DIR / "ablation_conf_sensitivity_cv1.csv").resolve()),
        "iou_summary": str((OUT_DIR / "ablation_iou_sensitivity_cv1.csv").resolve()),
        "report_md": str((OUT_DIR / "ablation_report_cv1.md").resolve()),
        "figures": {k: str(v.resolve()) for k, v in fig_paths.items()},
        "elapsed_sec": time.time() - t0,
    }
    with open(OUT_DIR / "ablation_manifest_cv1.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    print("[done] elapsed_sec=", round(payload["elapsed_sec"], 2))
    print("[done] report=", payload["report_md"])


if __name__ == "__main__":
    main()
