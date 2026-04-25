from __future__ import annotations

import json
import shutil
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from ultralytics import YOLO

ROOT = Path(r"C:\Users\lmhst\git\roboticsCV\CV_project#1")
OUT = ROOT / "outputs"
FIG = OUT / "figures"
RUNS = ROOT / "two_stage_runs"

DATA_TRAIN = ROOT / "voc3_large" / "data.yaml"
DATA_EVAL = ROOT / "voc3" / "data.yaml"

RESULT_CSV = OUT / "two_stage_coco_strategy_results_cv1.csv"
REPORT_MD = OUT / "two_stage_coco_strategy_report_cv1.md"
MANIFEST = OUT / "two_stage_coco_strategy_manifest_cv1.json"
FIG_METRICS = FIG / "two_stage_coco_strategy_metrics_cv1.png"
FIG_ERR = FIG / "two_stage_coco_strategy_error_cv1.png"
FIG_CM_COUNT = FIG / "two_stage_coco_strategy_confusion_counts_cv1.png"
FIG_CM_NORM = FIG / "two_stage_coco_strategy_confusion_normalized_cv1.png"

CLASSES = ["person", "car", "dog", "bg/other"]
BG_IDX = 3
RAW_CONF = 0.05
MATCH_IOU = 0.5


def safe_div(a: float, b: float) -> float:
    return float(a / b) if b > 0 else 0.0


def iou_xyxy(a, b) -> float:
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    iw = max(0.0, x2 - x1)
    ih = max(0.0, y2 - y1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    aa = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    bb = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    uu = aa + bb - inter
    return float(inter / uu) if uu > 0 else 0.0


def load_eval_data():
    cfg = yaml.safe_load(DATA_EVAL.read_text(encoding="utf-8"))
    names = cfg["names"]
    if isinstance(names, dict):
        names = [names[i] for i in sorted(names.keys())]
    val = Path(cfg["val"])
    if not val.is_absolute():
        val = (DATA_EVAL.parent / val).resolve()
    labels = val.parent.parent / "labels" / "val"
    imgs = sorted([p for p in val.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}])
    return names, imgs, labels


def load_gt(img_path: Path, labels_dir: Path):
    im = cv2.imread(str(img_path))
    h, w = im.shape[:2]
    lp = labels_dir / f"{img_path.stem}.txt"
    out = []
    if lp.exists():
        for ln in lp.read_text(encoding="utf-8").splitlines():
            ln = ln.strip()
            if not ln:
                continue
            p = ln.split()
            c = int(float(p[0]))
            x, y, bw, bh = [float(v) for v in p[1:5]]
            out.append((c, [(x - bw / 2) * w, (y - bh / 2) * h, (x + bw / 2) * w, (y + bh / 2) * h]))
    return out


def cache_preds(model_ref: str, names: list[str], imgs: list[Path], nms_iou: float):
    m = YOLO(model_ref)
    name2idx = {n: i for i, n in enumerate(names)}
    rs = m.predict(
        source=[str(p) for p in imgs],
        conf=RAW_CONF,
        iou=nms_iou,
        max_det=300,
        imgsz=640,
        verbose=False,
        save=False,
        device=0,
    )
    out = {}
    for p, r in zip(imgs, rs):
        arr = []
        if r.boxes is not None and len(r.boxes) > 0:
            cls = r.boxes.cls.cpu().numpy().astype(int)
            conf = r.boxes.conf.cpu().numpy().astype(float)
            box = r.boxes.xyxy.cpu().numpy().astype(float)
            for c, s, b in zip(cls, conf, box):
                cname = m.names[int(c)]
                if cname not in name2idx:
                    continue
                arr.append((name2idx[cname], float(s), b.tolist()))
        out[p.name] = arr
    return out


def eval_with_thr(names, imgs, labels_dir, cache, thr, max_det):
    tp = fp = fn = 0
    tp_c = np.zeros(3, dtype=np.int64)
    fp_c = np.zeros(3, dtype=np.int64)
    fn_c = np.zeros(3, dtype=np.int64)

    for img in imgs:
        gts = load_gt(img, labels_dir)
        prs = [(c, s, b) for (c, s, b) in cache[img.name] if s >= thr[names[c]]]
        prs.sort(key=lambda x: x[1], reverse=True)
        prs = prs[:max_det]
        used = set()

        for gc, gb in gts:
            bj = -1
            bi = 0.0
            for j, (pc, _ps, pb) in enumerate(prs):
                if j in used or pc != gc:
                    continue
                v = iou_xyxy(gb, pb)
                if v >= MATCH_IOU and v > bi:
                    bi = v
                    bj = j
            if bj >= 0:
                used.add(bj)
                tp += 1
                tp_c[gc] += 1
            else:
                fn += 1
                fn_c[gc] += 1

        for j, (pc, _ps, _pb) in enumerate(prs):
            if j not in used:
                fp += 1
                fp_c[pc] += 1

    p = safe_div(tp, tp + fp)
    r = safe_div(tp, tp + fn)
    f1 = safe_div(2 * p * r, p + r)
    fms = []
    for i in range(3):
        pp = safe_div(int(tp_c[i]), int(tp_c[i] + fp_c[i]))
        rr = safe_div(int(tp_c[i]), int(tp_c[i] + fn_c[i]))
        ff = safe_div(2 * pp * rr, pp + rr)
        fms.append(ff)
    return {
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "precision": p,
        "recall": r,
        "f1": f1,
        "f1_macro": float(np.mean(fms)),
    }


def sweep_best(model_ref: str, names: list[str], imgs: list[Path], labels_dir: Path) -> dict:
    nms_grid = [0.45, 0.60]
    maxdet_grid = [30, 50]
    tp_grid = [0.15, 0.20, 0.25, 0.30, 0.35, 0.40]
    tc_grid = [0.15, 0.20, 0.25, 0.30, 0.35]
    td_grid = [0.15, 0.20, 0.25, 0.30, 0.35]

    best = None
    for nms in nms_grid:
        cache = cache_preds(model_ref, names, imgs, nms_iou=nms)
        for md in maxdet_grid:
            for tp in tp_grid:
                for tc in tc_grid:
                    for td in td_grid:
                        thr = {"person": tp, "car": tc, "dog": td}
                        m = eval_with_thr(names, imgs, labels_dir, cache, thr, md)
                        row = {
                            "nms_iou": nms,
                            "max_det": md,
                            "thr_person": tp,
                            "thr_car": tc,
                            "thr_dog": td,
                            **m,
                        }
                        if best is None or row["f1"] > best["f1"] or (abs(row["f1"] - best["f1"]) < 1e-12 and row["fp"] < best["fp"]):
                            best = row
    return best


def run_two_stage(tag: str, s1: dict, s2: dict) -> Path:
    exp1 = RUNS / f"{tag}_stage1"
    exp2 = RUNS / f"{tag}_stage2"
    if exp1.exists():
        shutil.rmtree(exp1)
    if exp2.exists():
        shutil.rmtree(exp2)

    # Stage 1: COCO pretrained load + head replacement(auto by nc mismatch) + backbone freeze + head-only tuning
    m1 = YOLO("yolov8n.pt")
    m1.train(
        data=str(DATA_TRAIN),
        model="yolov8n.pt",
        project=str(RUNS),
        name=f"{tag}_stage1",
        exist_ok=True,
        verbose=False,
        workers=0,
        seed=42,
        deterministic=True,
        resume=False,
        optimizer="AdamW",
        **s1,
    )
    s1_best = exp1 / "weights" / "best.pt"
    if not s1_best.exists():
        s1_best = exp1 / "weights" / "last.pt"

    # Stage 2: full fine-tuning (unfreeze all)
    m2 = YOLO(str(s1_best))
    m2.train(
        data=str(DATA_TRAIN),
        model=str(s1_best),
        project=str(RUNS),
        name=f"{tag}_stage2",
        exist_ok=True,
        verbose=False,
        workers=0,
        seed=42,
        deterministic=True,
        resume=False,
        optimizer="AdamW",
        **s2,
    )
    s2_best = exp2 / "weights" / "best.pt"
    if not s2_best.exists():
        s2_best = exp2 / "weights" / "last.pt"
    return s2_best


def build_cm(model_ref: str, names: list[str], imgs: list[Path], labels_dir: Path, cfg: dict) -> tuple[np.ndarray, np.ndarray]:
    model = YOLO(model_ref)
    cm = np.zeros((4, 4), dtype=np.int64)
    thr = {"person": cfg["thr_person"], "car": cfg["thr_car"], "dog": cfg["thr_dog"]}

    for img in imgs:
        gts = load_gt(img, labels_dir)
        r = model.predict(
            source=str(img),
            conf=RAW_CONF,
            iou=float(cfg["nms_iou"]),
            max_det=300,
            imgsz=640,
            verbose=False,
            save=False,
            device=0,
        )[0]
        prs = []
        if r.boxes is not None and len(r.boxes) > 0:
            cls = r.boxes.cls.cpu().numpy().astype(int)
            conf = r.boxes.conf.cpu().numpy().astype(float)
            box = r.boxes.xyxy.cpu().numpy().astype(float)
            for c, s, b in zip(cls, conf, box):
                cname = model.names[int(c)]
                if cname not in names:
                    continue
                dc = names.index(cname)
                if s >= thr[cname]:
                    prs.append((dc, float(s), b.tolist()))
        prs.sort(key=lambda x: x[1], reverse=True)
        prs = prs[: int(cfg["max_det"])]

        pairs = []
        for gi, gt in enumerate(gts):
            for pi, pr in enumerate(prs):
                v = iou_xyxy(gt[1], pr[2])
                if v >= MATCH_IOU:
                    pairs.append((v, gi, pi))
        pairs.sort(key=lambda x: x[0], reverse=True)

        used_g = set()
        used_p = set()
        for _v, gi, pi in pairs:
            if gi in used_g or pi in used_p:
                continue
            used_g.add(gi)
            used_p.add(pi)
            cm[gts[gi][0], prs[pi][0]] += 1

        for gi, gt in enumerate(gts):
            if gi not in used_g:
                cm[gt[0], BG_IDX] += 1
        for pi, pr in enumerate(prs):
            if pi not in used_p:
                cm[BG_IDX, pr[0]] += 1

    norm = np.zeros_like(cm, dtype=np.float64)
    rs = cm.sum(axis=1, keepdims=True)
    np.divide(cm, np.maximum(rs, 1), out=norm, where=rs > 0)
    return cm, norm


def draw_cm_pair(cm_a, cm_b, title_a: str, title_b: str, out_path: Path, normalized: bool = False):
    fig, ax = plt.subplots(1, 2, figsize=(14, 5.5), dpi=170)
    for i, (cm, t) in enumerate([(cm_a, title_a), (cm_b, title_b)]):
        im = ax[i].imshow(cm, cmap="Blues", vmin=0.0 if normalized else None, vmax=1.0 if normalized else None)
        ax[i].set_title(t, fontsize=11, weight="bold")
        ax[i].set_xticks(range(4))
        ax[i].set_yticks(range(4))
        ax[i].set_xticklabels(CLASSES, rotation=20, ha="right")
        ax[i].set_yticklabels(CLASSES)
        ax[i].set_xlabel("Predicted")
        ax[i].set_ylabel("Ground Truth")
        vmax = float(cm.max()) if cm.size else 0.0
        for r in range(4):
            for c in range(4):
                txt = f"{cm[r, c]:.2f}" if normalized else str(int(cm[r, c]))
                thr = 0.5 if normalized else (0.45 * vmax if vmax > 0 else 0.0)
                color = "white" if cm[r, c] > thr else "black"
                ax[i].text(c, r, txt, ha="center", va="center", fontsize=9, color=color)
        fig.colorbar(im, ax=ax[i], shrink=0.82)
    fig.suptitle("COCO vs Two-Stage Fine-tuned (4x4 confusion)", fontsize=13, weight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out_path)
    plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    RUNS.mkdir(parents=True, exist_ok=True)

    names, imgs, labels = load_eval_data()
    coco_best = sweep_best("yolov8n.pt", names, imgs, labels)
    print("[coco_best]", coco_best)

    # Strategy variants: 1) head-only tuning (freeze backbone), 2) full fine-tuning
    candidates = [
        {
            "tag": "two_stage_a",
            "stage1": {"epochs": 8, "imgsz": 640, "batch": 16, "lr0": 0.0010, "freeze": 10, "mosaic": 0.2, "mixup": 0.0, "close_mosaic": 2},
            "stage2": {"epochs": 18, "imgsz": 640, "batch": 16, "lr0": 0.00018, "freeze": 0, "mosaic": 0.1, "mixup": 0.0, "close_mosaic": 2},
        },
        {
            "tag": "two_stage_b",
            "stage1": {"epochs": 10, "imgsz": 640, "batch": 16, "lr0": 0.0008, "freeze": 10, "mosaic": 0.15, "mixup": 0.0, "close_mosaic": 2},
            "stage2": {"epochs": 22, "imgsz": 640, "batch": 16, "lr0": 0.00012, "freeze": 0, "mosaic": 0.1, "mixup": 0.0, "close_mosaic": 2},
        },
        {
            "tag": "two_stage_c",
            "stage1": {"epochs": 6, "imgsz": 640, "batch": 16, "lr0": 0.0012, "freeze": 10, "mosaic": 0.25, "mixup": 0.0, "close_mosaic": 2},
            "stage2": {"epochs": 20, "imgsz": 640, "batch": 16, "lr0": 0.00022, "freeze": 0, "mosaic": 0.1, "mixup": 0.0, "close_mosaic": 2},
        },
    ]

    rows = []
    for c in candidates:
        print("[train-two-stage]", c["tag"])
        pt = run_two_stage(c["tag"], c["stage1"], c["stage2"])
        best = sweep_best(str(pt), names, imgs, labels)
        row = {
            "tag": c["tag"],
            "best_pt": str(pt),
            "f1": best["f1"],
            "precision": best["precision"],
            "recall": best["recall"],
            "f1_macro": best["f1_macro"],
            "tp": best["tp"],
            "fp": best["fp"],
            "fn": best["fn"],
            "nms_iou": best["nms_iou"],
            "max_det": best["max_det"],
            "thr_person": best["thr_person"],
            "thr_car": best["thr_car"],
            "thr_dog": best["thr_dog"],
            "delta_f1_vs_coco": best["f1"] - coco_best["f1"],
        }
        rows.append(row)
        print("[result]", row["tag"], "f1", row["f1"], "dF1", row["delta_f1_vs_coco"])

    df = pd.DataFrame(rows).sort_values(["f1", "fp"], ascending=[False, True]).reset_index(drop=True)
    df.to_csv(RESULT_CSV, index=False, encoding="utf-8-sig")
    best_row = df.iloc[0].to_dict()

    # figures
    metrics = ["precision", "recall", "f1", "f1_macro"]
    x = np.arange(len(metrics))
    w = 0.2
    fig, ax = plt.subplots(figsize=(10, 5), dpi=160)
    ax.bar(x - 1.5 * w, [coco_best[m] for m in metrics], width=w, label="COCO", color="#4e79a7")
    bestvals = [best_row[m] for m in metrics]
    ax.bar(x - 0.5 * w, bestvals, width=w, label="TwoStageBest", color="#f28e2b")
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.set_ylim(0, 1.0)
    ax.set_title("COCO vs Two-Stage Best (metrics)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG_METRICS)
    plt.close(fig)

    fig2, ax2 = plt.subplots(figsize=(10, 5), dpi=160)
    counts = ["tp", "fp", "fn"]
    x2 = np.arange(len(counts))
    ax2.bar(x2 - 0.2, [coco_best[k] for k in counts], width=0.4, label="COCO", color="#4e79a7")
    ax2.bar(x2 + 0.2, [best_row[k] for k in counts], width=0.4, label="TwoStageBest", color="#f28e2b")
    ax2.set_xticks(x2)
    ax2.set_xticklabels(counts)
    ax2.set_title("COCO vs Two-Stage Best (TP/FP/FN)")
    ax2.legend()
    fig2.tight_layout()
    fig2.savefig(FIG_ERR)
    plt.close(fig2)

    coco_cm, coco_norm = build_cm("yolov8n.pt", names, imgs, labels, coco_best)
    best_cm, best_norm = build_cm(str(best_row["best_pt"]), names, imgs, labels, best_row)
    draw_cm_pair(coco_cm, best_cm, "COCO (counts)", "TwoStageBest (counts)", FIG_CM_COUNT, normalized=False)
    draw_cm_pair(coco_norm, best_norm, "COCO (normalized)", "TwoStageBest (normalized)", FIG_CM_NORM, normalized=True)

    lines = [
        "# Two-Stage COCO Strategy Report (CV1)",
        "",
        "## Applied Strategy",
        "- 1) COCO pretrained load",
        "- 2) Detection head replacement by nc=3 (automatic during training)",
        "- 3) Backbone freeze (freeze=10) and head-focused tuning (Stage1)",
        "- 4) Full fine-tuning with freeze=0 (Stage2)",
        "",
        "## Baseline (COCO best operating)",
        f"- F1={coco_best['f1']:.4f}, P={coco_best['precision']:.4f}, R={coco_best['recall']:.4f}, TP={coco_best['tp']}, FP={coco_best['fp']}, FN={coco_best['fn']}",
        f"- params: nms_iou={coco_best['nms_iou']}, max_det={coco_best['max_det']}, thr=({coco_best['thr_person']:.2f},{coco_best['thr_car']:.2f},{coco_best['thr_dog']:.2f})",
        "",
        "## Best Two-Stage Result",
        f"- tag={best_row['tag']}, F1={best_row['f1']:.4f}, P={best_row['precision']:.4f}, R={best_row['recall']:.4f}, TP={int(best_row['tp'])}, FP={int(best_row['fp'])}, FN={int(best_row['fn'])}",
        f"- params: nms_iou={best_row['nms_iou']}, max_det={int(best_row['max_det'])}, thr=({best_row['thr_person']:.2f},{best_row['thr_car']:.2f},{best_row['thr_dog']:.2f})",
        f"- delta(F1 vs COCO)={best_row['delta_f1_vs_coco']:+.4f}",
        "",
        "## Visuals",
        "![metrics](C:/Users/lmhst/git/roboticsCV/CV_project#1/outputs/figures/two_stage_coco_strategy_metrics_cv1.png)",
        "",
        "![error](C:/Users/lmhst/git/roboticsCV/CV_project#1/outputs/figures/two_stage_coco_strategy_error_cv1.png)",
        "",
        "![cm_count](C:/Users/lmhst/git/roboticsCV/CV_project#1/outputs/figures/two_stage_coco_strategy_confusion_counts_cv1.png)",
        "",
        "![cm_norm](C:/Users/lmhst/git/roboticsCV/CV_project#1/outputs/figures/two_stage_coco_strategy_confusion_normalized_cv1.png)",
    ]
    REPORT_MD.write_text("\n".join(lines), encoding="utf-8")

    manifest = {
        "result_csv": str(RESULT_CSV),
        "report_md": str(REPORT_MD),
        "fig_metrics": str(FIG_METRICS),
        "fig_error": str(FIG_ERR),
        "fig_cm_count": str(FIG_CM_COUNT),
        "fig_cm_norm": str(FIG_CM_NORM),
        "coco_best": coco_best,
        "best_two_stage": best_row,
        "all_rows": rows,
    }
    MANIFEST.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print("[saved]", RESULT_CSV)
    print("[saved]", REPORT_MD)
    print("[saved]", MANIFEST)
    print("[summary]", json.dumps({"coco_f1": coco_best["f1"], "best_tag": best_row["tag"], "best_f1": best_row["f1"], "delta_f1": best_row["delta_f1_vs_coco"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
