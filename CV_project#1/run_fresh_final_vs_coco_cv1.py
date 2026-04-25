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
RUNS = ROOT / "fresh_final_runs"

DATA_TRAIN = ROOT / "voc3_large" / "data.yaml"
DATA_EVAL = ROOT / "voc3" / "data.yaml"

FRESH_TAG = "fresh_final_n1"
FRESH_DIR = RUNS / FRESH_TAG
FRESH_PT = FRESH_DIR / "weights" / "best.pt"

# unified eval search space
NMS_GRID = [0.45, 0.60]
MAXDET_GRID = [50]
TP_GRID = [0.15, 0.20, 0.25, 0.30, 0.35, 0.40]
TC_GRID = [0.15, 0.20, 0.25, 0.30, 0.35]
TD_GRID = [0.15, 0.20, 0.25, 0.30, 0.35]
RAW_CONF = 0.05
MATCH_IOU = 0.5

CSV = OUT / "fresh_final_vs_coco_metrics_cv1.csv"
MANIFEST = OUT / "fresh_final_vs_coco_manifest_cv1.json"
REPORT = OUT / "fresh_final_vs_coco_report_cv1.md"
FIG_METRIC = FIG / "fresh_vs_coco_metrics_cv1.png"
FIG_ERR = FIG / "fresh_vs_coco_error_counts_cv1.png"
FIG_CM_COUNT = FIG / "fresh_vs_coco_confusion_counts_cv1.png"
FIG_CM_NORM = FIG / "fresh_vs_coco_confusion_normalized_cv1.png"

CLASSES = ["person", "car", "dog", "background"]
BG_IDX = 3


def safe_div(a: float, b: float) -> float:
    return float(a / b) if b > 0 else 0.0


def iou_xyxy(a, b) -> float:
    x1 = max(a[0], b[0]); y1 = max(a[1], b[1]); x2 = min(a[2], b[2]); y2 = min(a[3], b[3])
    iw = max(0.0, x2 - x1); ih = max(0.0, y2 - y1)
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
        "tp": int(tp), "fp": int(fp), "fn": int(fn),
        "precision": p, "recall": r, "f1": f1,
        "f1_macro": float(np.mean(fms)),
    }


def sweep_best(model_ref: str, names, imgs, labels_dir):
    best = None
    for nms in NMS_GRID:
        cache = cache_preds(model_ref, names, imgs, nms)
        for md in MAXDET_GRID:
            for tp in TP_GRID:
                for tc in TC_GRID:
                    for td in TD_GRID:
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


def build_cm(model_ref: str, names, imgs, labels_dir, cfg):
    model = YOLO(model_ref)
    cm = np.zeros((4, 4), dtype=np.int64)

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
                t = cfg[f"thr_{cname}"]
                if s >= t:
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

        used_g = set(); used_p = set()
        for _v, gi, pi in pairs:
            if gi in used_g or pi in used_p:
                continue
            used_g.add(gi); used_p.add(pi)
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


def draw_cm_pair(cm_a, cm_b, title_a, title_b, out_path, normalized=False):
    fig, ax = plt.subplots(1, 2, figsize=(14, 5.5), dpi=170)
    for i, (cm, t) in enumerate([(cm_a, title_a), (cm_b, title_b)]):
        im = ax[i].imshow(cm, cmap="Blues", vmin=0.0 if normalized else None, vmax=1.0 if normalized else None)
        ax[i].set_title(t, fontsize=11, weight="bold")
        ax[i].set_xticks(range(4)); ax[i].set_yticks(range(4))
        ax[i].set_xticklabels(CLASSES, rotation=20, ha="right")
        ax[i].set_yticklabels(CLASSES)
        ax[i].set_xlabel("Predicted")
        ax[i].set_ylabel("Ground Truth")
        vmax = float(cm.max()) if cm.size else 0.0
        for r in range(4):
            for c in range(4):
                txt = f"{cm[r,c]:.2f}" if normalized else str(int(cm[r,c]))
                thr = 0.5 if normalized else (0.45 * vmax if vmax > 0 else 0.0)
                ax[i].text(c, r, txt, ha="center", va="center", fontsize=9, color=("white" if cm[r,c] > thr else "black"))
    fig.suptitle("COCO vs Fresh Fine-tuned (4x4 confusion)", fontsize=13, weight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out_path)
    plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    RUNS.mkdir(parents=True, exist_ok=True)

    if FRESH_DIR.exists():
        shutil.rmtree(FRESH_DIR)

    # 1) fresh training (new run, no resume)
    model = YOLO("yolov8n.pt")
    model.train(
        data=str(DATA_TRAIN),
        model="yolov8n.pt",
        epochs=20,
        imgsz=640,
        batch=16,
        lr0=0.00025,
        freeze=10,
        mosaic=0.2,
        mixup=0.0,
        close_mosaic=2,
        project=str(RUNS),
        name=FRESH_TAG,
        exist_ok=True,
        verbose=False,
        workers=0,
        seed=42,
        deterministic=True,
        resume=False,
    )

    if not FRESH_PT.exists():
        raise FileNotFoundError(FRESH_PT)

    # 2) unified evaluation + best operating search
    names, imgs, labels = load_eval_data()

    coco_best = sweep_best("yolov8n.pt", names, imgs, labels)
    fresh_best = sweep_best(str(FRESH_PT), names, imgs, labels)

    df = pd.DataFrame([
        {"model": "COCO", **coco_best},
        {"model": "FreshFineTune", **fresh_best},
    ])
    df.to_csv(CSV, index=False, encoding="utf-8-sig")

    # 3) visuals
    metrics = ["precision", "recall", "f1", "f1_macro"]
    x = np.arange(len(metrics))
    w = 0.35
    fig, ax = plt.subplots(figsize=(8.5, 5), dpi=150)
    ax.bar(x - w / 2, [coco_best[m] for m in metrics], width=w, label="COCO", color="#4e79a7")
    ax.bar(x + w / 2, [fresh_best[m] for m in metrics], width=w, label="FreshFineTune", color="#f28e2b")
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.set_ylim(0, 1.0)
    ax.set_title("Metric Comparison (best operating point)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG_METRIC)
    plt.close(fig)

    counts = ["tp", "fp", "fn"]
    x2 = np.arange(len(counts))
    fig2, ax2 = plt.subplots(figsize=(8.5, 5), dpi=150)
    ax2.bar(x2 - w / 2, [coco_best[c] for c in counts], width=w, label="COCO", color="#4e79a7")
    ax2.bar(x2 + w / 2, [fresh_best[c] for c in counts], width=w, label="FreshFineTune", color="#f28e2b")
    ax2.set_xticks(x2)
    ax2.set_xticklabels(counts)
    ax2.set_title("Error Count Comparison (best operating point)")
    ax2.legend()
    fig2.tight_layout()
    fig2.savefig(FIG_ERR)
    plt.close(fig2)

    coco_cm, coco_norm = build_cm("yolov8n.pt", names, imgs, labels, coco_best)
    fresh_cm, fresh_norm = build_cm(str(FRESH_PT), names, imgs, labels, fresh_best)

    draw_cm_pair(coco_cm, fresh_cm, "COCO (counts)", "FreshFineTune (counts)", FIG_CM_COUNT, normalized=False)
    draw_cm_pair(coco_norm, fresh_norm, "COCO (normalized)", "FreshFineTune (normalized)", FIG_CM_NORM, normalized=True)

    # 4) report
    lines = [
        "# Fresh Fine-tune vs COCO Report",
        "",
        "## Summary",
        "- 이 실험은 기존 학습 산출물을 사용하지 않고 새 폴더에서 신규 학습(no-resume)으로 수행함.",
        f"- fresh model: `{FRESH_PT}`",
        "- 비교는 동일 eval 프로토콜(같은 데이터/같은 운영값 탐색 공간)로 수행함.",
        "",
        "## Best Operating Results",
        f"- COCO: F1={coco_best['f1']:.4f}, P={coco_best['precision']:.4f}, R={coco_best['recall']:.4f}, TP={coco_best['tp']}, FP={coco_best['fp']}, FN={coco_best['fn']}",
        f"  - params: nms_iou={coco_best['nms_iou']}, max_det={coco_best['max_det']}, thr=({coco_best['thr_person']:.2f},{coco_best['thr_car']:.2f},{coco_best['thr_dog']:.2f})",
        f"- FreshFineTune: F1={fresh_best['f1']:.4f}, P={fresh_best['precision']:.4f}, R={fresh_best['recall']:.4f}, TP={fresh_best['tp']}, FP={fresh_best['fp']}, FN={fresh_best['fn']}",
        f"  - params: nms_iou={fresh_best['nms_iou']}, max_det={fresh_best['max_det']}, thr=({fresh_best['thr_person']:.2f},{fresh_best['thr_car']:.2f},{fresh_best['thr_dog']:.2f})",
        f"- delta(Fresh-COCO): dF1={fresh_best['f1']-coco_best['f1']:+.4f}, dTP={fresh_best['tp']-coco_best['tp']:+d}, dFP={fresh_best['fp']-coco_best['fp']:+d}",
        "",
        "## Graphs",
        "![metric_compare](C:/Users/lmhst/git/roboticsCV/CV_project#1/outputs/figures/fresh_vs_coco_metrics_cv1.png)",
        "",
        "![error_compare](C:/Users/lmhst/git/roboticsCV/CV_project#1/outputs/figures/fresh_vs_coco_error_counts_cv1.png)",
        "",
        "## Confusion Matrix (4x4, background 포함)",
        "![cm_counts](C:/Users/lmhst/git/roboticsCV/CV_project#1/outputs/figures/fresh_vs_coco_confusion_counts_cv1.png)",
        "",
        "![cm_norm](C:/Users/lmhst/git/roboticsCV/CV_project#1/outputs/figures/fresh_vs_coco_confusion_normalized_cv1.png)",
    ]
    REPORT.write_text("\n".join(lines), encoding="utf-8")

    manifest = {
        "fresh_pt": str(FRESH_PT),
        "csv": str(CSV),
        "report": str(REPORT),
        "fig_metric": str(FIG_METRIC),
        "fig_error": str(FIG_ERR),
        "fig_cm_count": str(FIG_CM_COUNT),
        "fig_cm_norm": str(FIG_CM_NORM),
        "coco_best": coco_best,
        "fresh_best": fresh_best,
        "delta": {
            "f1": float(fresh_best["f1"] - coco_best["f1"]),
            "tp": int(fresh_best["tp"] - coco_best["tp"]),
            "fp": int(fresh_best["fp"] - coco_best["fp"]),
        },
    }
    MANIFEST.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print("[saved]", CSV)
    print("[saved]", REPORT)
    print("[saved]", MANIFEST)
    print("[result]", json.dumps({"coco_best": coco_best, "fresh_best": fresh_best, "delta_f1": fresh_best["f1"] - coco_best["f1"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
