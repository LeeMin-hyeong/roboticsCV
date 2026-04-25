from __future__ import annotations

import json
import random
import shutil
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from scipy.stats import ttest_1samp, wilcoxon
from ultralytics import YOLO

ROOT = Path(r"C:\Users\lmhst\git\roboticsCV\CV_project#1")
OUT = ROOT / "outputs"
FIG = OUT / "figures"
RUNS = ROOT / "ordered_plan_runs"
DS_ROOT = ROOT / "datasets_ordered_plan"

VOC3 = ROOT / "voc3"
VOC3_LARGE = ROOT / "voc3_large"
EVAL_DATA = VOC3 / "data.yaml"
BASE_TRAIN_DATA = VOC3_LARGE / "data.yaml"

VOC2007 = ROOT / "data" / "VOCdevkit" / "VOC2007"
VOC_IMG = VOC2007 / "JPEGImages"
VOC_ANN = VOC2007 / "Annotations"
VOC_SPLIT = VOC2007 / "ImageSets" / "Main"

TARGET_CLASSES = ["person", "car", "dog"]
BG_LABEL = "bg/other"
BG_IDX = 3

RAW_CONF = 0.05
MATCH_IOU = 0.5

PLAN_MD = OUT / "ordered_plan_execution_cv1.md"
CANDIDATE_CSV = OUT / "ordered_plan_candidate_results_cv1.csv"
SEED_CSV = OUT / "ordered_plan_seedrepeat_cv1.csv"
REPORT_MD = OUT / "ordered_plan_report_cv1.md"
MANIFEST = OUT / "ordered_plan_manifest_cv1.json"

FIG_METRICS = FIG / "ordered_plan_candidate_metrics_cv1.png"
FIG_DELTA = FIG / "ordered_plan_seed_delta_cv1.png"
FIG_CM_COUNT = FIG / "ordered_plan_confusion_counts_cv1.png"
FIG_CM_NORM = FIG / "ordered_plan_confusion_normalized_cv1.png"


@dataclass
class EvalData:
    names: list[str]
    images: list[Path]
    labels_dir: Path
    gt_by_name: dict[str, list[tuple[int, list[float]]]]
    hw_by_name: dict[str, tuple[int, int]]


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


def write_plan_document() -> None:
    lines = [
        "# Ordered Optimization Plan (CV1)",
        "",
        "1. Fix baseline and objective",
        "- Baseline reference: COCO-pretrained `yolov8n.pt`.",
        "- Primary metric: F1 on voc3 val (class-matched IoU>=0.5).",
        "- Tie-breaker: lower FP, then higher TP.",
        "",
        "2. Data reconstruction",
        "- Add background negatives from VOC2007 images that do not contain person/car/dog.",
        "- Test low negative ratios and class-balance oversampling for underrepresented classes.",
        "",
        "3. Candidate training (from scratch)",
        "- Keep model fixed to `yolov8n`.",
        "- Train each candidate from pretrained COCO weights with `resume=False`.",
        "",
        "4. Operating tuning",
        "- Tune class thresholds + nms_iou + max_det.",
        "- Add person shape filter (min_area_frac and aspect ratio range).",
        "",
        "5. Statistical test",
        "- Repeat best candidate for 5 seeds.",
        "- One-sample tests on delta F1 vs COCO baseline (H1: delta>0).",
    ]
    PLAN_MD.write_text("\n".join(lines), encoding="utf-8")


def parse_split_ids(split_name: str) -> list[str]:
    p = VOC_SPLIT / f"{split_name}.txt"
    if not p.exists():
        return []
    out = []
    for ln in p.read_text(encoding="utf-8").splitlines():
        ln = ln.strip()
        if not ln:
            continue
        out.append(ln.split()[0])
    return out


def anno_objects(ann_xml: Path) -> set[str]:
    tree = ET.parse(ann_xml)
    root = tree.getroot()
    objs = set()
    for obj in root.findall("object"):
        nm = obj.findtext("name")
        if nm:
            objs.add(nm.strip())
    return objs


def collect_negative_pool() -> dict[str, list[str]]:
    pool = {"train": [], "val": []}
    split_map = {
        "train": parse_split_ids("trainval"),
        "val": parse_split_ids("test"),
    }
    target_set = set(TARGET_CLASSES)
    for split, ids in split_map.items():
        for vid in ids:
            ann = VOC_ANN / f"{vid}.xml"
            img = VOC_IMG / f"{vid}.jpg"
            if not ann.exists() or not img.exists():
                continue
            objs = anno_objects(ann)
            if len(objs.intersection(target_set)) == 0:
                pool[split].append(img.name)
    return pool


def parse_label_file(label_path: Path) -> list[tuple[int, list[float]]]:
    out = []
    if not label_path.exists():
        return out
    for ln in label_path.read_text(encoding="utf-8").splitlines():
        ln = ln.strip()
        if not ln:
            continue
        p = ln.split()
        cls = int(float(p[0]))
        xywh = [float(v) for v in p[1:5]]
        out.append((cls, xywh))
    return out


def image_contains_cls(label_path: Path, cls_idx: int) -> bool:
    for cls, _xywh in parse_label_file(label_path):
        if cls == cls_idx:
            return True
    return False


def copy_base_dataset(base_root: Path, dst_root: Path) -> None:
    if dst_root.exists():
        shutil.rmtree(dst_root)
    for split in ["train", "val"]:
        (dst_root / "images" / split).mkdir(parents=True, exist_ok=True)
        (dst_root / "labels" / split).mkdir(parents=True, exist_ok=True)
        for ip in sorted((base_root / "images" / split).glob("*.*")):
            if ip.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp"}:
                continue
            lp = base_root / "labels" / split / f"{ip.stem}.txt"
            if not lp.exists():
                continue
            shutil.copy2(ip, dst_root / "images" / split / ip.name)
            shutil.copy2(lp, dst_root / "labels" / split / lp.name)


def add_negatives(dst_root: Path, neg_pool: dict[str, list[str]], ratio: float, seed: int) -> dict[str, int]:
    rng = random.Random(seed)
    meta = {}
    for split in ["train", "val"]:
        img_dir = dst_root / "images" / split
        lb_dir = dst_root / "labels" / split
        pos_n = len(list(img_dir.glob("*.jpg")))
        need_n = int(round(pos_n * ratio))
        pool = list(neg_pool[split])
        rng.shuffle(pool)
        add_n = 0
        for fn in pool:
            if add_n >= need_n:
                break
            src = VOC_IMG / fn
            if not src.exists():
                continue
            stem = Path(fn).stem
            dst_img = img_dir / f"neg_{stem}.jpg"
            dst_lb = lb_dir / f"neg_{stem}.txt"
            if dst_img.exists():
                continue
            shutil.copy2(src, dst_img)
            dst_lb.write_text("", encoding="utf-8")
            add_n += 1
        meta[f"{split}_pos"] = pos_n
        meta[f"{split}_neg_added"] = add_n
    return meta


def collect_images_with_cls(labels_dir: Path, cls_idx: int) -> list[str]:
    out = []
    for lp in sorted(labels_dir.glob("*.txt")):
        if lp.stem.startswith("neg_"):
            continue
        if image_contains_cls(lp, cls_idx):
            out.append(lp.stem)
    return out


def count_presence(labels_dir: Path) -> dict[int, int]:
    cnt = {0: 0, 1: 0, 2: 0}
    for lp in sorted(labels_dir.glob("*.txt")):
        seen = set()
        for cls, _xywh in parse_label_file(lp):
            if cls in cnt:
                seen.add(cls)
        for c in seen:
            cnt[c] += 1
    return cnt


def duplicate_for_balance(dst_root: Path, target_fraction: float, seed: int) -> dict[str, int]:
    rng = random.Random(seed)
    train_img = dst_root / "images" / "train"
    train_lb = dst_root / "labels" / "train"
    presence = count_presence(train_lb)
    max_presence = max(presence.values())
    target = int(round(max_presence * target_fraction))
    added = {"dup_car": 0, "dup_dog": 0}

    for cls_idx, cls_name in [(1, "car"), (2, "dog")]:
        cur = presence.get(cls_idx, 0)
        if cur >= target:
            continue
        pool = collect_images_with_cls(train_lb, cls_idx)
        if len(pool) == 0:
            continue
        need = target - cur
        for i in range(need):
            stem = rng.choice(pool)
            src_img = train_img / f"{stem}.jpg"
            src_lb = train_lb / f"{stem}.txt"
            if not src_img.exists() or not src_lb.exists():
                continue
            dst_stem = f"{stem}_dup_{cls_name}_{i:03d}"
            dst_img = train_img / f"{dst_stem}.jpg"
            dst_lb = train_lb / f"{dst_stem}.txt"
            if dst_img.exists():
                continue
            shutil.copy2(src_img, dst_img)
            shutil.copy2(src_lb, dst_lb)
            added[f"dup_{cls_name}"] += 1
    return added


def write_data_yaml(ds_root: Path) -> Path:
    y = ds_root / "data.yaml"
    data = {
        "path": str(ds_root.resolve()),
        "train": "images/train",
        "val": "images/val",
        "nc": 3,
        "names": TARGET_CLASSES,
    }
    y.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    return y


def build_dataset_variant(tag: str, neg_ratio: float, balance_fraction: float, neg_pool: dict[str, list[str]], seed: int) -> dict:
    ds_dir = DS_ROOT / tag
    copy_base_dataset(VOC3_LARGE, ds_dir)
    neg_meta = add_negatives(ds_dir, neg_pool, ratio=neg_ratio, seed=seed + 11)
    bal_meta = {"dup_car": 0, "dup_dog": 0}
    if balance_fraction > 0:
        bal_meta = duplicate_for_balance(ds_dir, target_fraction=balance_fraction, seed=seed + 29)
    y = write_data_yaml(ds_dir)
    return {
        "tag": tag,
        "data_yaml": str(y),
        "neg_ratio": neg_ratio,
        "balance_fraction": balance_fraction,
        **neg_meta,
        **bal_meta,
    }


def load_eval_data() -> EvalData:
    cfg = yaml.safe_load(EVAL_DATA.read_text(encoding="utf-8"))
    names = cfg["names"]
    if isinstance(names, dict):
        names = [names[i] for i in sorted(names.keys())]
    val = Path(cfg["val"])
    if not val.is_absolute():
        val = (EVAL_DATA.parent / val).resolve()
    labels = val.parent.parent / "labels" / "val"
    images = sorted([p for p in val.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}])

    gt_by_name = {}
    hw_by_name = {}
    for ip in images:
        im = cv2.imread(str(ip))
        if im is None:
            continue
        h, w = im.shape[:2]
        hw_by_name[ip.name] = (h, w)
        lp = labels / f"{ip.stem}.txt"
        gts = []
        for cls, xywh in parse_label_file(lp):
            x, y, bw, bh = xywh
            xyxy = [(x - bw / 2) * w, (y - bh / 2) * h, (x + bw / 2) * w, (y + bh / 2) * h]
            gts.append((cls, xyxy))
        gt_by_name[ip.name] = gts
    return EvalData(names=names, images=images, labels_dir=labels, gt_by_name=gt_by_name, hw_by_name=hw_by_name)


def train_candidate(cfg: dict, seed: int, suffix: str = "") -> Path:
    tag = cfg["tag"] + suffix
    exp = RUNS / tag
    if exp.exists():
        shutil.rmtree(exp)
    model = YOLO("yolov8n.pt")
    model.train(
        data=str(cfg["data_yaml"]),
        model="yolov8n.pt",
        epochs=int(cfg["epochs"]),
        imgsz=int(cfg["imgsz"]),
        batch=int(cfg["batch"]),
        lr0=float(cfg["lr0"]),
        freeze=int(cfg["freeze"]),
        mosaic=float(cfg["mosaic"]),
        mixup=float(cfg["mixup"]),
        close_mosaic=int(cfg["close_mosaic"]),
        project=str(RUNS),
        name=tag,
        exist_ok=True,
        verbose=False,
        workers=0,
        seed=int(seed),
        deterministic=True,
        resume=False,
    )
    best = exp / "weights" / "best.pt"
    if not best.exists():
        best = exp / "weights" / "last.pt"
    return best


def cache_preds(model_ref: str, data: EvalData, nms_iou: float) -> dict[str, list[tuple[int, float, list[float]]]]:
    m = YOLO(model_ref)
    name2idx = {n: i for i, n in enumerate(data.names)}
    rs = m.predict(
        source=[str(p) for p in data.images],
        conf=RAW_CONF,
        iou=nms_iou,
        max_det=300,
        imgsz=640,
        verbose=False,
        save=False,
        device=0,
    )
    out = {}
    for p, r in zip(data.images, rs):
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


def filter_preds(
    preds: list[tuple[int, float, list[float]]],
    names: list[str],
    thresholds: dict[str, float],
    max_det: int,
    img_hw: tuple[int, int],
    min_area_frac: float,
    ar_min: float,
    ar_max: float,
) -> list[tuple[int, float, list[float]]]:
    h, w = img_hw
    img_area = float(max(h * w, 1))
    out = []
    for c, s, b in preds:
        cname = names[c]
        if s < thresholds[cname]:
            continue
        if cname == "person":
            bw = max(0.0, b[2] - b[0])
            bh = max(0.0, b[3] - b[1])
            af = (bw * bh) / img_area
            ar = (bw / bh) if bh > 1e-9 else 999.0
            if af < min_area_frac:
                continue
            if ar < ar_min or ar > ar_max:
                continue
        out.append((c, s, b))
    out.sort(key=lambda x: x[1], reverse=True)
    return out[:max_det]


def eval_with_cfg(
    data: EvalData,
    cache: dict[str, list[tuple[int, float, list[float]]]],
    thresholds: dict[str, float],
    max_det: int,
    min_area_frac: float,
    ar_min: float,
    ar_max: float,
) -> dict:
    tp = fp = fn = 0
    tp_c = np.zeros(3, dtype=np.int64)
    fp_c = np.zeros(3, dtype=np.int64)
    fn_c = np.zeros(3, dtype=np.int64)
    person_fp_bg = 0

    for ip in data.images:
        gts = data.gt_by_name.get(ip.name, [])
        prs = filter_preds(
            cache[ip.name],
            data.names,
            thresholds=thresholds,
            max_det=max_det,
            img_hw=data.hw_by_name[ip.name],
            min_area_frac=min_area_frac,
            ar_min=ar_min,
            ar_max=ar_max,
        )
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
                if gc < 3:
                    tp_c[gc] += 1
            else:
                fn += 1
                if gc < 3:
                    fn_c[gc] += 1

        for j, (pc, _ps, _pb) in enumerate(prs):
            if j not in used:
                fp += 1
                if pc < 3:
                    fp_c[pc] += 1
                if data.names[pc] == "person":
                    person_fp_bg += 1

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
        "person_fp_bg": int(person_fp_bg),
    }


def better_row(row: dict, best: dict | None) -> bool:
    if best is None:
        return True
    if row["f1"] > best["f1"] + 1e-12:
        return True
    if abs(row["f1"] - best["f1"]) <= 1e-12 and row["fp"] < best["fp"]:
        return True
    if abs(row["f1"] - best["f1"]) <= 1e-12 and row["fp"] == best["fp"] and row["tp"] > best["tp"]:
        return True
    return False


def search_best(model_ref: str, data: EvalData, grid: dict) -> dict:
    best = None
    for nms in grid["nms"]:
        cache = cache_preds(model_ref, data, nms_iou=float(nms))
        for md in grid["max_det"]:
            for tpv in grid["thr_person"]:
                for tcv in grid["thr_car"]:
                    for tdv in grid["thr_dog"]:
                        for min_area_frac, ar_min, ar_max in grid["person_filter"]:
                            thresholds = {"person": float(tpv), "car": float(tcv), "dog": float(tdv)}
                            m = eval_with_cfg(
                                data,
                                cache,
                                thresholds=thresholds,
                                max_det=int(md),
                                min_area_frac=float(min_area_frac),
                                ar_min=float(ar_min),
                                ar_max=float(ar_max),
                            )
                            row = {
                                "nms_iou": float(nms),
                                "max_det": int(md),
                                "thr_person": float(tpv),
                                "thr_car": float(tcv),
                                "thr_dog": float(tdv),
                                "min_area_frac": float(min_area_frac),
                                "ar_min": float(ar_min),
                                "ar_max": float(ar_max),
                                **m,
                            }
                            if better_row(row, best):
                                best = row
    return best


def build_confusion(model_ref: str, data: EvalData, op: dict) -> tuple[np.ndarray, np.ndarray]:
    model = YOLO(model_ref)
    cm = np.zeros((4, 4), dtype=np.int64)
    thr = {"person": op["thr_person"], "car": op["thr_car"], "dog": op["thr_dog"]}

    for ip in data.images:
        gts = data.gt_by_name.get(ip.name, [])
        r = model.predict(
            source=str(ip),
            conf=RAW_CONF,
            iou=float(op["nms_iou"]),
            max_det=300,
            imgsz=640,
            verbose=False,
            save=False,
            device=0,
        )[0]
        preds = []
        if r.boxes is not None and len(r.boxes) > 0:
            cls = r.boxes.cls.cpu().numpy().astype(int)
            conf = r.boxes.conf.cpu().numpy().astype(float)
            box = r.boxes.xyxy.cpu().numpy().astype(float)
            for c, s, b in zip(cls, conf, box):
                cname = model.names[int(c)]
                if cname not in data.names:
                    continue
                dc = data.names.index(cname)
                if s < thr[cname]:
                    continue
                if cname == "person":
                    h, w = data.hw_by_name[ip.name]
                    bw = max(0.0, b[2] - b[0])
                    bh = max(0.0, b[3] - b[1])
                    af = (bw * bh) / float(max(h * w, 1))
                    ar = (bw / bh) if bh > 1e-9 else 999.0
                    if af < float(op["min_area_frac"]):
                        continue
                    if ar < float(op["ar_min"]) or ar > float(op["ar_max"]):
                        continue
                preds.append((dc, float(s), b.tolist()))
        preds.sort(key=lambda x: x[1], reverse=True)
        preds = preds[: int(op["max_det"])]

        pairs = []
        for gi, gt in enumerate(gts):
            for pi, pr in enumerate(preds):
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
            cm[gts[gi][0], preds[pi][0]] += 1

        for gi, gt in enumerate(gts):
            if gi not in used_g:
                cm[gt[0], BG_IDX] += 1
        for pi, pr in enumerate(preds):
            if pi not in used_p:
                cm[BG_IDX, pr[0]] += 1

    norm = np.zeros_like(cm, dtype=np.float64)
    rs = cm.sum(axis=1, keepdims=True)
    np.divide(cm, np.maximum(rs, 1), out=norm, where=rs > 0)
    return cm, norm


def plot_candidate_metrics(df: pd.DataFrame, out_path: Path) -> None:
    labels = df["tag"].tolist()
    x = np.arange(len(labels))
    w = 0.2

    fig, ax = plt.subplots(figsize=(12, 5), dpi=160)
    ax.bar(x - 1.5 * w, df["f1"], width=w, label="F1")
    ax.bar(x - 0.5 * w, df["precision"], width=w, label="Precision")
    ax.bar(x + 0.5 * w, df["recall"], width=w, label="Recall")
    ax.bar(x + 1.5 * w, df["f1_macro"], width=w, label="F1_macro")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylim(0, 1.0)
    ax.set_title("Candidate Comparison (best operating point)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_seed_delta(seed_df: pd.DataFrame, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.5, 4.5), dpi=160)
    ax.plot(seed_df["seed"], seed_df["delta_f1_vs_coco"], marker="o", linewidth=1.5)
    ax.axhline(0.0, linestyle="--", linewidth=1)
    ax.set_xlabel("seed")
    ax.set_ylabel("delta F1 (candidate - COCO)")
    ax.set_title("Seed Repeat Delta F1")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def draw_cm_pair(cm_a, cm_b, title_a: str, title_b: str, out_path: Path, normalized: bool = False) -> None:
    labels = TARGET_CLASSES + [BG_LABEL]
    fig, ax = plt.subplots(1, 2, figsize=(14, 5.5), dpi=170)
    for i, (cm, title) in enumerate([(cm_a, title_a), (cm_b, title_b)]):
        im = ax[i].imshow(cm, cmap="Blues", vmin=0.0 if normalized else None, vmax=1.0 if normalized else None)
        ax[i].set_title(title, fontsize=11, weight="bold")
        ax[i].set_xticks(range(4))
        ax[i].set_yticks(range(4))
        ax[i].set_xticklabels(labels, rotation=20, ha="right")
        ax[i].set_yticklabels(labels)
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
    fig.suptitle("COCO vs Best Ordered Candidate (4x4 confusion)", fontsize=13, weight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out_path)
    plt.close(fig)


def around(v: float, low: float, high: float) -> list[float]:
    vals = []
    for d in [-0.10, -0.05, 0.0, 0.05, 0.10]:
        x = round(v + d, 2)
        x = max(low, min(high, x))
        vals.append(x)
    return sorted(set(vals))


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    RUNS.mkdir(parents=True, exist_ok=True)
    DS_ROOT.mkdir(parents=True, exist_ok=True)

    write_plan_document()
    eval_data = load_eval_data()
    neg_pool = collect_negative_pool()

    # step 2: data reconstruction variants
    ds_meta = []
    ds_meta.append(build_dataset_variant("plan_bg10", neg_ratio=0.10, balance_fraction=0.0, neg_pool=neg_pool, seed=42))
    ds_meta.append(build_dataset_variant("plan_bg10_bal075", neg_ratio=0.10, balance_fraction=0.75, neg_pool=neg_pool, seed=42))
    ds_meta.append(build_dataset_variant("plan_bg15_bal075", neg_ratio=0.15, balance_fraction=0.75, neg_pool=neg_pool, seed=42))
    ds_map = {d["tag"]: d["data_yaml"] for d in ds_meta}

    candidates = [
        {
            "tag": "plan_base_voc3large",
            "data_yaml": str(BASE_TRAIN_DATA),
            "epochs": 20,
            "imgsz": 640,
            "batch": 16,
            "lr0": 0.00025,
            "freeze": 10,
            "mosaic": 0.20,
            "mixup": 0.0,
            "close_mosaic": 2,
        },
        {
            "tag": "plan_bg10",
            "data_yaml": ds_map["plan_bg10"],
            "epochs": 20,
            "imgsz": 640,
            "batch": 16,
            "lr0": 0.00025,
            "freeze": 10,
            "mosaic": 0.20,
            "mixup": 0.0,
            "close_mosaic": 2,
        },
        {
            "tag": "plan_bg10_bal075",
            "data_yaml": ds_map["plan_bg10_bal075"],
            "epochs": 22,
            "imgsz": 640,
            "batch": 16,
            "lr0": 0.00022,
            "freeze": 12,
            "mosaic": 0.15,
            "mixup": 0.0,
            "close_mosaic": 2,
        },
        {
            "tag": "plan_bg15_bal075",
            "data_yaml": ds_map["plan_bg15_bal075"],
            "epochs": 22,
            "imgsz": 640,
            "batch": 16,
            "lr0": 0.00020,
            "freeze": 12,
            "mosaic": 0.15,
            "mixup": 0.0,
            "close_mosaic": 2,
        },
    ]

    coarse_grid = {
        "nms": [0.45, 0.60],
        "max_det": [50],
        "thr_person": [0.20, 0.25, 0.30, 0.35, 0.40, 0.45],
        "thr_car": [0.15, 0.20, 0.25, 0.30, 0.35],
        "thr_dog": [0.15, 0.20, 0.25, 0.30, 0.35],
        "person_filter": [(0.0, 0.0, 999.0)],
    }

    coco_best = search_best("yolov8n.pt", eval_data, coarse_grid)
    print("[baseline-coco]", coco_best)

    # step 3: candidate training + fair comparison
    cand_rows = []
    trained = {}
    for cfg in candidates:
        print("[train]", cfg["tag"])
        best_pt = train_candidate(cfg, seed=42)
        trained[cfg["tag"]] = str(best_pt)
        best_op = search_best(str(best_pt), eval_data, coarse_grid)
        row = {
            "tag": cfg["tag"],
            "best_pt": str(best_pt),
            "f1": best_op["f1"],
            "precision": best_op["precision"],
            "recall": best_op["recall"],
            "f1_macro": best_op["f1_macro"],
            "tp": best_op["tp"],
            "fp": best_op["fp"],
            "fn": best_op["fn"],
            "person_fp_bg": best_op["person_fp_bg"],
            "nms_iou": best_op["nms_iou"],
            "max_det": best_op["max_det"],
            "thr_person": best_op["thr_person"],
            "thr_car": best_op["thr_car"],
            "thr_dog": best_op["thr_dog"],
            "min_area_frac": best_op["min_area_frac"],
            "ar_min": best_op["ar_min"],
            "ar_max": best_op["ar_max"],
            "delta_f1_vs_coco": best_op["f1"] - coco_best["f1"],
        }
        cand_rows.append(row)
        print("[candidate]", row["tag"], "f1", f"{row['f1']:.4f}", "dF1", f"{row['delta_f1_vs_coco']:+.4f}")

    cand_df = pd.DataFrame(cand_rows).sort_values(["f1", "fp"], ascending=[False, True]).reset_index(drop=True)
    cand_df.to_csv(CANDIDATE_CSV, index=False, encoding="utf-8-sig")
    plot_candidate_metrics(cand_df, FIG_METRICS)

    best_row = cand_df.iloc[0].to_dict()
    best_cfg = next(c for c in candidates if c["tag"] == best_row["tag"])
    best_pt = best_row["best_pt"]

    # step 4: operating tuning around best candidate + person filter
    fine_grid = {
        "nms": [0.40, 0.45, 0.50, 0.55, 0.60],
        "max_det": [20, 30, 50],
        "thr_person": around(float(best_row["thr_person"]), 0.10, 0.70),
        "thr_car": around(float(best_row["thr_car"]), 0.10, 0.60),
        "thr_dog": around(float(best_row["thr_dog"]), 0.10, 0.60),
        "person_filter": [
            (0.0, 0.0, 999.0),
            (0.0005, 0.15, 2.0),
            (0.0010, 0.20, 1.8),
        ],
    }
    tuned_best = search_best(str(best_pt), eval_data, fine_grid)
    print("[tuned-best]", tuned_best)

    coco_cm, coco_norm = build_confusion("yolov8n.pt", eval_data, coco_best)
    tuned_cm, tuned_norm = build_confusion(str(best_pt), eval_data, tuned_best)
    draw_cm_pair(coco_cm, tuned_cm, "COCO (counts)", "Ordered Best (counts)", FIG_CM_COUNT, normalized=False)
    draw_cm_pair(coco_norm, tuned_norm, "COCO (normalized)", "Ordered Best (normalized)", FIG_CM_NORM, normalized=True)

    # step 5: 5-seed significance on selected recipe
    seeds = [101, 202, 303, 404, 505]
    seed_rows = []
    deltas = []
    for s in seeds:
        print("[seed-repeat]", best_cfg["tag"], "seed", s)
        s_pt = train_candidate(best_cfg, seed=s, suffix=f"_seed{s}")
        s_best = search_best(str(s_pt), eval_data, coarse_grid)
        delta = s_best["f1"] - coco_best["f1"]
        deltas.append(delta)
        seed_rows.append(
            {
                "seed": s,
                "best_pt": str(s_pt),
                "f1": s_best["f1"],
                "precision": s_best["precision"],
                "recall": s_best["recall"],
                "tp": s_best["tp"],
                "fp": s_best["fp"],
                "fn": s_best["fn"],
                "delta_f1_vs_coco": delta,
            }
        )
    seed_df = pd.DataFrame(seed_rows)
    seed_df.to_csv(SEED_CSV, index=False, encoding="utf-8-sig")
    plot_seed_delta(seed_df, FIG_DELTA)

    arr = np.array(deltas, dtype=float)
    t_res = ttest_1samp(arr, popmean=0.0, alternative="greater")
    nz = arr[np.abs(arr) > 1e-12]
    w_p = float(wilcoxon(nz, alternative="greater").pvalue) if len(nz) > 0 else 1.0
    sig = {
        "n": int(len(arr)),
        "mean_delta_f1": float(np.mean(arr)),
        "std_delta_f1": float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0,
        "ttest_p_greater": float(t_res.pvalue),
        "wilcoxon_p_greater": w_p,
        "significant_at_0_05": bool((float(t_res.pvalue) < 0.05) and (float(np.mean(arr)) > 0)),
    }

    report_lines = [
        "# Ordered Plan Execution Report (CV1)",
        "",
        "## Plan",
        "![plan_notebook](C:/Users/lmhst/git/roboticsCV/CV_project#1/outputs/figures/ordered_plan_candidate_metrics_cv1.png)",
        "",
        "## Baseline (COCO best in fair search)",
        (
            f"- F1={coco_best['f1']:.4f}, P={coco_best['precision']:.4f}, R={coco_best['recall']:.4f}, "
            f"TP={coco_best['tp']}, FP={coco_best['fp']}, FN={coco_best['fn']}, "
            f"thr=({coco_best['thr_person']:.2f},{coco_best['thr_car']:.2f},{coco_best['thr_dog']:.2f}), "
            f"nms_iou={coco_best['nms_iou']}, max_det={coco_best['max_det']}"
        ),
        "",
        "## Candidate Ranking",
        "![candidate_metrics](C:/Users/lmhst/git/roboticsCV/CV_project#1/outputs/figures/ordered_plan_candidate_metrics_cv1.png)",
        "",
        (
            f"- selected recipe: {best_row['tag']} "
            f"(coarse best F1={best_row['f1']:.4f}, dF1_vs_COCO={best_row['delta_f1_vs_coco']:+.4f})"
        ),
        (
            f"- tuned best (step4): F1={tuned_best['f1']:.4f}, P={tuned_best['precision']:.4f}, R={tuned_best['recall']:.4f}, "
            f"TP={tuned_best['tp']}, FP={tuned_best['fp']}, FN={tuned_best['fn']}, "
            f"person_fp_bg={tuned_best['person_fp_bg']}, "
            f"thr=({tuned_best['thr_person']:.2f},{tuned_best['thr_car']:.2f},{tuned_best['thr_dog']:.2f}), "
            f"filter(min_area={tuned_best['min_area_frac']:.4f}, ar=[{tuned_best['ar_min']:.2f},{tuned_best['ar_max']:.2f}]), "
            f"nms_iou={tuned_best['nms_iou']}, max_det={tuned_best['max_det']}"
        ),
        "",
        "## Confusion Matrix (4x4, with background)",
        "![cm_count](C:/Users/lmhst/git/roboticsCV/CV_project#1/outputs/figures/ordered_plan_confusion_counts_cv1.png)",
        "",
        "![cm_norm](C:/Users/lmhst/git/roboticsCV/CV_project#1/outputs/figures/ordered_plan_confusion_normalized_cv1.png)",
        "",
        "## Seed Repeat Significance (5 seeds)",
        "![seed_delta](C:/Users/lmhst/git/roboticsCV/CV_project#1/outputs/figures/ordered_plan_seed_delta_cv1.png)",
        "",
        f"- mean delta F1 (recipe - COCO) = {sig['mean_delta_f1']:+.4f} +/- {sig['std_delta_f1']:.4f}",
        f"- one-sample t-test (H1: delta>0): p={sig['ttest_p_greater']:.4g}",
        f"- Wilcoxon signed-rank (H1: delta>0): p={sig['wilcoxon_p_greater']:.4g}",
        f"- significant@0.05: {sig['significant_at_0_05']}",
    ]
    REPORT_MD.write_text("\n".join(report_lines), encoding="utf-8")

    manifest = {
        "plan_md": str(PLAN_MD),
        "candidate_csv": str(CANDIDATE_CSV),
        "seed_csv": str(SEED_CSV),
        "report_md": str(REPORT_MD),
        "fig_metrics": str(FIG_METRICS),
        "fig_seed_delta": str(FIG_DELTA),
        "fig_cm_count": str(FIG_CM_COUNT),
        "fig_cm_norm": str(FIG_CM_NORM),
        "datasets": ds_meta,
        "trained": trained,
        "coco_best": coco_best,
        "best_candidate_coarse": best_row,
        "best_candidate_tuned": tuned_best,
        "significance": sig,
    }
    MANIFEST.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

    print("[saved]", PLAN_MD)
    print("[saved]", CANDIDATE_CSV)
    print("[saved]", SEED_CSV)
    print("[saved]", REPORT_MD)
    print("[saved]", MANIFEST)
    print(
        "[summary]",
        json.dumps(
            {
                "coco_f1": coco_best["f1"],
                "best_coarse_tag": best_row["tag"],
                "best_coarse_f1": best_row["f1"],
                "best_tuned_f1": tuned_best["f1"],
                "significant": sig["significant_at_0_05"],
            },
            ensure_ascii=False,
        ),
    )


if __name__ == "__main__":
    main()
