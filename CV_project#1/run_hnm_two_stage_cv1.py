from __future__ import annotations

import json
import random
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
import torch
from ultralytics import YOLO

ROOT = Path(r"C:\Users\lmhst\git\roboticsCV\CV_project#1")
OUT = ROOT / "outputs"
FIG = OUT / "figures"
RUNS = ROOT / "hnm_two_stage_runs"
DS_ROOT = ROOT / "datasets_hnm"

VOC3 = ROOT / "voc3"
VOC3_LARGE = ROOT / "voc3_large"
DATA_TRAIN_BASE = VOC3_LARGE / "data.yaml"
DATA_EVAL = VOC3 / "data.yaml"

VOC2007 = ROOT / "data" / "VOCdevkit" / "VOC2007"
VOC_IMG = VOC2007 / "JPEGImages"
VOC_ANN = VOC2007 / "Annotations"
VOC_SPLIT = VOC2007 / "ImageSets" / "Main"

TARGET_CLASSES = ["person", "car", "dog"]

RAW_CONF = 0.05
MATCH_IOU = 0.5

CSV_RESULTS = OUT / "hnm_two_stage_results_cv1.csv"
REPORT_MD = OUT / "hnm_two_stage_report_cv1.md"
MANIFEST = OUT / "hnm_two_stage_manifest_cv1.json"
HNM_META_JSON = OUT / "hnm_selected_images_cv1.json"

FIG_METRICS = FIG / "hnm_two_stage_metrics_cv1.png"
FIG_ERRORS = FIG / "hnm_two_stage_errors_cv1.png"
FIG_CM_COUNT = FIG / "hnm_two_stage_confusion_counts_cv1.png"
FIG_CM_NORM = FIG / "hnm_two_stage_confusion_normalized_cv1.png"

CLASSES_4 = ["person", "car", "dog", "bg/other"]
BG_IDX = 3


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


def parse_split_ids(split_name: str) -> list[str]:
    p = VOC_SPLIT / f"{split_name}.txt"
    out = []
    if not p.exists():
        return out
    for ln in p.read_text(encoding="utf-8").splitlines():
        ln = ln.strip()
        if ln:
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
    split_map = {"train": parse_split_ids("trainval"), "val": parse_split_ids("test")}
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


def copy_base_dataset(dst_root: Path) -> None:
    if dst_root.exists():
        shutil.rmtree(dst_root)
    for split in ["train", "val"]:
        (dst_root / "images" / split).mkdir(parents=True, exist_ok=True)
        (dst_root / "labels" / split).mkdir(parents=True, exist_ok=True)
        for ip in sorted((VOC3_LARGE / "images" / split).glob("*.*")):
            if ip.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp"}:
                continue
            lp = VOC3_LARGE / "labels" / split / f"{ip.stem}.txt"
            if not lp.exists():
                continue
            shutil.copy2(ip, dst_root / "images" / split / ip.name)
            shutil.copy2(lp, dst_root / "labels" / split / lp.name)


def mine_hard_negatives(
    miner_model_ref: str,
    neg_filenames: list[str],
    sample_cap: int,
    topk: int,
    conf_mine: float,
    seed: int,
) -> list[dict]:
    rng = random.Random(seed)
    cands = list(neg_filenames)
    rng.shuffle(cands)
    cands = cands[: min(sample_cap, len(cands))]
    if len(cands) == 0:
        return []

    model = YOLO(miner_model_ref)
    rows = []

    # Use chunked CPU streaming to avoid GPU OOM during mining
    chunk_size = 96
    for i in range(0, len(cands), chunk_size):
        chunk = cands[i : i + chunk_size]
        paths = [str(VOC_IMG / fn) for fn in chunk]
        rs = model.predict(
            source=paths,
            conf=conf_mine,
            iou=0.5,
            max_det=200,
            imgsz=640,
            verbose=False,
            save=False,
            stream=True,
            device="cpu",
        )
        for fn, r in zip(chunk, rs):
            person_count = 0
            person_conf_sum = 0.0
            if r.boxes is not None and len(r.boxes) > 0:
                cls = r.boxes.cls.cpu().numpy().astype(int)
                conf = r.boxes.conf.cpu().numpy().astype(float)
                for c, s in zip(cls, conf):
                    cname = model.names[int(c)]
                    if cname == "person":
                        person_count += 1
                        person_conf_sum += float(s)
            score = person_conf_sum + 0.25 * person_count
            rows.append(
                {
                    "filename": fn,
                    "person_count": int(person_count),
                    "person_conf_sum": float(person_conf_sum),
                    "score": float(score),
                }
            )
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    rows = sorted(rows, key=lambda x: (x["score"], x["person_conf_sum"], x["person_count"]), reverse=True)
    rows = [r for r in rows if r["person_count"] > 0][:topk]
    return rows


def add_hard_negatives(ds_root: Path, selected_train: list[dict], selected_val: list[dict]) -> dict:
    meta = {}
    for split, rows in [("train", selected_train), ("val", selected_val)]:
        img_dir = ds_root / "images" / split
        lb_dir = ds_root / "labels" / split
        added = 0
        for i, row in enumerate(rows):
            fn = row["filename"]
            src = VOC_IMG / fn
            if not src.exists():
                continue
            stem = Path(fn).stem
            dst_stem = f"hnm_{i:03d}_{stem}"
            dst_img = img_dir / f"{dst_stem}.jpg"
            dst_lb = lb_dir / f"{dst_stem}.txt"
            if dst_img.exists():
                continue
            shutil.copy2(src, dst_img)
            dst_lb.write_text("", encoding="utf-8")
            added += 1
        meta[f"hnm_{split}_added"] = added
    return meta


def write_data_yaml(ds_root: Path) -> Path:
    y = ds_root / "data.yaml"
    y.write_text(
        yaml.safe_dump(
            {
                "path": str(ds_root.resolve()),
                "train": "images/train",
                "val": "images/val",
                "nc": 3,
                "names": TARGET_CLASSES,
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    return y


def build_hnm_dataset(miner_model_ref: str) -> tuple[Path, dict]:
    ds_dir = DS_ROOT / "hnm_voc3large_v1"
    copy_base_dataset(ds_dir)
    pool = collect_negative_pool()

    selected_train = mine_hard_negatives(
        miner_model_ref=miner_model_ref,
        neg_filenames=pool["train"],
        sample_cap=1200,
        topk=90,
        conf_mine=0.10,
        seed=42,
    )
    selected_val = mine_hard_negatives(
        miner_model_ref=miner_model_ref,
        neg_filenames=pool["val"],
        sample_cap=600,
        topk=30,
        conf_mine=0.10,
        seed=43,
    )
    add_meta = add_hard_negatives(ds_dir, selected_train, selected_val)
    y = write_data_yaml(ds_dir)

    hnm_meta = {
        "miner_model": miner_model_ref,
        "selected_train_count": len(selected_train),
        "selected_val_count": len(selected_val),
        "selected_train": selected_train,
        "selected_val": selected_val,
        **add_meta,
        "data_yaml": str(y.resolve()),
    }
    HNM_META_JSON.write_text(json.dumps(hnm_meta, indent=2), encoding="utf-8")
    return y, hnm_meta


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


def eval_with_cfg(names, imgs, labels_dir, cache, thr, max_det, person_filter):
    min_area_frac, ar_min, ar_max = person_filter
    tp = fp = fn = 0
    tp_c = np.zeros(3, dtype=np.int64)
    fp_c = np.zeros(3, dtype=np.int64)
    fn_c = np.zeros(3, dtype=np.int64)
    person_fp_bg = 0

    hw = {}
    for p in imgs:
        im = cv2.imread(str(p))
        hh, ww = im.shape[:2]
        hw[p.name] = (hh, ww)

    for img in imgs:
        gts = load_gt(img, labels_dir)
        h, w = hw[img.name]
        area = float(max(h * w, 1))

        prs = []
        for c, s, b in cache[img.name]:
            cname = names[c]
            if s < thr[cname]:
                continue
            if cname == "person":
                bw = max(0.0, b[2] - b[0])
                bh = max(0.0, b[3] - b[1])
                af = (bw * bh) / area
                ar = (bw / bh) if bh > 1e-9 else 999.0
                if af < min_area_frac:
                    continue
                if ar < ar_min or ar > ar_max:
                    continue
            prs.append((c, s, b))
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
                if names[pc] == "person":
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


def sweep_best(model_ref: str, names: list[str], imgs: list[Path], labels_dir: Path) -> dict:
    nms_grid = [0.35, 0.45, 0.55, 0.60]
    maxdet_grid = [10, 20, 30]
    tp_grid = [0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45]
    tc_grid = [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40]
    td_grid = [0.10, 0.15, 0.20, 0.25, 0.30, 0.35]
    person_filters = [
        (0.0, 0.0, 999.0),
        (0.0005, 0.15, 2.0),
        (0.0010, 0.20, 1.8),
    ]

    best = None
    for nms in nms_grid:
        cache = cache_preds(model_ref, names, imgs, nms_iou=nms)
        for md in maxdet_grid:
            for tp in tp_grid:
                for tc in tc_grid:
                    for td in td_grid:
                        thr = {"person": tp, "car": tc, "dog": td}
                        for pf in person_filters:
                            m = eval_with_cfg(names, imgs, labels_dir, cache, thr=thr, max_det=md, person_filter=pf)
                            row = {
                                "nms_iou": nms,
                                "max_det": md,
                                "thr_person": tp,
                                "thr_car": tc,
                                "thr_dog": td,
                                "min_area_frac": pf[0],
                                "ar_min": pf[1],
                                "ar_max": pf[2],
                                **m,
                            }
                            if best is None or row["f1"] > best["f1"] or (abs(row["f1"] - best["f1"]) < 1e-12 and row["fp"] < best["fp"]):
                                best = row
    return best


def run_two_stage(tag: str, train_yaml: Path, s1: dict, s2: dict) -> Path:
    exp1 = RUNS / f"{tag}_stage1"
    exp2 = RUNS / f"{tag}_stage2"
    if exp1.exists():
        shutil.rmtree(exp1)
    if exp2.exists():
        shutil.rmtree(exp2)

    # Stage1: head-focused tuning with backbone freeze (close to head-only)
    m1 = YOLO("yolov8n.pt")
    m1.train(
        data=str(train_yaml),
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

    # Stage2: full fine-tuning with lower LR
    m2 = YOLO(str(s1_best))
    m2.train(
        data=str(train_yaml),
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
    min_area_frac, ar_min, ar_max = float(cfg["min_area_frac"]), float(cfg["ar_min"]), float(cfg["ar_max"])

    hw = {}
    for p in imgs:
        im = cv2.imread(str(p))
        hh, ww = im.shape[:2]
        hw[p.name] = (hh, ww)

    for img in imgs:
        gts = load_gt(img, labels_dir)
        h, w = hw[img.name]
        img_area = float(max(h * w, 1))
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
                if s < thr[cname]:
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

        used_g, used_p = set(), set()
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


def draw_cm_pair(cm_a, cm_b, title_a: str, title_b: str, out_path: Path, normalized: bool):
    fig, ax = plt.subplots(1, 2, figsize=(14, 5.5), dpi=170)
    for i, (cm, title) in enumerate([(cm_a, title_a), (cm_b, title_b)]):
        im = ax[i].imshow(cm, cmap="Blues", vmin=0.0 if normalized else None, vmax=1.0 if normalized else None)
        ax[i].set_title(title, fontsize=11, weight="bold")
        ax[i].set_xticks(range(4))
        ax[i].set_yticks(range(4))
        ax[i].set_xticklabels(CLASSES_4, rotation=20, ha="right")
        ax[i].set_yticklabels(CLASSES_4)
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
    fig.suptitle("COCO vs HNM+2Stage (4x4 confusion)", fontsize=13, weight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out_path)
    plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    RUNS.mkdir(parents=True, exist_ok=True)
    DS_ROOT.mkdir(parents=True, exist_ok=True)

    names, imgs, labels_dir = load_eval_data()

    # baseline miner: previous strongest two-stage checkpoint if exists, else COCO
    miner_pt = ROOT / "two_stage_runs" / "two_stage_c_stage2" / "weights" / "best.pt"
    miner_model_ref = str(miner_pt) if miner_pt.exists() else "yolov8n.pt"

    train_yaml, hnm_meta = build_hnm_dataset(miner_model_ref=miner_model_ref)
    print("[hnm_dataset]", train_yaml)
    print("[hnm_meta]", json.dumps({"train_sel": hnm_meta["selected_train_count"], "val_sel": hnm_meta["selected_val_count"], "miner": miner_model_ref}, ensure_ascii=False))

    coco_best = sweep_best("yolov8n.pt", names, imgs, labels_dir)
    print("[coco_best]", coco_best)

    # combination of #1(HNM) + #3(improved 2-stage)
    candidates = [
        {
            "tag": "hnm_two_stage_a",
            "stage1": {
                "epochs": 8,
                "imgsz": 640,
                "batch": 16,
                "lr0": 0.0014,
                "freeze": 22,
                "mosaic": 0.05,
                "mixup": 0.0,
                "close_mosaic": 1,
                "cos_lr": False,
                "patience": 20,
            },
            "stage2": {
                "epochs": 14,
                "imgsz": 640,
                "batch": 16,
                "lr0": 0.00007,
                "freeze": 0,
                "mosaic": 0.0,
                "mixup": 0.0,
                "close_mosaic": 0,
                "cos_lr": True,
                "patience": 20,
            },
        },
        {
            "tag": "hnm_two_stage_b",
            "stage1": {
                "epochs": 10,
                "imgsz": 640,
                "batch": 16,
                "lr0": 0.0012,
                "freeze": 22,
                "mosaic": 0.10,
                "mixup": 0.0,
                "close_mosaic": 1,
                "cos_lr": False,
                "patience": 20,
            },
            "stage2": {
                "epochs": 16,
                "imgsz": 640,
                "batch": 16,
                "lr0": 0.00006,
                "freeze": 0,
                "mosaic": 0.0,
                "mixup": 0.0,
                "close_mosaic": 0,
                "cos_lr": True,
                "patience": 20,
            },
        },
    ]

    rows = []
    for c in candidates:
        print("[train]", c["tag"])
        best_pt = run_two_stage(c["tag"], train_yaml=train_yaml, s1=c["stage1"], s2=c["stage2"])
        best = sweep_best(str(best_pt), names, imgs, labels_dir)
        row = {
            "tag": c["tag"],
            "best_pt": str(best_pt),
            "f1": best["f1"],
            "precision": best["precision"],
            "recall": best["recall"],
            "f1_macro": best["f1_macro"],
            "tp": best["tp"],
            "fp": best["fp"],
            "fn": best["fn"],
            "person_fp_bg": best["person_fp_bg"],
            "nms_iou": best["nms_iou"],
            "max_det": best["max_det"],
            "thr_person": best["thr_person"],
            "thr_car": best["thr_car"],
            "thr_dog": best["thr_dog"],
            "min_area_frac": best["min_area_frac"],
            "ar_min": best["ar_min"],
            "ar_max": best["ar_max"],
            "delta_f1_vs_coco": best["f1"] - coco_best["f1"],
        }
        rows.append(row)
        print("[result]", row["tag"], "f1", row["f1"], "dF1", row["delta_f1_vs_coco"])

    df = pd.DataFrame(rows).sort_values(["f1", "fp"], ascending=[False, True]).reset_index(drop=True)
    df.to_csv(CSV_RESULTS, index=False, encoding="utf-8-sig")
    best_row = df.iloc[0].to_dict()

    # visuals
    metrics = ["precision", "recall", "f1", "f1_macro"]
    x = np.arange(len(metrics))
    w = 0.35
    fig, ax = plt.subplots(figsize=(8.8, 5), dpi=160)
    ax.bar(x - w / 2, [coco_best[m] for m in metrics], width=w, label="COCO", color="#4e79a7")
    ax.bar(x + w / 2, [best_row[m] for m in metrics], width=w, label="HNM+2Stage", color="#f28e2b")
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.set_ylim(0, 1.0)
    ax.set_title("COCO vs HNM+2Stage")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG_METRICS)
    plt.close(fig)

    fig2, ax2 = plt.subplots(figsize=(8.8, 5), dpi=160)
    counts = ["tp", "fp", "fn"]
    x2 = np.arange(len(counts))
    ax2.bar(x2 - w / 2, [coco_best[k] for k in counts], width=w, label="COCO", color="#4e79a7")
    ax2.bar(x2 + w / 2, [best_row[k] for k in counts], width=w, label="HNM+2Stage", color="#f28e2b")
    ax2.set_xticks(x2)
    ax2.set_xticklabels(counts)
    ax2.set_title("TP/FP/FN Comparison")
    ax2.legend()
    fig2.tight_layout()
    fig2.savefig(FIG_ERRORS)
    plt.close(fig2)

    coco_cm, coco_norm = build_cm("yolov8n.pt", names, imgs, labels_dir, coco_best)
    best_cm, best_norm = build_cm(str(best_row["best_pt"]), names, imgs, labels_dir, best_row)
    draw_cm_pair(coco_cm, best_cm, "COCO (counts)", "HNM+2Stage (counts)", FIG_CM_COUNT, normalized=False)
    draw_cm_pair(coco_norm, best_norm, "COCO (normalized)", "HNM+2Stage (normalized)", FIG_CM_NORM, normalized=True)

    lines = [
        "# HNM + Improved Two-Stage Report (CV1)",
        "",
        "## Applied combo",
        "- #1 Hard Negative Mining: background images with strong person FP are mined and added as empty labels.",
        "- #3 Improved Two-stage: Stage1(head-focused with high freeze) -> Stage2(full fine-tune with low LR + cosine).",
        "",
        "## Hard Negative Dataset",
        f"- miner_model: `{miner_model_ref}`",
        f"- train hard negatives selected: {hnm_meta['selected_train_count']} (added: {hnm_meta['hnm_train_added']})",
        f"- val hard negatives selected: {hnm_meta['selected_val_count']} (added: {hnm_meta['hnm_val_added']})",
        f"- train data yaml: `{train_yaml}`",
        "",
        "## Baseline (COCO best operating)",
        f"- F1={coco_best['f1']:.4f}, P={coco_best['precision']:.4f}, R={coco_best['recall']:.4f}, TP={coco_best['tp']}, FP={coco_best['fp']}, FN={coco_best['fn']}, person_fp_bg={coco_best['person_fp_bg']}",
        f"- params: nms_iou={coco_best['nms_iou']}, max_det={coco_best['max_det']}, thr=({coco_best['thr_person']:.2f},{coco_best['thr_car']:.2f},{coco_best['thr_dog']:.2f}), filter=({coco_best['min_area_frac']:.4f},{coco_best['ar_min']:.2f},{coco_best['ar_max']:.2f})",
        "",
        "## Best HNM+2Stage",
        f"- tag={best_row['tag']}, F1={best_row['f1']:.4f}, P={best_row['precision']:.4f}, R={best_row['recall']:.4f}, TP={int(best_row['tp'])}, FP={int(best_row['fp'])}, FN={int(best_row['fn'])}, person_fp_bg={int(best_row['person_fp_bg'])}",
        f"- params: nms_iou={best_row['nms_iou']}, max_det={int(best_row['max_det'])}, thr=({best_row['thr_person']:.2f},{best_row['thr_car']:.2f},{best_row['thr_dog']:.2f}), filter=({best_row['min_area_frac']:.4f},{best_row['ar_min']:.2f},{best_row['ar_max']:.2f})",
        f"- delta(F1 vs COCO)={best_row['delta_f1_vs_coco']:+.4f}",
        "",
        "## Visuals",
        "![metrics](C:/Users/lmhst/git/roboticsCV/CV_project#1/outputs/figures/hnm_two_stage_metrics_cv1.png)",
        "",
        "![errors](C:/Users/lmhst/git/roboticsCV/CV_project#1/outputs/figures/hnm_two_stage_errors_cv1.png)",
        "",
        "![cm_counts](C:/Users/lmhst/git/roboticsCV/CV_project#1/outputs/figures/hnm_two_stage_confusion_counts_cv1.png)",
        "",
        "![cm_norm](C:/Users/lmhst/git/roboticsCV/CV_project#1/outputs/figures/hnm_two_stage_confusion_normalized_cv1.png)",
    ]
    REPORT_MD.write_text("\n".join(lines), encoding="utf-8")

    manifest = {
        "csv_results": str(CSV_RESULTS),
        "report_md": str(REPORT_MD),
        "manifest": str(MANIFEST),
        "hnm_meta_json": str(HNM_META_JSON),
        "fig_metrics": str(FIG_METRICS),
        "fig_errors": str(FIG_ERRORS),
        "fig_cm_count": str(FIG_CM_COUNT),
        "fig_cm_norm": str(FIG_CM_NORM),
        "coco_best": coco_best,
        "best_hnm_two_stage": best_row,
        "all_rows": rows,
    }
    MANIFEST.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print("[saved]", CSV_RESULTS)
    print("[saved]", REPORT_MD)
    print("[saved]", MANIFEST)
    print(
        "[summary]",
        json.dumps(
            {
                "coco_f1": coco_best["f1"],
                "best_tag": best_row["tag"],
                "best_f1": best_row["f1"],
                "delta_f1": best_row["delta_f1_vs_coco"],
                "best_person_fp_bg": best_row["person_fp_bg"],
            },
            ensure_ascii=False,
        ),
    )


if __name__ == "__main__":
    main()
