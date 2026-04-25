from __future__ import annotations

import json
import os
import random
import shutil
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import yaml
from ultralytics import YOLO


ROOT = Path(r"C:\Users\lmhst\git\roboticsCV\CV_project#1")
OUT = ROOT / "outputs"
FIG = OUT / "figures"
RUNS = ROOT / "last_squeeze_runs"
DATASET = ROOT / "datasets_last_squeeze_voc07"

VOC07 = ROOT / "data" / "VOCdevkit" / "VOC2007"
PROTOCOL = ROOT / "protocol_split_cv1"

QUICK_CSV = OUT / "last_squeeze_quick_cv1.csv"
MANIFEST = OUT / "last_squeeze_manifest_cv1.json"
REPORT = OUT / "last_squeeze_report_cv1.md"
FIG_COMPARE = FIG / "last_squeeze_compare_cv1.png"
FIG_CLASS = FIG / "last_squeeze_class_ap_cv1.png"

TARGET = ["person", "car", "dog"]
NAME2IDX = {n: i for i, n in enumerate(TARGET)}
IOU_THRESHOLDS = np.arange(0.5, 0.96, 0.05)

SEED_BUILD = 20260419
SEED_QUICK = 777
SEED_FULL = 42


@dataclass
class BuildResult:
    data_yaml: Path
    train_txt: Path
    val_txt: Path
    test_txt: Path
    train_unique_count: int
    train_repeat_count: int
    train_pos_count: int
    train_neg_count: int
    train_dog_count: int
    val_count: int
    test_count: int


def safe_div(a: float, b: float) -> float:
    return float(a / b) if b > 0 else 0.0


def iou_xyxy(a: list[float], b: list[float]) -> float:
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


def compute_ap(recall: np.ndarray, precision: np.ndarray) -> float:
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([0.0], precision, [0.0]))
    mpre = np.maximum.accumulate(mpre[::-1])[::-1]
    x = np.linspace(0, 1, 101)
    y = np.interp(x, mrec, mpre)
    return float(np.trapezoid(y, x))


def eval_map(preds: list[list[dict]], gts: list[list[dict]]) -> dict:
    aps = np.zeros((3, len(IOU_THRESHOLDS)), dtype=np.float64)

    for cls_id in range(3):
        for ti, thr in enumerate(IOU_THRESHOLDS):
            confs = []
            tps = []
            n_gt = 0
            for idx in range(len(gts)):
                gt_c = [x["box"] for x in gts[idx] if x["cls"] == cls_id]
                pr_c = [x for x in preds[idx] if x["cls"] == cls_id]
                pr_c = sorted(pr_c, key=lambda x: x["conf"], reverse=True)
                n_gt += len(gt_c)
                used = [False] * len(gt_c)
                for p in pr_c:
                    confs.append(float(p["conf"]))
                    best_iou = 0.0
                    best_j = -1
                    for j, gb in enumerate(gt_c):
                        if used[j]:
                            continue
                        i = iou_xyxy(gb, p["box"])
                        if i >= thr and i > best_iou:
                            best_iou = i
                            best_j = j
                    if best_j >= 0:
                        used[best_j] = True
                        tps.append(1.0)
                    else:
                        tps.append(0.0)
            if n_gt == 0 or len(confs) == 0:
                aps[cls_id, ti] = 0.0
                continue

            conf_arr = np.array(confs, dtype=np.float64)
            tp_arr = np.array(tps, dtype=np.float64)
            order = np.argsort(-conf_arr)
            tp_arr = tp_arr[order]
            fp_arr = 1.0 - tp_arr
            ctp = np.cumsum(tp_arr)
            cfp = np.cumsum(fp_arr)
            rec = ctp / (n_gt + 1e-16)
            pre = ctp / (ctp + cfp + 1e-16)
            aps[cls_id, ti] = compute_ap(rec, pre)

    ap50_cls = aps[:, 0]
    ap75_idx = int(np.where(np.isclose(IOU_THRESHOLDS, 0.75))[0][0])
    ap75_cls = aps[:, ap75_idx]
    ap5095_cls = aps.mean(axis=1)
    return {
        "map50": float(ap50_cls.mean()),
        "map75": float(ap75_cls.mean()),
        "map50_95": float(ap5095_cls.mean()),
        "class_ap50_95": {
            "person": float(ap5095_cls[0]),
            "car": float(ap5095_cls[1]),
            "dog": float(ap5095_cls[2]),
        },
        "class_ap50": {
            "person": float(ap50_cls[0]),
            "car": float(ap50_cls[1]),
            "dog": float(ap50_cls[2]),
        },
    }


def load_gts(image_paths: list[Path]) -> list[list[dict]]:
    out = []
    for ip in image_paths:
        im = cv2.imread(str(ip))
        if im is None:
            out.append([])
            continue
        h, w = im.shape[:2]
        lp = Path(str(ip).replace("\\images\\", "\\labels\\")).with_suffix(".txt")
        one = []
        if lp.exists():
            for ln in lp.read_text(encoding="utf-8").splitlines():
                ln = ln.strip()
                if not ln:
                    continue
                p = ln.split()
                cls = int(float(p[0]))
                if cls not in {0, 1, 2}:
                    continue
                x, y, bw, bh = [float(v) for v in p[1:5]]
                one.append(
                    {
                        "cls": cls,
                        "box": [
                            (x - bw / 2) * w,
                            (y - bh / 2) * h,
                            (x + bw / 2) * w,
                            (y + bh / 2) * h,
                        ],
                    }
                )
        out.append(one)
    return out


def predict_target(model_ref: str | Path, image_paths: list[Path], imgsz: int = 640) -> list[list[dict]]:
    m = YOLO(str(model_ref))
    rs = m.predict(
        source=[str(x) for x in image_paths],
        conf=0.001,
        iou=0.7,
        max_det=300,
        imgsz=int(imgsz),
        verbose=False,
        save=False,
        device=0,
    )
    out = []
    for r in rs:
        one = []
        if r.boxes is not None and len(r.boxes) > 0:
            cls = r.boxes.cls.cpu().numpy().astype(int)
            conf = r.boxes.conf.cpu().numpy().astype(float)
            box = r.boxes.xyxy.cpu().numpy().astype(float)
            for c, s, b in zip(cls, conf, box):
                cname = m.names[int(c)]
                if cname not in NAME2IDX:
                    continue
                one.append({"cls": int(NAME2IDX[cname]), "conf": float(s), "box": b.tolist()})
        one.sort(key=lambda x: x["conf"], reverse=True)
        out.append(one)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return out


def link_or_copy(src: Path, dst: Path):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return
    try:
        os.link(str(src), str(dst))
    except Exception:
        shutil.copy2(src, dst)


def xml_to_yolo_label(xml_path: Path) -> tuple[str, set[str]]:
    root = ET.parse(xml_path).getroot()
    size = root.find("size")
    w = float(size.findtext("width"))
    h = float(size.findtext("height"))
    lines = []
    seen = set()
    for obj in root.findall("object"):
        cname = obj.findtext("name")
        if cname not in NAME2IDX:
            continue
        bb = obj.find("bndbox")
        x1 = max(0.0, float(bb.findtext("xmin")))
        y1 = max(0.0, float(bb.findtext("ymin")))
        x2 = min(w, float(bb.findtext("xmax")))
        y2 = min(h, float(bb.findtext("ymax")))
        if x2 <= x1 or y2 <= y1:
            continue
        cid = NAME2IDX[cname]
        cx = (x1 + x2) * 0.5 / w
        cy = (y1 + y2) * 0.5 / h
        bw = (x2 - x1) / w
        bh = (y2 - y1) / h
        lines.append(f"{cid} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
        seen.add(cname)
    return "\n".join(lines), seen


def read_list(p: Path) -> list[Path]:
    return [Path(x.strip()) for x in p.read_text(encoding="utf-8").splitlines() if x.strip()]


def prepare_dataset() -> BuildResult:
    if DATASET.exists():
        shutil.rmtree(DATASET)
    (DATASET / "images" / "train").mkdir(parents=True, exist_ok=True)
    (DATASET / "labels" / "train").mkdir(parents=True, exist_ok=True)
    (DATASET / "images" / "val").mkdir(parents=True, exist_ok=True)
    (DATASET / "labels" / "val").mkdir(parents=True, exist_ok=True)
    (DATASET / "images" / "test").mkdir(parents=True, exist_ok=True)
    (DATASET / "labels" / "test").mkdir(parents=True, exist_ok=True)

    proto_val = read_list(PROTOCOL / "val.txt")
    proto_test = read_list(PROTOCOL / "test.txt")
    holdout_stems = {x.stem for x in (proto_val + proto_test)}

    ids = [x.strip() for x in (VOC07 / "ImageSets" / "Main" / "trainval.txt").read_text(encoding="utf-8").splitlines() if x.strip()]

    pos_ids = []
    neg_ids = []
    dog_ids = []
    labels = {}

    for id_ in ids:
        if id_ in holdout_stems:
            continue
        xmlp = VOC07 / "Annotations" / f"{id_}.xml"
        lb, seen = xml_to_yolo_label(xmlp)
        labels[id_] = lb
        if seen:
            pos_ids.append(id_)
            if "dog" in seen:
                dog_ids.append(id_)
        else:
            neg_ids.append(id_)

    rng = random.Random(SEED_BUILD)
    rng.shuffle(pos_ids)
    rng.shuffle(neg_ids)

    neg_ratio = 0.45
    dog_repeat_extra = 3
    neg_pick = neg_ids[: int(len(pos_ids) * neg_ratio)]
    train_seq = list(pos_ids) + list(neg_pick) + list(dog_ids) * dog_repeat_extra

    unique_ids = sorted(set(train_seq))
    for id_ in unique_ids:
        src_img = VOC07 / "JPEGImages" / f"{id_}.jpg"
        dst_img = DATASET / "images" / "train" / f"{id_}.jpg"
        dst_lb = DATASET / "labels" / "train" / f"{id_}.txt"
        link_or_copy(src_img, dst_img)
        dst_lb.write_text(labels[id_], encoding="utf-8")

    train_txt = DATASET / "train.txt"
    train_txt.write_text(
        "\n".join([str((DATASET / "images" / "train" / f"{id_}.jpg").resolve()) for id_ in train_seq]),
        encoding="utf-8",
    )

    val_lines = []
    for src_ip in proto_val:
        src_lp = Path(str(src_ip).replace("\\images\\", "\\labels\\")).with_suffix(".txt")
        dst_ip = DATASET / "images" / "val" / src_ip.name
        dst_lp = DATASET / "labels" / "val" / f"{src_ip.stem}.txt"
        link_or_copy(src_ip, dst_ip)
        link_or_copy(src_lp, dst_lp)
        val_lines.append(str(dst_ip.resolve()))

    test_lines = []
    for src_ip in proto_test:
        src_lp = Path(str(src_ip).replace("\\images\\", "\\labels\\")).with_suffix(".txt")
        dst_ip = DATASET / "images" / "test" / src_ip.name
        dst_lp = DATASET / "labels" / "test" / f"{src_ip.stem}.txt"
        link_or_copy(src_ip, dst_ip)
        link_or_copy(src_lp, dst_lp)
        test_lines.append(str(dst_ip.resolve()))

    val_txt = DATASET / "val.txt"
    test_txt = DATASET / "test.txt"
    val_txt.write_text("\n".join(val_lines), encoding="utf-8")
    test_txt.write_text("\n".join(test_lines), encoding="utf-8")

    data_yaml = DATASET / "data.yaml"
    data_yaml.write_text(
        yaml.safe_dump(
            {
                "path": str(DATASET.resolve()),
                "train": str(train_txt.resolve()),
                "val": str(val_txt.resolve()),
                "test": str(test_txt.resolve()),
                "nc": 3,
                "names": TARGET,
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    return BuildResult(
        data_yaml=data_yaml,
        train_txt=train_txt,
        val_txt=val_txt,
        test_txt=test_txt,
        train_unique_count=len(unique_ids),
        train_repeat_count=len(train_seq),
        train_pos_count=len(pos_ids),
        train_neg_count=len(neg_pick),
        train_dog_count=len(dog_ids),
        val_count=len(val_lines),
        test_count=len(test_lines),
    )


def train_stage(
    tag: str,
    data_yaml: Path,
    cfg: dict,
    seed: int,
    head_epochs: int,
    fin_epochs: int,
    force: bool = False,
) -> Path:
    RUNS.mkdir(parents=True, exist_ok=True)

    head_best = RUNS / f"{tag}_head" / "weights" / "best.pt"
    fin_best = RUNS / f"{tag}_full" / "weights" / "best.pt"

    if fin_best.exists() and not force:
        return fin_best

    if force:
        for d in [RUNS / f"{tag}_head", RUNS / f"{tag}_full"]:
            if d.exists():
                shutil.rmtree(d)

    if head_epochs > 0 and not head_best.exists():
        YOLO("yolov8n.pt").train(
            data=str(data_yaml),
            model="yolov8n.pt",
            epochs=int(head_epochs),
            imgsz=int(cfg["imgsz"]),
            batch=int(cfg["batch"]),
            optimizer=str(cfg["optimizer"]),
            lr0=float(cfg["lr0_head"]),
            lrf=float(cfg["lrf"]),
            momentum=float(cfg["momentum"]),
            weight_decay=float(cfg["weight_decay"]),
            warmup_epochs=float(cfg["warmup_epochs"]),
            box=float(cfg["box"]),
            cls=float(cfg["cls"]),
            dfl=float(cfg["dfl"]),
            hsv_h=float(cfg["hsv_h"]),
            hsv_s=float(cfg["hsv_s"]),
            hsv_v=float(cfg["hsv_v"]),
            translate=float(cfg["translate"]),
            scale=float(cfg["scale"]),
            fliplr=float(cfg["fliplr"]),
            mosaic=float(cfg["mosaic_head"]),
            mixup=0.0,
            cutmix=0.0,
            close_mosaic=2,
            freeze=22,
            project=str(RUNS),
            name=f"{tag}_head",
            exist_ok=True,
            verbose=False,
            workers=0,
            seed=int(seed),
            deterministic=True,
            resume=False,
            device=0,
            amp=True,
            patience=20,
        )
    if not head_best.exists():
        head_best = RUNS / f"{tag}_head" / "weights" / "last.pt"

    if not fin_best.exists():
        YOLO(str(head_best if head_epochs > 0 else "yolov8n.pt")).train(
            data=str(data_yaml),
            model=str(head_best if head_epochs > 0 else "yolov8n.pt"),
            epochs=int(fin_epochs),
            imgsz=int(cfg["imgsz"]),
            batch=int(cfg["batch"]),
            optimizer=str(cfg["optimizer"]),
            lr0=float(cfg["lr0"]),
            lrf=float(cfg["lrf"]),
            momentum=float(cfg["momentum"]),
            weight_decay=float(cfg["weight_decay"]),
            warmup_epochs=float(cfg["warmup_epochs"]),
            box=float(cfg["box"]),
            cls=float(cfg["cls"]),
            dfl=float(cfg["dfl"]),
            hsv_h=float(cfg["hsv_h"]),
            hsv_s=float(cfg["hsv_s"]),
            hsv_v=float(cfg["hsv_v"]),
            translate=float(cfg["translate"]),
            scale=float(cfg["scale"]),
            fliplr=float(cfg["fliplr"]),
            mosaic=float(cfg["mosaic"]),
            mixup=0.0,
            cutmix=0.0,
            close_mosaic=int(cfg["close_mosaic"]),
            freeze=0,
            project=str(RUNS),
            name=f"{tag}_full",
            exist_ok=True,
            verbose=False,
            workers=0,
            seed=int(seed),
            deterministic=True,
            resume=False,
            device=0,
            amp=True,
            patience=25,
        )
    if not fin_best.exists():
        fin_best = RUNS / f"{tag}_full" / "weights" / "last.pt"
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return fin_best


def plot_compare(b0: dict, best: dict):
    fig, ax = plt.subplots(figsize=(7, 4.5), dpi=160)
    labels = ["mAP50", "mAP75", "mAP50-95"]
    x = np.arange(len(labels))
    w = 0.36
    b0v = [b0["map50"], b0["map75"], b0["map50_95"]]
    bv = [best["metrics_test"]["map50"], best["metrics_test"]["map75"], best["metrics_test"]["map50_95"]]
    ax.bar(x - w / 2, b0v, width=w, label="B0 (COCO raw)", color="#4e79a7")
    ax.bar(x + w / 2, bv, width=w, label="Last Squeeze", color="#f28e2b")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1)
    ax.set_title("B0 vs Last Squeeze (Test)")
    ax.legend()
    for i, v in enumerate(b0v):
        ax.text(i - w / 2, v + 0.012, f"{v:.3f}", ha="center", fontsize=9)
    for i, v in enumerate(bv):
        ax.text(i + w / 2, v + 0.012, f"{v:.3f}", ha="center", fontsize=9)
    fig.tight_layout()
    fig.savefig(FIG_COMPARE)
    plt.close(fig)


def plot_class_ap(b0: dict, best: dict):
    fig, ax = plt.subplots(figsize=(7, 4.5), dpi=160)
    x = np.arange(3)
    w = 0.36
    b0v = [b0["class_ap50_95"]["person"], b0["class_ap50_95"]["car"], b0["class_ap50_95"]["dog"]]
    bv = [
        best["metrics_test"]["class_ap50_95"]["person"],
        best["metrics_test"]["class_ap50_95"]["car"],
        best["metrics_test"]["class_ap50_95"]["dog"],
    ]
    ax.bar(x - w / 2, b0v, width=w, label="B0 (COCO raw)", color="#4e79a7")
    ax.bar(x + w / 2, bv, width=w, label="Last Squeeze", color="#f28e2b")
    ax.set_xticks(x)
    ax.set_xticklabels(TARGET)
    ax.set_ylim(0, 1)
    ax.set_title("Class AP50-95 (Test)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG_CLASS)
    plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    RUNS.mkdir(parents=True, exist_ok=True)

    build = prepare_dataset()

    test_images = read_list(build.test_txt)
    val_images = read_list(build.val_txt)
    test_gts = load_gts(test_images)
    val_gts = load_gts(val_images)

    b0_pred_test = predict_target("yolov8n.pt", test_images, imgsz=640)
    b0_test = eval_map(b0_pred_test, test_gts)
    b0_pred_val = predict_target("yolov8n.pt", val_images, imgsz=640)
    b0_val = eval_map(b0_pred_val, val_gts)

    quick_cfgs = [
        {
            "name": "q_sgd_bal",
            "optimizer": "SGD",
            "imgsz": 640,
            "batch": 16,
            "lr0_head": 0.0014,
            "lr0": 0.00055,
            "lrf": 0.05,
            "momentum": 0.937,
            "weight_decay": 0.00045,
            "warmup_epochs": 2.0,
            "box": 7.8,
            "cls": 0.45,
            "dfl": 1.35,
            "hsv_h": 0.015,
            "hsv_s": 0.65,
            "hsv_v": 0.35,
            "translate": 0.08,
            "scale": 0.45,
            "fliplr": 0.5,
            "mosaic_head": 0.20,
            "mosaic": 0.10,
            "close_mosaic": 1,
        },
        {
            "name": "q_adamw_lowaug",
            "optimizer": "AdamW",
            "imgsz": 640,
            "batch": 16,
            "lr0_head": 0.0010,
            "lr0": 0.00035,
            "lrf": 0.02,
            "momentum": 0.90,
            "weight_decay": 0.00080,
            "warmup_epochs": 2.5,
            "box": 8.0,
            "cls": 0.40,
            "dfl": 1.45,
            "hsv_h": 0.010,
            "hsv_s": 0.55,
            "hsv_v": 0.30,
            "translate": 0.06,
            "scale": 0.40,
            "fliplr": 0.5,
            "mosaic_head": 0.15,
            "mosaic": 0.05,
            "close_mosaic": 1,
        },
    ]

    quick_rows = []
    for cfg in quick_cfgs:
        tag = f"quick_{cfg['name']}"
        print("[quick-train]", tag)
        pt = train_stage(
            tag=tag,
            data_yaml=build.data_yaml,
            cfg=cfg,
            seed=SEED_QUICK,
            head_epochs=3,
            fin_epochs=7,
            force=False,
        )
        pred_val = predict_target(str(pt), val_images, imgsz=640)
        met_val = eval_map(pred_val, val_gts)
        quick_rows.append({"tag": tag, "pt": str(pt), **cfg, **{f"val_{k}": v for k, v in met_val.items() if not isinstance(v, dict)}})
        print("[quick-val]", tag, "map50-95=", f"{met_val['map50_95']:.4f}")

    quick_df = pd.DataFrame(quick_rows).sort_values("val_map50_95", ascending=False).reset_index(drop=True)
    quick_df.to_csv(QUICK_CSV, index=False, encoding="utf-8-sig")
    best_cfg = quick_df.iloc[0].to_dict()

    cfg_full = next(cfg for cfg in quick_cfgs if f"quick_{cfg['name']}" == best_cfg["tag"])
    final_tag = f"final_{cfg_full['name']}"
    print("[full-train]", final_tag)
    pt_final = train_stage(
        tag=final_tag,
        data_yaml=build.data_yaml,
        cfg=cfg_full,
        seed=SEED_FULL,
        head_epochs=8,
        fin_epochs=24,
        force=False,
    )

    pred_test = predict_target(str(pt_final), test_images, imgsz=640)
    met_test = eval_map(pred_test, test_gts)
    pred_val = predict_target(str(pt_final), val_images, imgsz=640)
    met_val = eval_map(pred_val, val_gts)

    best = {
        "tag": final_tag,
        "pt": str(pt_final),
        "cfg": cfg_full,
        "metrics_val": met_val,
        "metrics_test": met_test,
        "delta_test_vs_b0": {
            "map50": met_test["map50"] - b0_test["map50"],
            "map75": met_test["map75"] - b0_test["map75"],
            "map50_95": met_test["map50_95"] - b0_test["map50_95"],
            "person_ap50_95": met_test["class_ap50_95"]["person"] - b0_test["class_ap50_95"]["person"],
            "car_ap50_95": met_test["class_ap50_95"]["car"] - b0_test["class_ap50_95"]["car"],
            "dog_ap50_95": met_test["class_ap50_95"]["dog"] - b0_test["class_ap50_95"]["dog"],
        },
    }

    plot_compare(b0_test, best)
    plot_class_ap(b0_test, best)

    manifest = {
        "dataset": {
            "root": str(DATASET.resolve()),
            "data_yaml": str(build.data_yaml.resolve()),
            "train_unique_count": build.train_unique_count,
            "train_repeat_count": build.train_repeat_count,
            "train_pos_count": build.train_pos_count,
            "train_neg_count": build.train_neg_count,
            "train_dog_count": build.train_dog_count,
            "val_count": build.val_count,
            "test_count": build.test_count,
        },
        "b0_val": b0_val,
        "b0_test": b0_test,
        "quick_csv": str(QUICK_CSV.resolve()),
        "quick_winner_tag": best_cfg["tag"],
        "final": best,
        "fig_compare": str(FIG_COMPARE.resolve()),
        "fig_class_ap": str(FIG_CLASS.resolve()),
        "beats_b0_on_test_map50_95": bool(best["delta_test_vs_b0"]["map50_95"] > 0.0),
    }
    MANIFEST.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    report_lines = [
        "# Last Squeeze Report (CV1)",
        "",
        "## Setup",
        "- Model fixed: `yolov8n.pt`",
        "- Goal: surpass B0(COCO raw) on protocol test mAP50-95",
        f"- Train set expanded from VOC2007 trainval (target+background): unique={build.train_unique_count}, repeated={build.train_repeat_count}",
        f"- Holdout: val={build.val_count}, test={build.test_count}",
        "",
        "## B0 (fair mapping eval)",
        f"- val: mAP50={b0_val['map50']:.4f}, mAP75={b0_val['map75']:.4f}, mAP50-95={b0_val['map50_95']:.4f}",
        f"- test: mAP50={b0_test['map50']:.4f}, mAP75={b0_test['map75']:.4f}, mAP50-95={b0_test['map50_95']:.4f}",
        "",
        "## Final Model",
        f"- run: `{final_tag}`",
        f"- test: mAP50={met_test['map50']:.4f}, mAP75={met_test['map75']:.4f}, mAP50-95={met_test['map50_95']:.4f}",
        f"- delta vs B0(test): mAP50={best['delta_test_vs_b0']['map50']:+.4f}, mAP75={best['delta_test_vs_b0']['map75']:+.4f}, mAP50-95={best['delta_test_vs_b0']['map50_95']:+.4f}",
        f"- beats B0 on mAP50-95: `{manifest['beats_b0_on_test_map50_95']}`",
        "",
        f"![compare]({FIG_COMPARE.as_posix()})",
        "",
        f"![class_ap]({FIG_CLASS.as_posix()})",
    ]
    REPORT.write_text("\n".join(report_lines), encoding="utf-8")

    print("[saved]", QUICK_CSV)
    print("[saved]", MANIFEST)
    print("[saved]", REPORT)
    print("[summary]", json.dumps({"b0_test_map50_95": b0_test["map50_95"], "final_test_map50_95": met_test["map50_95"], "delta": best["delta_test_vs_b0"]["map50_95"]}))


if __name__ == "__main__":
    main()
