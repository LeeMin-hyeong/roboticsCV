from __future__ import annotations

import json
import math
import random
import shutil
from dataclasses import dataclass
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

VOC3L = ROOT / "voc3_large"
TARGET = ["person", "car", "dog"]
TARGET_SET = set(TARGET)

PROTOCOL_DIR = ROOT / "protocol_split_cv1"
RUNS_STAGE1 = ROOT / "protocol_stage1_runs"
RUNS_STAGE2 = ROOT / "protocol_stage2_runs"
RUNS_FINAL = ROOT / "protocol_final_runs"

SPLIT_META_JSON = OUT / "protocol_split_meta_cv1.json"
STAGE1_CSV = OUT / "protocol_stage1_screen_cv1.csv"
STAGE2_CSV = OUT / "protocol_stage2_refine_cv1.csv"
SEED_CSV = OUT / "protocol_seed_results_cv1.csv"
BOOT_JSON = OUT / "protocol_bootstrap_cv1.json"
REPORT_MD = OUT / "protocol_report_cv1.md"
MANIFEST = OUT / "protocol_manifest_cv1.json"

FIG_STAGE1 = FIG / "protocol_stage1_top_cv1.png"
FIG_STAGE2 = FIG / "protocol_stage2_compare_cv1.png"
FIG_TEST = FIG / "protocol_test_compare_cv1.png"
FIG_CLASS = FIG / "protocol_class_ap_cv1.png"
FIG_SEED = FIG / "protocol_seed_delta_cv1.png"
FIG_BOOT = FIG / "protocol_bootstrap_delta_cv1.png"

SEED_SPLIT = 20260418
SEEDS = [101, 202, 303, 404, 505]
BOOT_N = 10000
MIN_EFFECT = 0.005

N_STAGE1_TRIALS = 8
STAGE1_EPOCHS = 24
STAGE2_FULL_EPOCHS = 45
STAGE2_HEAD_EPOCHS = 10
TOPK_STAGE2 = 3

IOU_THRESHOLDS = np.arange(0.5, 0.96, 0.05)


@dataclass
class ProtocolPaths:
    data_yaml: Path
    train_txt: Path
    val_txt: Path
    test_txt: Path
    train_images: list[Path]
    val_images: list[Path]
    test_images: list[Path]


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


def label_presence_signature(label_path: Path) -> str:
    seen = set()
    for cls, _xywh in parse_label_file(label_path):
        if cls in {0, 1, 2}:
            seen.add(cls)
    bits = ["1" if i in seen else "0" for i in [0, 1, 2]]
    return "".join(bits)


def split_protocol() -> ProtocolPaths:
    PROTOCOL_DIR.mkdir(parents=True, exist_ok=True)
    train_src = sorted([p for p in (VOC3L / "images" / "train").glob("*.jpg")])
    test_src = sorted([p for p in (VOC3L / "images" / "val").glob("*.jpg")])

    rng = random.Random(SEED_SPLIT)
    by_sig: dict[str, list[Path]] = {}
    for ip in train_src:
        lp = VOC3L / "labels" / "train" / f"{ip.stem}.txt"
        sig = label_presence_signature(lp)
        by_sig.setdefault(sig, []).append(ip)

    train_sel: list[Path] = []
    val_sel: list[Path] = []
    for sig, imgs in sorted(by_sig.items()):
        imgs = list(imgs)
        rng.shuffle(imgs)
        n_val = max(1, int(round(len(imgs) * 0.2)))
        val_sel.extend(imgs[:n_val])
        train_sel.extend(imgs[n_val:])

    train_sel = sorted(train_sel)
    val_sel = sorted(val_sel)
    test_sel = test_src

    train_txt = PROTOCOL_DIR / "train.txt"
    val_txt = PROTOCOL_DIR / "val.txt"
    test_txt = PROTOCOL_DIR / "test.txt"
    train_txt.write_text("\n".join([str(p.resolve()) for p in train_sel]), encoding="utf-8")
    val_txt.write_text("\n".join([str(p.resolve()) for p in val_sel]), encoding="utf-8")
    test_txt.write_text("\n".join([str(p.resolve()) for p in test_sel]), encoding="utf-8")

    data_yaml = PROTOCOL_DIR / "data_protocol.yaml"
    data_yaml.write_text(
        yaml.safe_dump(
            {
                "path": str(ROOT.resolve()),
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

    split_meta = {
        "seed_split": SEED_SPLIT,
        "train_count": len(train_sel),
        "val_count": len(val_sel),
        "test_count": len(test_sel),
        "train_txt": str(train_txt.resolve()),
        "val_txt": str(val_txt.resolve()),
        "test_txt": str(test_txt.resolve()),
        "data_yaml": str(data_yaml.resolve()),
        "signature_counts_train_src": {k: len(v) for k, v in by_sig.items()},
    }
    SPLIT_META_JSON.write_text(json.dumps(split_meta, indent=2), encoding="utf-8")

    return ProtocolPaths(
        data_yaml=data_yaml,
        train_txt=train_txt,
        val_txt=val_txt,
        test_txt=test_txt,
        train_images=train_sel,
        val_images=val_sel,
        test_images=test_sel,
    )


def maybe_remove_dir(d: Path):
    if d.exists():
        shutil.rmtree(d)


def train_once(
    run_dir: Path,
    run_name: str,
    data_yaml: Path,
    base_model: str | Path,
    cfg: dict,
    epochs: int,
    seed: int,
    freeze: int,
    force: bool = False,
) -> Path:
    exp = run_dir / run_name
    best = exp / "weights" / "best.pt"
    if best.exists() and not force:
        return best
    if exp.exists():
        shutil.rmtree(exp)

    model = YOLO(str(base_model))
    kwargs = dict(
        data=str(data_yaml),
        model=str(base_model),
        epochs=int(epochs),
        imgsz=int(cfg["imgsz"]),
        batch=int(cfg["batch"]),
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
        mixup=float(cfg["mixup"]),
        cutmix=float(cfg["cutmix"]),
        close_mosaic=int(cfg["close_mosaic"]),
        freeze=int(freeze),
        optimizer=str(cfg["optimizer"]),
        project=str(run_dir),
        name=run_name,
        exist_ok=True,
        verbose=False,
        workers=0,
        seed=int(seed),
        deterministic=True,
        resume=False,
        device=0,
        amp=True,
        patience=30,
    )
    model.train(**kwargs)
    if not best.exists():
        best = exp / "weights" / "last.pt"
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return best


def val_map(weights: Path | str, data_yaml: Path, imgsz: int, batch: int) -> dict:
    m = YOLO(str(weights)).val(
        data=str(data_yaml),
        split="val",
        imgsz=int(imgsz),
        batch=int(batch),
        workers=0,
        verbose=False,
        device=0,
    )
    out = {
        "val_map50": float(m.box.map50),
        "val_map75": float(m.box.map75),
        "val_map50_95": float(m.box.map),
        "val_precision": float(m.box.mp),
        "val_recall": float(m.box.mr),
    }
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return out


def random_cfg(rng: random.Random) -> dict:
    imgsz = rng.choice([640, 800])
    batch = 16 if imgsz == 640 else 12
    cfg = {
        "optimizer": rng.choice(["AdamW", "SGD"]),
        "imgsz": imgsz,
        "batch": batch,
        "lr0": 10 ** rng.uniform(-3.7, -2.4),
        "lrf": rng.uniform(0.03, 0.2),
        "momentum": rng.uniform(0.86, 0.97),
        "weight_decay": 10 ** rng.uniform(-5.0, -3.0),
        "warmup_epochs": rng.uniform(1.0, 4.0),
        "box": rng.uniform(5.5, 8.5),
        "cls": rng.uniform(0.25, 0.8),
        "dfl": rng.uniform(1.0, 2.0),
        "hsv_h": rng.uniform(0.0, 0.03),
        "hsv_s": rng.uniform(0.45, 0.9),
        "hsv_v": rng.uniform(0.2, 0.5),
        "translate": rng.uniform(0.0, 0.2),
        "scale": rng.uniform(0.35, 0.8),
        "fliplr": rng.uniform(0.0, 0.5),
        "mosaic": rng.uniform(0.0, 0.7),
        "mixup": rng.uniform(0.0, 0.25),
        "cutmix": rng.uniform(0.0, 0.25),
        "close_mosaic": rng.randint(1, 5),
        "freeze_screen": rng.choice([0, 10]),
    }
    return cfg


def stage1_screen(paths: ProtocolPaths) -> pd.DataFrame:
    RUNS_STAGE1.mkdir(parents=True, exist_ok=True)
    rng = random.Random(3407)
    rows = []
    for i in range(N_STAGE1_TRIALS):
        cfg = random_cfg(rng)
        tag = f"s1_t{i+1:02d}"
        print("[stage1-train]", tag)
        best = train_once(
            run_dir=RUNS_STAGE1,
            run_name=tag,
            data_yaml=paths.data_yaml,
            base_model="yolov8n.pt",
            cfg=cfg,
            epochs=STAGE1_EPOCHS,
            seed=42,
            freeze=int(cfg["freeze_screen"]),
            force=False,
        )
        vm = val_map(best, paths.data_yaml, imgsz=int(cfg["imgsz"]), batch=int(cfg["batch"]))
        row = {
            "trial": tag,
            "best_pt": str(best),
            **cfg,
            **vm,
        }
        rows.append(row)
        print("[stage1-val]", tag, vm["val_map50_95"])
    df = pd.DataFrame(rows).sort_values("val_map50_95", ascending=False).reset_index(drop=True)
    df.to_csv(STAGE1_CSV, index=False, encoding="utf-8-sig")
    return df


def stage2_refine(paths: ProtocolPaths, stage1_df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    RUNS_STAGE2.mkdir(parents=True, exist_ok=True)
    top = stage1_df.head(TOPK_STAGE2).to_dict("records")
    rows = []
    for r in top:
        base_cfg = {
            "optimizer": r["optimizer"],
            "imgsz": int(r["imgsz"]),
            "batch": int(r["batch"]),
            "lr0": float(r["lr0"]),
            "lrf": float(r["lrf"]),
            "momentum": float(r["momentum"]),
            "weight_decay": float(r["weight_decay"]),
            "warmup_epochs": float(r["warmup_epochs"]),
            "box": float(r["box"]),
            "cls": float(r["cls"]),
            "dfl": float(r["dfl"]),
            "hsv_h": float(r["hsv_h"]),
            "hsv_s": float(r["hsv_s"]),
            "hsv_v": float(r["hsv_v"]),
            "translate": float(r["translate"]),
            "scale": float(r["scale"]),
            "fliplr": float(r["fliplr"]),
            "mosaic": float(r["mosaic"]),
            "mixup": float(r["mixup"]),
            "cutmix": float(r["cutmix"]),
            "close_mosaic": int(r["close_mosaic"]),
        }
        # branch A: full finetune
        tag_full = f"s2_{r['trial']}_full"
        print("[stage2-train]", tag_full)
        pt_full = train_once(
            run_dir=RUNS_STAGE2,
            run_name=tag_full,
            data_yaml=paths.data_yaml,
            base_model="yolov8n.pt",
            cfg=base_cfg,
            epochs=STAGE2_FULL_EPOCHS,
            seed=42,
            freeze=0,
            force=False,
        )
        vm_full = val_map(pt_full, paths.data_yaml, imgsz=base_cfg["imgsz"], batch=base_cfg["batch"])
        rows.append(
            {
                "source_trial": r["trial"],
                "branch": "full",
                "run": tag_full,
                "best_pt": str(pt_full),
                **base_cfg,
                "freeze_head": 0,
                "freeze_final": 0,
                **vm_full,
            }
        )
        # branch B: freeze then unfreeze
        tag_head = f"s2_{r['trial']}_head"
        tag_staged = f"s2_{r['trial']}_staged"
        cfg_head = dict(base_cfg)
        cfg_head["mosaic"] = min(cfg_head["mosaic"], 0.2)
        cfg_head["mixup"] = 0.0
        cfg_head["cutmix"] = 0.0
        print("[stage2-train]", tag_head)
        pt_head = train_once(
            run_dir=RUNS_STAGE2,
            run_name=tag_head,
            data_yaml=paths.data_yaml,
            base_model="yolov8n.pt",
            cfg=cfg_head,
            epochs=STAGE2_HEAD_EPOCHS,
            seed=42,
            freeze=22,
            force=False,
        )
        cfg_stage = dict(base_cfg)
        cfg_stage["lr0"] = base_cfg["lr0"] * 0.5
        cfg_stage["mosaic"] = min(base_cfg["mosaic"], 0.1)
        cfg_stage["mixup"] = 0.0
        cfg_stage["cutmix"] = 0.0
        cfg_stage["close_mosaic"] = 1
        print("[stage2-train]", tag_staged)
        pt_staged = train_once(
            run_dir=RUNS_STAGE2,
            run_name=tag_staged,
            data_yaml=paths.data_yaml,
            base_model=pt_head,
            cfg=cfg_stage,
            epochs=STAGE2_FULL_EPOCHS - STAGE2_HEAD_EPOCHS,
            seed=42,
            freeze=0,
            force=False,
        )
        vm_staged = val_map(pt_staged, paths.data_yaml, imgsz=base_cfg["imgsz"], batch=base_cfg["batch"])
        rows.append(
            {
                "source_trial": r["trial"],
                "branch": "staged",
                "run": tag_staged,
                "best_pt": str(pt_staged),
                **cfg_stage,
                "freeze_head": 22,
                "freeze_final": 0,
                **vm_staged,
            }
        )
    df = pd.DataFrame(rows).sort_values("val_map50_95", ascending=False).reset_index(drop=True)
    df.to_csv(STAGE2_CSV, index=False, encoding="utf-8-sig")
    best = df.iloc[0].to_dict()
    return df, best


def train_b1_seed(paths: ProtocolPaths, seed: int, epochs: int = STAGE2_FULL_EPOCHS) -> Path:
    RUNS_FINAL.mkdir(parents=True, exist_ok=True)
    tag = f"b1_seed{seed}"
    exp = RUNS_FINAL / tag
    best = exp / "weights" / "best.pt"
    if best.exists():
        return best
    if exp.exists():
        shutil.rmtree(exp)
    model = YOLO("yolov8n.pt")
    model.train(
        data=str(paths.data_yaml),
        model="yolov8n.pt",
        epochs=int(epochs),
        imgsz=640,
        batch=16,
        project=str(RUNS_FINAL),
        name=tag,
        exist_ok=True,
        verbose=False,
        workers=0,
        seed=int(seed),
        deterministic=True,
        resume=False,
        device=0,
        amp=True,
    )
    if not best.exists():
        best = exp / "weights" / "last.pt"
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return best


def train_mstar_seed(paths: ProtocolPaths, best_cfg: dict, seed: int) -> Path:
    RUNS_FINAL.mkdir(parents=True, exist_ok=True)
    branch = str(best_cfg["branch"])
    run = str(best_cfg["run"])
    cfg = {
        "optimizer": str(best_cfg["optimizer"]),
        "imgsz": int(best_cfg["imgsz"]),
        "batch": int(best_cfg["batch"]),
        "lr0": float(best_cfg["lr0"]),
        "lrf": float(best_cfg["lrf"]),
        "momentum": float(best_cfg["momentum"]),
        "weight_decay": float(best_cfg["weight_decay"]),
        "warmup_epochs": float(best_cfg["warmup_epochs"]),
        "box": float(best_cfg["box"]),
        "cls": float(best_cfg["cls"]),
        "dfl": float(best_cfg["dfl"]),
        "hsv_h": float(best_cfg["hsv_h"]),
        "hsv_s": float(best_cfg["hsv_s"]),
        "hsv_v": float(best_cfg["hsv_v"]),
        "translate": float(best_cfg["translate"]),
        "scale": float(best_cfg["scale"]),
        "fliplr": float(best_cfg["fliplr"]),
        "mosaic": float(best_cfg["mosaic"]),
        "mixup": float(best_cfg["mixup"]),
        "cutmix": float(best_cfg["cutmix"]),
        "close_mosaic": int(best_cfg["close_mosaic"]),
    }

    if branch == "full":
        tag = f"mstar_full_seed{seed}"
        pt = train_once(
            run_dir=RUNS_FINAL,
            run_name=tag,
            data_yaml=paths.data_yaml,
            base_model="yolov8n.pt",
            cfg=cfg,
            epochs=STAGE2_FULL_EPOCHS,
            seed=seed,
            freeze=0,
            force=False,
        )
        return pt

    # staged branch
    cfg_head = dict(cfg)
    cfg_head["mosaic"] = min(cfg_head["mosaic"], 0.2)
    cfg_head["mixup"] = 0.0
    cfg_head["cutmix"] = 0.0
    head_tag = f"mstar_staged_head_seed{seed}"
    head_pt = train_once(
        run_dir=RUNS_FINAL,
        run_name=head_tag,
        data_yaml=paths.data_yaml,
        base_model="yolov8n.pt",
        cfg=cfg_head,
        epochs=STAGE2_HEAD_EPOCHS,
        seed=seed,
        freeze=22,
        force=False,
    )
    cfg_stage = dict(cfg)
    cfg_stage["lr0"] = cfg["lr0"] * 0.5
    cfg_stage["mosaic"] = min(cfg["mosaic"], 0.1)
    cfg_stage["mixup"] = 0.0
    cfg_stage["cutmix"] = 0.0
    cfg_stage["close_mosaic"] = 1
    fin_tag = f"mstar_staged_seed{seed}"
    fin_pt = train_once(
        run_dir=RUNS_FINAL,
        run_name=fin_tag,
        data_yaml=paths.data_yaml,
        base_model=head_pt,
        cfg=cfg_stage,
        epochs=STAGE2_FULL_EPOCHS - STAGE2_HEAD_EPOCHS,
        seed=seed,
        freeze=0,
        force=False,
    )
    return fin_pt


def img_to_label_path(img_path: Path) -> Path:
    p = str(img_path)
    p = p.replace("\\images\\train\\", "\\labels\\train\\")
    p = p.replace("\\images\\val\\", "\\labels\\val\\")
    p = p.replace("/images/train/", "/labels/train/")
    p = p.replace("/images/val/", "/labels/val/")
    return Path(Path(p).with_suffix(".txt"))


def load_test_gts(test_images: list[Path]) -> list[list[dict]]:
    out = []
    for ip in test_images:
        im = cv2.imread(str(ip))
        h, w = im.shape[:2]
        lp = img_to_label_path(ip)
        one = []
        if lp.exists():
            for cls, xywh in parse_label_file(lp):
                if cls not in {0, 1, 2}:
                    continue
                x, y, bw, bh = xywh
                box = [(x - bw / 2) * w, (y - bh / 2) * h, (x + bw / 2) * w, (y + bh / 2) * h]
                one.append({"cls": int(cls), "box": box})
        out.append(one)
    return out


def predict_target(model_ref: str | Path, test_images: list[Path]) -> list[list[dict]]:
    m = YOLO(str(model_ref))
    name2idx = {"person": 0, "car": 1, "dog": 2}
    rs = m.predict(
        source=[str(p) for p in test_images],
        conf=0.001,
        iou=0.7,
        max_det=300,
        imgsz=640,
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
                if cname not in name2idx:
                    continue
                one.append({"cls": int(name2idx[cname]), "conf": float(s), "box": b.tolist()})
        one.sort(key=lambda x: x["conf"], reverse=True)
        out.append(one)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return out


def compute_ap(recall: np.ndarray, precision: np.ndarray) -> float:
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([0.0], precision, [0.0]))
    mpre = np.maximum.accumulate(mpre[::-1])[::-1]
    x = np.linspace(0, 1, 101)
    y = np.interp(x, mrec, mpre)
    return float(np.trapezoid(y, x))


def eval_map(preds: list[list[dict]], gts: list[list[dict]], indices: np.ndarray | None = None) -> dict:
    if indices is None:
        indices = np.arange(len(gts), dtype=np.int64)
    aps = np.zeros((3, len(IOU_THRESHOLDS)), dtype=np.float64)

    for cls_id in range(3):
        for ti, thr in enumerate(IOU_THRESHOLDS):
            confs = []
            tps = []
            n_gt = 0
            for idx in indices.tolist():
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

    out = {
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
    return out


def eval_image_map(preds_one: list[dict], gts_one: list[dict]) -> float:
    aps = np.zeros((3, len(IOU_THRESHOLDS)), dtype=np.float64)
    for cls_id in range(3):
        gt_c = [x["box"] for x in gts_one if x["cls"] == cls_id]
        pr_c = [x for x in preds_one if x["cls"] == cls_id]
        pr_c = sorted(pr_c, key=lambda x: x["conf"], reverse=True)
        for ti, thr in enumerate(IOU_THRESHOLDS):
            n_gt = len(gt_c)
            if n_gt == 0 or len(pr_c) == 0:
                aps[cls_id, ti] = 0.0
                continue
            used = [False] * n_gt
            confs = []
            tps = []
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
    return float(aps.mean())


def image_presence_mask(gts_one: list[dict]) -> int:
    seen = set([g["cls"] for g in gts_one])
    mask = 0
    for c in [0, 1, 2]:
        if c in seen:
            mask |= (1 << c)
    return int(mask)


def stratified_bootstrap_mean(delta: np.ndarray, strata: np.ndarray, n_boot: int, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    uniq = np.unique(strata)
    idx_map = {u: np.where(strata == u)[0] for u in uniq}
    boots = np.zeros(n_boot, dtype=np.float64)
    for b in range(n_boot):
        picks = []
        for u in uniq:
            idxs = idx_map[u]
            sel = rng.choice(idxs, size=len(idxs), replace=True)
            picks.append(sel)
        all_idx = np.concatenate(picks)
        boots[b] = float(np.mean(delta[all_idx]))
    lo, hi = np.percentile(boots, [2.5, 97.5]).tolist()
    p_one_sided = float(np.mean(boots <= 0.0))
    return {
        "mean": float(np.mean(delta)),
        "ci95_low": float(lo),
        "ci95_high": float(hi),
        "p_boot_one_sided": p_one_sided,
        "n_boot": int(n_boot),
    }


def plot_stage1(df: pd.DataFrame):
    top = df.head(8).copy()
    fig, ax = plt.subplots(figsize=(10, 5), dpi=160)
    ax.bar(top["trial"], top["val_map50_95"], color="#4e79a7")
    ax.set_ylim(0, 1)
    ax.set_title("Stage1 Screening Top Trials (val mAP50-95)")
    ax.set_ylabel("val mAP50-95")
    ax.tick_params(axis="x", rotation=25)
    fig.tight_layout()
    fig.savefig(FIG_STAGE1)
    plt.close(fig)


def plot_stage2(df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(10, 5), dpi=160)
    for b, c in [("full", "#59a14f"), ("staged", "#f28e2b")]:
        sub = df[df["branch"] == b]
        ax.scatter(sub["val_map50_95"], sub["val_map50"], label=b, color=c, s=80)
    ax.set_xlabel("val mAP50-95")
    ax.set_ylabel("val mAP50")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title("Stage2 Refine: full vs staged")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG_STAGE2)
    plt.close(fig)


def plot_test_compare(m_b0: dict, m_b1: dict, m_m: dict):
    metrics = ["map50", "map75", "map50_95"]
    labels = ["B0(COCO raw)", "B1(basic FT)", "M*(tuned)"]
    vals = np.array(
        [
            [m_b0[k] for k in metrics],
            [m_b1[k] for k in metrics],
            [m_m[k] for k in metrics],
        ]
    )
    x = np.arange(len(metrics))
    w = 0.25
    fig, ax = plt.subplots(figsize=(9, 5), dpi=160)
    ax.bar(x - w, vals[0], width=w, label=labels[0], color="#4e79a7")
    ax.bar(x, vals[1], width=w, label=labels[1], color="#f28e2b")
    ax.bar(x + w, vals[2], width=w, label=labels[2], color="#59a14f")
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.set_ylim(0, 1)
    ax.set_title("Test Metrics Comparison")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG_TEST)
    plt.close(fig)


def plot_class_ap(m_b0: dict, m_b1: dict, m_m: dict):
    classes = ["person", "car", "dog"]
    x = np.arange(len(classes))
    w = 0.25
    fig, ax = plt.subplots(figsize=(9, 5), dpi=160)
    ax.bar(x - w, [m_b0["class_ap50_95"][c] for c in classes], width=w, label="B0", color="#4e79a7")
    ax.bar(x, [m_b1["class_ap50_95"][c] for c in classes], width=w, label="B1", color="#f28e2b")
    ax.bar(x + w, [m_m["class_ap50_95"][c] for c in classes], width=w, label="M*", color="#59a14f")
    ax.set_xticks(x)
    ax.set_xticklabels(classes)
    ax.set_ylim(0, 1)
    ax.set_title("Class-wise AP50-95 (test)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG_CLASS)
    plt.close(fig)


def plot_seed_delta(seed_df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(9, 5), dpi=160)
    ax.plot(seed_df["seed"], seed_df["delta_vs_b0"], marker="o", label="M*-B0")
    ax.plot(seed_df["seed"], seed_df["delta_vs_b1"], marker="s", label="M*-B1")
    ax.axhline(0.0, linestyle="--", color="black", linewidth=1)
    ax.set_xlabel("seed")
    ax.set_ylabel("delta mAP50-95")
    ax.set_title("Seed-wise Delta on Test")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG_SEED)
    plt.close(fig)


def plot_bootstrap(boot_b0: dict, boot_b1: dict):
    fig, ax = plt.subplots(figsize=(9, 5), dpi=160)
    ax.hist(boot_b0["samples"], bins=40, alpha=0.55, label="Delta(M*-B0)", color="#4e79a7")
    ax.hist(boot_b1["samples"], bins=40, alpha=0.55, label="Delta(M*-B1)", color="#f28e2b")
    ax.axvline(0.0, linestyle="--", color="black", linewidth=1)
    ax.set_title("Bootstrap Delta Distribution (paired stratified)")
    ax.set_xlabel("delta")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG_BOOT)
    plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    RUNS_STAGE1.mkdir(parents=True, exist_ok=True)
    RUNS_STAGE2.mkdir(parents=True, exist_ok=True)
    RUNS_FINAL.mkdir(parents=True, exist_ok=True)

    # 0) protocol split
    paths = split_protocol()

    # 1) stage1 screening
    s1_df = stage1_screen(paths)
    plot_stage1(s1_df)

    # 2) stage2 refine
    s2_df, best_cfg = stage2_refine(paths, s1_df)
    plot_stage2(s2_df)

    # 3) seed=42 baseline B1 and M*
    b1_seed42 = train_b1_seed(paths, seed=42)
    m_seed42 = train_mstar_seed(paths, best_cfg, seed=42)

    # 4) test evaluation pipeline (fair for B0)
    test_images = paths.test_images
    gts = load_test_gts(test_images)

    preds_b0 = predict_target("yolov8n.pt", test_images)
    preds_b1_42 = predict_target(str(b1_seed42), test_images)
    preds_m_42 = predict_target(str(m_seed42), test_images)

    met_b0 = eval_map(preds_b0, gts)
    met_b1_42 = eval_map(preds_b1_42, gts)
    met_m_42 = eval_map(preds_m_42, gts)

    # 5) seed repeats
    rows_seed = []
    preds_b1_seed = {}
    preds_m_seed = {}
    for s in SEEDS:
        print("[seed-train] B1", s)
        pt_b1 = train_b1_seed(paths, seed=s)
        print("[seed-train] M*", s)
        pt_m = train_mstar_seed(paths, best_cfg, seed=s)
        pr_b1 = predict_target(str(pt_b1), test_images)
        pr_m = predict_target(str(pt_m), test_images)
        preds_b1_seed[s] = pr_b1
        preds_m_seed[s] = pr_m
        m_b1 = eval_map(pr_b1, gts)
        m_m = eval_map(pr_m, gts)
        rows_seed.append(
            {
                "seed": s,
                "b1_pt": str(pt_b1),
                "mstar_pt": str(pt_m),
                "b1_map50_95": m_b1["map50_95"],
                "mstar_map50_95": m_m["map50_95"],
                "delta_vs_b0": m_m["map50_95"] - met_b0["map50_95"],
                "delta_vs_b1": m_m["map50_95"] - m_b1["map50_95"],
            }
        )
    seed_df = pd.DataFrame(rows_seed)
    seed_df.to_csv(SEED_CSV, index=False, encoding="utf-8-sig")
    plot_seed_delta(seed_df)

    # mean metrics across seeds for B1 and M*
    b1_seed_maps = []
    m_seed_maps = []
    b1_class = []
    m_class = []
    for s in SEEDS:
        mb1 = eval_map(preds_b1_seed[s], gts)
        mm = eval_map(preds_m_seed[s], gts)
        b1_seed_maps.append(mb1)
        m_seed_maps.append(mm)
        b1_class.append([mb1["class_ap50_95"]["person"], mb1["class_ap50_95"]["car"], mb1["class_ap50_95"]["dog"]])
        m_class.append([mm["class_ap50_95"]["person"], mm["class_ap50_95"]["car"], mm["class_ap50_95"]["dog"]])

    met_b1_mean = {
        "map50": float(np.mean([x["map50"] for x in b1_seed_maps])),
        "map75": float(np.mean([x["map75"] for x in b1_seed_maps])),
        "map50_95": float(np.mean([x["map50_95"] for x in b1_seed_maps])),
        "class_ap50_95": {
            "person": float(np.mean([x[0] for x in b1_class])),
            "car": float(np.mean([x[1] for x in b1_class])),
            "dog": float(np.mean([x[2] for x in b1_class])),
        },
    }
    met_m_mean = {
        "map50": float(np.mean([x["map50"] for x in m_seed_maps])),
        "map75": float(np.mean([x["map75"] for x in m_seed_maps])),
        "map50_95": float(np.mean([x["map50_95"] for x in m_seed_maps])),
        "class_ap50_95": {
            "person": float(np.mean([x[0] for x in m_class])),
            "car": float(np.mean([x[1] for x in m_class])),
            "dog": float(np.mean([x[2] for x in m_class])),
        },
    }

    plot_test_compare(met_b0, met_b1_mean, met_m_mean)
    plot_class_ap(met_b0, met_b1_mean, met_m_mean)

    # 6) paired stratified bootstrap on image-level AP50-95 deltas (pooled seed-image units)
    image_masks = np.array([image_presence_mask(x) for x in gts], dtype=np.int64)
    img_metric_b0 = np.array([eval_image_map(preds_b0[i], gts[i]) for i in range(len(gts))], dtype=np.float64)
    delta_pool_b0 = []
    strata_pool_b0 = []
    delta_pool_b1 = []
    strata_pool_b1 = []
    for s in SEEDS:
        img_m = np.array([eval_image_map(preds_m_seed[s][i], gts[i]) for i in range(len(gts))], dtype=np.float64)
        img_b1 = np.array([eval_image_map(preds_b1_seed[s][i], gts[i]) for i in range(len(gts))], dtype=np.float64)
        delta_pool_b0.append(img_m - img_metric_b0)
        delta_pool_b1.append(img_m - img_b1)
        strata_pool_b0.append(image_masks.copy())
        strata_pool_b1.append(image_masks.copy())
    delta_pool_b0 = np.concatenate(delta_pool_b0)
    strata_pool_b0 = np.concatenate(strata_pool_b0)
    delta_pool_b1 = np.concatenate(delta_pool_b1)
    strata_pool_b1 = np.concatenate(strata_pool_b1)

    # bootstrap samples
    def bootstrap_with_samples(delta: np.ndarray, strata: np.ndarray, n_boot: int, seed: int):
        rng = np.random.default_rng(seed)
        uniq = np.unique(strata)
        idx_map = {u: np.where(strata == u)[0] for u in uniq}
        boots = np.zeros(n_boot, dtype=np.float64)
        for bi in range(n_boot):
            picks = []
            for u in uniq:
                idxs = idx_map[u]
                picks.append(rng.choice(idxs, size=len(idxs), replace=True))
            all_idx = np.concatenate(picks)
            boots[bi] = float(np.mean(delta[all_idx]))
        lo, hi = np.percentile(boots, [2.5, 97.5]).tolist()
        p = float(np.mean(boots <= 0.0))
        return {
            "mean": float(np.mean(delta)),
            "ci95_low": float(lo),
            "ci95_high": float(hi),
            "p_boot_one_sided": p,
            "n_boot": int(n_boot),
            "samples": boots,
        }

    boot_b0 = bootstrap_with_samples(delta_pool_b0, strata_pool_b0, BOOT_N, seed=1901)
    boot_b1 = bootstrap_with_samples(delta_pool_b1, strata_pool_b1, BOOT_N, seed=1902)
    plot_bootstrap(boot_b0, boot_b1)

    # hypothesis criteria
    mean_delta_vs_b0_seed = float(seed_df["delta_vs_b0"].mean())
    mean_delta_vs_b1_seed = float(seed_df["delta_vs_b1"].mean())
    pos_vs_b0 = int((seed_df["delta_vs_b0"] > 0).sum())
    pos_vs_b1 = int((seed_df["delta_vs_b1"] > 0).sum())

    accept_vs_b0 = (
        (boot_b0["ci95_low"] > 0.0)
        and (mean_delta_vs_b0_seed > 0.0)
        and (pos_vs_b0 >= 4)
        and (mean_delta_vs_b0_seed >= MIN_EFFECT)
    )
    accept_vs_b1 = (
        (boot_b1["ci95_low"] > 0.0)
        and (mean_delta_vs_b1_seed > 0.0)
        and (pos_vs_b1 >= 4)
        and (mean_delta_vs_b1_seed >= MIN_EFFECT)
    )

    boot_out = {
        "vs_b0": {k: v for k, v in boot_b0.items() if k != "samples"},
        "vs_b1": {k: v for k, v in boot_b1.items() if k != "samples"},
        "criteria": {
            "min_effect": MIN_EFFECT,
            "seed_count": len(SEEDS),
            "require_positive_seeds": 4,
        },
        "seed_summary": {
            "mean_delta_vs_b0": mean_delta_vs_b0_seed,
            "mean_delta_vs_b1": mean_delta_vs_b1_seed,
            "positive_seeds_vs_b0": pos_vs_b0,
            "positive_seeds_vs_b1": pos_vs_b1,
            "accept_vs_b0": bool(accept_vs_b0),
            "accept_vs_b1": bool(accept_vs_b1),
        },
    }
    BOOT_JSON.write_text(json.dumps(boot_out, indent=2), encoding="utf-8")

    # 7) report
    lines = [
        "# YOLOv8n 3-Class Protocol Report (CV1)",
        "",
        "## Protocol",
        "- Goal: statistically significant improvement over COCO pretrained baseline (B0).",
        "- Primary metric: test mAP50-95. Secondary: mAP50, mAP75, class-wise AP50-95.",
        "- Data protocol: tune on train/val split from voc3_large train; keep voc3_large val as sealed test.",
        "- Baselines: B0(raw COCO), B1(basic fine-tune), M*(tuned).",
        "- Search: 2-stage (screening -> refine) with full vs staged freeze branch.",
        "",
        "## Stage1/Stage2",
        f"- stage1 trials: {N_STAGE1_TRIALS}, epochs={STAGE1_EPOCHS}",
        f"- stage2 topK: {TOPK_STAGE2}, full_epochs={STAGE2_FULL_EPOCHS}, head_epochs={STAGE2_HEAD_EPOCHS}",
        f"- selected M* run: `{best_cfg['run']}` ({best_cfg['branch']}) val mAP50-95={best_cfg['val_map50_95']:.4f}",
        "",
        "![stage1](C:/Users/lmhst/git/roboticsCV/CV_project#1/outputs/figures/protocol_stage1_top_cv1.png)",
        "",
        "![stage2](C:/Users/lmhst/git/roboticsCV/CV_project#1/outputs/figures/protocol_stage2_compare_cv1.png)",
        "",
        "## Test Metrics",
        f"- B0(raw COCO): mAP50={met_b0['map50']:.4f}, mAP75={met_b0['map75']:.4f}, mAP50-95={met_b0['map50_95']:.4f}",
        f"- B1(mean over {len(SEEDS)} seeds): mAP50={met_b1_mean['map50']:.4f}, mAP75={met_b1_mean['map75']:.4f}, mAP50-95={met_b1_mean['map50_95']:.4f}",
        f"- M*(mean over {len(SEEDS)} seeds): mAP50={met_m_mean['map50']:.4f}, mAP75={met_m_mean['map75']:.4f}, mAP50-95={met_m_mean['map50_95']:.4f}",
        "",
        f"- class AP50-95 B0: person={met_b0['class_ap50_95']['person']:.4f}, car={met_b0['class_ap50_95']['car']:.4f}, dog={met_b0['class_ap50_95']['dog']:.4f}",
        f"- class AP50-95 B1(mean): person={met_b1_mean['class_ap50_95']['person']:.4f}, car={met_b1_mean['class_ap50_95']['car']:.4f}, dog={met_b1_mean['class_ap50_95']['dog']:.4f}",
        f"- class AP50-95 M*(mean): person={met_m_mean['class_ap50_95']['person']:.4f}, car={met_m_mean['class_ap50_95']['car']:.4f}, dog={met_m_mean['class_ap50_95']['dog']:.4f}",
        "",
        "![test_compare](C:/Users/lmhst/git/roboticsCV/CV_project#1/outputs/figures/protocol_test_compare_cv1.png)",
        "",
        "![class_ap](C:/Users/lmhst/git/roboticsCV/CV_project#1/outputs/figures/protocol_class_ap_cv1.png)",
        "",
        "## Seed-wise Delta",
        f"- mean delta vs B0: {mean_delta_vs_b0_seed:+.4f} ({pos_vs_b0}/{len(SEEDS)} positive seeds)",
        f"- mean delta vs B1: {mean_delta_vs_b1_seed:+.4f} ({pos_vs_b1}/{len(SEEDS)} positive seeds)",
        "",
        "![seed_delta](C:/Users/lmhst/git/roboticsCV/CV_project#1/outputs/figures/protocol_seed_delta_cv1.png)",
        "",
        "## Paired Stratified Bootstrap (seed-image pooled, image-level AP50-95)",
        f"- vs B0: mean={boot_b0['mean']:+.4f}, 95% CI=[{boot_b0['ci95_low']:+.4f}, {boot_b0['ci95_high']:+.4f}], p(one-sided)={boot_b0['p_boot_one_sided']:.4g}",
        f"- vs B1: mean={boot_b1['mean']:+.4f}, 95% CI=[{boot_b1['ci95_low']:+.4f}, {boot_b1['ci95_high']:+.4f}], p(one-sided)={boot_b1['p_boot_one_sided']:.4g}",
        "",
        "![boot](C:/Users/lmhst/git/roboticsCV/CV_project#1/outputs/figures/protocol_bootstrap_delta_cv1.png)",
        "",
        "## Acceptance Criteria",
        f"- Criteria: CI_low>0 and seed_mean>0 and positive_seeds>=4/5 and seed_mean>={MIN_EFFECT:.3f}",
        f"- Result vs B0: {accept_vs_b0}",
        f"- Result vs B1: {accept_vs_b1}",
    ]
    REPORT_MD.write_text("\n".join(lines), encoding="utf-8")

    manifest = {
        "split_meta_json": str(SPLIT_META_JSON),
        "stage1_csv": str(STAGE1_CSV),
        "stage2_csv": str(STAGE2_CSV),
        "seed_csv": str(SEED_CSV),
        "boot_json": str(BOOT_JSON),
        "report_md": str(REPORT_MD),
        "fig_stage1": str(FIG_STAGE1),
        "fig_stage2": str(FIG_STAGE2),
        "fig_test": str(FIG_TEST),
        "fig_class": str(FIG_CLASS),
        "fig_seed": str(FIG_SEED),
        "fig_boot": str(FIG_BOOT),
        "best_cfg": best_cfg,
        "metrics": {
            "b0": met_b0,
            "b1_mean": met_b1_mean,
            "mstar_mean": met_m_mean,
        },
        "bootstrap": boot_out,
    }
    MANIFEST.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print("[saved]", SPLIT_META_JSON)
    print("[saved]", STAGE1_CSV)
    print("[saved]", STAGE2_CSV)
    print("[saved]", SEED_CSV)
    print("[saved]", BOOT_JSON)
    print("[saved]", REPORT_MD)
    print("[saved]", MANIFEST)
    print(
        "[summary]",
        json.dumps(
            {
                "b0_map5095": met_b0["map50_95"],
                "b1_mean_map5095": met_b1_mean["map50_95"],
                "mstar_mean_map5095": met_m_mean["map50_95"],
                "accept_vs_b0": accept_vs_b0,
                "accept_vs_b1": accept_vs_b1,
            },
            ensure_ascii=False,
        ),
    )


if __name__ == "__main__":
    main()
