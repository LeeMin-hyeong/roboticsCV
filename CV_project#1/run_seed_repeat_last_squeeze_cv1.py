from __future__ import annotations

import itertools
import json
import math
import shutil
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from ultralytics import YOLO

ROOT = Path(r"C:\Users\lmhst\git\roboticsCV\CV_project#1")
OUT = ROOT / "outputs"
FIG = OUT / "figures"
TABLES = OUT / "tables"
METRICS = OUT / "metrics"
REPORTS = OUT / "reports"
MANIFESTS = OUT / "manifests"

RUNS = ROOT / "last_squeeze_runs"
DATA_YAML = ROOT / "datasets_last_squeeze_voc07" / "data.yaml"
TEST_TXT = ROOT / "datasets_last_squeeze_voc07" / "test.txt"
BASE_PT = ROOT / "last_squeeze_runs" / "quick_q_sgd_bal_full" / "weights" / "best.pt"

SEEDS = [101, 202, 303, 404, 505]
BOOT_N = 20000
IOU_THRESHOLDS = np.arange(0.5, 0.96, 0.05)

SEED_CSV = TABLES / "last_squeeze_seed_repeat_cv1.csv"
BOOT_JSON = METRICS / "last_squeeze_seed_repeat_bootstrap_cv1.json"
MANIFEST_JSON = MANIFESTS / "last_squeeze_seed_repeat_manifest_cv1.json"
REPORT_MD = REPORTS / "last_squeeze_seed_repeat_report_cv1.md"
FIG_SEED = FIG / "last_squeeze_seed_repeat_delta_cv1.png"
FIG_BOOT = FIG / "last_squeeze_seed_repeat_bootstrap_cv1.png"

TARGET = ["person", "car", "dog"]
NAME2IDX = {n: i for i, n in enumerate(TARGET)}


TRAIN_CFG = {
    "epochs": 18,
    "imgsz": 640,
    "batch": 16,
    "optimizer": "SGD",
    "lr0": 0.00035,
    "lrf": 0.02,
    "momentum": 0.937,
    "weight_decay": 0.00045,
    "warmup_epochs": 1.0,
    "box": 7.8,
    "cls": 0.45,
    "dfl": 1.35,
    "hsv_h": 0.01,
    "hsv_s": 0.6,
    "hsv_v": 0.3,
    "translate": 0.05,
    "scale": 0.35,
    "fliplr": 0.5,
    "mosaic": 0.05,
    "mixup": 0.0,
    "cutmix": 0.0,
    "close_mosaic": 1,
    "freeze": 0,
    "patience": 20,
}


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


def load_list(p: Path) -> list[Path]:
    return [Path(x.strip()) for x in p.read_text(encoding="utf-8").splitlines() if x.strip()]


def load_gts(image_paths: list[Path]) -> list[list[dict]]:
    out: list[list[dict]] = []
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
    }


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


def train_seed(seed: int, force: bool = False) -> Path:
    run_name = f"squeeze_refine1_seed{seed}"
    exp = RUNS / run_name
    best = exp / "weights" / "best.pt"
    if best.exists() and not force:
        return best
    if exp.exists():
        shutil.rmtree(exp)

    YOLO(str(BASE_PT)).train(
        data=str(DATA_YAML),
        model=str(BASE_PT),
        epochs=int(TRAIN_CFG["epochs"]),
        imgsz=int(TRAIN_CFG["imgsz"]),
        batch=int(TRAIN_CFG["batch"]),
        optimizer=str(TRAIN_CFG["optimizer"]),
        lr0=float(TRAIN_CFG["lr0"]),
        lrf=float(TRAIN_CFG["lrf"]),
        momentum=float(TRAIN_CFG["momentum"]),
        weight_decay=float(TRAIN_CFG["weight_decay"]),
        warmup_epochs=float(TRAIN_CFG["warmup_epochs"]),
        box=float(TRAIN_CFG["box"]),
        cls=float(TRAIN_CFG["cls"]),
        dfl=float(TRAIN_CFG["dfl"]),
        hsv_h=float(TRAIN_CFG["hsv_h"]),
        hsv_s=float(TRAIN_CFG["hsv_s"]),
        hsv_v=float(TRAIN_CFG["hsv_v"]),
        translate=float(TRAIN_CFG["translate"]),
        scale=float(TRAIN_CFG["scale"]),
        fliplr=float(TRAIN_CFG["fliplr"]),
        mosaic=float(TRAIN_CFG["mosaic"]),
        mixup=float(TRAIN_CFG["mixup"]),
        cutmix=float(TRAIN_CFG["cutmix"]),
        close_mosaic=int(TRAIN_CFG["close_mosaic"]),
        freeze=int(TRAIN_CFG["freeze"]),
        project=str(RUNS),
        name=run_name,
        exist_ok=True,
        verbose=False,
        workers=0,
        seed=int(seed),
        deterministic=True,
        resume=False,
        device=0,
        amp=True,
        patience=int(TRAIN_CFG["patience"]),
    )
    if not best.exists():
        best = exp / "weights" / "last.pt"
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return best


def sign_test_p_one_sided(deltas: np.ndarray) -> tuple[int, int, float]:
    n = int(deltas.size)
    k = int(np.sum(deltas > 0))
    # H1: p(positive) > 0.5
    p = 0.0
    for i in range(k, n + 1):
        p += math.comb(n, i) * (0.5 ** n)
    return k, n, float(p)


def permutation_signflip_p_one_sided(deltas: np.ndarray) -> float:
    n = int(deltas.size)
    obs = float(np.mean(deltas))
    vals = []
    for bits in itertools.product([-1.0, 1.0], repeat=n):
        signs = np.array(bits, dtype=np.float64)
        vals.append(float(np.mean(signs * deltas)))
    vals = np.array(vals, dtype=np.float64)
    if obs >= 0:
        return float(np.mean(vals >= obs))
    return 1.0


def bootstrap_seed_mean(deltas: np.ndarray, n_boot: int, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    n = int(deltas.size)
    boots = np.zeros(n_boot, dtype=np.float64)
    for bi in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boots[bi] = float(np.mean(deltas[idx]))
    lo, hi = np.percentile(boots, [2.5, 97.5]).tolist()
    return {
        "mean": float(np.mean(deltas)),
        "ci95_low": float(lo),
        "ci95_high": float(hi),
        "p_boot_one_sided": float(np.mean(boots <= 0.0)),
        "n_boot": int(n_boot),
    }


def paired_bootstrap_image_seed(delta_img: np.ndarray, strata: np.ndarray, n_boot: int, seed: int) -> dict:
    # delta_img shape: [n_seed, n_image]
    rng = np.random.default_rng(seed)
    n_seed, _n_img = delta_img.shape
    uniq = np.unique(strata)
    idx_map = {u: np.where(strata == u)[0] for u in uniq}
    boots = np.zeros(n_boot, dtype=np.float64)
    for bi in range(n_boot):
        picks = []
        for u in uniq:
            idxs = idx_map[u]
            picks.append(rng.choice(idxs, size=len(idxs), replace=True))
        all_idx = np.concatenate(picks)
        # paired: same sampled image indices applied to all seeds
        boots[bi] = float(np.mean(delta_img[:, all_idx]))
    lo, hi = np.percentile(boots, [2.5, 97.5]).tolist()
    return {
        "mean": float(np.mean(delta_img)),
        "ci95_low": float(lo),
        "ci95_high": float(hi),
        "p_boot_one_sided": float(np.mean(boots <= 0.0)),
        "n_boot": int(n_boot),
    }


def plot_seed_delta(seed_df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(9, 5), dpi=170)
    ax.plot(seed_df["seed"], seed_df["delta_map50_95_vs_b0"], marker="o", color="#1f77b4")
    ax.axhline(0.0, linestyle="--", color="black", linewidth=1)
    ax.set_title("Seed-wise Delta vs B0 (mAP50-95)")
    ax.set_xlabel("seed")
    ax.set_ylabel("delta mAP50-95")
    fig.tight_layout()
    fig.savefig(FIG_SEED)
    plt.close(fig)


def plot_bootstrap_hist(seed_boot: dict, image_boot: dict):
    fig, ax = plt.subplots(figsize=(9, 5), dpi=170)
    txt1 = f"Seed-level bootstrap CI: [{seed_boot['ci95_low']:+.4f}, {seed_boot['ci95_high']:+.4f}]"
    txt2 = f"Image-paired bootstrap CI: [{image_boot['ci95_low']:+.4f}, {image_boot['ci95_high']:+.4f}]"
    ax.axis("off")
    ax.text(0.02, 0.75, txt1, fontsize=12)
    ax.text(0.02, 0.55, txt2, fontsize=12)
    ax.text(
        0.02,
        0.30,
        f"Seed bootstrap p(one-sided)={seed_boot['p_boot_one_sided']:.4g}\n"
        f"Image paired bootstrap p(one-sided)={image_boot['p_boot_one_sided']:.4g}",
        fontsize=12,
    )
    fig.tight_layout()
    fig.savefig(FIG_BOOT)
    plt.close(fig)


def main():
    for d in [OUT, FIG, TABLES, METRICS, REPORTS, MANIFESTS, RUNS]:
        d.mkdir(parents=True, exist_ok=True)

    if not BASE_PT.exists():
        raise FileNotFoundError(f"base checkpoint not found: {BASE_PT}")
    if not DATA_YAML.exists():
        raise FileNotFoundError(f"data yaml not found: {DATA_YAML}")
    if not TEST_TXT.exists():
        raise FileNotFoundError(f"test list not found: {TEST_TXT}")

    test_images = load_list(TEST_TXT)
    gts = load_gts(test_images)

    print("[eval] B0 predictions")
    preds_b0 = predict_target("yolov8n.pt", test_images, imgsz=640)
    met_b0 = eval_map(preds_b0, gts)

    rows = []
    preds_seed = {}
    for s in SEEDS:
        print(f"[train] seed={s}")
        pt = train_seed(seed=s, force=False)
        print(f"[eval] seed={s}")
        pr = predict_target(str(pt), test_images, imgsz=640)
        preds_seed[s] = pr
        met = eval_map(pr, gts)
        rows.append(
            {
                "seed": int(s),
                "weights": str(pt),
                "b0_map50_95": float(met_b0["map50_95"]),
                "model_map50": float(met["map50"]),
                "model_map75": float(met["map75"]),
                "model_map50_95": float(met["map50_95"]),
                "delta_map50_95_vs_b0": float(met["map50_95"] - met_b0["map50_95"]),
            }
        )

    seed_df = pd.DataFrame(rows).sort_values("seed").reset_index(drop=True)
    seed_df.to_csv(SEED_CSV, index=False, encoding="utf-8-sig")
    plot_seed_delta(seed_df)

    deltas = seed_df["delta_map50_95_vs_b0"].to_numpy(dtype=np.float64)
    k_pos, n_seed, p_sign = sign_test_p_one_sided(deltas)
    p_perm = permutation_signflip_p_one_sided(deltas)
    seed_boot = bootstrap_seed_mean(deltas, n_boot=BOOT_N, seed=2901)

    # image-level paired bootstrap (seed x image)
    img_metric_b0 = np.array([eval_image_map(preds_b0[i], gts[i]) for i in range(len(gts))], dtype=np.float64)
    delta_mat = []
    for s in SEEDS:
        img_m = np.array([eval_image_map(preds_seed[s][i], gts[i]) for i in range(len(gts))], dtype=np.float64)
        delta_mat.append(img_m - img_metric_b0)
    delta_mat = np.array(delta_mat, dtype=np.float64)

    strata = np.array([image_presence_mask(x) for x in gts], dtype=np.int64)
    img_boot = paired_bootstrap_image_seed(delta_mat, strata=strata, n_boot=BOOT_N, seed=2902)
    plot_bootstrap_hist(seed_boot, img_boot)

    out = {
        "seeds": SEEDS,
        "train_cfg": TRAIN_CFG,
        "test_count": len(test_images),
        "b0": met_b0,
        "seed_level": {
            "mean_delta_map50_95": float(np.mean(deltas)),
            "std_delta_map50_95": float(np.std(deltas, ddof=1)) if len(deltas) > 1 else 0.0,
            "positive_seed_count": int(k_pos),
            "seed_count": int(n_seed),
            "p_sign_test_one_sided": float(p_sign),
            "p_permutation_signflip_one_sided": float(p_perm),
            "bootstrap": seed_boot,
        },
        "paired_bootstrap_image_level": img_boot,
        "paths": {
            "seed_csv": str(SEED_CSV.resolve()),
            "bootstrap_json": str(BOOT_JSON.resolve()),
            "seed_plot": str(FIG_SEED.resolve()),
            "bootstrap_plot": str(FIG_BOOT.resolve()),
        },
    }
    BOOT_JSON.write_text(json.dumps(out, indent=2), encoding="utf-8")

    lines = [
        "# Last Squeeze Seed Repeat Significance (CV1)",
        "",
        "## Protocol",
        f"- Model: `yolov8n` + last_squeeze refine config",
        f"- Start checkpoint: `{BASE_PT}`",
        f"- Seeds: `{SEEDS}`",
        f"- Test set: `{TEST_TXT}` ({len(test_images)} images)",
        "",
        "## B0",
        f"- mAP50={met_b0['map50']:.4f}, mAP75={met_b0['map75']:.4f}, mAP50-95={met_b0['map50_95']:.4f}",
        "",
        "## Seed-Level (vs B0)",
        f"- mean delta(mAP50-95) = {np.mean(deltas):+.4f}",
        f"- std delta = {np.std(deltas, ddof=1):.4f}",
        f"- positive seeds = {k_pos}/{n_seed}",
        f"- one-sided sign test p = {p_sign:.6f}",
        f"- one-sided exact sign-flip permutation p = {p_perm:.6f}",
        f"- seed bootstrap 95% CI = [{seed_boot['ci95_low']:+.4f}, {seed_boot['ci95_high']:+.4f}]",
        "",
        "## Paired Bootstrap (Image-Level, Stratified by class-presence mask)",
        f"- mean delta = {img_boot['mean']:+.4f}",
        f"- 95% CI = [{img_boot['ci95_low']:+.4f}, {img_boot['ci95_high']:+.4f}]",
        f"- one-sided p = {img_boot['p_boot_one_sided']:.6f}",
        "",
        "## Figures",
        f"![seed_delta](C:/Users/lmhst/git/roboticsCV/CV_project#1/outputs/figures/{FIG_SEED.name})",
        "",
        f"![bootstrap](C:/Users/lmhst/git/roboticsCV/CV_project#1/outputs/figures/{FIG_BOOT.name})",
    ]
    REPORT_MD.write_text("\n".join(lines), encoding="utf-8")

    manifest = {
        "seed_csv": str(SEED_CSV.resolve()),
        "bootstrap_json": str(BOOT_JSON.resolve()),
        "report_md": str(REPORT_MD.resolve()),
        "figure_seed": str(FIG_SEED.resolve()),
        "figure_bootstrap": str(FIG_BOOT.resolve()),
    }
    MANIFEST_JSON.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print("[saved]", SEED_CSV)
    print("[saved]", BOOT_JSON)
    print("[saved]", REPORT_MD)
    print("[saved]", MANIFEST_JSON)
    print(
        json.dumps(
            {
                "mean_delta_map50_95": float(np.mean(deltas)),
                "p_sign_test_one_sided": float(p_sign),
                "p_permutation_one_sided": float(p_perm),
                "paired_boot_ci95": [img_boot["ci95_low"], img_boot["ci95_high"]],
                "paired_boot_p_one_sided": float(img_boot["p_boot_one_sided"]),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
