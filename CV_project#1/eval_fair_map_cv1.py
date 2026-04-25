from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

TARGET = ["person", "car", "dog"]
NAME2IDX = {n: i for i, n in enumerate(TARGET)}
IOU_THRESHOLDS = np.arange(0.5, 0.96, 0.05)


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test-txt", required=True)
    ap.add_argument("--b0", default="yolov8n.pt")
    ap.add_argument("--model", required=True)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--out-json", default="")
    args = ap.parse_args()

    test_images = load_list(Path(args.test_txt))
    gts = load_gts(test_images)

    met_b0 = eval_map(predict_target(args.b0, test_images, imgsz=args.imgsz), gts)
    met_m = eval_map(predict_target(args.model, test_images, imgsz=args.imgsz), gts)

    out = {
        "test_count": len(test_images),
        "b0": met_b0,
        "model": met_m,
        "delta_model_minus_b0": {
            "map50": met_m["map50"] - met_b0["map50"],
            "map75": met_m["map75"] - met_b0["map75"],
            "map50_95": met_m["map50_95"] - met_b0["map50_95"],
            "person_ap50_95": met_m["class_ap50_95"]["person"] - met_b0["class_ap50_95"]["person"],
            "car_ap50_95": met_m["class_ap50_95"]["car"] - met_b0["class_ap50_95"]["car"],
            "dog_ap50_95": met_m["class_ap50_95"]["dog"] - met_b0["class_ap50_95"]["dog"],
        },
    }
    print(json.dumps(out, indent=2))
    if args.out_json:
        Path(args.out_json).write_text(json.dumps(out, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
