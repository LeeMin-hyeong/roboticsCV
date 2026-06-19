import argparse
import csv
import json
import random
import time
import zipfile
from collections import Counter, defaultdict
from io import BytesIO
from pathlib import Path

import torch
from PIL import Image, ImageOps
from ultralytics import YOLO


TARGET_CLASSES = ["drill", "hammer", "pliers", "screwdriver", "wrench"]
PROMPTS = {
    "drill": "a photo of a drill",
    "hammer": "a photo of a hammer",
    "pliers": "a photo of pliers",
    "screwdriver": "a photo of a screwdriver",
    "wrench": "a photo of a wrench",
}
COCO_CLASS_ALIASES = {
    "Drill": "drill",
    "Hammer": "hammer",
    "Pliers": "pliers",
    "plier": "pliers",
    "Screwdriver": "screwdriver",
    "Screw Driver": "screwdriver",
    "Wrench": "wrench",
}
DEFAULT_FOUNDATIONS = [
    "owlvit:google/owlvit-base-patch32:OWL-ViT",
    "owlv2:google/owlv2-base-patch16-ensemble:OWLv2",
    "grounding_dino:IDEA-Research/grounding-dino-tiny:Grounding DINO",
]


def model_key(label):
    return (
        label.lower()
        .replace(" ", "_")
        .replace("-", "_")
        .replace("/", "_")
        .replace(".", "_")
        .replace("(", "")
        .replace(")", "")
    )


def load_image(data):
    image = Image.open(BytesIO(data))
    image = ImageOps.exif_transpose(image)
    return image.convert("RGB")


def xywh_to_xyxy(box):
    x, y, w, h = [float(v) for v in box]
    return [x, y, x + w, y + h]


def box_iou(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def load_coco_from_zip(zip_path):
    with zipfile.ZipFile(zip_path) as zf:
        ann_entry = next(e for e in zf.infolist() if e.filename.endswith("_annotations.coco.json"))
        coco = json.loads(zf.read(ann_entry).decode("utf-8"))
        image_entries = {entry.filename: entry for entry in zf.infolist() if not entry.is_dir()}

    categories = {cat["id"]: cat["name"] for cat in coco["categories"]}
    images = {img["id"]: img for img in coco["images"]}
    anns_by_image = defaultdict(list)
    for ann in coco["annotations"]:
        cls = COCO_CLASS_ALIASES.get(categories.get(ann["category_id"], ""))
        if cls is None:
            continue
        anns_by_image[ann["image_id"]].append(
            {
                "class": cls,
                "bbox": xywh_to_xyxy(ann["bbox"]),
                "area": float(ann.get("area", float(ann["bbox"][2]) * float(ann["bbox"][3]))),
            }
        )

    return coco, images, anns_by_image, image_entries


def select_subset(images, anns_by_image, sample_per_class, seed, target_classes):
    rng = random.Random(seed)
    image_ids_by_class = defaultdict(list)
    for image_id, anns in anns_by_image.items():
        classes = {ann["class"] for ann in anns}
        for cls in classes:
            image_ids_by_class[cls].append(image_id)

    selected = []
    selected_set = set()
    for cls in target_classes:
        ids = image_ids_by_class.get(cls, [])
        rng.shuffle(ids)
        picked = 0
        for image_id in ids:
            if image_id in selected_set:
                continue
            selected.append(image_id)
            selected_set.add(image_id)
            picked += 1
            if picked >= sample_per_class:
                break

    rng.shuffle(selected)
    return selected


class YoloRunner:
    def __init__(self, model_path):
        self.model = YOLO(model_path)
        self.names = self.model.names

    def detect(self, image, threshold):
        started = time.perf_counter()
        results = self.model.predict(image, conf=threshold, verbose=False)
        elapsed = time.perf_counter() - started
        detections = []
        if results and results[0].boxes is not None:
            boxes = results[0].boxes
            for cls_id, score, xyxy in zip(boxes.cls.tolist(), boxes.conf.tolist(), boxes.xyxy.tolist()):
                cls = self.names[int(cls_id)]
                if cls in TARGET_CLASSES:
                    detections.append({"class": cls, "score": float(score), "bbox": [float(v) for v in xyxy]})
        return detections, elapsed


class OwlRunner:
    def __init__(self, kind, model_name, device):
        self.kind = kind
        self.device = device
        if kind == "owlvit":
            from transformers import AutoTokenizer, OwlViTForObjectDetection, OwlViTImageProcessor, OwlViTProcessor

            model_cls = OwlViTForObjectDetection
            proc_cls = OwlViTProcessor
            image_proc_cls = OwlViTImageProcessor
        elif kind == "owlv2":
            from transformers import AutoTokenizer, Owlv2ForObjectDetection, Owlv2ImageProcessor, Owlv2Processor

            model_cls = Owlv2ForObjectDetection
            proc_cls = Owlv2Processor
            image_proc_cls = Owlv2ImageProcessor
        else:
            raise ValueError(kind)

        try:
            self.processor = proc_cls.from_pretrained(model_name)
            self.post_processor = self.processor
        except OSError:
            image_processor = image_proc_cls.from_pretrained(model_name)
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.processor = proc_cls(image_processor=image_processor, tokenizer=tokenizer)
            self.post_processor = image_processor
        self.model = model_cls.from_pretrained(model_name).to(device)
        self.model.eval()
        self.labels = TARGET_CLASSES
        self.texts = [[PROMPTS[cls] for cls in self.labels]]

    def detect(self, image, threshold):
        started = time.perf_counter()
        inputs = self.processor(text=self.texts, images=image, return_tensors="pt")
        inputs = {key: value.to(self.device) for key, value in inputs.items()}
        with torch.inference_mode():
            outputs = self.model(**inputs)
        target_sizes = torch.tensor([image.size[::-1]], device=self.device)
        if hasattr(self.post_processor, "post_process_object_detection"):
            processed = self.post_processor.post_process_object_detection(
                outputs=outputs,
                target_sizes=target_sizes,
                threshold=threshold,
            )[0]
        else:
            processed = self.post_processor.post_process_grounded_object_detection(
                outputs=outputs,
                target_sizes=target_sizes,
                threshold=threshold,
            )[0]
        elapsed = time.perf_counter() - started
        return detections_from_prompt_output(processed, self.labels), elapsed


class GroundingDinoRunner:
    def __init__(self, model_name, device):
        from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

        self.device = device
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(model_name).to(device)
        self.model.eval()
        self.labels = TARGET_CLASSES
        self.text = ". ".join(PROMPTS[cls] for cls in self.labels) + "."
        self.text_labels = [[PROMPTS[cls] for cls in self.labels]]

    def detect(self, image, threshold):
        started = time.perf_counter()
        inputs = self.processor(images=image, text=self.text, return_tensors="pt")
        inputs = {key: value.to(self.device) for key, value in inputs.items()}
        with torch.inference_mode():
            outputs = self.model(**inputs)
        target_sizes = torch.tensor([image.size[::-1]], device=self.device)
        processed = self.processor.post_process_grounded_object_detection(
            outputs,
            input_ids=inputs.get("input_ids"),
            threshold=threshold,
            text_threshold=threshold,
            target_sizes=target_sizes,
            text_labels=self.text_labels,
        )[0]
        elapsed = time.perf_counter() - started
        return detections_from_grounding_output(processed), elapsed


def detections_from_prompt_output(processed, labels):
    scores = processed["scores"].detach().cpu().tolist()
    label_ids = processed["labels"].detach().cpu().tolist()
    boxes = processed["boxes"].detach().cpu().tolist()
    detections = []
    for score, label_id, box in zip(scores, label_ids, boxes):
        detections.append({"class": labels[int(label_id)], "score": float(score), "bbox": [float(v) for v in box]})
    return detections


def detections_from_grounding_output(processed):
    scores = processed["scores"].detach().cpu().tolist()
    boxes = processed["boxes"].detach().cpu().tolist()
    label_values = processed.get("text_labels") or processed.get("labels")
    if hasattr(label_values, "detach"):
        label_values = label_values.detach().cpu().tolist()

    detections = []
    for score, label_value, box in zip(scores, label_values, boxes):
        if isinstance(label_value, int):
            cls = TARGET_CLASSES[label_value]
        else:
            normalized = str(label_value).lower()
            cls = ""
            for candidate, prompt in PROMPTS.items():
                if candidate in normalized or prompt in normalized:
                    cls = candidate
                    break
        if cls:
            detections.append({"class": cls, "score": float(score), "bbox": [float(v) for v in box]})
    return detections


def parse_foundation_spec(spec):
    parts = spec.split(":", 2)
    if len(parts) != 3:
        raise ValueError(f"Foundation spec must be kind:model_id:label, got {spec!r}")
    return parts[0].strip(), parts[1].strip(), parts[2].strip()


def build_runner(kind, model_name, device):
    if kind in {"owlvit", "owlv2"}:
        return OwlRunner(kind, model_name, device)
    if kind in {"grounding_dino", "grounding-dino", "groundingdino"}:
        return GroundingDinoRunner(model_name, device)
    raise ValueError(f"Unsupported foundation kind: {kind}")


def average_precision(recalls, precisions):
    points = [(0.0, 1.0)] + sorted(zip(recalls, precisions)) + [(1.0, 0.0)]
    for i in range(len(points) - 2, -1, -1):
        points[i] = (points[i][0], max(points[i][1], points[i + 1][1]))
    ap = 0.0
    for i in range(1, len(points)):
        ap += (points[i][0] - points[i - 1][0]) * points[i][1]
    return ap


def evaluate_at_iou(gt_by_image, pred_by_image, iou_threshold, target_classes):
    per_class = {}
    aps = []
    total_tp = total_fp = total_gt = 0

    for cls in target_classes:
        gt_records = {
            image_id: [{"bbox": ann["bbox"], "matched": False} for ann in anns if ann["class"] == cls]
            for image_id, anns in gt_by_image.items()
        }
        n_gt = sum(len(items) for items in gt_records.values())
        preds = []
        for image_id, detections in pred_by_image.items():
            for det in detections:
                if det["class"] == cls:
                    preds.append((image_id, det["score"], det["bbox"]))
        preds.sort(key=lambda item: item[1], reverse=True)

        tp = []
        fp = []
        for image_id, _score, pred_box in preds:
            candidates = gt_records.get(image_id, [])
            best_iou = 0.0
            best_idx = -1
            for idx, gt in enumerate(candidates):
                if gt["matched"]:
                    continue
                iou = box_iou(pred_box, gt["bbox"])
                if iou > best_iou:
                    best_iou = iou
                    best_idx = idx
            if best_iou >= iou_threshold and best_idx >= 0:
                candidates[best_idx]["matched"] = True
                tp.append(1)
                fp.append(0)
            else:
                tp.append(0)
                fp.append(1)

        cum_tp = []
        cum_fp = []
        running_tp = running_fp = 0
        for t, f in zip(tp, fp):
            running_tp += t
            running_fp += f
            cum_tp.append(running_tp)
            cum_fp.append(running_fp)

        recalls = [v / n_gt if n_gt else 0.0 for v in cum_tp]
        precisions = [
            cum_tp[i] / (cum_tp[i] + cum_fp[i]) if (cum_tp[i] + cum_fp[i]) else 0.0
            for i in range(len(cum_tp))
        ]
        ap = average_precision(recalls, precisions) if n_gt else 0.0
        final_tp = sum(tp)
        final_fp = sum(fp)
        recall = final_tp / n_gt if n_gt else 0.0
        precision = final_tp / (final_tp + final_fp) if (final_tp + final_fp) else 0.0
        per_class[cls] = {
            "gt": n_gt,
            "pred": len(preds),
            "tp": final_tp,
            "fp": final_fp,
            "precision": precision,
            "recall": recall,
            "ap": ap,
        }
        if n_gt:
            aps.append(ap)
            total_gt += n_gt
            total_tp += final_tp
            total_fp += final_fp

    return {
        "map": sum(aps) / len(aps) if aps else 0.0,
        "precision": total_tp / (total_tp + total_fp) if (total_tp + total_fp) else 0.0,
        "recall": total_tp / total_gt if total_gt else 0.0,
        "per_class": per_class,
    }


def evaluate(gt_by_image, pred_by_image, target_classes):
    iou_thresholds = [round(0.5 + i * 0.05, 2) for i in range(10)]
    at_50 = evaluate_at_iou(gt_by_image, pred_by_image, 0.5, target_classes)
    maps = [evaluate_at_iou(gt_by_image, pred_by_image, thr, target_classes)["map"] for thr in iou_thresholds]
    return {
        "map50": at_50["map"],
        "map50_95": sum(maps) / len(maps),
        "precision50": at_50["precision"],
        "recall50": at_50["recall"],
        "per_class50": at_50["per_class"],
    }


def write_report(path, summary, model_labels):
    target_classes = summary["classes"]
    lines = [
        "# Roboflow COCO BBox Evaluation",
        "",
        f"- Images: {summary['num_images']}",
        f"- Ground-truth boxes: {summary['num_gt_boxes']}",
        f"- Classes: {', '.join(target_classes)}",
        "",
        "## Overall",
        "",
        "| Model | mAP@0.5 | mAP@0.5:0.95 | Precision@0.5 | Recall@0.5 | Avg Time/Image |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for key, label in model_labels:
        item = summary["models"][key]
        lines.append(
            f"| {label} | {item['map50']:.3f} | {item['map50_95']:.3f} | "
            f"{item['precision50']:.3f} | {item['recall50']:.3f} | {item['avg_time_sec']:.3f}s |"
        )

    lines.extend(["", "## AP@0.5 By Class", ""])
    lines.append("| Class | " + " | ".join(label for _key, label in model_labels) + " |")
    lines.append("|---|" + "|".join("---:" for _ in model_labels) + "|")
    for cls in target_classes:
        vals = []
        for key, _label in model_labels:
            vals.append(f"{summary['models'][key]['per_class50'][cls]['ap']:.3f}")
        lines.append(f"| {cls} | " + " | ".join(vals) + " |")

    path.write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--zip", required=True)
    parser.add_argument("--yolo", required=True)
    parser.add_argument("--out", default="roboflow_bbox_results")
    parser.add_argument("--sample-per-class", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--yolo-conf", type=float, default=0.001)
    parser.add_argument("--foundation-threshold", type=float, default=0.01)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--foundation", action="append", default=[])
    parser.add_argument("--skip-foundation", action="store_true")
    parser.add_argument(
        "--classes",
        default=",".join(TARGET_CLASSES),
        help="Comma-separated evaluation classes. Defaults to all YOLO tool classes.",
    )
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    coco, images, anns_by_image, image_entries = load_coco_from_zip(args.zip)
    requested_classes = [cls.strip() for cls in args.classes.split(",") if cls.strip()]
    available_classes = sorted({ann["class"] for anns in anns_by_image.values() for ann in anns})
    target_classes = [cls for cls in requested_classes if cls in available_classes]
    if not target_classes:
        raise ValueError(f"No requested classes found. Requested={requested_classes}, available={available_classes}")
    missing_classes = [cls for cls in requested_classes if cls not in available_classes]
    if missing_classes:
        print(f"Skipping classes not present in dataset: {', '.join(missing_classes)}")

    selected_ids = select_subset(images, anns_by_image, args.sample_per_class, args.seed, target_classes)
    gt_by_image = {image_id: anns_by_image[image_id] for image_id in selected_ids}

    model_labels = [("yolo", "YOLO fine-tuned")]
    runners = [("yolo", "YOLO fine-tuned", YoloRunner(args.yolo), args.yolo_conf)]

    foundation_specs = [] if args.skip_foundation else (args.foundation or DEFAULT_FOUNDATIONS)
    for spec in foundation_specs:
        kind, model_name, label = parse_foundation_spec(spec)
        key = model_key(label)
        print(f"Loading foundation model: {label} ({model_name})")
        runners.append((key, label, build_runner(kind, model_name, args.device), args.foundation_threshold))
        model_labels.append((key, label))

    pred_by_model = {key: {} for key, _label, _runner, _threshold in runners}
    timing = Counter()
    rows = []

    with zipfile.ZipFile(args.zip) as zf:
        for idx, image_id in enumerate(selected_ids, 1):
            image_info = images[image_id]
            entry_name = "train/" + image_info["file_name"]
            if entry_name not in image_entries:
                entry_name = next(name for name in image_entries if name.endswith("/" + image_info["file_name"]))
            image = load_image(zf.read(entry_name))

            for key, label, runner, threshold in runners:
                detections, elapsed = runner.detect(image, threshold)
                pred_by_model[key][image_id] = detections
                timing[key] += elapsed
                rows.append(
                    {
                        "image_id": image_id,
                        "file": image_info["file_name"],
                        "model": label,
                        "num_gt": len(gt_by_image[image_id]),
                        "num_pred": len(detections),
                        "time_sec": f"{elapsed:.6f}",
                    }
                )
            if idx % 10 == 0:
                print(f"Processed {idx}/{len(selected_ids)} images")

    summary = {
        "num_images": len(selected_ids),
        "num_gt_boxes": sum(len(v) for v in gt_by_image.values()),
        "sample_per_class": args.sample_per_class,
        "seed": args.seed,
        "classes": target_classes,
        "missing_requested_classes": missing_classes,
        "models": {},
    }
    for key, _label in model_labels:
        metrics = evaluate(gt_by_image, pred_by_model[key], target_classes)
        metrics["avg_time_sec"] = timing[key] / len(selected_ids) if selected_ids else 0.0
        summary["models"][key] = metrics

    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(out_dir / "summary.md", summary, model_labels)
    with (out_dir / "image_model_counts.csv").open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else ["image_id"])
        writer.writeheader()
        writer.writerows(rows)

    selected_manifest = [
        {
            "image_id": image_id,
            "file": images[image_id]["file_name"],
            "classes": sorted({ann["class"] for ann in gt_by_image[image_id]}),
            "num_gt": len(gt_by_image[image_id]),
        }
        for image_id in selected_ids
    ]
    (out_dir / "selected_images.json").write_text(
        json.dumps(selected_manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
