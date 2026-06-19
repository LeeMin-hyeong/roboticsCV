import argparse
import csv
import json
import time
import zipfile
from collections import Counter, defaultdict
from io import BytesIO
from pathlib import Path

import torch
from PIL import Image, ImageOps
from ultralytics import YOLO


CLASS_MAP = {
    "망치": "hammer",
    "드라이버": "screwdriver",
    "집게류": "pliers",
}

PROMPTS = {
    "hammer": "a photo of a hammer",
    "screwdriver": "a photo of a screwdriver",
    "pliers": "a photo of pliers",
}

DEFAULT_FOUNDATIONS = [
    "owlvit:google/owlvit-base-patch32:OWL-ViT",
    "owlv2:google/owlv2-base-patch16-ensemble:OWLv2",
    "grounding_dino:IDEA-Research/grounding-dino-tiny:Grounding DINO",
]


def try_register_heif():
    try:
        from pillow_heif import register_heif_opener

        register_heif_opener()
        return True
    except Exception:
        return False


def iter_zip_images(zip_path, skip_suffixes=None):
    skip_suffixes = skip_suffixes or set()
    suffixes = {".jpg", ".jpeg", ".png", ".heic", ".heif", ".webp"}

    with zipfile.ZipFile(zip_path) as zf:
        for entry in zf.infolist():
            if entry.is_dir():
                continue
            path = Path(entry.filename)
            suffix = path.suffix.lower()
            if suffix not in suffixes or suffix in skip_suffixes:
                continue

            top = entry.filename.split("/")[0]
            true_class = CLASS_MAP.get(top)
            if true_class is None:
                continue
            with zf.open(entry) as f:
                data = f.read()
            yield entry.filename, true_class, data


def load_image(image_bytes):
    img = Image.open(BytesIO(image_bytes))
    img = ImageOps.exif_transpose(img)
    return img.convert("RGB")


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


def run_yolo(model, image, conf):
    started = time.perf_counter()
    results = model.predict(image, conf=conf, verbose=False)
    elapsed = time.perf_counter() - started
    names = model.names
    detections = []

    if results and results[0].boxes is not None:
        boxes = results[0].boxes
        for cls_id, score in zip(boxes.cls.tolist(), boxes.conf.tolist()):
            detections.append((names[int(cls_id)], float(score)))

    if not detections:
        return "", 0.0, False, elapsed

    pred_class, pred_conf = max(detections, key=lambda item: item[1])
    return pred_class, pred_conf, True, elapsed


class OwlVitRunner:
    def __init__(self, model_name, device):
        from transformers import AutoTokenizer, OwlViTForObjectDetection, OwlViTImageProcessor, OwlViTProcessor

        self.device = device
        try:
            self.processor = OwlViTProcessor.from_pretrained(model_name)
            self.post_processor = self.processor
        except OSError:
            image_processor = OwlViTImageProcessor.from_pretrained(model_name)
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.processor = OwlViTProcessor(image_processor=image_processor, tokenizer=tokenizer)
            self.post_processor = image_processor
        self.model = OwlViTForObjectDetection.from_pretrained(model_name).to(device)
        self.model.eval()
        self.labels = list(PROMPTS.keys())
        self.texts = [list(PROMPTS.values())]

    def predict(self, image, threshold):
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
        return best_prompt_prediction(processed, self.labels, elapsed)


class Owlv2Runner:
    def __init__(self, model_name, device):
        from transformers import AutoTokenizer, Owlv2ForObjectDetection, Owlv2ImageProcessor, Owlv2Processor

        self.device = device
        try:
            self.processor = Owlv2Processor.from_pretrained(model_name)
            self.post_processor = self.processor
        except OSError:
            image_processor = Owlv2ImageProcessor.from_pretrained(model_name)
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.processor = Owlv2Processor(image_processor=image_processor, tokenizer=tokenizer)
            self.post_processor = image_processor
        self.model = Owlv2ForObjectDetection.from_pretrained(model_name).to(device)
        self.model.eval()
        self.labels = list(PROMPTS.keys())
        self.texts = [list(PROMPTS.values())]

    def predict(self, image, threshold):
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
        return best_prompt_prediction(processed, self.labels, elapsed)


class GroundingDinoRunner:
    def __init__(self, model_name, device):
        from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

        self.device = device
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(model_name).to(device)
        self.model.eval()
        self.labels = list(PROMPTS.keys())
        self.text = ". ".join(PROMPTS.values()) + "."
        self.text_labels = [list(PROMPTS.values())]

    def predict(self, image, threshold):
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

        scores = processed["scores"].detach().cpu().tolist()
        if not scores:
            return "", 0.0, False, elapsed

        label_values = processed.get("text_labels") or processed.get("labels")
        if hasattr(label_values, "detach"):
            label_values = label_values.detach().cpu().tolist()

        best_index = max(range(len(scores)), key=lambda i: scores[i])
        label_value = label_values[best_index]
        if isinstance(label_value, int):
            pred_class = self.labels[label_value]
        else:
            normalized = str(label_value).lower()
            pred_class = ""
            for cls, prompt in PROMPTS.items():
                if cls in normalized or prompt in normalized:
                    pred_class = cls
                    break
        return pred_class, float(scores[best_index]), bool(pred_class), elapsed


def best_prompt_prediction(processed, labels, elapsed):
    scores = processed["scores"].detach().cpu().tolist()
    label_ids = processed["labels"].detach().cpu().tolist()
    if not scores:
        return "", 0.0, False, elapsed

    best_index = max(range(len(scores)), key=lambda i: scores[i])
    return labels[label_ids[best_index]], float(scores[best_index]), True, elapsed


def parse_foundation_spec(spec):
    parts = spec.split(":", 2)
    if len(parts) != 3:
        raise ValueError(f"Foundation spec must be kind:model_id:label, got {spec!r}")
    return parts[0].strip(), parts[1].strip(), parts[2].strip()


def build_foundation_runner(kind, model_name, device):
    if kind == "owlvit":
        return OwlVitRunner(model_name, device)
    if kind == "owlv2":
        return Owlv2Runner(model_name, device)
    if kind in {"grounding_dino", "grounding-dino", "groundingdino"}:
        return GroundingDinoRunner(model_name, device)
    raise ValueError(f"Unsupported foundation kind: {kind}")


def summarize(rows, keys):
    total = len(rows)
    by_class = defaultdict(list)
    for row in rows:
        by_class[row["true_class"]].append(row)

    def acc(prefix, items):
        valid = [r for r in items if r.get(f"{prefix}_pred")]
        correct = [r for r in items if r.get(f"{prefix}_correct") == "1"]
        return {
            "correct": len(correct),
            "total": len(items),
            "accuracy": len(correct) / len(items) if items else 0.0,
            "coverage": len(valid) / len(items) if items else 0.0,
            "avg_time_sec": sum(float(r.get(f"{prefix}_time_sec", 0.0)) for r in items) / len(items)
            if items
            else 0.0,
        }

    return {
        "total_images": total,
        "models": {key: acc(key, rows) for key in keys},
        "by_class": {
            cls: {
                "count": len(items),
                "models": {key: acc(key, items) for key in keys},
            }
            for cls, items in sorted(by_class.items())
        },
    }


def write_markdown(summary, output_path, model_labels, foundation_specs, yolo_path):
    lines = [
        "# Foundation Models vs YOLO Comparison",
        "",
        "This evaluation uses folder names as image-level ground truth because bounding-box labels are not available.",
        "",
        f"- YOLO model: `{yolo_path}`",
        "- Foundation models:",
    ]
    for spec in foundation_specs:
        lines.append(f"  - `{spec}`")
    lines.extend(
        [
            f"- Total images: {summary['total_images']}",
            "",
            "## Overall",
            "",
            "| Model | Correct / Total | Accuracy | Detection Coverage | Avg Time/Image |",
            "|---|---:|---:|---:|---:|",
        ]
    )

    for key, label in model_labels:
        item = summary["models"][key]
        lines.append(
            f"| {label} | {item['correct']} / {item['total']} | "
            f"{item['accuracy']:.3f} | {item['coverage']:.3f} | {item['avg_time_sec']:.3f}s |"
        )

    lines.extend(["", "## By Class", ""])
    for cls, item in summary["by_class"].items():
        lines.extend(
            [
                f"### {cls}",
                "",
                "| Model | Correct / Total | Accuracy | Detection Coverage | Avg Time/Image |",
                "|---|---:|---:|---:|---:|",
            ]
        )
        for key, label in model_labels:
            metric = item["models"][key]
            lines.append(
                f"| {label} | {metric['correct']} / {metric['total']} | "
                f"{metric['accuracy']:.3f} | {metric['coverage']:.3f} | {metric['avg_time_sec']:.3f}s |"
            )
        lines.append("")

    output_path.write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--zip", required=True, help="Dataset zip path")
    parser.add_argument("--yolo", required=True, help="Fine-tuned YOLO .pt path")
    parser.add_argument("--out", default="comparison_results", help="Output directory")
    parser.add_argument(
        "--foundation",
        action="append",
        default=[],
        help="Foundation spec: kind:model_id:label. Repeatable.",
    )
    parser.add_argument("--sample-per-class", type=int, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--yolo-conf", type=float, default=0.25)
    parser.add_argument("--foundation-threshold", type=float, default=0.05)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--skip-foundation", action="store_true")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    heif_ok = try_register_heif()

    yolo = YOLO(args.yolo)
    foundation_specs = [] if args.skip_foundation else (args.foundation or DEFAULT_FOUNDATIONS)

    runners = []
    model_labels = [("yolo", "YOLO fine-tuned")]
    for spec in foundation_specs:
        kind, model_name, label = parse_foundation_spec(spec)
        key = model_key(label)
        print(f"Loading foundation model: {label} ({model_name})")
        runners.append((key, label, build_foundation_runner(kind, model_name, args.device)))
        model_labels.append((key, label))

    rows = []
    skipped = []
    skip_suffixes = set()
    if not heif_ok:
        skip_suffixes.update({".heic", ".heif"})

    accepted_counts = Counter()
    for filename, true_class, image_bytes in iter_zip_images(args.zip, skip_suffixes):
        if args.sample_per_class is not None and accepted_counts[true_class] >= args.sample_per_class:
            continue
        if args.limit is not None and len(rows) >= args.limit:
            break

        try:
            image = load_image(image_bytes)
        except Exception as exc:
            skipped.append({"file": filename, "reason": str(exc), "heif_registered": heif_ok})
            continue

        accepted_counts[true_class] += 1

        row = {"file": filename, "true_class": true_class}
        yolo_pred, yolo_conf, yolo_found, yolo_time = run_yolo(yolo, image, args.yolo_conf)
        row.update(
            {
                "yolo_pred": yolo_pred,
                "yolo_conf": f"{yolo_conf:.6f}",
                "yolo_found": "1" if yolo_found else "0",
                "yolo_correct": "1" if yolo_pred == true_class else "0",
                "yolo_time_sec": f"{yolo_time:.6f}",
            }
        )

        for key, _label, runner in runners:
            pred, conf, found, elapsed = runner.predict(image, args.foundation_threshold)
            row.update(
                {
                    f"{key}_pred": pred,
                    f"{key}_conf": f"{conf:.6f}",
                    f"{key}_found": "1" if found else "0",
                    f"{key}_correct": "1" if pred == true_class else "0",
                    f"{key}_time_sec": f"{elapsed:.6f}",
                }
            )

        rows.append(row)

    csv_path = out_dir / "predictions.csv"
    fieldnames = list(rows[0].keys()) if rows else ["file"]
    with csv_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    keys = [key for key, _label in model_labels]
    summary = summarize(rows, keys)
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_markdown(summary, out_dir / "summary.md", model_labels, foundation_specs, args.yolo)

    if skipped:
        (out_dir / "skipped.json").write_text(json.dumps(skipped, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if skipped:
        print(f"Skipped {len(skipped)} images. See {out_dir / 'skipped.json'}")


if __name__ == "__main__":
    main()
