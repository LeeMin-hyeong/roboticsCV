import argparse
import random
import textwrap
import zipfile
from collections import defaultdict
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, ImageOps

from roboflow_bbox_eval import (
    COCO_CLASS_ALIASES,
    DEFAULT_FOUNDATIONS,
    TARGET_CLASSES,
    YoloRunner,
    build_runner,
    load_coco_from_zip,
    parse_foundation_spec,
)


ROOT = Path(__file__).resolve().parent
MODEL_COLORS = {
    "YOLO fine-tuned": "#2563EB",
    "OWL-ViT": "#16A34A",
    "OWLv2": "#F59E0B",
    "Grounding DINO": "#DC2626",
}
GT_COLOR = "#22C55E"


def load_image(data):
    image = Image.open(data)
    image = ImageOps.exif_transpose(image)
    return image.convert("RGB")


def get_font(size):
    candidates = [
        "C:/Windows/Fonts/malgun.ttf",
        "C:/Windows/Fonts/arial.ttf",
    ]
    for path in candidates:
        if Path(path).exists():
            return ImageFont.truetype(path, size=size)
    return ImageFont.load_default()


def draw_label(draw, xy, text, fill, font):
    x, y = xy
    bbox = draw.textbbox((x, y), text, font=font)
    pad = 4
    bg = [bbox[0] - pad, bbox[1] - pad, bbox[2] + pad, bbox[3] + pad]
    draw.rectangle(bg, fill=fill)
    draw.text((x, y), text, fill="white", font=font)


def draw_boxes(image, gt_boxes, pred_boxes, model_label, target_class):
    canvas = image.copy()
    draw = ImageDraw.Draw(canvas)
    small = get_font(12)

    for gt in gt_boxes:
        x1, y1, x2, y2 = gt["bbox"]
        draw.rectangle([x1, y1, x2, y2], outline=GT_COLOR, width=4)
        draw_label(draw, (x1, max(0, y1 - 24)), f"GT {target_class}", GT_COLOR, small)

    color = MODEL_COLORS[model_label]
    for det in pred_boxes[:4]:
        x1, y1, x2, y2 = det["bbox"]
        draw.rectangle([x1, y1, x2, y2], outline=color, width=4)
        draw_label(draw, (x1, y1), f"{det['class']} {det['score']:.2f}", color, small)

    if not pred_boxes:
        draw_label(draw, (8, 8), "no prediction", "#111827", small)
    return canvas


def make_tile(image, max_w=520, max_h=390):
    img = image.copy()
    img.thumbnail((max_w, max_h), Image.LANCZOS)
    tile = Image.new("RGB", (max_w, max_h), "white")
    x = (max_w - img.width) // 2
    y = (max_h - img.height) // 2
    tile.paste(img, (x, y))
    return tile


def make_grid(images_by_model_class, model_labels, classes, out_path):
    tile_w, tile_h = 520, 390
    header_h = 48
    left_w = 155
    font = get_font(17)
    small = get_font(14)
    grid = Image.new(
        "RGB",
        (left_w + tile_w * len(classes), header_h + tile_h * len(model_labels)),
        "white",
    )
    draw = ImageDraw.Draw(grid)
    draw.rectangle([0, 0, grid.width, header_h], fill="#E8EEF5")
    draw.rectangle([0, 0, left_w, grid.height], fill="#F2F4F7")
    for c_idx, cls in enumerate(classes):
        x = left_w + c_idx * tile_w
        draw.text((x + 16, 14), cls, fill="#111827", font=font)
    for r_idx, model in enumerate(model_labels):
        y = header_h + r_idx * tile_h
        wrapped = "\n".join(textwrap.wrap(model, width=14))
        draw.text((12, y + 18), wrapped, fill="#111827", font=small)
        for c_idx, cls in enumerate(classes):
            x = left_w + c_idx * tile_w
            tile = make_tile(images_by_model_class[(model, cls)], tile_w, tile_h)
            grid.paste(tile, (x, y))
            draw.rectangle([x, y, x + tile_w, y + tile_h], outline="#D1D5DB", width=1)
    grid.save(out_path)


def choose_images(images, anns_by_image, target_classes, seed):
    rng = random.Random(seed)
    ids_by_class = defaultdict(list)
    for image_id, anns in anns_by_image.items():
        classes = {ann["class"] for ann in anns}
        for cls in target_classes:
            if cls in classes:
                ids_by_class[cls].append(image_id)
    chosen = {}
    for cls in target_classes:
        if not ids_by_class[cls]:
            continue
        chosen[cls] = rng.choice(ids_by_class[cls])
    return chosen


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--zip", default="C:/Users/lmhst/Downloads/aiHub.coco.zip")
    parser.add_argument("--yolo", default="C:/Users/lmhst/Downloads/yolov11_best.pt")
    parser.add_argument("--out-dir", default="bbox_prediction_examples_aihub")
    parser.add_argument("--seed", type=int, default=31)
    parser.add_argument("--foundation-threshold", type=float, default=0.05)
    parser.add_argument("--yolo-conf", type=float, default=0.25)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    out_dir = ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    _coco, images, anns_by_image, image_entries = load_coco_from_zip(args.zip)
    available_classes = sorted({ann["class"] for anns in anns_by_image.values() for ann in anns})
    target_classes = [cls for cls in TARGET_CLASSES if cls in available_classes]
    chosen = choose_images(images, anns_by_image, target_classes, args.seed)

    runners = [("YOLO fine-tuned", YoloRunner(args.yolo), args.yolo_conf)]
    for spec in DEFAULT_FOUNDATIONS:
        kind, model_name, label = parse_foundation_spec(spec)
        print(f"Loading {label}")
        runners.append((label, build_runner(kind, model_name, args.device), args.foundation_threshold))

    images_by_model_class = {}
    with zipfile.ZipFile(args.zip) as zf:
        for cls, image_id in chosen.items():
            info = images[image_id]
            entry_name = "train/" + info["file_name"]
            if entry_name not in image_entries:
                entry_name = next(name for name in image_entries if name.endswith("/" + info["file_name"]))
            with zf.open(entry_name) as f:
                base = load_image(f)
            gt_boxes = [ann for ann in anns_by_image[image_id] if ann["class"] == cls]

            for model_label, runner, threshold in runners:
                detections, _elapsed = runner.detect(base, threshold)
                same_class = [det for det in detections if det["class"] == cls]
                same_class.sort(key=lambda det: det["score"], reverse=True)
                annotated = draw_boxes(base, gt_boxes, same_class, model_label, cls)
                images_by_model_class[(model_label, cls)] = annotated
                safe = model_label.lower().replace(" ", "_").replace("-", "_")
                annotated.save(out_dir / f"{cls}_{safe}.png")

    model_labels = [label for label, _runner, _threshold in runners]
    grid_path = out_dir / "aihub_bbox_prediction_examples_grid.png"
    make_grid(images_by_model_class, model_labels, target_classes, grid_path)
    print(grid_path)


if __name__ == "__main__":
    main()
