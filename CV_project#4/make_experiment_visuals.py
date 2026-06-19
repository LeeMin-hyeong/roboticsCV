import json
import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parent
IMAGE_SUMMARY = ROOT / "comparison_results_multi_full" / "summary.json"
BBOX_SUMMARY = ROOT / "roboflow_bbox_subset_20" / "summary.json"
OUT_DIR = ROOT / "report_visuals"

MODEL_LABELS = {
    "yolo": "YOLO",
    "owl_vit": "OWL-ViT",
    "owlv2": "OWLv2",
    "grounding_dino": "Grounding DINO",
}
MODEL_ORDER = ["yolo", "owl_vit", "owlv2", "grounding_dino"]
MODEL_COLORS = {
    "yolo": "#2563eb",
    "owl_vit": "#16a34a",
    "owlv2": "#f59e0b",
    "grounding_dino": "#dc2626",
}


def load_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def finish(path):
    plt.tight_layout()
    plt.savefig(path, dpi=180, bbox_inches="tight")
    plt.close()


def grouped_bar(title, categories, values_by_model, ylabel, path, ylim=None):
    x = np.arange(len(categories))
    width = 0.18
    offsets = np.linspace(-1.5 * width, 1.5 * width, len(MODEL_ORDER))

    plt.figure(figsize=(9.5, 5.2))
    for offset, key in zip(offsets, MODEL_ORDER):
        vals = values_by_model[key]
        plt.bar(
            x + offset,
            vals,
            width,
            label=MODEL_LABELS[key],
            color=MODEL_COLORS[key],
            edgecolor="#111827",
            linewidth=0.4,
        )
        for xi, yi in zip(x + offset, vals):
            plt.text(xi, yi + 0.015, f"{yi:.2f}", ha="center", va="bottom", fontsize=8)

    plt.title(title, fontsize=14, weight="bold")
    plt.ylabel(ylabel)
    plt.xticks(x, categories)
    if ylim:
        plt.ylim(*ylim)
    plt.legend(frameon=False, ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.09))
    plt.grid(axis="y", alpha=0.25)
    finish(path)


def heatmap(title, row_labels, col_labels, matrix, path, cbar_label):
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    im = ax.imshow(matrix, cmap="YlGnBu", vmin=0, vmax=1)
    ax.set_title(title, fontsize=14, weight="bold")
    ax.set_xticks(np.arange(len(col_labels)), labels=col_labels, rotation=20, ha="right")
    ax.set_yticks(np.arange(len(row_labels)), labels=row_labels)

    for i in range(len(row_labels)):
        for j in range(len(col_labels)):
            val = matrix[i, j]
            color = "white" if val > 0.65 else "#111827"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center", color=color, fontsize=9)

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.set_ylabel(cbar_label, rotation=-90, va="bottom")
    finish(path)


def make_image_level_visuals(summary):
    overall_values = {
        key: [
            summary["models"][key]["accuracy"],
            summary["models"][key]["coverage"],
        ]
        for key in MODEL_ORDER
    }
    grouped_bar(
        "Image-level prediction: accuracy and coverage",
        ["Accuracy", "Coverage"],
        overall_values,
        "Score",
        OUT_DIR / "image_level_overall.png",
        ylim=(0, 1.12),
    )

    classes = list(summary["by_class"].keys())
    matrix = np.array(
        [
            [summary["by_class"][cls]["models"][key]["accuracy"] for key in MODEL_ORDER]
            for cls in classes
        ]
    )
    heatmap(
        "Image-level class accuracy",
        classes,
        [MODEL_LABELS[key] for key in MODEL_ORDER],
        matrix,
        OUT_DIR / "image_level_class_accuracy.png",
        "Accuracy",
    )


def make_bbox_visuals(summary):
    overall_values = {
        key: [
            summary["models"][key]["map50"],
            summary["models"][key]["map50_95"],
            summary["models"][key]["recall50"],
        ]
        for key in MODEL_ORDER
    }
    grouped_bar(
        "BBox evaluation: mAP and recall",
        ["mAP@0.5", "mAP@0.5:0.95", "Recall@0.5"],
        overall_values,
        "Score",
        OUT_DIR / "bbox_overall_map.png",
        ylim=(0, 1.12),
    )

    classes = list(summary["models"]["yolo"]["per_class50"].keys())
    matrix = np.array(
        [
            [summary["models"][key]["per_class50"][cls]["ap"] for key in MODEL_ORDER]
            for cls in classes
        ]
    )
    heatmap(
        "BBox AP@0.5 by class",
        classes,
        [MODEL_LABELS[key] for key in MODEL_ORDER],
        matrix,
        OUT_DIR / "bbox_class_ap.png",
        "AP@0.5",
    )


def make_time_visual(image_summary, bbox_summary):
    categories = ["No-bbox\nimage-level", "COCO bbox\nmAP eval"]
    x = np.arange(len(categories))
    width = 0.18
    offsets = np.linspace(-1.5 * width, 1.5 * width, len(MODEL_ORDER))

    plt.figure(figsize=(9.5, 5.2))
    for offset, key in zip(offsets, MODEL_ORDER):
        vals = [
            image_summary["models"][key]["avg_time_sec"],
            bbox_summary["models"][key]["avg_time_sec"],
        ]
        plt.bar(
            x + offset,
            vals,
            width,
            label=MODEL_LABELS[key],
            color=MODEL_COLORS[key],
            edgecolor="#111827",
            linewidth=0.4,
        )
        for xi, yi in zip(x + offset, vals):
            plt.text(xi, yi + 0.015, f"{yi:.2f}s", ha="center", va="bottom", fontsize=8)

    plt.title("Average inference time per image", fontsize=14, weight="bold")
    plt.ylabel("Seconds")
    plt.xticks(x, categories)
    plt.legend(frameon=False, ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.12))
    plt.grid(axis="y", alpha=0.25)
    finish(OUT_DIR / "inference_time_comparison.png")


def main():
    global OUT_DIR
    parser = argparse.ArgumentParser()
    parser.add_argument("--image-summary", default=str(IMAGE_SUMMARY))
    parser.add_argument("--bbox-summary", default=str(BBOX_SUMMARY))
    parser.add_argument("--out-dir", default=str(OUT_DIR))
    args = parser.parse_args()

    OUT_DIR = Path(args.out_dir)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    image_summary = load_json(Path(args.image_summary))
    bbox_summary = load_json(Path(args.bbox_summary))
    make_image_level_visuals(image_summary)
    make_bbox_visuals(bbox_summary)
    make_time_visual(image_summary, bbox_summary)

    for path in sorted(OUT_DIR.glob("*.png")):
        print(path)


if __name__ == "__main__":
    main()
