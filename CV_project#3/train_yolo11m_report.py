import json
import shutil
import random
from pathlib import Path
from datetime import datetime

import yaml
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from ultralytics import YOLO


ROOT = Path(__file__).resolve().parent

ARGS_YAML = ROOT / "yolov11_args.yaml"
COCO_JSON = ROOT / "dataset" / "train" / "_annotations.coco.json"

SOURCE_IMAGE_DIR = ROOT / "dataset" / "train"

PREPARED_DATASET_DIR = ROOT / "dataset_yolo_report"
REPORT_DIR = ROOT / "report_yolo11m"

TRAIN_RATIO = 0.8
SEED = 0


IMAGE_EXTS = {
    ".jpg", ".jpeg", ".png", ".bmp", ".webp"
}


def load_yaml(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"YAML 파일을 찾을 수 없습니다: {path}")

    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def load_coco(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"COCO annotation 파일을 찾을 수 없습니다: {path}")

    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def make_clean_dir(path: Path):
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def coco_bbox_to_yolo(bbox, img_w, img_h):
    """
    COCO bbox: [x_min, y_min, width, height]
    YOLO bbox: x_center y_center width height, normalized

    Roboflow COCO 데이터에서 bbox 값 일부가 문자열로 들어오는 경우가 있어서
    float 변환을 먼저 수행한다.
    """
    if bbox is None or len(bbox) != 4:
        raise ValueError(f"Invalid bbox: {bbox}")

    try:
        x, y, w, h = [float(v) for v in bbox]
        img_w = float(img_w)
        img_h = float(img_h)
    except Exception as e:
        raise ValueError(f"bbox 값을 float으로 변환할 수 없습니다. bbox={bbox}") from e

    if img_w <= 0 or img_h <= 0:
        raise ValueError(f"Invalid image size: width={img_w}, height={img_h}")

    if w <= 0 or h <= 0:
        raise ValueError(f"Invalid bbox size: bbox={bbox}")

    x_center = x + w / 2.0
    y_center = y + h / 2.0

    return [
        x_center / img_w,
        y_center / img_h,
        w / img_w,
        h / img_h,
    ]


def copy_image(src: Path, dst: Path):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if not src.exists():
        pass
        # raise FileNotFoundError(f"이미지 파일을 찾을 수 없습니다: {src}")
    shutil.copy2(src, dst)


def prepare_yolo_dataset():
    """
    ./dataset/train/_annotations.coco.json 을 읽어서
    Ultralytics YOLO용 dataset 구조로 변환한다.
    """
    coco = load_coco(COCO_JSON)

    images = coco.get("images", [])
    annotations = coco.get("annotations", [])
    categories = coco.get("categories", [])

    if not images:
        raise ValueError("COCO JSON 안에 images 항목이 없습니다.")

    if not categories:
        raise ValueError("COCO JSON 안에 categories 항목이 없습니다.")

    category_id_to_yolo_id = {
        cat["id"]: idx for idx, cat in enumerate(sorted(categories, key=lambda x: x["id"]))
    }

    yolo_names = {
        idx: cat["name"] for idx, cat in enumerate(sorted(categories, key=lambda x: x["id"]))
    }

    image_id_to_info = {img["id"]: img for img in images}

    ann_by_image = {}
    for ann in annotations:
        image_id = ann["image_id"]
        ann_by_image.setdefault(image_id, []).append(ann)

    image_ids = [img["id"] for img in images]

    train_ids, val_ids = train_test_split(
        image_ids,
        train_size=TRAIN_RATIO,
        random_state=SEED,
        shuffle=True,
    )

    train_ids = set(train_ids)
    val_ids = set(val_ids)

    make_clean_dir(PREPARED_DATASET_DIR)

    for split in ["train", "val"]:
        (PREPARED_DATASET_DIR / "images" / split).mkdir(parents=True, exist_ok=True)
        (PREPARED_DATASET_DIR / "labels" / split).mkdir(parents=True, exist_ok=True)

    for image_id, img in image_id_to_info.items():
        file_name = img["file_name"]
        img_w = img["width"]
        img_h = img["height"]

        split = "train" if image_id in train_ids else "val"

        src_image = SOURCE_IMAGE_DIR / file_name
        dst_image = PREPARED_DATASET_DIR / "images" / split / Path(file_name).name
        dst_label = PREPARED_DATASET_DIR / "labels" / split / f"{Path(file_name).stem}.txt"

        copy_image(src_image, dst_image)

        label_lines = []

        for ann in ann_by_image.get(image_id, []):
            if ann.get("iscrowd", 0) == 1:
                continue

            category_id = ann["category_id"]
            if category_id not in category_id_to_yolo_id:
                continue

            cls_id = category_id_to_yolo_id[category_id]
            try:
                x_center, y_center, w, h = coco_bbox_to_yolo(
                    bbox=ann["bbox"],
                    img_w=img_w,
                    img_h=img_h,
                )
            except ValueError as e:
                print(f"[WARN] bbox 변환 실패, annotation 스킵: {e}")
                continue

            # 잘못된 bbox 방어
            if w <= 0 or h <= 0:
                continue

            x_center = min(max(x_center, 0.0), 1.0)
            y_center = min(max(y_center, 0.0), 1.0)
            w = min(max(w, 0.0), 1.0)
            h = min(max(h, 0.0), 1.0)

            label_lines.append(
                f"{cls_id} {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}"
            )

        dst_label.write_text("\n".join(label_lines), encoding="utf-8")

    data_yaml = {
        "path": str(PREPARED_DATASET_DIR),
        "train": "images/train",
        "val": "images/val",
        "names": yolo_names,
    }

    data_yaml_path = PREPARED_DATASET_DIR / "data.yaml"

    with open(data_yaml_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(data_yaml, f, sort_keys=False, allow_unicode=True)

    summary = {
        "total_images": len(image_ids),
        "train_images": len(train_ids),
        "val_images": len(val_ids),
        "total_annotations": len(annotations),
        "classes": yolo_names,
        "prepared_dataset_dir": str(PREPARED_DATASET_DIR),
        "data_yaml": str(data_yaml_path),
    }

    return data_yaml_path, summary


def filter_train_args(args: dict) -> dict:
    """
    yolov11_args.yaml에서 YOLO train에 넘길 주요 인자만 정리한다.
    save_dir, data 등 환경 의존 값은 현재 프로젝트 기준으로 덮어쓴다.
    """
    allowed_keys = {
        "epochs",
        "patience",
        "batch",
        "imgsz",
        "save",
        "save_period",
        "cache",
        "device",
        "workers",
        "project",
        "name",
        "exist_ok",
        "pretrained",
        "optimizer",
        "verbose",
        "seed",
        "deterministic",
        "single_cls",
        "rect",
        "cos_lr",
        "close_mosaic",
        "resume",
        "amp",
        "fraction",
        "profile",
        "freeze",
        "multi_scale",
        "overlap_mask",
        "mask_ratio",
        "dropout",
        "val",
        "split",
        "conf",
        "iou",
        "max_det",
        "half",
        "plots",
        "augment",
        "agnostic_nms",
        "classes",
        "lr0",
        "lrf",
        "momentum",
        "weight_decay",
        "warmup_epochs",
        "warmup_momentum",
        "warmup_bias_lr",
        "box",
        "cls",
        "dfl",
        "nbs",
        "hsv_h",
        "hsv_s",
        "hsv_v",
        "degrees",
        "translate",
        "scale",
        "shear",
        "perspective",
        "flipud",
        "fliplr",
        "mosaic",
        "mixup",
        "copy_paste",
        "auto_augment",
        "erasing",
    }

    train_args = {k: v for k, v in args.items() if k in allowed_keys}

    # 업로드된 yaml 기준값 반영
    train_args.setdefault("epochs", 150)
    train_args.setdefault("patience", 35)
    train_args.setdefault("batch", 16)
    train_args.setdefault("imgsz", 640)
    train_args.setdefault("plots", True)
    train_args.setdefault("val", True)
    train_args.setdefault("exist_ok", True)

    # 현재 프로젝트 폴더 기준으로 저장 위치 고정
    train_args["project"] = str(ROOT / "runs" / "yolov11_report")
    train_args["name"] = "tools_report"
    train_args["exist_ok"] = True

    # GPU 번호가 안 맞으면 에러가 날 수 있으므로 필요 시 아래를 "0" 또는 "cpu"로 수정
    train_args["device"] = "0"
    # train_args["device"] = "cpu"

    return train_args


def train_model(data_yaml_path: Path):
    args = load_yaml(ARGS_YAML)
    train_args = filter_train_args(args)

    model_path = args.get("model", "yolo11m.pt")
    model = YOLO(model_path)

    print("\n================ Train Config ================")
    print(f"model: {model_path}")
    print(f"data : {data_yaml_path}")
    for k, v in train_args.items():
        print(f"{k}: {v}")
    print("==============================================\n")

    results = model.train(
        data=str(data_yaml_path),
        **train_args,
    )

    save_dir = Path(results.save_dir)

    return save_dir, train_args


def find_column(df: pd.DataFrame, candidates):
    columns = {c.strip(): c for c in df.columns}

    for candidate in candidates:
        if candidate in columns:
            return columns[candidate]

    for col in df.columns:
        normalized = col.strip().lower()
        for candidate in candidates:
            if normalized == candidate.lower():
                return col

    return None


def plot_metric(df: pd.DataFrame, x_col: str, y_cols: list[str], title: str, ylabel: str, save_path: Path):
    plt.figure(figsize=(10, 6))

    for col in y_cols:
        if col in df.columns:
            plt.plot(df[x_col], df[col], label=col)

    plt.title(title)
    plt.xlabel("epoch")
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def create_training_report_assets(save_dir: Path, dataset_summary: dict, train_args: dict):
    """
    Ultralytics가 생성한 results.csv를 기반으로
    보고서용 그래프, 표, 요약 파일을 생성한다.
    """
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    results_csv = save_dir / "results.csv"
    if not results_csv.exists():
        raise FileNotFoundError(f"results.csv를 찾을 수 없습니다: {results_csv}")

    df = pd.read_csv(results_csv)
    df.columns = [c.strip() for c in df.columns]

    epoch_col = find_column(df, ["epoch"])
    if epoch_col is None:
        df.insert(0, "epoch", range(1, len(df) + 1))
        epoch_col = "epoch"

    # 원본 학습 로그 복사
    df.to_csv(REPORT_DIR / "training_results_full.csv", index=False)

    # 보고서용 마지막 10 epoch 표
    last_table = df.tail(10).copy()
    last_table.to_csv(REPORT_DIR / "training_last_10_epochs.csv", index=False)

    # 최고 epoch 요약용 컬럼 후보
    map50_col = find_column(df, ["metrics/mAP50(B)", "metrics/mAP50"])
    map5095_col = find_column(df, ["metrics/mAP50-95(B)", "metrics/mAP50-95"])
    precision_col = find_column(df, ["metrics/precision(B)", "metrics/precision"])
    recall_col = find_column(df, ["metrics/recall(B)", "metrics/recall"])

    best_metric_col = map5095_col or map50_col
    if best_metric_col:
        best_idx = df[best_metric_col].idxmax()
        best_row = df.loc[best_idx].to_dict()
    else:
        best_idx = len(df) - 1
        best_row = df.iloc[-1].to_dict()

    summary_rows = []

    for label, col in [
        ("best_mAP50_95", map5095_col),
        ("best_mAP50", map50_col),
        ("best_precision", precision_col),
        ("best_recall", recall_col),
    ]:
        if col and col in df.columns:
            summary_rows.append({
                "metric": label,
                "best_value": float(df[col].max()),
                "best_epoch": int(df.loc[df[col].idxmax(), epoch_col]),
                "source_column": col,
            })

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(REPORT_DIR / "training_summary_table.csv", index=False)

    # loss 그래프
    train_loss_cols = [
        col for col in [
            "train/box_loss",
            "train/cls_loss",
            "train/dfl_loss",
        ] if col in df.columns
    ]

    val_loss_cols = [
        col for col in [
            "val/box_loss",
            "val/cls_loss",
            "val/dfl_loss",
        ] if col in df.columns
    ]

    metric_cols = [
        col for col in [
            precision_col,
            recall_col,
            map50_col,
            map5095_col,
        ] if col is not None and col in df.columns
    ]

    if train_loss_cols:
        plot_metric(
            df=df,
            x_col=epoch_col,
            y_cols=train_loss_cols,
            title="Training Loss Curve",
            ylabel="loss",
            save_path=REPORT_DIR / "graph_train_loss.png",
        )

    if val_loss_cols:
        plot_metric(
            df=df,
            x_col=epoch_col,
            y_cols=val_loss_cols,
            title="Validation Loss Curve",
            ylabel="loss",
            save_path=REPORT_DIR / "graph_val_loss.png",
        )

    if metric_cols:
        plot_metric(
            df=df,
            x_col=epoch_col,
            y_cols=metric_cols,
            title="Detection Metrics Curve",
            ylabel="score",
            save_path=REPORT_DIR / "graph_detection_metrics.png",
        )

    if map50_col and map5095_col:
        plot_metric(
            df=df,
            x_col=epoch_col,
            y_cols=[map50_col, map5095_col],
            title="mAP Curve",
            ylabel="mAP",
            save_path=REPORT_DIR / "graph_map.png",
        )

    if precision_col and recall_col:
        plot_metric(
            df=df,
            x_col=epoch_col,
            y_cols=[precision_col, recall_col],
            title="Precision / Recall Curve",
            ylabel="score",
            save_path=REPORT_DIR / "graph_precision_recall.png",
        )

    # Ultralytics가 자동 생성한 주요 이미지도 보고서 폴더로 복사
    auto_plot_names = [
        "results.png",
        "confusion_matrix.png",
        "confusion_matrix_normalized.png",
        "F1_curve.png",
        "P_curve.png",
        "R_curve.png",
        "PR_curve.png",
        "labels.jpg",
        "labels_correlogram.jpg",
    ]

    for name in auto_plot_names:
        src = save_dir / name
        if src.exists():
            shutil.copy2(src, REPORT_DIR / name)

    report_summary = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "source_coco_json": str(COCO_JSON),
        "prepared_dataset": dataset_summary,
        "train_args": train_args,
        "ultralytics_save_dir": str(save_dir),
        "report_dir": str(REPORT_DIR),
        "best_epoch_index": int(best_idx),
        "best_epoch_row": {
            str(k): float(v) if isinstance(v, (int, float)) else str(v)
            for k, v in best_row.items()
        },
    }

    with open(REPORT_DIR / "report_summary.json", "w", encoding="utf-8") as f:
        json.dump(report_summary, f, indent=2, ensure_ascii=False)

    report_txt = make_report_text(
        dataset_summary=dataset_summary,
        train_args=train_args,
        summary_df=summary_df,
        save_dir=save_dir,
    )

    (REPORT_DIR / "report_summary.txt").write_text(report_txt, encoding="utf-8")

    print("\n================ Report Assets Saved ================")
    print(f"Ultralytics run dir : {save_dir}")
    print(f"Report dir          : {REPORT_DIR}")
    print("Saved files:")
    for path in sorted(REPORT_DIR.iterdir()):
        print(f"- {path.name}")
    print("=====================================================\n")


def make_report_text(dataset_summary: dict, train_args: dict, summary_df: pd.DataFrame, save_dir: Path) -> str:
    class_text = ", ".join(
        [f"{idx}: {name}" for idx, name in dataset_summary["classes"].items()]
    )

    metric_lines = []
    for _, row in summary_df.iterrows():
        metric_lines.append(
            f"- {row['metric']}: {row['best_value']:.4f} at epoch {int(row['best_epoch'])}"
        )

    metric_text = "\n".join(metric_lines) if metric_lines else "- metric summary unavailable"

    text = f"""
YOLO11m Domain Fine-tuning Training Report Summary

1. Dataset
- Source annotation: {COCO_JSON}
- Total images: {dataset_summary['total_images']}
- Train images: {dataset_summary['train_images']}
- Validation images: {dataset_summary['val_images']}
- Total annotations: {dataset_summary['total_annotations']}
- Classes: {class_text}

2. Training Configuration
- Model: {load_yaml(ARGS_YAML).get('model', 'yolo11m.pt')}
- Epochs: {train_args.get('epochs')}
- Batch size: {train_args.get('batch')}
- Image size: {train_args.get('imgsz')}
- Optimizer: {train_args.get('optimizer')}
- Learning rate lr0: {train_args.get('lr0')}
- Weight decay: {train_args.get('weight_decay')}
- Patience: {train_args.get('patience')}
- Augmentation: mosaic={train_args.get('mosaic')}, mixup={train_args.get('mixup')}, copy_paste={train_args.get('copy_paste')}

3. Best Training Metrics
{metric_text}

4. Generated Report Assets
- training_results_full.csv
- training_last_10_epochs.csv
- training_summary_table.csv
- graph_train_loss.png
- graph_val_loss.png
- graph_detection_metrics.png
- graph_map.png
- graph_precision_recall.png
- results.png
- confusion_matrix.png
- PR_curve.png
- F1_curve.png

5. Original Ultralytics Output
- {save_dir}
""".strip()

    return text


def main():
    random.seed(SEED)

    data_yaml_path, dataset_summary = prepare_yolo_dataset()
    save_dir, train_args = train_model(data_yaml_path)
    create_training_report_assets(
        save_dir=save_dir,
        dataset_summary=dataset_summary,
        train_args=train_args,
    )


if __name__ == "__main__":
    main()