import argparse
import json
from pathlib import Path
from datetime import datetime

import pandas as pd
import yaml
from ultralytics import YOLO


VALID_VAL_KEYS = {
    "data",
    "split",
    "imgsz",
    "batch",
    "conf",
    "iou",
    "device",
    "half",
    "plots",
    "save_json",
    "save_txt",
    "save_conf",
    "project",
    "name",
    "workers",
    "verbose",
}


def load_yaml(path: str | Path) -> dict:
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"YAML file not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    return data or {}


def filter_val_args(hyp: dict) -> dict:
    """
    YOLO 학습용 hyp.yaml에는 validation에서 쓰지 않는 값들이 섞여 있을 수 있다.
    예: lr0, lrf, momentum, weight_decay, warmup_epochs 등
    따라서 model.val()에 넣을 수 있는 key만 필터링한다.
    """
    return {k: v for k, v in hyp.items() if k in VALID_VAL_KEYS}


def safe_float(value):
    try:
        return float(value)
    except Exception:
        return None


def make_output_dir(project: str, name: str | None, model_path: Path, split: str) -> Path:
    if name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        name = f"eval_{model_path.stem}_{split}_{timestamp}"

    output_dir = Path(project) / name
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def save_summary(metrics, model: YOLO, output_dir: Path, model_path: Path, data_yaml: Path, val_args: dict):
    box = metrics.box

    summary = {
        "model_path": str(model_path),
        "data_yaml": str(data_yaml),
        "val_args": val_args,
        "mAP50_95": safe_float(box.map),
        "mAP50": safe_float(box.map50),
        "mAP75": safe_float(box.map75),
        "mean_precision": safe_float(box.mp),
        "mean_recall": safe_float(box.mr),
        "fitness": safe_float(metrics.fitness),
        "speed_ms_per_image": metrics.speed,
        "class_names": model.names,
    }

    with open(output_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    pd.DataFrame([{
        "model_path": summary["model_path"],
        "data_yaml": summary["data_yaml"],
        "mAP50_95": summary["mAP50_95"],
        "mAP50": summary["mAP50"],
        "mAP75": summary["mAP75"],
        "mean_precision": summary["mean_precision"],
        "mean_recall": summary["mean_recall"],
        "fitness": summary["fitness"],
    }]).to_csv(output_dir / "summary.csv", index=False)

    return summary


def save_per_class_metrics(metrics, model: YOLO, output_dir: Path):
    box = metrics.box
    class_maps = list(box.maps) if box.maps is not None else []

    rows = []

    for class_id, class_name in model.names.items():
        class_id = int(class_id)

        rows.append({
            "class_id": class_id,
            "class_name": class_name,
            "mAP50_95": safe_float(class_maps[class_id]) if class_id < len(class_maps) else None,
        })

    pd.DataFrame(rows).to_csv(output_dir / "per_class_metrics.csv", index=False)


def save_confusion_matrix(metrics, output_dir: Path):
    try:
        cm_df = metrics.confusion_matrix.to_df()

        # Ultralytics 버전에 따라 pandas DataFrame 또는 polars DataFrame일 수 있음
        if hasattr(cm_df, "to_csv"):
            cm_df.to_csv(output_dir / "confusion_matrix.csv", index=False)
        elif hasattr(cm_df, "write_csv"):
            cm_df.write_csv(output_dir / "confusion_matrix.csv")
        else:
            with open(output_dir / "confusion_matrix.txt", "w", encoding="utf-8") as f:
                f.write(str(cm_df))

    except Exception as e:
        with open(output_dir / "confusion_matrix_error.txt", "w", encoding="utf-8") as f:
            f.write(str(e))


def validate_tuned_model(
    model_path: str,
    data_yaml: str,
    hyp_yaml: str,
    output_project: str,
    output_name: str | None,
):
    model_path = Path(model_path)
    data_yaml = Path(data_yaml)
    hyp_yaml = Path(hyp_yaml)

    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    if not data_yaml.exists():
        raise FileNotFoundError(f"Dataset YAML not found: {data_yaml}")

    hyp = load_yaml(hyp_yaml)
    val_args = filter_val_args(hyp)

    # CLI에서 받은 data.yaml을 우선 사용
    val_args["data"] = str(data_yaml)

    # 기본값 보정
    val_args.setdefault("split", "test")
    val_args.setdefault("imgsz", 640)
    val_args.setdefault("batch", 16)
    val_args.setdefault("conf", 0.25)
    val_args.setdefault("iou", 0.7)
    val_args.setdefault("plots", True)
    val_args.setdefault("save_json", True)
    val_args.setdefault("save_txt", True)
    val_args.setdefault("save_conf", True)
    val_args.setdefault("verbose", True)

    split = val_args["split"]

    output_dir = make_output_dir(
        project=output_project,
        name=output_name,
        model_path=model_path,
        split=split,
    )

    # Ultralytics 결과 저장 경로 지정
    val_args["project"] = str(output_dir.parent)
    val_args["name"] = output_dir.name

    print("\n================ Validation Config ================")
    print(f"Model : {model_path}")
    print(f"Data  : {data_yaml}")
    print(f"Hyp   : {hyp_yaml}")
    print(f"Split : {split}")
    print("Validation args:")
    for k, v in val_args.items():
        print(f"  {k}: {v}")
    print("===================================================\n")

    model = YOLO(str(model_path))

    metrics = model.val(**val_args)

    summary = save_summary(
        metrics=metrics,
        model=model,
        output_dir=output_dir,
        model_path=model_path,
        data_yaml=data_yaml,
        val_args=val_args,
    )

    save_per_class_metrics(metrics, model, output_dir)
    save_confusion_matrix(metrics, output_dir)

    try:
        with open(output_dir / "metrics.json", "w", encoding="utf-8") as f:
            f.write(metrics.to_json())
    except Exception as e:
        with open(output_dir / "metrics_json_error.txt", "w", encoding="utf-8") as f:
            f.write(str(e))

    print("\n================ Validation Summary ================")
    print(f"mAP50-95      : {summary['mAP50_95']:.4f}")
    print(f"mAP50         : {summary['mAP50']:.4f}")
    print(f"mAP75         : {summary['mAP75']:.4f}")
    print(f"Mean Precision: {summary['mean_precision']:.4f}")
    print(f"Mean Recall   : {summary['mean_recall']:.4f}")
    print(f"Output Dir    : {output_dir}")
    print("====================================================\n")

    return summary


def parse_args():
    parser = argparse.ArgumentParser(
        description="Validate a fine-tuned YOLO11m .pt model using hyperparameters from YAML."
    )

    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to fine-tuned .pt model. Example: runs/detect/train/weights/best.pt",
    )

    parser.add_argument(
        "--data",
        type=str,
        required=True,
        help="Path to dataset data.yaml.",
    )

    parser.add_argument(
        "--hyp",
        type=str,
        required=True,
        help="Path to hyperparameter YAML file.",
    )

    parser.add_argument(
        "--project",
        type=str,
        default="runs/validate",
        help="Output directory.",
    )

    parser.add_argument(
        "--name",
        type=str,
        default=None,
        help="Run name.",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    validate_tuned_model(
        model_path=args.model,
        data_yaml=args.data,
        hyp_yaml=args.hyp,
        output_project=args.project,
        output_name=args.name,
    )


if __name__ == "__main__":
    main()