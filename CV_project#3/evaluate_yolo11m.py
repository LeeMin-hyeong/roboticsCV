import argparse
import json
from pathlib import Path
from datetime import datetime

import pandas as pd
from ultralytics import YOLO


def safe_float(value):
    try:
        return float(value)
    except Exception:
        return None


def save_text(path: Path, text: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def evaluate_model(
    model_path: str,
    data_yaml: str,
    split: str,
    imgsz: int,
    batch: int,
    conf: float,
    iou: float,
    device: str,
    project: str,
    name: str,
):
    model_path = Path(model_path)
    data_yaml = Path(data_yaml)

    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    if not data_yaml.exists():
        raise FileNotFoundError(f"Dataset YAML not found: {data_yaml}")

    run_name = name or f"eval_{model_path.stem}_{split}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    output_dir = Path(project) / run_name
    output_dir.mkdir(parents=True, exist_ok=True)

    model = YOLO(str(model_path))

    metrics = model.val(
        data=str(data_yaml),
        split=split,
        imgsz=imgsz,
        batch=batch,
        conf=conf,
        iou=iou,
        device=device,
        plots=True,
        save_json=True,
        save_txt=True,
        save_conf=True,
        project=project,
        name=run_name,
        verbose=True,
    )

    # -----------------------------
    # 1) 전체 성능 요약 저장
    # -----------------------------
    box = metrics.box

    summary = {
        "model_path": str(model_path),
        "data_yaml": str(data_yaml),
        "split": split,
        "imgsz": imgsz,
        "batch": batch,
        "conf": conf,
        "iou": iou,
        "device": device,
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

    pd.DataFrame([summary]).to_csv(output_dir / "summary.csv", index=False)

    # -----------------------------
    # 2) 클래스별 성능 저장
    # -----------------------------
    class_rows = []

    class_maps = list(box.maps) if box.maps is not None else []

    for class_id, class_name in model.names.items():
        row = {
            "class_id": int(class_id),
            "class_name": class_name,
            "mAP50_95": safe_float(class_maps[int(class_id)]) if int(class_id) < len(class_maps) else None,
        }
        class_rows.append(row)

    class_df = pd.DataFrame(class_rows)
    class_df.to_csv(output_dir / "per_class_metrics.csv", index=False)

    # -----------------------------
    # 3) Ultralytics 기본 export 저장
    # -----------------------------
    try:
        save_text(output_dir / "ultralytics_metrics.csv", metrics.to_csv())
    except Exception as e:
        save_text(output_dir / "ultralytics_metrics_csv_error.txt", str(e))

    try:
        save_text(output_dir / "ultralytics_metrics.json", metrics.to_json())
    except Exception as e:
        save_text(output_dir / "ultralytics_metrics_json_error.txt", str(e))

    # -----------------------------
    # 4) 이미지별 precision / recall / F1 저장
    # -----------------------------
    image_metrics = getattr(box, "image_metrics", None)

    if image_metrics:
        image_rows = []
        for image_name, values in image_metrics.items():
            row = {"image": image_name}
            row.update(values)
            image_rows.append(row)

        pd.DataFrame(image_rows).to_csv(output_dir / "per_image_metrics.csv", index=False)

    # -----------------------------
    # 5) 혼동 행렬 저장
    # -----------------------------
    try:
        cm_df = metrics.confusion_matrix.to_df()
        cm_df.write_csv(output_dir / "confusion_matrix.csv")
    except Exception as e:
        save_text(output_dir / "confusion_matrix_error.txt", str(e))

    print("\n================ Evaluation Summary ================")
    print(f"Model      : {model_path}")
    print(f"Dataset    : {data_yaml}")
    print(f"Split      : {split}")
    print(f"mAP50-95   : {summary['mAP50_95']:.4f}")
    print(f"mAP50      : {summary['mAP50']:.4f}")
    print(f"mAP75      : {summary['mAP75']:.4f}")
    print(f"Precision  : {summary['mean_precision']:.4f}")
    print(f"Recall     : {summary['mean_recall']:.4f}")
    print(f"Output dir : {output_dir}")
    print("====================================================\n")

    return summary


def compare_models(
    baseline_path: str,
    finetuned_path: str,
    data_yaml: str,
    split: str,
    imgsz: int,
    batch: int,
    conf: float,
    iou: float,
    device: str,
    project: str,
):
    baseline = evaluate_model(
        model_path=baseline_path,
        data_yaml=data_yaml,
        split=split,
        imgsz=imgsz,
        batch=batch,
        conf=conf,
        iou=iou,
        device=device,
        project=project,
        name="baseline_eval",
    )

    finetuned = evaluate_model(
        model_path=finetuned_path,
        data_yaml=data_yaml,
        split=split,
        imgsz=imgsz,
        batch=batch,
        conf=conf,
        iou=iou,
        device=device,
        project=project,
        name="finetuned_eval",
    )

    comparison = {
        "baseline_model": baseline_path,
        "finetuned_model": finetuned_path,
        "split": split,
        "baseline_mAP50_95": baseline["mAP50_95"],
        "finetuned_mAP50_95": finetuned["mAP50_95"],
        "delta_mAP50_95": finetuned["mAP50_95"] - baseline["mAP50_95"],
        "baseline_mAP50": baseline["mAP50"],
        "finetuned_mAP50": finetuned["mAP50"],
        "delta_mAP50": finetuned["mAP50"] - baseline["mAP50"],
        "baseline_precision": baseline["mean_precision"],
        "finetuned_precision": finetuned["mean_precision"],
        "delta_precision": finetuned["mean_precision"] - baseline["mean_precision"],
        "baseline_recall": baseline["mean_recall"],
        "finetuned_recall": finetuned["mean_recall"],
        "delta_recall": finetuned["mean_recall"] - baseline["mean_recall"],
    }

    output_dir = Path(project) / "comparison"
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir / "comparison.json", "w", encoding="utf-8") as f:
        json.dump(comparison, f, indent=2, ensure_ascii=False)

    pd.DataFrame([comparison]).to_csv(output_dir / "comparison.csv", index=False)

    print("\n================ Model Comparison ================")
    print(f"Baseline mAP50-95 : {comparison['baseline_mAP50_95']:.4f}")
    print(f"Finetuned mAP50-95: {comparison['finetuned_mAP50_95']:.4f}")
    print(f"Delta mAP50-95    : {comparison['delta_mAP50_95']:+.4f}")
    print(f"Baseline mAP50    : {comparison['baseline_mAP50']:.4f}")
    print(f"Finetuned mAP50   : {comparison['finetuned_mAP50']:.4f}")
    print(f"Delta mAP50       : {comparison['delta_mAP50']:+.4f}")
    print(f"Output dir        : {output_dir}")
    print("==================================================\n")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate YOLO11m fine-tuned model on a domain-specific dataset."
    )

    parser.add_argument(
        "--model",
        type=str,
        default="runs/detect/train/weights/best.pt",
        help="Path to fine-tuned YOLO model.",
    )

    parser.add_argument(
        "--baseline",
        type=str,
        default=None,
        help="Optional baseline model path, e.g. yolo11m.pt.",
    )

    parser.add_argument(
        "--data",
        type=str,
        required=True,
        help="Path to YOLO dataset YAML.",
    )

    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["train", "val", "test"],
        help="Dataset split to evaluate.",
    )

    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.7)
    parser.add_argument("--device", type=str, default="0")
    parser.add_argument("--project", type=str, default="runs/evaluate")

    return parser.parse_args()


def main():
    args = parse_args()

    if args.baseline:
        compare_models(
            baseline_path=args.baseline,
            finetuned_path=args.model,
            data_yaml=args.data,
            split=args.split,
            imgsz=args.imgsz,
            batch=args.batch,
            conf=args.conf,
            iou=args.iou,
            device=args.device,
            project=args.project,
        )
    else:
        evaluate_model(
            model_path=args.model,
            data_yaml=args.data,
            split=args.split,
            imgsz=args.imgsz,
            batch=args.batch,
            conf=args.conf,
            iou=args.iou,
            device=args.device,
            project=args.project,
            name=None,
        )


if __name__ == "__main__":
    main()