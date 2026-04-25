from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from run_two_stage_coco_strategy_cv1 import (
    FIG,
    ROOT,
    build_cm,
    draw_cm_pair,
    load_eval_data,
    run_two_stage,
    sweep_best,
)

OUT = ROOT / "outputs"

CSV_PREV = OUT / "two_stage_coco_strategy_results_cv1.csv"
CSV_NEW = OUT / "two_stage_coco_strategy_round2_results_cv1.csv"
REPORT_MD = OUT / "two_stage_coco_strategy_round2_report_cv1.md"
MANIFEST = OUT / "two_stage_coco_strategy_round2_manifest_cv1.json"

FIG_METRICS = FIG / "two_stage_coco_strategy_round2_metrics_cv1.png"
FIG_ERR = FIG / "two_stage_coco_strategy_round2_error_cv1.png"
FIG_CM_COUNT = FIG / "two_stage_coco_strategy_round2_confusion_counts_cv1.png"
FIG_CM_NORM = FIG / "two_stage_coco_strategy_round2_confusion_normalized_cv1.png"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)

    names, imgs, labels = load_eval_data()
    coco_best = sweep_best("yolov8n.pt", names, imgs, labels)

    new_candidates = [
        {
            "tag": "two_stage_d",
            "stage1": {"epochs": 10, "imgsz": 640, "batch": 16, "lr0": 0.0018, "freeze": 22, "mosaic": 0.05, "mixup": 0.0, "close_mosaic": 1},
            "stage2": {"epochs": 14, "imgsz": 640, "batch": 16, "lr0": 0.00006, "freeze": 0, "mosaic": 0.0, "mixup": 0.0, "close_mosaic": 0},
        },
        {
            "tag": "two_stage_e",
            "stage1": {"epochs": 12, "imgsz": 640, "batch": 16, "lr0": 0.0015, "freeze": 22, "mosaic": 0.1, "mixup": 0.0, "close_mosaic": 1},
            "stage2": {"epochs": 16, "imgsz": 640, "batch": 16, "lr0": 0.00008, "freeze": 0, "mosaic": 0.05, "mixup": 0.0, "close_mosaic": 1},
        },
        {
            "tag": "two_stage_f",
            "stage1": {"epochs": 10, "imgsz": 640, "batch": 16, "lr0": 0.0012, "freeze": 15, "mosaic": 0.1, "mixup": 0.0, "close_mosaic": 1},
            "stage2": {"epochs": 14, "imgsz": 640, "batch": 16, "lr0": 0.00007, "freeze": 0, "mosaic": 0.0, "mixup": 0.0, "close_mosaic": 0},
        },
    ]

    rows_new = []
    for c in new_candidates:
        print("[train-two-stage-round2]", c["tag"])
        pt = run_two_stage(c["tag"], c["stage1"], c["stage2"])
        best = sweep_best(str(pt), names, imgs, labels)
        row = {
            "tag": c["tag"],
            "best_pt": str(pt),
            "f1": best["f1"],
            "precision": best["precision"],
            "recall": best["recall"],
            "f1_macro": best["f1_macro"],
            "tp": best["tp"],
            "fp": best["fp"],
            "fn": best["fn"],
            "nms_iou": best["nms_iou"],
            "max_det": best["max_det"],
            "thr_person": best["thr_person"],
            "thr_car": best["thr_car"],
            "thr_dog": best["thr_dog"],
            "delta_f1_vs_coco": best["f1"] - coco_best["f1"],
        }
        rows_new.append(row)
        print("[round2-result]", row["tag"], "f1", row["f1"], "dF1", row["delta_f1_vs_coco"])

    df_new = pd.DataFrame(rows_new)
    if CSV_PREV.exists():
        df_prev = pd.read_csv(CSV_PREV)
        df_all = pd.concat([df_prev, df_new], ignore_index=True)
    else:
        df_all = df_new.copy()

    df_all = df_all.sort_values(["f1", "fp"], ascending=[False, True]).reset_index(drop=True)
    df_all.to_csv(CSV_NEW, index=False, encoding="utf-8-sig")

    best_row = df_all.iloc[0].to_dict()

    metrics = ["precision", "recall", "f1", "f1_macro"]
    x = np.arange(len(metrics))
    w = 0.2
    fig, ax = plt.subplots(figsize=(10, 5), dpi=160)
    ax.bar(x - 1.5 * w, [coco_best[m] for m in metrics], width=w, label="COCO", color="#4e79a7")
    ax.bar(x - 0.5 * w, [best_row[m] for m in metrics], width=w, label="TwoStageBest", color="#f28e2b")
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.set_ylim(0, 1.0)
    ax.set_title("COCO vs Two-Stage Best (after round2)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG_METRICS)
    plt.close(fig)

    fig2, ax2 = plt.subplots(figsize=(10, 5), dpi=160)
    counts = ["tp", "fp", "fn"]
    x2 = np.arange(len(counts))
    ax2.bar(x2 - 0.2, [coco_best[k] for k in counts], width=0.4, label="COCO", color="#4e79a7")
    ax2.bar(x2 + 0.2, [best_row[k] for k in counts], width=0.4, label="TwoStageBest", color="#f28e2b")
    ax2.set_xticks(x2)
    ax2.set_xticklabels(counts)
    ax2.set_title("COCO vs Two-Stage Best (after round2)")
    ax2.legend()
    fig2.tight_layout()
    fig2.savefig(FIG_ERR)
    plt.close(fig2)

    coco_cm, coco_norm = build_cm("yolov8n.pt", names, imgs, labels, coco_best)
    best_cm, best_norm = build_cm(str(best_row["best_pt"]), names, imgs, labels, best_row)
    draw_cm_pair(coco_cm, best_cm, "COCO (counts)", "TwoStageBest (counts)", FIG_CM_COUNT, normalized=False)
    draw_cm_pair(coco_norm, best_norm, "COCO (normalized)", "TwoStageBest (normalized)", FIG_CM_NORM, normalized=True)

    lines = [
        "# Two-Stage COCO Strategy Round2 Report (CV1)",
        "",
        "- round2 adds conservative settings: stronger stage1 freeze, lower stage2 lr.",
        "",
        "## Baseline (COCO best)",
        f"- F1={coco_best['f1']:.4f}, P={coco_best['precision']:.4f}, R={coco_best['recall']:.4f}, TP={coco_best['tp']}, FP={coco_best['fp']}, FN={coco_best['fn']}",
        "",
        "## Best After Round2",
        f"- tag={best_row['tag']}, F1={best_row['f1']:.4f}, P={best_row['precision']:.4f}, R={best_row['recall']:.4f}, TP={int(best_row['tp'])}, FP={int(best_row['fp'])}, FN={int(best_row['fn'])}",
        f"- params: nms_iou={best_row['nms_iou']}, max_det={int(best_row['max_det'])}, thr=({best_row['thr_person']:.2f},{best_row['thr_car']:.2f},{best_row['thr_dog']:.2f})",
        f"- delta(F1 vs COCO)={best_row['delta_f1_vs_coco']:+.4f}",
        "",
        "![metrics](C:/Users/lmhst/git/roboticsCV/CV_project#1/outputs/figures/two_stage_coco_strategy_round2_metrics_cv1.png)",
        "",
        "![error](C:/Users/lmhst/git/roboticsCV/CV_project#1/outputs/figures/two_stage_coco_strategy_round2_error_cv1.png)",
        "",
        "![cm_count](C:/Users/lmhst/git/roboticsCV/CV_project#1/outputs/figures/two_stage_coco_strategy_round2_confusion_counts_cv1.png)",
        "",
        "![cm_norm](C:/Users/lmhst/git/roboticsCV/CV_project#1/outputs/figures/two_stage_coco_strategy_round2_confusion_normalized_cv1.png)",
    ]
    REPORT_MD.write_text("\n".join(lines), encoding="utf-8")

    manifest = {
        "csv_prev": str(CSV_PREV),
        "csv_new": str(CSV_NEW),
        "report_md": str(REPORT_MD),
        "fig_metrics": str(FIG_METRICS),
        "fig_error": str(FIG_ERR),
        "fig_cm_count": str(FIG_CM_COUNT),
        "fig_cm_norm": str(FIG_CM_NORM),
        "coco_best": coco_best,
        "best_after_round2": best_row,
    }
    MANIFEST.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print("[saved]", CSV_NEW)
    print("[saved]", REPORT_MD)
    print("[saved]", MANIFEST)
    print("[summary]", json.dumps({"coco_f1": coco_best["f1"], "best_tag": best_row["tag"], "best_f1": best_row["f1"], "delta_f1": best_row["delta_f1_vs_coco"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
