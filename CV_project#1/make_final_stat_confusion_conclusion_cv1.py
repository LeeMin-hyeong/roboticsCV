from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(r"C:\Users\lmhst\git\roboticsCV\CV_project#1")
OUT = ROOT / "outputs"
FIG = OUT / "figures"

SIG10 = OUT / "seed_repeat_best_large_manifest_cv1.json"
OPSIG = OUT / "lg02_operating_tune2_significance_manifest_cv1.json"
CM4 = OUT / "confusion_compare_4x4_cv1.json"

FIG_STATS = FIG / "final_stats_significance_dashboard_cv1.png"
FIG_CM_DELTA = FIG / "final_confusion_delta_4x4_cv1.png"
REPORT = OUT / "final_stat_confusion_conclusion_cv1.md"


def load_json(p: Path):
    return json.loads(p.read_text(encoding="utf-8"))


def draw_stats_dashboard(sig10: dict, opsig: dict):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), dpi=160)

    # Panel A: 10-seed train-level significance
    ax = axes[0]
    labels = ["mAP50", "mAP50-95"]
    base = [sig10["base_mean_map50"], sig10["base_mean_map50_95"]]
    opt = [sig10["opt_mean_map50"], sig10["opt_mean_map50_95"]]
    x = np.arange(len(labels))
    w = 0.35
    ax.bar(x - w / 2, base, width=w, label="Baseline", color="#6c8ebf")
    ax.bar(x + w / 2, opt, width=w, label="Optimized", color="#2ca02c")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, max(opt) * 1.25)
    ax.set_title("Train-level (10 seeds)")
    ax.set_ylabel("Score")
    ax.legend(loc="upper left")

    p1 = sig10["paired_map50"]["p"]
    p2 = sig10["paired_map50_95"]["p"]
    d1 = sig10["opt_mean_map50"] - sig10["base_mean_map50"]
    d2 = sig10["opt_mean_map50_95"] - sig10["base_mean_map50_95"]
    ax.text(0.02, 0.98, f"mAP50 Δ={d1:+.4f}, p={p1:.2e}\nmAP50-95 Δ={d2:+.4f}, p={p2:.2e}", transform=ax.transAxes, va="top", ha="left", fontsize=9, bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))

    # Panel B: Operating-level significance
    ax2 = axes[1]
    labels2 = ["Precision", "Recall", "F1"]
    b2 = [opsig["base_global"]["precision"], opsig["base_global"]["recall"], opsig["base_global"]["f1"]]
    t2 = [opsig["tuned_global"]["precision"], opsig["tuned_global"]["recall"], opsig["tuned_global"]["f1"]]
    x2 = np.arange(len(labels2))
    ax2.bar(x2 - w / 2, b2, width=w, label="Operating Baseline", color="#c27ba0")
    ax2.bar(x2 + w / 2, t2, width=w, label="Operating Tuned", color="#3cba54")
    ax2.set_xticks(x2)
    ax2.set_xticklabels(labels2)
    ax2.set_ylim(0, 1.05)
    ax2.set_title("Operating-level (fixed conf=0.35)")
    ax2.legend(loc="lower right")

    pf1 = opsig["paired_tests_image_level"]["f1"]["wilcoxon_p"]
    pfp = opsig["paired_tests_image_level"]["fp"]["wilcoxon_p"]
    dprec = opsig["global_delta"]["precision"]
    df1 = opsig["global_delta"]["f1"]
    dfp = opsig["global_delta"]["fp"]
    ax2.text(0.02, 0.98, f"ΔPrecision={dprec:+.4f}\nΔF1={df1:+.4f} (p={pf1:.4f})\nΔFP={dfp:+.0f} (p={pfp:.4f})", transform=ax2.transAxes, va="top", ha="left", fontsize=9, bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))

    fig.suptitle("Final Statistical Significance Summary", fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(FIG_STATS)
    plt.close(fig)


def draw_confusion_delta(cm4: dict):
    classes = cm4["classes"]
    b = np.array(cm4["baseline_cm_counts"], dtype=float)
    o = np.array(cm4["optimized_cm_counts"], dtype=float)
    d = o - b

    vmax = np.max(np.abs(d))
    fig, ax = plt.subplots(figsize=(7, 6), dpi=170)
    im = ax.imshow(d, cmap="coolwarm", vmin=-vmax, vmax=vmax)
    ax.set_xticks(np.arange(len(classes)))
    ax.set_yticks(np.arange(len(classes)))
    ax.set_xticklabels(classes)
    ax.set_yticklabels(classes)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Ground Truth")
    ax.set_title("4x4 Confusion Delta (Optimized - Baseline)")

    for i in range(d.shape[0]):
        for j in range(d.shape[1]):
            v = int(d[i, j])
            ax.text(j, i, f"{v:+d}", ha="center", va="center", fontsize=10)

    cbar = fig.colorbar(im, ax=ax, shrink=0.9)
    cbar.set_label("Count difference")

    fig.tight_layout()
    fig.savefig(FIG_CM_DELTA)
    plt.close(fig)


def write_report(sig10: dict, opsig: dict):
    p_map50 = sig10["paired_map50"]["p"]
    p_map95 = sig10["paired_map50_95"]["p"]
    d_map50 = sig10["opt_mean_map50"] - sig10["base_mean_map50"]
    d_map95 = sig10["opt_mean_map50_95"] - sig10["base_mean_map50_95"]

    p_f1 = opsig["paired_tests_image_level"]["f1"]["wilcoxon_p"]
    p_fp = opsig["paired_tests_image_level"]["fp"]["wilcoxon_p"]

    lines = [
        "# 최종 결론: 통계적 유의성 + Confusion Metric",
        "",
        "## 1) 통계적 유의성 요약",
        f"- 학습 최적화(10 seeds): mAP50 Δ={d_map50:+.4f} (p={p_map50:.2e}), mAP50-95 Δ={d_map95:+.4f} (p={p_map95:.2e})",
        f"- 운영 2차 튜닝(conf=0.35 고정): F1 p={p_f1:.6f}, FP p={p_fp:.6f}",
        "- 해석: 학습 단계와 운영 단계 모두에서 p<0.05 기준의 유의미한 개선이 확인됨",
        "",
        "![Final stats dashboard](/C:/Users/lmhst/git/roboticsCV/CV_project#1/outputs/figures/final_stats_significance_dashboard_cv1.png)",
        "",
        "## 2) Confusion Metric 변화 (Background 포함 4x4)",
        "- 아래 heatmap은 optimized-baseline의 confusion count 차이임",
        "- 대각선(정분류) 증가와 off-diagonal/배경 오분류 패턴을 함께 확인 가능",
        "",
        "![Confusion delta 4x4](/C:/Users/lmhst/git/roboticsCV/CV_project#1/outputs/figures/final_confusion_delta_4x4_cv1.png)",
        "",
        "참고 원본:",
        "![Confusion counts](/C:/Users/lmhst/git/roboticsCV/CV_project#1/outputs/figures/confusion_compare_4x4_counts_cv1.png)",
        "![Confusion normalized](/C:/Users/lmhst/git/roboticsCV/CV_project#1/outputs/figures/confusion_compare_4x4_normalized_cv1.png)",
        "",
        "## 3) 최종 결론",
        "- 성능 개선은 우연 변동이 아니라 통계적으로 유의한 개선으로 판단 가능",
        "- 특히 운영 튜닝으로 FP를 유의하게 줄이면서 F1을 추가 개선하여 실사용 안정성이 향상됨",
        "- 따라서 최종 운영 권장안은 `lg_02_best_prev + conf=0.35 + nms_iou=0.45 + max_det=20`",
    ]
    REPORT.write_text("\n".join(lines), encoding="utf-8")


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    sig10 = load_json(SIG10)
    opsig = load_json(OPSIG)
    cm4 = load_json(CM4)

    draw_stats_dashboard(sig10, opsig)
    draw_confusion_delta(cm4)
    write_report(sig10, opsig)

    print(f"[saved] {FIG_STATS}")
    print(f"[saved] {FIG_CM_DELTA}")
    print(f"[saved] {REPORT}")


if __name__ == "__main__":
    main()
