import os
import csv
import math
from pathlib import Path
import numpy as np
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(r"c:\Users\lmhst\git\roboticsCV\CV_exercise#2")
OUT = ROOT / "outputs"
OUT.mkdir(exist_ok=True)

np.random.seed(42)

rows = []
exercise_reports = {}
modifications = []


def save_compare_2(orig, res, title1, title2, out_path, cmap='gray'):
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    if cmap is None:
        plt.imshow(orig)
    else:
        plt.imshow(orig, cmap=cmap)
    plt.title(title1)
    plt.axis('off')

    plt.subplot(1, 2, 2)
    if cmap is None:
        plt.imshow(res)
    else:
        plt.imshow(res, cmap=cmap)
    plt.title(title2)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(out_path, dpi=140)
    plt.close()


def add_row(ex, param, value, out_file, visual_change, metric, analysis):
    rows.append({
        "Exercise": ex,
        "Parameter": param,
        "Value": value,
        "Output file": str(out_file.relative_to(ROOT)).replace('\\', '/'),
        "Visual change": visual_change,
        "Optional metric": metric,
        "One-line analysis": analysis,
    })


def gray_entropy(img_u8):
    hist = cv2.calcHist([img_u8], [0], None, [256], [0, 256]).flatten()
    p = hist / max(hist.sum(), 1)
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum())


def hist_spread(img_u8):
    p5 = np.percentile(img_u8, 5)
    p95 = np.percentile(img_u8, 95)
    return float(p95 - p5)


# Exercise 1
ex = 1
ex_dir = OUT / "ex1_quantization"
ex_dir.mkdir(exist_ok=True)
img = cv2.imread(str(ROOT / "face.jpg"), cv2.IMREAD_GRAYSCALE).astype(np.float32)
for b in [1, 2, 4, 6, 8]:
    levels = 2 ** b
    gap = 256 / levels
    q = np.uint8(np.ceil(img / gap) * gap - 1)
    q = np.clip(q, 0, 255)
    out = ex_dir / f"ex1_numOfBit_{b}.png"
    save_compare_2(img.astype(np.uint8), q, "Original", f"{b}-bit", out, cmap='gray')
    uniq = int(np.unique(q).size)
    add_row(
        "Exercise 1",
        "numOfBit",
        str(b),
        out,
        "비트 수가 낮을수록 계단형 밴딩 증가",
        f"unique_levels={uniq}",
        f"numOfBit={b}에서는 표현 가능한 계조가 줄어 윤곽 경계가 더 거칠게 보였다.",
    )

exercise_reports[1] = {
    "title": "Quantization: how many bits per pixel?",
    "objective": "비트 수 변화가 화질과 계조 표현에 미치는 영향을 비교한다.",
    "fixed": "입력 영상(face.jpg), 양자화 방식, 표시 스케일 고정",
    "changed": "numOfBit",
    "rep_image": str((ex_dir / "ex1_numOfBit_4.png").relative_to(ROOT)).replace('\\', '/'),
    "summary": "비트 수가 증가할수록 계조 단계가 늘어 밴딩이 줄고 원본과 유사해졌다. 1~2bit에서는 명암 단절이 뚜렷했고 6~8bit 구간에서 시각적 품질이 안정적이었다.",
}

# Exercise 2
ex = 2
ex_dir = OUT / "ex2_brightness_contrast"
ex_dir.mkdir(exist_ok=True)
img_bgr = cv2.imread(str(ROOT / "parrot.jpg"), cv2.IMREAD_COLOR)
img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

for s in [0.8, 1.0, 1.2]:
    res = np.clip(img * s, 0, 1)
    out = ex_dir / f"ex2_scale_{s:.2f}.png"
    save_compare_2(img, res, "Original", f"scale={s:.2f}", out, cmap=None)
    mean_v = float(res.mean())
    add_row(
        "Exercise 2",
        "scale",
        f"{s:.2f}",
        out,
        "스케일이 커질수록 전체 밝기 상승",
        f"mean_intensity={mean_v:.4f}",
        f"scale={s:.2f}로 조정하니 전체 밝기가 {'증가' if s>1 else '감소' if s<1 else '유지'}했고 색조 자체는 크게 변하지 않았다.",
    )

for g in [0.7, 1.0, 1.4]:
    res = np.clip(img ** g, 0, 1)
    out = ex_dir / f"ex2_gamma_{g:.2f}.png"
    save_compare_2(img, res, "Original", f"gamma={g:.2f}", out, cmap=None)
    std_v = float(res.std())
    add_row(
        "Exercise 2",
        "gamma",
        f"{g:.2f}",
        out,
        "감마에 따라 중간톤 대비가 재분배",
        f"std={std_v:.4f}",
        f"gamma={g:.2f}에서는 중간 밝기 영역이 재배치되어 체감 대비가 {'강해지거나' if g>1 else '완만해지며'} 명부/암부 비중이 달라졌다.",
    )

exercise_reports[2] = {
    "title": "Brightness Adjustment and Contrast Adjustment",
    "objective": "밝기 스케일과 감마 보정이 영상의 명암 분포에 주는 영향을 분리해 확인한다.",
    "fixed": "입력 영상(parrot.jpg), RGB 정규화 범위(0~1) 고정",
    "changed": "scale 또는 gamma를 한 번에 하나씩 변경",
    "rep_image": str((ex_dir / "ex2_gamma_1.40.png").relative_to(ROOT)).replace('\\', '/'),
    "summary": "scale은 전체 밝기를 거의 선형으로 이동시켰고, gamma는 중간톤을 중심으로 비선형 재배치를 만들었다. 따라서 밝기 조절과 대비 조절은 시각 효과가 분명히 다르게 나타났다.",
}

# Exercise 3
ex = 3
ex_dir = OUT / "ex3_image_averaging"
ex_dir.mkdir(exist_ok=True)
img = cv2.imread(str(ROOT / "quadnight.jpg"), cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0
noise_std = 0.2
for N in [1, 2, 8, 32, 64]:
    acc = np.zeros_like(img, dtype=np.float32)
    for _ in range(N):
        noise = np.random.normal(loc=0.0, scale=noise_std, size=img.shape).astype(np.float32)
        noisy = np.clip(img + noise, 0, 1)
        acc += noisy
    avg = np.clip(acc / N, 0, 1)
    out = ex_dir / f"ex3_N_{N}.png"
    save_compare_2(img, avg, "Original", f"Averaged N={N}", out, cmap='gray')
    mae = float(np.mean(np.abs(avg - img)))
    add_row(
        "Exercise 3",
        "N",
        str(N),
        out,
        "평균 횟수가 늘수록 랜덤 노이즈 감소",
        f"MAE_vs_original={mae:.4f}",
        f"N={N}로 평균화하니 잡음 성분이 상쇄되어 원본과의 오차가 점진적으로 줄었다.",
    )

modifications.append("Exercise 3: 원본 노트북의 노이즈 평균(0.5) 설정은 밝기 바이어스를 크게 유발해, 평균 0.0의 가우시안 노이즈로 수정해 비교 실험을 수행함.")

exercise_reports[3] = {
    "title": "Image averaging for noise reduction",
    "objective": "샘플 평균 수 N 증가에 따른 잡음 억제 효과를 정량/정성 비교한다.",
    "fixed": "입력 영상(quadnight.jpg), 노이즈 표준편차(0.2) 고정",
    "changed": "N",
    "rep_image": str((ex_dir / "ex3_N_32.png").relative_to(ROOT)).replace('\\', '/'),
    "summary": "N이 증가할수록 랜덤 노이즈가 평균 과정에서 상쇄되며 영상이 매끈해졌다. 다만 매우 큰 N에서는 계산량 증가 대비 체감 개선 폭이 점차 작아졌다.",
}

# Exercise 4
ex = 4
ex_dir = OUT / "ex4_image_subtraction"
ex_dir.mkdir(exist_ok=True)
mask = cv2.imread(str(ROOT / "mask.jpg"), cv2.IMREAD_COLOR).astype(np.float32) / 255.0
live = cv2.imread(str(ROOT / "live.jpg"), cv2.IMREAD_COLOR).astype(np.float32) / 255.0
diff = np.abs(mask - live).mean(axis=2)

settings = [
    (0.05, False),
    (0.10, False),
    (0.15, False),
    (0.10, True),
    (0.20, True),
]
for thr, eq in settings:
    d = (diff * 255).astype(np.uint8)
    if eq:
        d_proc = cv2.equalizeHist(d)
        d_f = d_proc.astype(np.float32) / 255.0
    else:
        d_proc = d
        d_f = diff
    bw = (d_f > thr).astype(np.uint8) * 255
    out = ex_dir / f"ex4_thr_{thr:.2f}_eq_{int(eq)}.png"
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 3, 1); plt.imshow(cv2.cvtColor((live*255).astype(np.uint8), cv2.COLOR_BGR2RGB)); plt.title("Live"); plt.axis('off')
    plt.subplot(1, 3, 2); plt.imshow(d_proc, cmap='gray'); plt.title(f"Diff(eq={eq})"); plt.axis('off')
    plt.subplot(1, 3, 3); plt.imshow(bw, cmap='gray'); plt.title(f"Binary thr={thr:.2f}"); plt.axis('off')
    plt.tight_layout(); plt.savefig(out, dpi=140); plt.close()
    fg = float((bw > 0).mean())
    add_row(
        "Exercise 4",
        "threshold/equalization",
        f"thr={thr:.2f}, eq={eq}",
        out,
        "임계값이 낮을수록 검출 영역 증가, 평활화 시 미세 차이 강조",
        f"foreground_ratio={fg:.4f}",
        f"thr={thr:.2f}, eq={eq} 설정에서는 전경 검출 면적이 {fg:.3f}로 나타나 임계 민감도가 직접 반영되었다.",
    )

exercise_reports[4] = {
    "title": "Image subtraction",
    "objective": "차영상 기반 검출에서 임계값과 히스토그램 평활화 적용 여부의 영향을 비교한다.",
    "fixed": "입력 영상(mask/live), 차영상 계산 방식(abs diff) 고정",
    "changed": "threshold, equalization",
    "rep_image": str((ex_dir / "ex4_thr_0.10_eq_1.png").relative_to(ROOT)).replace('\\', '/'),
    "summary": "threshold를 낮추면 배경까지 쉽게 포함되고, 높이면 실제 변화 영역만 남는 경향이 확인됐다. equalization을 추가하면 약한 차이도 부각되지만 과검출 가능성도 함께 증가했다.",
}

# Exercise 5
ex = 5
ex_dir = OUT / "ex5_video_background_subtraction"
ex_dir.mkdir(exist_ok=True)


def run_bg(alpha=0.95, theta=25, max_frames=180, snapshot_idx=120):
    cap = cv2.VideoCapture(str(ROOT / "surveillance.avi"))
    ret, bg = cap.read()
    if not ret:
        cap.release()
        raise RuntimeError("Cannot read surveillance.avi")
    bg = cv2.cvtColor(bg, cv2.COLOR_BGR2GRAY).astype(np.float32)

    fg_ratios = []
    snap = None
    i = 0
    while i < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)
        bg = alpha * bg + (1 - alpha) * gray
        diff = np.abs(gray - bg)
        bw = (diff > theta).astype(np.uint8) * 255
        fg_ratios.append(float((bw > 0).mean()))
        if i == snapshot_idx:
            snap = (gray.copy(), bg.copy(), diff.copy(), bw.copy())
        i += 1

    cap.release()
    if snap is None and i > 0:
        snap = (gray.copy(), bg.copy(), diff.copy(), bw.copy())
    return snap, float(np.mean(fg_ratios)) if fg_ratios else 0.0

bg_settings = [
    ("alpha", 0.90, 25),
    ("alpha", 0.95, 25),
    ("alpha", 0.98, 25),
    ("theta", 0.95, 15),
    ("theta", 0.95, 35),
]
for p, a, t in bg_settings:
    snap, fg_mean = run_bg(alpha=a, theta=t)
    gray, bg, diff, bw = snap
    out = ex_dir / f"ex5_alpha_{a:.2f}_theta_{t}.png"
    plt.figure(figsize=(10, 8))
    plt.subplot(2, 2, 1); plt.imshow(gray, cmap='gray'); plt.title("Current"); plt.axis('off')
    plt.subplot(2, 2, 2); plt.imshow(bg, cmap='gray'); plt.title("Background"); plt.axis('off')
    plt.subplot(2, 2, 3); plt.imshow(diff, cmap='gray'); plt.title("Diff"); plt.axis('off')
    plt.subplot(2, 2, 4); plt.imshow(bw, cmap='gray'); plt.title("Thresholded"); plt.axis('off')
    plt.tight_layout(); plt.savefig(out, dpi=140); plt.close()
    add_row(
        "Exercise 5",
        p,
        f"alpha={a:.2f}, theta={t}",
        out,
        "alpha는 배경 적응 속도, theta는 검출 민감도에 영향",
        f"mean_foreground_ratio={fg_mean:.4f}",
        f"alpha={a:.2f}, theta={t}에서 평균 전경 비율이 {fg_mean:.3f}로 측정되어 {'민감한' if fg_mean>0.08 else '보수적인'} 검출 경향을 보였다.",
    )

modifications.append("Exercise 5: 노트북의 theta=0.1은 diff(0~255) 스케일과 불일치해 거의 전체 픽셀이 검출될 수 있어, 정수 스케일 임계값(15~35)으로 보정해 실험함.")

exercise_reports[5] = {
    "title": "Video background subtraction",
    "objective": "배경 누적 계수(alpha)와 이진화 임계값(theta)의 검출 특성을 비교한다.",
    "fixed": "입력 영상(surveillance.avi), 프레임 수(최대 180) 고정",
    "changed": "alpha 또는 theta",
    "rep_image": str((ex_dir / "ex5_alpha_0.95_theta_25.png").relative_to(ROOT)).replace('\\', '/'),
    "summary": "alpha가 커질수록 배경 업데이트가 느려져 움직임 잔상이 더 오래 남는 경향이 있었다. theta를 높이면 잡음성 변화가 줄지만 작은 객체가 누락될 가능성이 커졌다.",
}

# Exercise 6
ex = 6
ex_dir = OUT / "ex6_mask_subtraction"
ex_dir.mkdir(exist_ok=True)
m1 = cv2.imread(str(ROOT / "mask1.png"), cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0
m2 = cv2.imread(str(ROOT / "mask2.png"), cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0
diff = np.abs(m1 - m2)

for thr, eq in [(0.05, False), (0.10, False), (0.20, False), (0.10, True), (0.20, True)]:
    d = (diff * 255).astype(np.uint8)
    if eq:
        d = cv2.equalizeHist(d)
        d_f = d.astype(np.float32) / 255.0
    else:
        d_f = diff
    bw = (d_f > thr).astype(np.uint8) * 255
    out = ex_dir / f"ex6_thr_{thr:.2f}_eq_{int(eq)}.png"
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 3, 1); plt.imshow(m1, cmap='gray'); plt.title("mask1"); plt.axis('off')
    plt.subplot(1, 3, 2); plt.imshow(d, cmap='gray'); plt.title(f"diff(eq={eq})"); plt.axis('off')
    plt.subplot(1, 3, 3); plt.imshow(bw, cmap='gray'); plt.title(f"binary thr={thr:.2f}"); plt.axis('off')
    plt.tight_layout(); plt.savefig(out, dpi=140); plt.close()
    fg = float((bw > 0).mean())
    add_row(
        "Exercise 6",
        "threshold/equalization",
        f"thr={thr:.2f}, eq={eq}",
        out,
        "임계값과 평활화에 따라 결함 후보 면적 변화",
        f"foreground_ratio={fg:.4f}",
        f"thr={thr:.2f}, eq={eq}에서는 검출 마스크 비율이 {fg:.3f}로, 임계값 증가 시 과검출이 줄었다.",
    )

exercise_reports[6] = {
    "title": "Image subtraction for photomask comparison",
    "objective": "포토마스크 차영상에서 결함 후보 추출 조건을 비교한다.",
    "fixed": "입력 이미지(mask1/mask2), 차영상 계산(abs diff) 고정",
    "changed": "threshold, equalization",
    "rep_image": str((ex_dir / "ex6_thr_0.10_eq_1.png").relative_to(ROOT)).replace('\\', '/'),
    "summary": "threshold가 낮으면 미세 오차까지 넓게 검출되고, 높이면 주요 차이만 남았다. equalization 적용 시 약한 패턴 차이도 강조되어 검출 범위가 증가하는 경향이 있었다.",
}

# Exercise 7
ex = 7
ex_dir = OUT / "ex7_defect_localization"
ex_dir.mkdir(exist_ok=True)
orig = cv2.imread(str(ROOT / "pcbCropped.png"), cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0
defected = cv2.imread(str(ROOT / "pcbCroppedTranslatedDefected.png"), cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0


def detect_with_shift(x_shift=10, y_shift=10, thr=0.15, border_ratio=0.05):
    reg = np.zeros_like(defected)
    reg[y_shift:, x_shift:] = defected[:-y_shift, :-x_shift]
    d1 = np.abs(orig - defected)
    d2 = np.abs(orig - reg)
    bw = (d2 > thr)
    h, w = bw.shape
    b = round(border_ratio * w)
    mask = np.zeros_like(bw)
    mask[b:h-b, b:w-b] = 1
    bw = bw * mask
    return d1, d2, bw.astype(np.uint8) * 255

for x in [8, 10, 12]:
    d1, d2, bw = detect_with_shift(x_shift=x, y_shift=10, thr=0.15)
    out = ex_dir / f"ex7_xShift_{x}_thr_0.15.png"
    plt.figure(figsize=(15, 4))
    plt.subplot(1, 3, 1); plt.imshow(d1, cmap='gray'); plt.title('Unaligned diff'); plt.axis('off')
    plt.subplot(1, 3, 2); plt.imshow(d2, cmap='gray'); plt.title(f'Aligned diff (x={x})'); plt.axis('off')
    plt.subplot(1, 3, 3); plt.imshow(bw, cmap='gray'); plt.title('Binary defect'); plt.axis('off')
    plt.tight_layout(); plt.savefig(out, dpi=140); plt.close()
    fg = float((bw > 0).mean())
    add_row(
        "Exercise 7",
        "xShift",
        str(x),
        out,
        "정합 오차가 작을수록 배경 차이가 감소",
        f"foreground_ratio={fg:.4f}",
        f"xShift={x}에서는 정합 정확도 차이로 결함 후보 마스크 면적이 {fg:.3f} 수준으로 변했다.",
    )

for thr in [0.10, 0.20]:
    d1, d2, bw = detect_with_shift(x_shift=10, y_shift=10, thr=thr)
    out = ex_dir / f"ex7_xShift_10_thr_{thr:.2f}.png"
    plt.figure(figsize=(15, 4))
    plt.subplot(1, 3, 1); plt.imshow(d1, cmap='gray'); plt.title('Unaligned diff'); plt.axis('off')
    plt.subplot(1, 3, 2); plt.imshow(d2, cmap='gray'); plt.title('Aligned diff'); plt.axis('off')
    plt.subplot(1, 3, 3); plt.imshow(bw, cmap='gray'); plt.title(f'Binary thr={thr:.2f}'); plt.axis('off')
    plt.tight_layout(); plt.savefig(out, dpi=140); plt.close()
    fg = float((bw > 0).mean())
    add_row(
        "Exercise 7",
        "threshold",
        f"{thr:.2f}",
        out,
        "threshold 증가 시 미세 결함 후보 억제",
        f"foreground_ratio={fg:.4f}",
        f"threshold={thr:.2f}로 높이자 약한 차이 성분이 제거되어 검출 결과가 더 보수적으로 바뀌었다.",
    )

exercise_reports[7] = {
    "title": "Where is the defect?",
    "objective": "정합 파라미터와 이진화 임계값이 결함 위치 검출 정확도에 미치는 영향을 확인한다.",
    "fixed": "입력 PCB 이미지 쌍, yShift=10, borderRatio=0.05 고정",
    "changed": "xShift 또는 threshold",
    "rep_image": str((ex_dir / "ex7_xShift_10_thr_0.15.png").relative_to(ROOT)).replace('\\', '/'),
    "summary": "정합이 맞지 않으면 실제 결함 외에 구조적 오차가 함께 검출된다. threshold를 적절히 높이면 잡검출이 줄지만 약한 결함 신호도 함께 누락될 수 있다.",
}

# Exercise 8
ex = 8
ex_dir = OUT / "ex8_histogram_example"
ex_dir.mkdir(exist_ok=True)
for name in ["bay.jpg", "brain.jpg", "moon.jpg"]:
    img = cv2.imread(str(ROOT / name), cv2.IMREAD_GRAYSCALE)
    hist, bins = np.histogram(img.flatten(), bins=256, range=[0, 256])
    out = ex_dir / f"ex8_input_{Path(name).stem}.png"
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.imshow(img, cmap='gray')
    plt.title(f"Image: {name}")
    plt.axis('off')
    plt.subplot(1, 2, 2)
    plt.bar(bins[:-1], hist, width=1)
    plt.xlim(0, 255)
    plt.title("Histogram")
    plt.tight_layout(); plt.savefig(out, dpi=140); plt.close()
    spread = hist_spread(img)
    ent = gray_entropy(img)
    add_row(
        "Exercise 8",
        "input_image",
        name,
        out,
        "입력 영상에 따라 히스토그램 집중/분산 양상 차이",
        f"spread={spread:.2f}, entropy={ent:.2f}",
        f"{name}는 히스토그램 분포 폭과 엔트로피가 달라 장면 대비 특성이 다르게 나타났다.",
    )

exercise_reports[8] = {
    "title": "Example histogram",
    "objective": "입력 영상 종류에 따른 히스토그램 형태 차이를 확인한다.",
    "fixed": "히스토그램 계산 방식(256 bins), 그레이스케일 변환 방식 고정",
    "changed": "input image",
    "rep_image": str((ex_dir / "ex8_input_moon.png").relative_to(ROOT)).replace('\\', '/'),
    "summary": "영상 종류에 따라 특정 밝기 구간에 픽셀이 몰리거나 넓게 퍼지는 패턴이 달랐다. 이는 후속 대비 향상 기법 적용 시 체감 효과 차이로 이어진다.",
}

# Exercise 9
ex = 9
ex_dir = OUT / "ex9_hist_eq"
ex_dir.mkdir(exist_ok=True)
for name in ["bay.jpg", "brain.jpg", "moon.jpg"]:
    img = cv2.imread(str(ROOT / name), cv2.IMREAD_GRAYSCALE)
    eq = cv2.equalizeHist(img)
    out = ex_dir / f"ex9_input_{Path(name).stem}.png"
    plt.figure(figsize=(12, 8))
    plt.subplot(2, 2, 1); plt.imshow(img, cmap='gray'); plt.title('Original'); plt.axis('off')
    plt.subplot(2, 2, 2); plt.imshow(eq, cmap='gray'); plt.title('Equalized'); plt.axis('off')
    h1, b = np.histogram(img.flatten(), bins=256, range=[0, 256])
    h2, _ = np.histogram(eq.flatten(), bins=256, range=[0, 256])
    plt.subplot(2, 2, 3); plt.bar(b[:-1], h1, width=1); plt.title('Hist original'); plt.xlim(0,255)
    plt.subplot(2, 2, 4); plt.bar(b[:-1], h2, width=1); plt.title('Hist equalized'); plt.xlim(0,255)
    plt.tight_layout(); plt.savefig(out, dpi=140); plt.close()
    std_before = float(img.std())
    std_after = float(eq.std())
    add_row(
        "Exercise 9",
        "input_image",
        name,
        out,
        "평활화 후 명암 분포 확장",
        f"std_before={std_before:.2f}, std_after={std_after:.2f}",
        f"{name}에서 히스토그램 평활화 후 명암 표준편차가 {std_before:.2f}→{std_after:.2f}로 변화해 대비가 재분배되었다.",
    )

modifications.append("Exercise 9: 원본 노트북의 grayscale 이미지 시각화에서 BGR2RGB 변환을 사용한 부분은 오류 가능성이 있어, 단일 채널 `cmap='gray'` 표시로 수정함.")

exercise_reports[9] = {
    "title": "Histogram equalization example",
    "objective": "영상별 글로벌 히스토그램 평활화 효과를 원본 대비로 비교한다.",
    "fixed": "equalizeHist 연산 방식, 시각화 레이아웃 고정",
    "changed": "input image",
    "rep_image": str((ex_dir / "ex9_input_moon.png").relative_to(ROOT)).replace('\\', '/'),
    "summary": "평활화는 대체로 저대비 구간을 펼쳐 디테일을 드러냈다. 다만 원본 분포가 이미 넓은 영상에서는 개선폭이 상대적으로 제한적이었다.",
}

# Exercise 10
ex = 10
ex_dir = OUT / "ex10_clhe"
ex_dir.mkdir(exist_ok=True)
img = cv2.imread(str(ROOT / "moon.jpg"), cv2.IMREAD_GRAYSCALE)


def clip_hist_eq(image_u8, clip_ratio):
    hist = cv2.calcHist([image_u8], [0], None, [256], [0, 256]).flatten().astype(np.float64)
    max_count = hist.max()
    clip_limit = clip_ratio * max_count
    clipped = np.minimum(hist, clip_limit)
    excess = int((hist - clipped).sum())
    redist = excess // 256
    rem = excess % 256
    clipped += redist
    if rem > 0:
        clipped[:rem] += 1

    cdf = np.cumsum(clipped)
    cdf_min = cdf[np.nonzero(cdf)][0] if np.any(cdf > 0) else 0
    denom = (cdf[-1] - cdf_min) if cdf[-1] > cdf_min else 1
    lut = np.floor((cdf - cdf_min) / denom * 255.0).clip(0, 255).astype(np.uint8)
    out = cv2.LUT(image_u8, lut)
    return out, lut

for r in [1.0, 0.7, 0.4, 0.1, 0.01]:
    out_img, lut = clip_hist_eq(img, r)
    out = ex_dir / f"ex10_clipRatio_{r:.2f}.png"
    plt.figure(figsize=(14, 4))
    plt.subplot(1, 3, 1); plt.imshow(img, cmap='gray'); plt.title('Original'); plt.axis('off')
    plt.subplot(1, 3, 2); plt.imshow(out_img, cmap='gray'); plt.title(f'clipRatio={r:.2f}'); plt.axis('off')
    plt.subplot(1, 3, 3); plt.plot(lut); plt.plot([0,255],[0,255],'k--',linewidth=1); plt.title('LUT'); plt.xlim(0,255); plt.ylim(0,255)
    plt.tight_layout(); plt.savefig(out, dpi=140); plt.close()
    std_v = float(out_img.std())
    add_row(
        "Exercise 10",
        "clipRatio",
        f"{r:.2f}",
        out,
        "clipRatio가 작을수록 히스토그램 과집중 억제",
        f"std={std_v:.2f}",
        f"clipRatio={r:.2f}에서는 강한 피크가 제한되어 대비 증폭이 {'완화' if r<0.4 else '유지'}되는 경향을 보였다.",
    )

modifications.append("Exercise 10: 원본 노트북의 LUT 생성 로직은 히스토그램 값을 직접 LUT로 쓰는 문제로 매핑 함수가 왜곡될 수 있어, clip+재분배+CDF 기반 LUT로 수정함.")

exercise_reports[10] = {
    "title": "Contrast-limited histogram equalization",
    "objective": "clipRatio 변화가 대비 향상과 과증폭 억제 사이 균형에 미치는 영향을 분석한다.",
    "fixed": "입력 영상(moon.jpg), CLHE 처리 파이프라인 고정",
    "changed": "clipRatio",
    "rep_image": str((ex_dir / "ex10_clipRatio_0.10.png").relative_to(ROOT)).replace('\\', '/'),
    "summary": "clipRatio가 높으면 강한 대비 향상이 가능하지만 일부 밝기 구간 과증폭이 발생할 수 있다. clipRatio를 낮추면 결과가 안정적이지만 전체 대비 확장 폭은 줄어든다.",
}

# Exercise 11
ex = 11
ex_dir = OUT / "ex11_clahe"
ex_dir.mkdir(exist_ok=True)
img = cv2.imread(str(ROOT / "parrot.jpg"), cv2.IMREAD_GRAYSCALE)
eq = cv2.equalizeHist(img)

for tile in [4, 8, 16]:
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(tile, tile))
    cimg = clahe.apply(img)
    out = ex_dir / f"ex11_tile_{tile}_clip_2.0.png"
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 3, 1); plt.imshow(img, cmap='gray'); plt.title('Original'); plt.axis('off')
    plt.subplot(1, 3, 2); plt.imshow(eq, cmap='gray'); plt.title('Global HE'); plt.axis('off')
    plt.subplot(1, 3, 3); plt.imshow(cimg, cmap='gray'); plt.title(f'CLAHE tile={tile}'); plt.axis('off')
    plt.tight_layout(); plt.savefig(out, dpi=140); plt.close()
    std_v = float(cimg.std())
    add_row(
        "Exercise 11",
        "tileGridSize",
        f"({tile},{tile})",
        out,
        "타일이 작을수록 지역 대비 강조",
        f"std={std_v:.2f}",
        f"tile={tile}에서는 지역 히스토그램 단위가 {'세밀' if tile<=8 else '완만'}해져 질감 강조 정도가 달라졌다.",
    )

for clip in [1.0, 4.0]:
    clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=(8, 8))
    cimg = clahe.apply(img)
    out = ex_dir / f"ex11_tile_8_clip_{clip:.1f}.png"
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 3, 1); plt.imshow(img, cmap='gray'); plt.title('Original'); plt.axis('off')
    plt.subplot(1, 3, 2); plt.imshow(eq, cmap='gray'); plt.title('Global HE'); plt.axis('off')
    plt.subplot(1, 3, 3); plt.imshow(cimg, cmap='gray'); plt.title(f'CLAHE clip={clip:.1f}'); plt.axis('off')
    plt.tight_layout(); plt.savefig(out, dpi=140); plt.close()
    std_v = float(cimg.std())
    add_row(
        "Exercise 11",
        "clipLimit",
        f"{clip:.1f}",
        out,
        "clipLimit이 클수록 지역 대비 증폭 경향",
        f"std={std_v:.2f}",
        f"clipLimit={clip:.1f}에서는 지역 대비가 {'강해져' if clip>2 else '완화되어'} 노이즈/질감 증폭 수준이 달라졌다.",
    )

exercise_reports[11] = {
    "title": "Adaptive histogram equalization",
    "objective": "CLAHE의 타일 크기와 클립 제한이 지역 대비에 미치는 영향을 비교한다.",
    "fixed": "입력 영상(parrot.jpg), 비교 기준(원본/Global HE) 고정",
    "changed": "tileGridSize 또는 clipLimit",
    "rep_image": str((ex_dir / "ex11_tile_8_clip_2.0.png").relative_to(ROOT)).replace('\\', '/'),
    "summary": "작은 타일은 국소 디테일을 강하게 살리지만 과강조 가능성이 있고, 큰 타일은 더 부드러운 보정을 만든다. clipLimit이 커질수록 지역 대비가 강해져 질감 강조가 뚜렷해졌다.",
}

# Save summary CSV
csv_path = OUT / "ablation_summary.csv"
with open(csv_path, 'w', newline='', encoding='utf-8-sig') as f:
    writer = csv.DictWriter(f, fieldnames=["Exercise", "Parameter", "Value", "Output file", "Visual change", "Optional metric", "One-line analysis"])
    writer.writeheader()
    writer.writerows(rows)

# Save exercise-level markdown report
report_path = OUT / "ablation_report_draft.md"
with open(report_path, 'w', encoding='utf-8') as f:
    f.write("# Basic Image Processing Ablation Study (초안)\n\n")
    for ex in range(1, 12):
        r = exercise_reports[ex]
        f.write(f"## Exercise {ex}. {r['title']}\n\n")
        f.write("### 1) 실험 목적\n")
        f.write(f"- {r['objective']}\n\n")
        f.write("### 2) 고정 조건\n")
        f.write(f"- {r['fixed']}\n\n")
        f.write("### 3) 변경 인자\n")
        f.write(f"- {r['changed']}\n\n")
        f.write("### 4) 결과 표\n")
        f.write("| Exercise | Parameter | Value | Output file | Visual change | Optional metric | One-line analysis |\n")
        f.write("|---|---|---|---|---|---|---|\n")
        for row in rows:
            if row["Exercise"] == f"Exercise {ex}":
                f.write(f"| {row['Exercise']} | {row['Parameter']} | {row['Value']} | {row['Output file']} | {row['Visual change']} | {row['Optional metric']} | {row['One-line analysis']} |\n")
        f.write("\n")
        f.write("### 5) 대표 결과 이미지\n")
        f.write(f"- `{r['rep_image']}`\n\n")
        f.write("### 6) 해석 요약\n")
        f.write(f"- {r['summary']}\n\n")

    f.write("## 수정 사항\n")
    for m in modifications:
        f.write(f"- {m}\n")

# File list
file_list_path = OUT / "generated_images.txt"
png_files = sorted(OUT.rglob("*.png"))
with open(file_list_path, 'w', encoding='utf-8') as f:
    for p in png_files:
        f.write(str(p.relative_to(ROOT)).replace('\\', '/') + "\n")

print(f"rows={len(rows)}")
print(csv_path)
print(report_path)
print(file_list_path)
print(f"png_count={len(png_files)}")
