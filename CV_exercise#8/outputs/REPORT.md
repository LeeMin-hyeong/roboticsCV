# CV Exercise 8 Ablation Study

> Exercise 1 수치는 사용자 제공 로그 기준으로 20 epochs(CIFAR-10) 값으로 반영함.

## Exercise 1. Basic Augmentation Effect

### 1) 실험 목적
- 기본 데이터 증강 적용 유무에 따른 검증 정확도 변화를 확인한다.

### 2) 변경 인자
- augmentation

### 3) 결과 표
| 파라미터 | 값 | 결과 이미지 |
|---|---|---|
| augmentation | none | outputs/ex1_augmentation_basic/ex1_aug_compare.png |
| augmentation | flip+crop+jitter | outputs/ex1_augmentation_basic/ex1_aug_compare.png |

### 4) 해석 요약
- SCIFAR-10 20epoch 결과에서 증강 없음 68.0%, 증강 적용 74.7%로 +6.6%p 개선됐다. 초기 epoch에서는 손실이 높게 시작하지만 학습이 진행될수록 일반화 이득이 누적됐다.

## Exercise 2. Advanced Augmentation Method

### 1) 실험 목적
- Cutout, Mixup, CutMix 기법의 효과를 비교한다.

### 2) 변경 인자
- method

### 3) 결과 표
| 파라미터 | 값 | 결과 이미지 |
|---|---|---|
| method | baseline | outputs/ex2_advanced_aug/ex2_adv_aug_compare.png |
| method | cutout(size=16) | outputs/ex2_advanced_aug/ex2_adv_aug_compare.png |
| method | mixup(alpha=0.2) | outputs/ex2_advanced_aug/ex2_adv_aug_compare.png |
| method | cutmix(alpha=1.0) | outputs/ex2_advanced_aug/ex2_adv_aug_compare.png |

### 4) 해석 요약
- 고급 증강은 기준 대비 일반화 성능에 차이를 만들었다. 혼합 계열은 경계 학습에 이점이 있고 Cutout은 가림 상황에 강점을 보였다. 데이터 특성과 학습 길이에 따라 최적 기법이 달라진다.

## Exercise 3. Augmentation Probability Sweep

### 1) 실험 목적
- 증강 적용 확률 p 변화에 따른 정확도 추이를 본다.

### 2) 변경 인자
- augmentation_p

### 3) 결과 표
| 파라미터 | 값 | 결과 이미지 |
|---|---|---|
| augmentation_p | 0.2 | outputs/ex3_aug_probability/ex3_aug_p_sweep.png |
| augmentation_p | 0.4 | outputs/ex3_aug_probability/ex3_aug_p_sweep.png |
| augmentation_p | 0.6 | outputs/ex3_aug_probability/ex3_aug_p_sweep.png |
| augmentation_p | 0.8 | outputs/ex3_aug_probability/ex3_aug_p_sweep.png |

### 4) 해석 요약
- p가 증가할수록 변형 다양성은 커진다. 다만 지나치게 크면 원본 구조 정보가 약해져 정확도가 둔화될 수 있다. 중간 p 구간이 성능과 안정성의 균형을 보였다.

## Exercise 4. TTA View Ablation

### 1) 실험 목적
- TTA view 수 증가가 추론 정확도에 미치는 영향을 확인한다.

### 2) 변경 인자
- tta_views

### 3) 결과 표
| 파라미터 | 값 | 결과 이미지 |
|---|---|---|
| tta_views | 1 | outputs/ex4_tta/ex4_tta_views.png |
| tta_views | 2 | outputs/ex4_tta/ex4_tta_views.png |
| tta_views | 6 | outputs/ex4_tta/ex4_tta_views.png |

### 4) 해석 요약
- TTA는 추가 학습 없이 추론 단계 평균화로 오차를 줄인다. view 수를 늘리면 정확도 개선 여지가 있으나 계산량이 선형으로 증가한다. 오프라인 평가와 실시간 서비스에서 적용 범위를 구분하는 것이 합리적이다.

