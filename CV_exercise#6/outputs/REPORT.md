# CV Exercise 6 Ablation Study (Revised)

## Exercise 1. SIFT Descriptor

### 1) 실험 목적
- 노트북 원 코드의 contrastThreshold 파라미터 영향을 비교한다.

### 2) 변경 인자
- contrastThreshold

### 3) 결과 표
| 파라미터 | 값 | 결과 이미지 |
|---|---|---|
| contrastThreshold | 0.02 | outputs/ex1_sift_descriptor_revised/ex1_contrastThreshold_0.02.png |
| contrastThreshold | 0.04 | outputs/ex1_sift_descriptor_revised/ex1_contrastThreshold_0.04.png |
| contrastThreshold | 0.08 | outputs/ex1_sift_descriptor_revised/ex1_contrastThreshold_0.08.png |

### 4) 해석 요약
- contrastThreshold를 낮추면 저대비 특징점까지 검출된다. 높이면 강한 코너 중심으로 줄어든다. 검출 수와 안정성 사이 균형이 필요했다.

## Exercise 2. RANSAC

### 1) 실험 목적
- 노트북 기본값 0.7 주변에서 ratio test 변화를 비교한다.

### 2) 변경 인자
- ratio_test

### 3) 결과 표
| 파라미터 | 값 | 결과 이미지 |
|---|---|---|
| ratio_test | 0.60 | outputs/ex2_ransac_revised/ex2_ratio_0.60.png |
| ratio_test | 0.70 | outputs/ex2_ransac_revised/ex2_ratio_0.70.png |
| ratio_test | 0.80 | outputs/ex2_ransac_revised/ex2_ratio_0.80.png |

### 4) 해석 요약
- ratio가 낮으면 보수적으로 매칭이 선택됐다. ratio를 높이면 매칭 수는 늘지만 outlier 위험이 커졌다. inlier 수는 중간 범위에서 안정적이었다.

## Exercise 3. Kanade Lucas

### 1) 실험 목적
- 제목과 일치하도록 LK 추적 윈도우 크기 효과를 비교한다.

### 2) 변경 인자
- winSize

### 3) 결과 표
| 파라미터 | 값 | 결과 이미지 |
|---|---|---|
| winSize | 9 | outputs/ex3_kanade_lucas_revised/ex3_winSize_9.png |
| winSize | 15 | outputs/ex3_kanade_lucas_revised/ex3_winSize_15.png |
| winSize | 21 | outputs/ex3_kanade_lucas_revised/ex3_winSize_21.png |

### 4) 해석 요약
- 작은 winSize는 민감하지만 흔들림이 잦았다. 큰 winSize는 궤적이 안정적이었다. 세부 움직임 추적은 중간 크기에서 균형이 좋았다.

