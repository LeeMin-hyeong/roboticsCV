# CV Exercise 3 Ablation Study

## Exercise 1. Graylevel thresholding

### 1) 실험 목적
- 단일 임계값 변화에 따른 이진 분할 결과 변화를 확인한다.

### 3) 변경 인자
- threshold

### 4) 결과 표
| 파라미터 | 값 | 결과 이미지 |
|---|---|---|
| threshold | 90 | outputs/ex1_graylevel_thresholding/ex1_level_90.png |
| threshold | 105 | outputs/ex1_graylevel_thresholding/ex1_level_105.png |
| threshold | 120 | outputs/ex1_graylevel_thresholding/ex1_level_120.png |
| threshold | 140 | outputs/ex1_graylevel_thresholding/ex1_level_140.png |

### 6) 해석 요약
- threshold가 증가할수록 전경으로 분류되는 영역이 넓어졌다. 낮은 값에서는 세부 객체가 일부 끊겼다. 높은 값에서는 배경 노이즈가 함께 포함됐다. 과제 목적에 맞는 중간 임계값 선택이 중요했다.

## Exercise 2. Global Thresholding

### 1) 실험 목적
- 전역 임계값 설정이 분할 결과에 주는 영향을 비교한다.

### 3) 변경 인자
- global_threshold

### 4) 결과 표
| 파라미터 | 값 | 결과 이미지 |
|---|---|---|
| global_threshold | 120 | outputs/ex2_global_thresholding/ex2_threshold_120.png |
| global_threshold | 140 | outputs/ex2_global_thresholding/ex2_threshold_140.png |
| global_threshold | 160 | outputs/ex2_global_thresholding/ex2_threshold_160.png |
| global_threshold | 180 | outputs/ex2_global_thresholding/ex2_threshold_180.png |

### 6) 해석 요약
- 전역 임계값을 높일수록 밝은 영역만 남고 어두운 디테일은 빠르게 사라졌다. Otsu 근처 값에서 전경과 배경 균형이 상대적으로 안정적이었다. 과도한 임계값은 객체 내부 정보 손실을 유발했다.

## Exercise 3. Locally Adaptive Thresholding

### 1) 실험 목적
- 지역 분산 조건에 따른 적응 임계 분할 효과를 비교한다.

### 3) 변경 인자
- var_thresh

### 4) 결과 표
| 파라미터 | 값 | 결과 이미지 |
|---|---|---|
| var_thresh | 0.0001 | outputs/ex3_local_adaptive_thresholding/ex3_varThresh_0.0001.png |
| var_thresh | 0.0005 | outputs/ex3_local_adaptive_thresholding/ex3_varThresh_0.0005.png |
| var_thresh | 0.001 | outputs/ex3_local_adaptive_thresholding/ex3_varThresh_0.0010.png |
| var_thresh | 0.005 | outputs/ex3_local_adaptive_thresholding/ex3_varThresh_0.0050.png |

### 6) 해석 요약
- var_thresh가 낮을 때는 더 많은 타일에서 지역 임계가 적용됐다. 지역 텍스트와 경계는 잘 보였지만 잡음도 증가했다. var_thresh를 높이면 결과가 전역 임계 형태에 가까워졌다. 지역성과 안정성 사이의 절충이 필요했다.

## Exercise 4. MAP Skin Detector

### 1) 실험 목적
- 색상 히스토그램 bin 크기가 피부 분류 성능에 미치는 영향을 비교한다.

### 3) 변경 인자
- color_bin_step

### 4) 결과 표
| 파라미터 | 값 | 결과 이미지 |
|---|---|---|
| color_bin_step | 8 (test1) | outputs/ex4_map_skin_detector/ex4_step_8_test_1.png |
| color_bin_step | 8 (test2) | outputs/ex4_map_skin_detector/ex4_step_8_test_2.png |
| color_bin_step | 16 (test1) | outputs/ex4_map_skin_detector/ex4_step_16_test_1.png |
| color_bin_step | 16 (test2) | outputs/ex4_map_skin_detector/ex4_step_16_test_2.png |
| color_bin_step | 32 (test1) | outputs/ex4_map_skin_detector/ex4_step_32_test_1.png |
| color_bin_step | 32 (test2) | outputs/ex4_map_skin_detector/ex4_step_32_test_2.png |

### 6) 해석 요약
- bin step이 작으면 색상 구분은 세밀해진다. 학습 샘플 수가 제한되면 오분류도 늘 수 있다. bin step이 크면 분류 경계가 단순해져 얼굴 외 영역이 섞일 수 있다. 테스트 영상별로 적정 bin 크기가 다르게 나타났다.

## Exercise 5. Region Labeling

### 1) 실험 목적
- 형태학적 closing 크기에 따른 연결 성분 라벨링 결과를 비교한다.

### 3) 변경 인자
- closing_kernel

### 4) 결과 표
| 파라미터 | 값 | 결과 이미지 |
|---|---|---|
| closing_kernel | 1 | outputs/ex5_region_labeling/ex5_closing_1.png |
| closing_kernel | 3 | outputs/ex5_region_labeling/ex5_closing_3.png |
| closing_kernel | 5 | outputs/ex5_region_labeling/ex5_closing_5.png |
| closing_kernel | 7 | outputs/ex5_region_labeling/ex5_closing_7.png |

### 6) 해석 요약
- closing 커널이 커질수록 끊긴 전경이 연결됐다. 작은 객체 간 간격이 메워지면서 라벨 수가 감소했다. 커널이 너무 크면 서로 다른 객체가 합쳐질 수 있다. 객체 분리 목적이면 작은 커널이 더 유리했다.

## Exercise 6. Hole Filling

### 1) 실험 목적
- 이진화 임계값에 따른 hole filling 결과 변화를 비교한다.

### 3) 변경 인자
- threshold

### 4) 결과 표
| 파라미터 | 값 | 결과 이미지 |
|---|---|---|
| threshold | 90 | outputs/ex6_hole_filling/ex6_level_90.png |
| threshold | 105 | outputs/ex6_hole_filling/ex6_level_105.png |
| threshold | 120 | outputs/ex6_hole_filling/ex6_level_120.png |
| threshold | 140 | outputs/ex6_hole_filling/ex6_level_140.png |

### 6) 해석 요약
- threshold가 낮으면 내부 hole 자체가 적게 형성됐다. threshold가 올라가면 공극 후보가 늘어 채움 면적이 증가했다. 너무 높은 값은 외곽 잡영까지 포함해 과도한 채움으로 이어졌다. 채움 단계 전 이진화 품질이 최종 결과를 좌우했다.

