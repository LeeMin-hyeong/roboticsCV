# Basic Image Processing Ablation Study (요약본)

## Exercise 1. Quantization: how many bits per pixel?

### 1) 실험 목적
- 비트 수 변화가 화질과 계조 표현에 미치는 영향을 비교한다.

### 3) 변경 인자
- numOfBit

### 4) 결과 표
| 파라미터 | 값 | 결과 이미지 |
|---|---|---|
| numOfBit | 1 | outputs/ex1_quantization/ex1_numOfBit_1.png |
| numOfBit | 2 | outputs/ex1_quantization/ex1_numOfBit_2.png |
| numOfBit | 4 | outputs/ex1_quantization/ex1_numOfBit_4.png |
| numOfBit | 6 | outputs/ex1_quantization/ex1_numOfBit_6.png |
| numOfBit | 8 | outputs/ex1_quantization/ex1_numOfBit_8.png |

### 6) 해석 요약
- numOfBit가 1~2인 구간에서는 계조 단계가 부족했다. 얼굴 경계 주변에 밴딩이 뚜렷했다. 4bit 이상부터는 구조 정보가 복원됐다. 8bit에서는 미세 톤이 유지되어 원본과 가장 유사했다.

## Exercise 2. Brightness Adjustment and Contrast Adjustment

### 1) 실험 목적
- 밝기 스케일과 감마 보정이 명암 분포에 주는 영향을 분리해 확인한다.

### 3) 변경 인자
- scale 또는 gamma

### 4) 결과 표
| 파라미터 | 값 | 결과 이미지 |
|---|---|---|
| scale | 0.80 | outputs/ex2_brightness_contrast/ex2_scale_0.80.png |
| scale | 1.00 | outputs/ex2_brightness_contrast/ex2_scale_1.00.png |
| scale | 1.20 | outputs/ex2_brightness_contrast/ex2_scale_1.20.png |
| gamma | 0.70 | outputs/ex2_brightness_contrast/ex2_gamma_0.70.png |
| gamma | 1.00 | outputs/ex2_brightness_contrast/ex2_gamma_1.00.png |
| gamma | 1.40 | outputs/ex2_brightness_contrast/ex2_gamma_1.40.png |

### 6) 해석 요약
- scale은 영상 전체 밝기를 선형으로 이동시켰다. 평균 밝기 변화가 직접적으로 관찰됐다. gamma는 중간톤을 비선형으로 재배치했다. 두 방법은 목적과 결과가 분명히 달랐다.

## Exercise 3. Image averaging for noise reduction

### 1) 실험 목적
- 평균 횟수 증가에 따른 노이즈 감소 효과를 비교한다.

### 3) 변경 인자
- N

### 4) 결과 표
| 파라미터 | 값 | 결과 이미지 |
|---|---|---|
| N | 1 | outputs/ex3_image_averaging/ex3_N_1.png |
| N | 2 | outputs/ex3_image_averaging/ex3_N_2.png |
| N | 8 | outputs/ex3_image_averaging/ex3_N_8.png |
| N | 32 | outputs/ex3_image_averaging/ex3_N_32.png |
| N | 64 | outputs/ex3_image_averaging/ex3_N_64.png |

### 6) 해석 요약
- N이 증가할수록 랜덤 노이즈가 평균 과정에서 상쇄됐다. 배경과 평탄 영역이 더 안정적으로 보였다. MAE도 N=1에서 N=64로 갈수록 감소했다. 큰 N에서는 개선 폭이 점차 완만해졌다.

## Exercise 4. Image subtraction

### 1) 실험 목적
- 차영상 검출에서 임계값과 평활화의 영향을 비교한다.

### 3) 변경 인자
- threshold, equalization

### 4) 결과 표
| 파라미터 | 값 | 결과 이미지 |
|---|---|---|
| threshold/equalization | thr=0.05, eq=False | outputs/ex4_image_subtraction/ex4_thr_0.05_eq_0.png |
| threshold/equalization | thr=0.10, eq=False | outputs/ex4_image_subtraction/ex4_thr_0.10_eq_0.png |
| threshold/equalization | thr=0.15, eq=False | outputs/ex4_image_subtraction/ex4_thr_0.15_eq_0.png |
| threshold/equalization | thr=0.10, eq=True | outputs/ex4_image_subtraction/ex4_thr_0.10_eq_1.png |
| threshold/equalization | thr=0.20, eq=True | outputs/ex4_image_subtraction/ex4_thr_0.20_eq_1.png |

### 6) 해석 요약
- threshold를 낮추면 작은 차이까지 검출된다. 전경 비율이 증가하지만 배경 잡검출도 늘어난다. threshold를 높이면 주요 변화만 남는다. equalization을 적용하면 미세 차이는 잘 보이지만 과검출 위험이 커진다.

## Exercise 5. Video background subtraction

### 1) 실험 목적
- 배경모델 파라미터가 움직임 검출에 미치는 영향을 확인한다.

### 3) 변경 인자
- alpha 또는 theta

### 4) 결과 표
| 파라미터 | 값 | 결과 이미지 |
|---|---|---|
| alpha | alpha=0.90, theta=25 | outputs/ex5_video_background_subtraction/ex5_alpha_0.90_theta_25.png |
| alpha | alpha=0.95, theta=25 | outputs/ex5_video_background_subtraction/ex5_alpha_0.95_theta_25.png |
| alpha | alpha=0.98, theta=25 | outputs/ex5_video_background_subtraction/ex5_alpha_0.98_theta_25.png |
| theta | alpha=0.95, theta=15 | outputs/ex5_video_background_subtraction/ex5_alpha_0.95_theta_15.png |
| theta | alpha=0.95, theta=35 | outputs/ex5_video_background_subtraction/ex5_alpha_0.95_theta_35.png |

### 6) 해석 요약
- alpha가 클수록 배경 갱신이 느리다. 이동 객체의 잔상이 오래 남는다. theta를 낮추면 사람 윤곽 검출은 민감해진다. 노이즈성 픽셀도 함께 증가한다.

## Exercise 6. Image subtraction for photomask comparison

### 1) 실험 목적
- 포토마스크 차영상에서 결함 후보 추출 조건을 비교한다.

### 3) 변경 인자
- threshold, equalization

### 4) 결과 표
| 파라미터 | 값 | 결과 이미지 |
|---|---|---|
| threshold/equalization | thr=0.05, eq=False | outputs/ex6_mask_subtraction/ex6_thr_0.05_eq_0.png |
| threshold/equalization | thr=0.10, eq=False | outputs/ex6_mask_subtraction/ex6_thr_0.10_eq_0.png |
| threshold/equalization | thr=0.20, eq=False | outputs/ex6_mask_subtraction/ex6_thr_0.20_eq_0.png |
| threshold/equalization | thr=0.10, eq=True | outputs/ex6_mask_subtraction/ex6_thr_0.10_eq_1.png |
| threshold/equalization | thr=0.20, eq=True | outputs/ex6_mask_subtraction/ex6_thr_0.20_eq_1.png |

### 6) 해석 요약
- 임계값이 낮을 때는 미세 패턴 차이까지 대량 검출됐다. 후보 영역이 넓어졌다. 임계값을 높이면 확실한 결함 후보만 남는다. equalization은 약한 차이를 잘 살리지만 잡검출 민감도도 높인다.

## Exercise 7. Where is the defect?

### 1) 실험 목적
- 정합 및 임계값 파라미터에 따른 결함 위치 검출 변화를 확인한다.

### 3) 변경 인자
- xShift 또는 threshold

### 4) 결과 표
| 파라미터 | 값 | 결과 이미지 |
|---|---|---|
| xShift | 8 | outputs/ex7_defect_localization/ex7_xShift_8_thr_0.15.png |
| xShift | 10 | outputs/ex7_defect_localization/ex7_xShift_10_thr_0.15.png |
| xShift | 12 | outputs/ex7_defect_localization/ex7_xShift_12_thr_0.15.png |
| threshold | 0.10 | outputs/ex7_defect_localization/ex7_xShift_10_thr_0.10.png |
| threshold | 0.20 | outputs/ex7_defect_localization/ex7_xShift_10_thr_0.20.png |

### 6) 해석 요약
- xShift가 실제 이동량과 맞지 않으면 오정합이 남는다. 결함이 아닌 영역까지 반응한다. 정합이 맞으면 결함 주변이 상대적으로 선명해진다. threshold 조정으로 민감도와 정밀도 균형을 맞출 수 있다.

## Exercise 8. Example histogram

### 1) 실험 목적
- 입력 영상별 히스토그램 분포 특성을 비교한다.

### 3) 변경 인자
- input_image

### 4) 결과 표
| 파라미터 | 값 | 결과 이미지 |
|---|---|---|
| input_image | bay.jpg | outputs/ex8_histogram_example/ex8_input_bay.png |
| input_image | brain.jpg | outputs/ex8_histogram_example/ex8_input_brain.png |
| input_image | moon.jpg | outputs/ex8_histogram_example/ex8_input_moon.png |

### 6) 해석 요약
- 입력 이미지마다 히스토그램 집중 구간이 다르다. 분산 폭도 영상마다 다르게 나타났다. 어떤 영상은 좁은 밝기 대역에 픽셀이 몰렸다. 이런 차이는 후속 대비 향상 효과 크기에 직접 영향을 준다.

## Exercise 9. Histogram equalization example

### 1) 실험 목적
- 영상별 히스토그램 평활화 효과를 비교한다.

### 3) 변경 인자
- input_image

### 4) 결과 표
| 파라미터 | 값 | 결과 이미지 |
|---|---|---|
| input_image | bay.jpg | outputs/ex9_hist_eq/ex9_input_bay.png |
| input_image | brain.jpg | outputs/ex9_hist_eq/ex9_input_brain.png |
| input_image | moon.jpg | outputs/ex9_hist_eq/ex9_input_moon.png |

### 6) 해석 요약
- 평활화 후 픽셀 분포가 더 넓은 밝기 구간으로 퍼졌다. 전반적인 대비가 상승했다. 저대비 영상에서는 경계와 질감 향상이 더 크게 보였다. 원본 분포가 넓은 영상은 개선 폭이 상대적으로 작았다.

## Exercise 10. Contrast-limited histogram equalization

### 1) 실험 목적
- clipRatio에 따른 대비 증폭/억제 균형을 비교한다.

### 3) 변경 인자
- clipRatio

### 4) 결과 표
| 파라미터 | 값 | 결과 이미지 |
|---|---|---|
| clipRatio | 1.00 | outputs/ex10_clhe/ex10_clipRatio_1.00.png |
| clipRatio | 0.70 | outputs/ex10_clhe/ex10_clipRatio_0.70.png |
| clipRatio | 0.40 | outputs/ex10_clhe/ex10_clipRatio_0.40.png |
| clipRatio | 0.10 | outputs/ex10_clhe/ex10_clipRatio_0.10.png |
| clipRatio | 0.01 | outputs/ex10_clhe/ex10_clipRatio_0.01.png |

### 6) 해석 요약
- clipRatio가 큰 설정은 대비 향상이 강하다. 일부 톤이 과하게 강조될 수 있다. clipRatio를 낮추면 피크가 제한된다. 결과는 더 안정적이고 자연스럽게 유지된다.

## Exercise 11. Adaptive histogram equalization

### 1) 실험 목적
- CLAHE 타일/클립 파라미터가 지역 대비에 미치는 영향을 비교한다.

### 3) 변경 인자
- tileGridSize 또는 clipLimit

### 4) 결과 표
| 파라미터 | 값 | 결과 이미지 |
|---|---|---|
| tileGridSize | (4,4) | outputs/ex11_clahe/ex11_tile_4_clip_2.0.png |
| tileGridSize | (8,8) | outputs/ex11_clahe/ex11_tile_8_clip_2.0.png |
| tileGridSize | (16,16) | outputs/ex11_clahe/ex11_tile_16_clip_2.0.png |
| clipLimit | 1.0 | outputs/ex11_clahe/ex11_tile_8_clip_1.0.png |
| clipLimit | 4.0 | outputs/ex11_clahe/ex11_tile_8_clip_4.0.png |

### 6) 해석 요약
- 작은 타일은 미세 질감과 경계를 강하게 드러낸다. 큰 타일은 전역적으로 부드러운 결과를 만든다. clipLimit을 높이면 지역 대비가 더 강해진다. 잡음성 패턴도 함께 강조될 수 있다.

