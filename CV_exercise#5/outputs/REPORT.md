# CV Exercise 5 Ablation Study (Revised)

## Exercise 1. Template Matching

### 1) 실험 목적
- 상관 기반 템플릿 매칭에서 임계 백분위를 비교한다.

### 2) 변경 인자
- corr_percentile

### 3) 결과 표
| 파라미터 | 값 | 결과 이미지 |
|---|---|---|
| corr_percentile | 99.5 | outputs/ex1_template_matching_revised/ex1_percentile_99.5.png |
| corr_percentile | 99.7 | outputs/ex1_template_matching_revised/ex1_percentile_99.7.png |
| corr_percentile | 99.9 | outputs/ex1_template_matching_revised/ex1_percentile_99.9.png |

### 4) 해석 요약
- 백분위를 높이면 후보 수가 감소한다. 낮은 백분위는 민감도가 높지만 오검출 가능성이 커진다. 목표 물체가 명확할수록 높은 백분위가 유리했다.

## Exercise 2. Matched Filtering

### 1) 실험 목적
- 숫자 템플릿 선택에 따른 matched filtering 응답을 비교한다.

### 2) 변경 인자
- template_digit

### 3) 결과 표
| 파라미터 | 값 | 결과 이미지 |
|---|---|---|
| template_digit | 8 | outputs/ex2_matched_filtering_revised/ex2_template_8.png |
| template_digit | 3 | outputs/ex2_matched_filtering_revised/ex2_template_3.png |
| template_digit | 0 | outputs/ex2_matched_filtering_revised/ex2_template_0.png |

### 4) 해석 요약
- 템플릿 숫자에 따라 응답 분포가 달라졌다. 목표 패턴과 유사한 템플릿에서 피크가 집중됐다. 불일치 템플릿은 응답이 분산됐다.

## Exercise 3. Gender Recognition

### 1) 실험 목적
- PCA 차원 수가 성별 분류 성능에 미치는 영향을 비교한다.

### 2) 변경 인자
- num_components

### 3) 결과 표
| 파라미터 | 값 | 결과 이미지 |
|---|---|---|
| num_components | 10 | outputs/ex3_gender_recognition_revised/ex3_pca_10.png |
| num_components | 20 | outputs/ex3_gender_recognition_revised/ex3_pca_20.png |
| num_components | 40 | outputs/ex3_gender_recognition_revised/ex3_pca_40.png |

### 4) 해석 요약
- 차원이 너무 낮으면 성별 특징 분리가 약해진다. 적정 차원에서는 평균 얼굴 기반 분류 성능이 개선된다. 과도한 차원은 개선 폭이 제한적이었다.

## Exercise 4. Illumination Variations

### 1) 실험 목적
- 조명 왜곡 계수 theta 변화가 얼굴 대비에 미치는 영향을 비교한다.

### 2) 변경 인자
- theta

### 3) 결과 표
| 파라미터 | 값 | 결과 이미지 |
|---|---|---|
| theta | 0.1 | outputs/ex4_illumination_variation_revised/ex4_theta_0.1.png |
| theta | 0.2 | outputs/ex4_illumination_variation_revised/ex4_theta_0.2.png |
| theta | 0.4 | outputs/ex4_illumination_variation_revised/ex4_theta_0.4.png |

### 4) 해석 요약
- theta가 커질수록 비균일 조명 왜곡이 강해졌다. 균일하지 않은 명암 기울기가 얼굴 특징을 가렸다. 평활화 보정은 일부 회복에 도움을 줬다.

## Exercise 5. Person Identification

### 1) 실험 목적
- 원 코드의 PCA 기반 식별에서 차원 수를 점검한다.

### 2) 변경 인자
- num_components

### 3) 결과 표
| 파라미터 | 값 | 결과 이미지 |
|---|---|---|
| num_components | 80 | outputs/ex5_person_identification_revised/ex5_pca_80.png |
| num_components | 120 | outputs/ex5_person_identification_revised/ex5_pca_120.png |
| num_components | 150 | outputs/ex5_person_identification_revised/ex5_pca_150.png |

### 4) 해석 요약
- 성분 수가 증가하면 분리력이 개선되는 구간이 있었다. 원 코드의 높은 차원 설정은 조명 변동 데이터에서 성능 안정화에 도움을 줬다. 다만 일정 수준 이후 이득은 완만해졌다.

