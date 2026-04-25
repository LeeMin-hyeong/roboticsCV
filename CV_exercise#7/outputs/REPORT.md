# CV Exercise 7 Ablation Study

> 실행 환경 제약으로 CIFAR-10 대신 sklearn digits 데이터를 32x32로 리사이즈해 동일 구조 실험을 수행했다.

## Exercise 1. Kernel Size Study

### 1) 실험 목적
- 커널 크기 변화가 성능과 파라미터 수에 미치는 영향을 확인한다.

### 2) 변경 인자
- kernel_size

### 3) 결과 표
| 파라미터 | 값 | 결과 이미지 |
|---|---|---|
| kernel_size | 3 | outputs/ex1_kernel_size/ex1_kernel_summary.png |
| kernel_size | 5 | outputs/ex1_kernel_size/ex1_kernel_summary.png |
| kernel_size | 7 | outputs/ex1_kernel_size/ex1_kernel_summary.png |
| kernel_size | 9 | outputs/ex1_kernel_size/ex1_kernel_summary.png |

### 4) 해석 요약
- 3x3에서 Test Acc 76.1%로 가장 높은 성능을 기록했다.
- 5x5는 파라미터가 증가했지만 정확도는 75.4%로 소폭 하락했다.
- 7x7, 9x9로 갈수록 정확도가 62.2%, 58.1%까지 크게 감소했다.
- 커널 크기 확대에 따른 표현력 증가보다 최적화 난이도와 과대 수용영역의 불이익이 더 크게 작용했다.

## Exercise 2. Dropout Study

### 1) 실험 목적
- Dropout 비율 변화가 일반화 성능과 과적합 갭에 주는 영향을 비교한다.

### 2) 변경 인자
- dropout_p

### 3) 결과 표
| 파라미터 | 값 | 결과 이미지 |
|---|---|---|
| dropout_p | 0.0 | outputs/ex2_dropout/ex2_dropout_summary.png |
| dropout_p | 0.25 | outputs/ex2_dropout/ex2_dropout_summary.png |
| dropout_p | 0.5 | outputs/ex2_dropout/ex2_dropout_summary.png |
| dropout_p | 0.75 | outputs/ex2_dropout/ex2_dropout_summary.png |

### 4) 해석 요약
- p=0.0에서는 Train 98.5%, Test 75.7%, Gap 22.8%p로 과적합이 크게 나타났다.
- p=0.25에서는 Test 76.6%로 소폭 개선됐지만 Gap 22.4%p로 여전히 컸다.
- p=0.5에서는 Test 77.4%로 최고 성능을 기록했고 Gap 19.2%p로 균형이 가장 좋았다.
- p=0.75에서는 Gap이 9.7%p까지 줄었지만 Train 85.4%로 하락해 과한 규제 경향이 확인됐다.

## Exercise 3. Depth Study

### 1) 실험 목적
- 합성곱 층 깊이 변화에 따른 정확도와 학습 시간의 트레이드오프를 비교한다.

### 2) 변경 인자
- num_conv_layers

### 3) 결과 표
| 파라미터 | 값 | 결과 이미지 |
|---|---|---|
| num_conv_layers | 2 | outputs/ex3_depth/ex3_depth_summary.png |
| num_conv_layers | 4 | outputs/ex3_depth/ex3_depth_summary.png |
| num_conv_layers | 6 | outputs/ex3_depth/ex3_depth_summary.png |
| num_conv_layers | 8 | outputs/ex3_depth/ex3_depth_summary.png |

### 4) 해석 요약
- 깊이 2에서 Test 72.8%, 118.3s로 가장 빠르지만 정확도는 가장 낮았다.
- 깊이 4와 6에서 Test 79.2% → 81.7%로 꾸준히 상승했고 시간 증가는 123.2s → 128.1s로 완만했다.
- 깊이 8에서 Test 83.4%로 최고 성능을 기록했지만 학습 시간 142.6s, 파라미터 2,806,858로 비용이 가장 컸다.
- 정확도 최우선이면 깊이 8이 유리하고, 효율을 함께 보면 깊이 6이 타협점으로 해석된다.

## Exercise 4. Learning Rate Study

### 1) 실험 목적
- 학습률 변화가 수렴 안정성과 최종 정확도에 미치는 영향을 비교한다.

### 2) 변경 인자
- learning_rate

### 3) 결과 표
| 파라미터 | 값 | 결과 이미지 |
|---|---|---|
| learning_rate | 0.1 | outputs/ex4_learning_rate/ex4_lr_summary.png |
| learning_rate | 0.01 | outputs/ex4_learning_rate/ex4_lr_summary.png |
| learning_rate | 0.001 | outputs/ex4_learning_rate/ex4_lr_summary.png |
| learning_rate | 0.0001 | outputs/ex4_learning_rate/ex4_lr_summary.png |

### 4) 해석 요약
- lr=0.1은 E1 loss가 18.4264로 매우 높고 최종 Test Acc도 19.5%에 그쳐 학습이 사실상 실패했다.
- lr=0.01은 학습이 진행되지만 최종 Test Acc 47.5%로 수렴 속도와 성능이 모두 부족했다.
- lr=0.001에서 최종 Test Acc 77.2%로 최고 성능을 기록했고 final loss도 0.4638로 가장 낮았다.
- lr=0.0001은 안정적이지만 Test Acc 74.1%로 lr=0.001 대비 성능이 낮아, 이번 설정의 최적 학습률은 0.001로 해석된다.
