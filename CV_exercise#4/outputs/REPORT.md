# CV Exercise 4 Ablation Study (Revised)

## Exercise 1. Binary_dilation

### 1) 실험 목적
- 이진 dilation 크기에 따른 연결 효과를 비교한다.

### 2) 변경 인자
- disk_size

### 3) 결과 표
| 파라미터 | 값 | 결과 이미지 |
|---|---|---|
| disk_size | 10 | outputs/ex1_binary_dilation/ex1_disk_10.png |
| disk_size | 20 | outputs/ex1_binary_dilation/ex1_disk_20.png |
| disk_size | 30 | outputs/ex1_binary_dilation/ex1_disk_30.png |

### 4) 해석 요약
- 원 코드처럼 disk 크기를 키우면 전경이 확장됐다. 작은 결손 복원에는 유리했다. 큰 disk에서는 객체 병합이 쉽게 발생했다.

## Exercise 2. Binary_Erosion

### 1) 실험 목적
- 이진 erosion 크기 변화에 따른 객체 축소를 확인한다.

### 2) 변경 인자
- disk_size

### 3) 결과 표
| 파라미터 | 값 | 결과 이미지 |
|---|---|---|
| disk_size | 3 | outputs/ex2_binary_erosion/ex2_disk_3.png |
| disk_size | 7 | outputs/ex2_binary_erosion/ex2_disk_7.png |
| disk_size | 11 | outputs/ex2_binary_erosion/ex2_disk_11.png |

### 4) 해석 요약
- 원 코드의 침식 실험에서 disk가 커질수록 구조가 빠르게 감소했다. 얇은 전경은 먼저 사라졌다. 잡음 억제와 정보 보존 사이 절충이 필요했다.

## Exercise 3. Binary_Erosion_Coins

### 1) 실험 목적
- 동전 분리를 위한 structuring element 형태와 크기를 비교한다.

### 2) 변경 인자
- SE(shape,size)

### 3) 결과 표
| 파라미터 | 값 | 결과 이미지 |
|---|---|---|
| SE(shape,size) | square,30 | outputs/ex3_binary_erosion_coins/ex3_square_30.png |
| SE(shape,size) | square,70 | outputs/ex3_binary_erosion_coins/ex3_square_70.png |
| SE(shape,size) | square,96 | outputs/ex3_binary_erosion_coins/ex3_square_96.png |
| SE(shape,size) | disk,30 | outputs/ex3_binary_erosion_coins/ex3_disk_30.png |
| SE(shape,size) | disk,70 | outputs/ex3_binary_erosion_coins/ex3_disk_70.png |
| SE(shape,size) | disk,96 | outputs/ex3_binary_erosion_coins/ex3_disk_96.png |

### 4) 해석 요약
- square와 disk는 같은 크기에서도 분리 양상이 달랐다. 크기를 키우면 접촉 영역은 줄었다. 과도한 침식은 객체 자체를 약화시켰다.

## Exercise 4. Binary_Erosion_Fence

### 1) 실험 목적
- cross 길이에 따른 fence 검출 변화를 비교한다.

### 2) 변경 인자
- cross_length

### 3) 결과 표
| 파라미터 | 값 | 결과 이미지 |
|---|---|---|
| cross_length | 101 | outputs/ex4_binary_erosion_fence/ex4_length_101.png |
| cross_length | 151 | outputs/ex4_binary_erosion_fence/ex4_length_151.png |
| cross_length | 201 | outputs/ex4_binary_erosion_fence/ex4_length_201.png |

### 4) 해석 요약
- 길이가 길수록 구조요소 방향성이 강해졌다. fence 패턴 강조는 증가했다. 동시에 검출 영역은 더 보수적으로 변했다.

## Exercise 5. Small_Hole_Removal

### 1) 실험 목적
- small hole removal에서 이진화 임계값 영향을 비교한다.

### 2) 변경 인자
- threshold

### 3) 결과 표
| 파라미터 | 값 | 결과 이미지 |
|---|---|---|
| threshold | 90 | outputs/ex5_small_hole_removal/ex5_threshold_90.png |
| threshold | 100 | outputs/ex5_small_hole_removal/ex5_threshold_100.png |
| threshold | 110 | outputs/ex5_small_hole_removal/ex5_threshold_110.png |

### 4) 해석 요약
- 커널은 원 코드 값으로 고정했다. threshold 변화만으로 hole 후보 영역이 달라졌다. 채움 결과도 임계값에 민감하게 변했다.

## Exercise 6. Binary_Edge_Detection

### 1) 실험 목적
- binary edge detection에서 edge 조합 방식 차이를 비교한다.

### 2) 변경 인자
- edge_mode

### 3) 결과 표
| 파라미터 | 값 | 결과 이미지 |
|---|---|---|
| edge_mode | edge1_dilated_minus_orig | outputs/ex6_binary_edge_detection/ex6_edge1_dilated_minus_orig.png |
| edge_mode | edge2_orig_minus_eroded | outputs/ex6_binary_edge_detection/ex6_edge2_orig_minus_eroded.png |
| edge_mode | edge3_combined | outputs/ex6_binary_edge_detection/ex6_edge3_combined.png |

### 4) 해석 요약
- 커널은 원 코드 값으로 고정했다. edge1과 edge2는 강조 방향이 다르다. combined 결과는 경계 범위가 가장 넓게 나타났다.

## Exercise 7. 1D Gray Morph

### 1) 실험 목적
- 1D 회색조 morphology에서 창 길이 효과를 비교한다.

### 2) 변경 인자
- SE_length

### 3) 결과 표
| 파라미터 | 값 | 결과 이미지 |
|---|---|---|
| SE_length | 1 | outputs/ex7_1d_gray_morph/ex7_length_1.png |
| SE_length | 11 | outputs/ex7_1d_gray_morph/ex7_length_11.png |
| SE_length | 21 | outputs/ex7_1d_gray_morph/ex7_length_21.png |

### 4) 해석 요약
- 원 코드 길이 증가에 따라 envelope 폭이 커졌다. peak와 valley 왜곡도 함께 증가했다. 길이 선택이 신호 형태 보존에 중요했다.

## Exercise 8. 2D Gray Morph

### 1) 실험 목적
- 2D 회색조 morphology에서 구조요소 형태 차이를 비교한다.

### 2) 변경 인자
- SE_type

### 3) 결과 표
| 파라미터 | 값 | 결과 이미지 |
|---|---|---|
| SE_type | square_10 | outputs/ex8_2d_gray_morph/ex8_square_10.png |
| SE_type | disk_17 | outputs/ex8_2d_gray_morph/ex8_disk_17.png |
| SE_type | double_line | outputs/ex8_2d_gray_morph/ex8_double_line.png |

### 4) 해석 요약
- shape가 달라지면 강조 방향이 달라졌다. line 계열은 방향성 반응이 강했다. disk와 square는 보다 균일한 팽창 특성을 보였다.

## Exercise 9. Coin Separation by Graylevel Dilation

### 1) 실험 목적
- gray dilation 크기에 따른 coin 분리 변화를 비교한다.

### 2) 변경 인자
- disk_size

### 3) 결과 표
| 파라미터 | 값 | 결과 이미지 |
|---|---|---|
| disk_size | 41 | outputs/ex9_coin_separation/ex9_disk_41.png |
| disk_size | 61 | outputs/ex9_coin_separation/ex9_disk_61.png |
| disk_size | 81 | outputs/ex9_coin_separation/ex9_disk_81.png |

### 4) 해석 요약
- dilation 크기에 따라 동전 blob 결합 정도가 바뀌었다. 분리 성능은 크기 선택에 민감했다. 과도한 크기는 객체 결합을 유발했다.

## Exercise 10. Hole Detection by Graylevel Erosion

### 1) 실험 목적
- cross 길이에 따른 hole 강조 효과를 비교한다.

### 2) 변경 인자
- cross_length

### 3) 결과 표
| 파라미터 | 값 | 결과 이미지 |
|---|---|---|
| cross_length | 101 | outputs/ex10_hole_detection/ex10_length_101.png |
| cross_length | 151 | outputs/ex10_hole_detection/ex10_length_151.png |
| cross_length | 201 | outputs/ex10_hole_detection/ex10_length_201.png |

### 4) 해석 요약
- 길이가 커질수록 침식 기반 hole 응답이 강화됐다. threshold 결과도 함께 변했다. 지나치게 긴 길이는 배경 영향까지 확대했다.

## Exercise 11. Graylevel Morphological Edge Detector

### 1) 실험 목적
- edge threshold에 따른 경계 선택 강도를 비교한다.

### 2) 변경 인자
- edge_threshold

### 3) 결과 표
| 파라미터 | 값 | 결과 이미지 |
|---|---|---|
| edge_threshold | 0.10 | outputs/ex11_gray_edge_detector/ex11_threshold_0.10.png |
| edge_threshold | 0.15 | outputs/ex11_gray_edge_detector/ex11_threshold_0.15.png |
| edge_threshold | 0.20 | outputs/ex11_gray_edge_detector/ex11_threshold_0.20.png |

### 4) 해석 요약
- 커널은 원 코드 값으로 고정했다. threshold를 높이면 강한 경계만 남았다. 낮은 threshold에서는 배경 잡응답이 늘었다.

## Exercise 12. Cascaded Graylevel Dilations

### 1) 실험 목적
- cascaded dilation 단계별 누적 효과를 비교한다.

### 2) 변경 인자
- iterations

### 3) 결과 표
| 파라미터 | 값 | 결과 이미지 |
|---|---|---|
| iterations | 1 | outputs/ex12_cascaded_dilations/ex12_iter_1.png |
| iterations | 2 | outputs/ex12_cascaded_dilations/ex12_iter_2.png |
| iterations | 3 | outputs/ex12_cascaded_dilations/ex12_iter_3.png |

### 4) 해석 요약
- 단계가 늘수록 방향성 팽창이 누적됐다. 작은 틈은 메워졌다. 세부 경계는 점차 완화됐다.

## Exercise 13. Majority Filter and Median Filter

### 1) 실험 목적
- 원 코드 노이즈 조건에서 median filtering 효과를 비교한다.

### 2) 변경 인자
- noise_level / median_kernel

### 3) 결과 표
| 파라미터 | 값 | 결과 이미지 |
|---|---|---|
| noise_level | noise5 | outputs/ex13_majority_median/ex13_noise5_k3.png |
| noise_level | noise20 | outputs/ex13_majority_median/ex13_noise20_k3.png |
| median_kernel | 3 | outputs/ex13_majority_median/ex13_sculpture_k3.png |
| median_kernel | 7 | outputs/ex13_majority_median/ex13_sculpture_k7.png |

### 4) 해석 요약
- 노이즈 강도가 높을수록 원본 손상이 커졌다. median 필터는 잡음 감소에 효과적이었다. 큰 커널은 노이즈를 더 줄이지만 텍스처도 완화됐다.

## Exercise 14. Nonuniform Lighting Compensation

### 1) 실험 목적
- 조명 보정 window 크기 변화 효과를 비교한다.

### 2) 변경 인자
- window_size

### 3) 결과 표
| 파라미터 | 값 | 결과 이미지 |
|---|---|---|
| window_size | 31 | outputs/ex14_lighting_comp/ex14_window_31.png |
| window_size | 61 | outputs/ex14_lighting_comp/ex14_window_61.png |
| window_size | 91 | outputs/ex14_lighting_comp/ex14_window_91.png |

### 4) 해석 요약
- window가 커질수록 저주파 조명 성분 제거가 강해졌다. 문자 대비는 개선될 수 있다. 과도한 window는 세부 정보 약화를 유발했다.

