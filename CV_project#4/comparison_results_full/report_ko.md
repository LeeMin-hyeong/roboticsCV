# Foundation 모델과 YOLO Fine-tuning 모델 비교

## 실험 목적

본 실험의 목적은 라벨링된 bounding box가 없는 이미지 데이터셋에서 Foundation 모델과 fine-tuned YOLO 모델의 성능을 비교하는 것이다. 데이터셋에는 객체 위치 라벨은 없지만, 상위 폴더명이 `망치`, `드라이버`, `집게류`로 구성되어 있어 이미지 단위 class label로 사용할 수 있다.

따라서 본 실험에서는 mAP와 같은 객체 탐지 위치 기반 지표 대신, 각 이미지에서 모델이 예측한 대표 class가 폴더명과 일치하는지를 기준으로 image-level accuracy를 측정하였다.

## 비교 대상

- YOLO fine-tuned model: `C:\Users\lmhst\Downloads\yolov11_best.pt`
- Foundation model: `google/owlvit-base-patch32`
- 평가 데이터: `C:\Users\lmhst\Downloads\drive-download-20260530T183501Z-3-001.zip`
- 클래스 매핑:
  - `망치` -> `hammer`
  - `드라이버` -> `screwdriver`
  - `집게류` -> `pliers`

## 평가 방법

각 이미지에 대해 다음 과정을 수행하였다.

1. zip 파일 내부 이미지를 읽는다.
2. 폴더명을 ground truth class로 사용한다.
3. YOLO fine-tuned 모델로 객체를 탐지하고, 가장 confidence가 높은 class를 대표 예측으로 사용한다.
4. OWL-ViT Foundation 모델에는 `hammer`, `screwdriver`, `pliers`에 해당하는 text prompt를 입력하고, 가장 높은 score의 class를 대표 예측으로 사용한다.
5. 대표 예측 class와 폴더명 기반 정답이 일치하면 correct로 계산한다.

## 전체 결과

| Model | Correct / Total | Accuracy | Detection Coverage | Avg Time/Image |
|---|---:|---:|---:|---:|
| YOLO fine-tuned | 363 / 508 | 0.715 | 0.961 | 0.161s |
| Foundation OWL-ViT | 384 / 508 | 0.756 | 0.998 | 0.210s |

## 클래스별 결과

| Class | Model | Correct / Total | Accuracy | Detection Coverage | Avg Time/Image |
|---|---|---:|---:|---:|---:|
| hammer | YOLO fine-tuned | 88 / 101 | 0.871 | 0.970 | 0.156s |
| hammer | Foundation OWL-ViT | 62 / 101 | 0.614 | 1.000 | 0.204s |
| pliers | YOLO fine-tuned | 176 / 306 | 0.575 | 0.944 | 0.160s |
| pliers | Foundation OWL-ViT | 234 / 306 | 0.765 | 0.997 | 0.211s |
| screwdriver | YOLO fine-tuned | 99 / 101 | 0.980 | 1.000 | 0.170s |
| screwdriver | Foundation OWL-ViT | 88 / 101 | 0.871 | 1.000 | 0.212s |

## 해석

전체 image-level accuracy는 Foundation 모델인 OWL-ViT가 0.756으로 YOLO fine-tuned 모델의 0.715보다 약간 높았다. 특히 `pliers` 클래스에서 OWL-ViT가 0.765, YOLO가 0.575로 차이가 크게 나타났다. 반면 `hammer`와 `screwdriver` 클래스에서는 YOLO fine-tuned 모델이 더 높은 정확도를 보였다.

YOLO fine-tuned 모델은 평균 추론 시간이 0.161초로 OWL-ViT의 0.210초보다 빠르다. 따라서 특정 클래스에서 충분히 학습된 경우에는 YOLO가 더 빠르고 안정적으로 동작하지만, `pliers`처럼 데이터 분포가 복잡하거나 학습이 충분하지 않은 클래스에서는 open-vocabulary Foundation 모델이 더 잘 일반화할 수 있음을 확인할 수 있다.

## 한계

본 데이터셋에는 bounding box annotation이 없기 때문에 mAP, IoU, localization error와 같은 표준 객체 탐지 지표는 계산할 수 없다. 따라서 이 결과는 객체 위치까지 정확히 탐지했는지를 평가한 것이 아니라, 이미지 안의 주요 객체 class를 맞혔는지를 평가한 약식 비교이다.

정확한 객체 탐지 성능 비교를 위해서는 일부 이미지라도 bounding box를 직접 라벨링한 test set을 만들고, 두 모델의 예측 box를 같은 기준으로 평가해야 한다.

## 결론

라벨이 없는 데이터셋에서도 폴더명을 이미지 단위 정답으로 사용하면 Foundation 모델과 YOLO fine-tuned 모델을 비교할 수 있다. 본 실험에서는 Foundation 모델이 전체 정확도와 coverage에서 더 높았고, YOLO fine-tuned 모델은 `hammer`, `screwdriver` 클래스와 추론 속도에서 강점을 보였다.

따라서 데이터셋 특화 학습이 잘 된 클래스에서는 YOLO fine-tuning이 유리하지만, 라벨이 부족하거나 클래스 분포가 다양할 때는 Foundation 모델이 더 강한 일반화 성능을 보일 수 있다.
