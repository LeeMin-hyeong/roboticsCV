# 여러 Foundation 모델과 YOLO Fine-tuning 모델 비교

## 실험 목적

본 실험은 bounding box 라벨이 없는 공구 이미지 데이터셋에서 fine-tuned YOLO 모델과 여러 Foundation 모델의 성능을 비교하는 것을 목표로 한다. 데이터셋에는 객체 위치 annotation은 없지만, 폴더명이 `망치`, `드라이버`, `집게류`로 구성되어 있어 이를 image-level ground truth로 사용하였다.

따라서 본 실험은 mAP/IoU 기반의 정식 object detection 평가가 아니라, 각 이미지의 주요 객체 class를 맞혔는지 평가하는 image-level 비교 실험이다.

## 비교 모델

| 구분 | 모델 |
|---|---|
| Fine-tuned detector | YOLOv11 fine-tuned model |
| Foundation model 1 | OWL-ViT (`google/owlvit-base-patch32`) |
| Foundation model 2 | OWLv2 (`google/owlv2-base-patch16-ensemble`) |
| Foundation model 3 | Grounding DINO (`IDEA-Research/grounding-dino-tiny`) |

사용한 class prompt는 다음과 같다.

| Dataset folder | Evaluation class | Text prompt |
|---|---|---|
| 망치 | hammer | a photo of a hammer |
| 드라이버 | screwdriver | a photo of a screwdriver |
| 집게류 | pliers | a photo of pliers |

## 평가 방법

각 이미지에 대해 다음 과정을 수행하였다.

1. zip 내부 이미지를 읽고 폴더명을 정답 class로 사용한다.
2. YOLO fine-tuned 모델은 confidence가 가장 높은 detection class를 대표 예측으로 사용한다.
3. Foundation 모델들은 동일한 text prompt를 입력하고, 가장 높은 score의 detection class를 대표 예측으로 사용한다.
4. 대표 예측 class가 폴더명 기반 정답과 일치하면 correct로 계산한다.
5. 모델별 accuracy, detection coverage, 평균 추론 시간을 비교한다.

## 전체 결과

| Model | Correct / Total | Accuracy | Detection Coverage | Avg Time/Image |
|---|---:|---:|---:|---:|
| YOLO fine-tuned | 363 / 508 | 0.715 | 0.961 | 0.150s |
| OWL-ViT | 385 / 508 | 0.758 | 1.000 | 0.178s |
| OWLv2 | 435 / 508 | 0.856 | 1.000 | 0.888s |
| Grounding DINO | 138 / 508 | 0.272 | 0.906 | 0.520s |

## 클래스별 결과

| Class | Model | Correct / Total | Accuracy | Detection Coverage | Avg Time/Image |
|---|---|---:|---:|---:|---:|
| hammer | YOLO fine-tuned | 88 / 101 | 0.871 | 0.970 | 0.139s |
| hammer | OWL-ViT | 62 / 101 | 0.614 | 1.000 | 0.170s |
| hammer | OWLv2 | 55 / 101 | 0.545 | 1.000 | 0.885s |
| hammer | Grounding DINO | 72 / 101 | 0.713 | 0.970 | 0.573s |
| pliers | YOLO fine-tuned | 176 / 306 | 0.575 | 0.944 | 0.152s |
| pliers | OWL-ViT | 235 / 306 | 0.768 | 1.000 | 0.181s |
| pliers | OWLv2 | 280 / 306 | 0.915 | 1.000 | 0.904s |
| pliers | Grounding DINO | 64 / 306 | 0.209 | 0.882 | 0.518s |
| screwdriver | YOLO fine-tuned | 99 / 101 | 0.980 | 1.000 | 0.158s |
| screwdriver | OWL-ViT | 88 / 101 | 0.871 | 1.000 | 0.181s |
| screwdriver | OWLv2 | 100 / 101 | 0.990 | 1.000 | 0.844s |
| screwdriver | Grounding DINO | 2 / 101 | 0.020 | 0.911 | 0.476s |

## 결과 해석

전체 accuracy는 OWLv2가 0.856으로 가장 높았다. 특히 `pliers` 클래스에서 OWLv2는 0.915의 accuracy를 보이며 YOLO fine-tuned 모델의 0.575보다 크게 높았다. 이는 open-vocabulary Foundation 모델이 특정 클래스에 대해 더 강한 일반화 성능을 보일 수 있음을 보여준다.

YOLO fine-tuned 모델은 전체 accuracy에서는 OWLv2보다 낮았지만, 평균 추론 시간이 0.150초로 가장 빨랐다. 또한 `hammer`와 `screwdriver` 클래스에서는 높은 정확도를 보였으며, 특히 `screwdriver`에서는 0.980으로 OWLv2와 거의 비슷한 성능을 보였다.

Grounding DINO는 `hammer`에서는 비교적 동작했지만, `screwdriver`와 `pliers`에서 낮은 class matching accuracy를 보였다. 이는 모델 자체가 나쁘다는 의미라기보다는, 본 실험처럼 짧은 class prompt와 image-level 대표 예측만 사용하는 조건에서는 OWL 계열 모델이 더 적합했음을 의미한다.

## 한계

본 실험은 bounding box annotation이 없기 때문에 mAP, IoU, localization error를 계산하지 않았다. 따라서 모델이 객체 위치를 얼마나 정확히 찾았는지는 평가하지 못했고, 이미지 안의 주요 객체 class를 맞혔는지만 평가하였다.

정확한 object detection 비교를 위해서는 일부 test image에 대해 사람이 직접 bounding box를 라벨링한 뒤, 동일한 test set에서 mAP@0.5와 mAP@0.5:0.95를 계산해야 한다.

## 결론

라벨이 없는 데이터셋에서도 폴더명을 image-level label로 사용하면 YOLO fine-tuning 모델과 여러 Foundation 모델을 비교할 수 있다. 본 실험에서는 OWLv2가 가장 높은 전체 accuracy를 보였고, YOLO fine-tuned 모델은 가장 빠른 추론 속도와 특정 클래스에서의 높은 정확도를 보였다.

따라서 데이터셋 특화 모델이 필요한 실시간 응용에서는 YOLO가 적합하고, 라벨이 부족하거나 클래스 분포가 다양할 때는 OWLv2와 같은 Foundation 모델이 더 강한 후보가 될 수 있다.
