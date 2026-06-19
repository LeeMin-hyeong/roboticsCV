# Roboflow Tools 데이터셋 기반 BBox 비교 실험

## 실험 목적

본 실험은 Roboflow Universe의 Tools object detection 데이터셋 일부를 test subset으로 사용하여, fine-tuned YOLO 모델과 여러 Foundation object detection 모델의 bounding box 성능을 비교하는 것을 목표로 한다.

이전 실험은 bbox 라벨이 없는 데이터셋을 사용했기 때문에 image-level class matching만 가능했다. 반면 본 실험에서는 COCO annotation의 bounding box를 ground truth로 사용하므로, mAP와 IoU 기반의 object detection 평가가 가능하다.

## 데이터셋

- Source: Roboflow Universe Tools dataset
- Export format: COCO
- Original classes: 12 classes
- Evaluation classes: `drill`, `hammer`, `pliers`, `screwdriver`, `wrench`
- Test subset 구성: 각 클래스별 20장씩 샘플링
- 최종 평가 이미지 수: 100장
- Ground-truth bbox 수: 192개

Roboflow의 `Pliers`와 `plier` 클래스는 같은 의미로 보고 `pliers`로 통합하였다. `Hardhat`, `Rope`, `Measuring Tape`, `Toolbox`, `0`, `1`, `Tools` 등 현재 YOLO 모델과 겹치지 않는 클래스는 평가에서 제외하였다.

## 비교 모델

| 구분 | 모델 |
|---|---|
| Fine-tuned detector | YOLOv11 fine-tuned model |
| Foundation model 1 | OWL-ViT (`google/owlvit-base-patch32`) |
| Foundation model 2 | OWLv2 (`google/owlv2-base-patch16-ensemble`) |
| Foundation model 3 | Grounding DINO (`IDEA-Research/grounding-dino-tiny`) |

Foundation 모델에는 다음 text prompt를 사용하였다.

| Class | Prompt |
|---|---|
| drill | a photo of a drill |
| hammer | a photo of a hammer |
| pliers | a photo of pliers |
| screwdriver | a photo of a screwdriver |
| wrench | a photo of a wrench |

## 평가 방법

COCO annotation의 bounding box를 ground truth로 사용하고, 각 모델의 predicted bounding box와 비교하였다.

평가 지표는 다음과 같다.

- mAP@0.5: IoU threshold 0.5에서의 mean Average Precision
- mAP@0.5:0.95: IoU threshold 0.5부터 0.95까지 0.05 간격으로 계산한 mAP 평균
- Precision@0.5
- Recall@0.5
- Average inference time per image

## 전체 결과

| Model | mAP@0.5 | mAP@0.5:0.95 | Precision@0.5 | Recall@0.5 | Avg Time/Image |
|---|---:|---:|---:|---:|---:|
| YOLO fine-tuned | 0.910 | 0.686 | 0.176 | 0.948 | 0.036s |
| OWL-ViT | 0.339 | 0.215 | 0.108 | 0.552 | 0.053s |
| OWLv2 | 0.679 | 0.453 | 0.095 | 0.875 | 0.633s |
| Grounding DINO | 0.033 | 0.021 | 0.022 | 0.411 | 0.295s |

## 클래스별 AP@0.5

| Class | YOLO fine-tuned | OWL-ViT | OWLv2 | Grounding DINO |
|---|---:|---:|---:|---:|
| drill | 0.865 | 0.465 | 0.768 | 0.078 |
| hammer | 0.970 | 0.361 | 0.437 | 0.008 |
| pliers | 0.940 | 0.272 | 0.857 | 0.019 |
| screwdriver | 0.913 | 0.188 | 0.612 | 0.033 |
| wrench | 0.864 | 0.411 | 0.722 | 0.026 |

## 결과 해석

Bounding box 기준 평가에서는 YOLO fine-tuned 모델이 가장 높은 성능을 보였다. YOLO는 mAP@0.5 0.910, mAP@0.5:0.95 0.686으로 모든 Foundation 모델보다 높았다. 이는 특정 공구 클래스에 대해 fine-tuning된 detector가 실제 bbox localization 성능에서 강하다는 것을 보여준다.

Foundation 모델 중에서는 OWLv2가 가장 좋은 결과를 보였다. OWLv2는 mAP@0.5 0.679로 OWL-ViT보다 높았으며, 특히 `pliers`와 `drill`에서 비교적 높은 AP를 기록하였다. 하지만 평균 추론 시간은 0.633초로 YOLO보다 훨씬 느렸다.

OWL-ViT는 빠른 편이지만 bbox mAP는 낮았다. Grounding DINO는 본 실험 설정에서는 가장 낮은 mAP를 보였는데, 이는 짧은 class prompt와 본 데이터셋의 공구 이미지 조건에서 false positive가 많이 발생했기 때문이다.

Precision 값이 전반적으로 낮게 나타나는 이유는 mAP 계산을 위해 낮은 confidence prediction까지 포함했기 때문이다. 따라서 이 실험에서는 Precision 단독보다 mAP와 Recall을 중심으로 해석하는 것이 적절하다.

## 이전 실험과의 차이

이전 자체 데이터셋 실험은 bbox annotation이 없어 폴더명을 image-level label로 사용했다. 그래서 객체 위치가 맞는지는 평가할 수 없었다.

이번 Roboflow 실험은 COCO bbox annotation을 사용했기 때문에 object detection 관점에서 더 정식적인 비교가 가능하다. 결과적으로 image-level 비교에서는 Foundation 모델이 강하게 보였지만, bbox localization까지 포함하면 fine-tuned YOLO가 더 높은 성능을 보였다.

## 한계

본 실험은 Roboflow 데이터셋 전체가 아니라 클래스별 20장씩 샘플링한 subset으로 수행하였다. 또한 다운로드된 export가 `train` split 중심이므로, 엄밀한 의미의 공식 test set은 아니다. 하지만 모델 학습에 이 subset을 사용하지 않고 평가에만 사용했으므로, 과제용 비교 실험으로는 충분히 타당하다.

더 엄밀한 실험을 위해서는 다음을 추가할 수 있다.

- 공식 train/valid/test split이 포함된 export 사용
- 더 큰 test subset 사용
- confidence threshold별 PR curve 분석
- 예측 bbox 시각화 예시 추가

## 결론

Roboflow Tools 데이터셋의 bbox annotation을 사용한 비교 결과, fine-tuned YOLO 모델이 mAP와 추론 속도 측면에서 가장 우수했다. Foundation 모델 중에서는 OWLv2가 가장 높은 성능을 보였지만, YOLO에 비해 localization 성능과 속도 모두에서 차이가 있었다.

따라서 bbox 정확도가 중요한 object detection 과제에서는 task-specific fine-tuning된 YOLO가 유리하며, Foundation 모델은 라벨이 부족하거나 새로운 클래스를 빠르게 실험할 때 보조적인 baseline으로 활용하는 것이 적절하다.
