# AIHub COCO 데이터셋 기반 독립 BBox 비교 실험

## 실험 목적

이전 Roboflow Tools 데이터셋은 YOLO fine-tuning에 사용된 데이터와 겹칠 가능성이 있어 YOLO 성능이 과대평가될 수 있다. 이를 보완하기 위해 새 데이터셋 `aiHub.coco.zip`을 독립 평가셋으로 사용하여 YOLO fine-tuned 모델과 Foundation object detection 모델의 bbox 성능을 다시 비교하였다.

## 데이터셋

- Dataset: `C:\Users\lmhst\Downloads\aiHub.coco.zip`
- Format: COCO
- Images in export: 255
- Annotations in export: 422
- Evaluation subset: 클래스별 최대 40장 샘플링
- Final evaluation images: 157
- Ground-truth boxes: 295

평가에는 YOLO 모델과 겹치는 4개 클래스를 사용하였다. `Drill` 클래스는 새 데이터셋에 없어 제외하였다.

| COCO class | Evaluation class |
|---|---|
| Hammer | hammer |
| Pliers | pliers |
| Screw Driver | screwdriver |
| Wrench | wrench |

## 비교 모델

| 구분 | 모델 |
|---|---|
| Fine-tuned detector | YOLOv11 fine-tuned model |
| Foundation model 1 | OWL-ViT (`google/owlvit-base-patch32`) |
| Foundation model 2 | OWLv2 (`google/owlv2-base-patch16-ensemble`) |
| Foundation model 3 | Grounding DINO (`IDEA-Research/grounding-dino-tiny`) |

## 전체 결과

| Model | mAP@0.5 | mAP@0.5:0.95 | Precision@0.5 | Recall@0.5 | Avg Time/Image |
|---|---:|---:|---:|---:|---:|
| YOLO fine-tuned | 0.857 | 0.587 | 0.380 | 0.892 | 0.030s |
| OWL-ViT | 0.398 | 0.307 | 0.222 | 0.624 | 0.119s |
| OWLv2 | 0.759 | 0.591 | 0.185 | 0.898 | 2.812s |
| Grounding DINO | 0.022 | 0.012 | 0.058 | 0.281 | 0.408s |

![AIHub bbox overall result](../report_visuals_aihub/bbox_overall_map.png)

새 독립 평가셋에서도 YOLO fine-tuned 모델은 mAP@0.5 기준으로 가장 높은 성능을 보였다. 다만 mAP@0.5:0.95에서는 OWLv2가 0.591로 YOLO의 0.587과 거의 동일하게 나타났다. 이는 엄격한 IoU 범위까지 고려하면 OWLv2의 bbox 품질도 경쟁력이 있음을 보여준다.

## 클래스별 AP@0.5

| Class | YOLO fine-tuned | OWL-ViT | OWLv2 | Grounding DINO |
|---|---:|---:|---:|---:|
| hammer | 0.938 | 0.290 | 0.631 | 0.025 |
| pliers | 0.904 | 0.367 | 0.884 | 0.015 |
| screwdriver | 0.785 | 0.408 | 0.821 | 0.020 |
| wrench | 0.802 | 0.526 | 0.702 | 0.025 |

![AIHub bbox class AP](../report_visuals_aihub/bbox_class_ap.png)

클래스별로 보면 YOLO는 `hammer`, `pliers`, `wrench`에서 가장 높은 AP@0.5를 보였다. 반면 `screwdriver`에서는 OWLv2가 0.821로 YOLO의 0.785보다 높았다. 이는 새 데이터셋에서 Foundation 모델이 특정 클래스에 대해 더 강하게 일반화할 수 있음을 보여준다.

## 이전 Roboflow 평가와 비교

| Dataset | Model | mAP@0.5 | mAP@0.5:0.95 | Recall@0.5 |
|---|---|---:|---:|---:|
| Roboflow Tools subset | YOLO fine-tuned | 0.910 | 0.686 | 0.948 |
| Roboflow Tools subset | OWLv2 | 0.679 | 0.453 | 0.875 |
| AIHub subset | YOLO fine-tuned | 0.857 | 0.587 | 0.892 |
| AIHub subset | OWLv2 | 0.759 | 0.591 | 0.898 |

Roboflow 평가에서는 YOLO가 OWLv2보다 큰 차이로 높았다. 하지만 독립 AIHub 평가에서는 YOLO의 mAP@0.5가 0.910에서 0.857로 낮아졌고, OWLv2는 0.679에서 0.759로 상승하였다. 특히 mAP@0.5:0.95에서는 AIHub 평가에서 OWLv2가 YOLO와 거의 같은 수준이었다.

이 차이는 이전 Roboflow 데이터셋이 YOLO 학습 데이터와 겹쳤을 가능성이 있으며, 그로 인해 YOLO에 유리한 평가였을 수 있음을 시사한다.

## 해석

독립 데이터셋 기준으로도 YOLO fine-tuned 모델은 빠른 추론 속도와 높은 mAP@0.5를 보였다. 따라서 특정 공구 탐지 task에 대해 fine-tuning된 YOLO는 여전히 강력한 detector이다.

하지만 새 데이터셋에서는 OWLv2가 YOLO와의 격차를 크게 줄였고, mAP@0.5:0.95와 `screwdriver` class에서는 YOLO와 비슷하거나 더 높은 성능을 보였다. 이는 Foundation 모델이 학습 데이터에 직접 포함되지 않은 새로운 도메인에서 더 나은 일반화 성능을 보일 수 있음을 의미한다.

Grounding DINO는 본 실험 조건에서 낮은 mAP를 보였다. 짧은 class prompt와 공구 이미지 조건에서 false positive가 많았던 것으로 해석된다.

## 결론

새 AIHub COCO 데이터셋을 사용한 독립 bbox 평가 결과, YOLO fine-tuned 모델은 mAP@0.5와 추론 속도에서 가장 우수했다. 그러나 OWLv2는 독립 데이터셋에서 성능이 크게 향상되었고, mAP@0.5:0.95에서는 YOLO와 거의 동일한 수준을 보였다.

따라서 최종 결론은 다음과 같다.

| 관점 | 결론 |
|---|---|
| 학습 데이터와 유사한 데이터 | YOLO fine-tuned 모델이 매우 강함 |
| 독립 데이터셋 일반화 | OWLv2가 YOLO와 경쟁 가능 |
| 실시간성 | YOLO가 가장 유리 |
| bbox localization 정밀도 | YOLO와 OWLv2 모두 강점이 있으며, 데이터셋에 따라 차이가 있음 |

즉, YOLO는 배포와 속도 측면에서 유리하고, Foundation 모델은 새로운 데이터셋에서의 일반화 성능을 확인하는 baseline으로 중요하다.
