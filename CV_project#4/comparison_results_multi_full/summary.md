# Foundation Models vs YOLO Comparison

This evaluation uses folder names as image-level ground truth because bounding-box labels are not available.

- YOLO model: `C:\Users\lmhst\Downloads\yolov11_best.pt`
- Foundation models:
  - `owlvit:google/owlvit-base-patch32:OWL-ViT`
  - `owlv2:google/owlv2-base-patch16-ensemble:OWLv2`
  - `grounding_dino:IDEA-Research/grounding-dino-tiny:Grounding DINO`
- Total images: 508

## Overall

| Model | Correct / Total | Accuracy | Detection Coverage | Avg Time/Image |
|---|---:|---:|---:|---:|
| YOLO fine-tuned | 363 / 508 | 0.715 | 0.961 | 0.150s |
| OWL-ViT | 385 / 508 | 0.758 | 1.000 | 0.178s |
| OWLv2 | 435 / 508 | 0.856 | 1.000 | 0.888s |
| Grounding DINO | 138 / 508 | 0.272 | 0.906 | 0.520s |

## By Class

### hammer

| Model | Correct / Total | Accuracy | Detection Coverage | Avg Time/Image |
|---|---:|---:|---:|---:|
| YOLO fine-tuned | 88 / 101 | 0.871 | 0.970 | 0.139s |
| OWL-ViT | 62 / 101 | 0.614 | 1.000 | 0.170s |
| OWLv2 | 55 / 101 | 0.545 | 1.000 | 0.885s |
| Grounding DINO | 72 / 101 | 0.713 | 0.970 | 0.573s |

### pliers

| Model | Correct / Total | Accuracy | Detection Coverage | Avg Time/Image |
|---|---:|---:|---:|---:|
| YOLO fine-tuned | 176 / 306 | 0.575 | 0.944 | 0.152s |
| OWL-ViT | 235 / 306 | 0.768 | 1.000 | 0.181s |
| OWLv2 | 280 / 306 | 0.915 | 1.000 | 0.904s |
| Grounding DINO | 64 / 306 | 0.209 | 0.882 | 0.518s |

### screwdriver

| Model | Correct / Total | Accuracy | Detection Coverage | Avg Time/Image |
|---|---:|---:|---:|---:|
| YOLO fine-tuned | 99 / 101 | 0.980 | 1.000 | 0.158s |
| OWL-ViT | 88 / 101 | 0.871 | 1.000 | 0.181s |
| OWLv2 | 100 / 101 | 0.990 | 1.000 | 0.844s |
| Grounding DINO | 2 / 101 | 0.020 | 0.911 | 0.476s |
