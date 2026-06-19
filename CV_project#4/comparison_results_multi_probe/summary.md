# Foundation Models vs YOLO Comparison

This evaluation uses folder names as image-level ground truth because bounding-box labels are not available.

- YOLO model: `C:\Users\lmhst\Downloads\yolov11_best.pt`
- Foundation models:
  - `owlvit:google/owlvit-base-patch32:OWL-ViT`
  - `owlv2:google/owlv2-base-patch16-ensemble:OWLv2`
  - `grounding_dino:IDEA-Research/grounding-dino-tiny:Grounding DINO`
- Total images: 6

## Overall

| Model | Correct / Total | Accuracy | Detection Coverage | Avg Time/Image |
|---|---:|---:|---:|---:|
| YOLO fine-tuned | 6 / 6 | 1.000 | 1.000 | 0.279s |
| OWL-ViT | 5 / 6 | 0.833 | 1.000 | 0.231s |
| OWLv2 | 5 / 6 | 0.833 | 1.000 | 1.278s |
| Grounding DINO | 1 / 6 | 0.167 | 0.167 | 0.410s |

## By Class

### hammer

| Model | Correct / Total | Accuracy | Detection Coverage | Avg Time/Image |
|---|---:|---:|---:|---:|
| YOLO fine-tuned | 2 / 2 | 1.000 | 1.000 | 0.207s |
| OWL-ViT | 2 / 2 | 1.000 | 1.000 | 0.251s |
| OWLv2 | 1 / 2 | 0.500 | 1.000 | 1.498s |
| Grounding DINO | 1 / 2 | 0.500 | 0.500 | 0.402s |

### pliers

| Model | Correct / Total | Accuracy | Detection Coverage | Avg Time/Image |
|---|---:|---:|---:|---:|
| YOLO fine-tuned | 2 / 2 | 1.000 | 1.000 | 0.199s |
| OWL-ViT | 1 / 2 | 0.500 | 1.000 | 0.244s |
| OWLv2 | 2 / 2 | 1.000 | 1.000 | 1.372s |
| Grounding DINO | 0 / 2 | 0.000 | 0.000 | 0.365s |

### screwdriver

| Model | Correct / Total | Accuracy | Detection Coverage | Avg Time/Image |
|---|---:|---:|---:|---:|
| YOLO fine-tuned | 2 / 2 | 1.000 | 1.000 | 0.430s |
| OWL-ViT | 2 / 2 | 1.000 | 1.000 | 0.200s |
| OWLv2 | 2 / 2 | 1.000 | 1.000 | 0.965s |
| Grounding DINO | 0 / 2 | 0.000 | 0.000 | 0.464s |
