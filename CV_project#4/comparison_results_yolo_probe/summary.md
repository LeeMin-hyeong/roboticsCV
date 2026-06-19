# Foundation vs YOLO Comparison

This evaluation uses folder names as image-level ground truth because bounding-box labels are not available.

- YOLO model: `C:\Users\lmhst\Downloads\yolov11_best.pt`
- Foundation model: `google/owlvit-base-patch32`
- Total images: 15

## Overall

| Model | Correct / Total | Accuracy | Detection Coverage | Avg Time/Image |
|---|---:|---:|---:|---:|
| YOLO fine-tuned | 8 / 15 | 0.533 | 1.000 | 0.165s |
| Foundation | 0 / 15 | 0.000 | 0.000 | 0.000s |

## By Class

### hammer

| Model | Correct / Total | Accuracy | Detection Coverage | Avg Time/Image |
|---|---:|---:|---:|---:|
| YOLO fine-tuned | 3 / 5 | 0.600 | 1.000 | 0.074s |
| Foundation | 0 / 5 | 0.000 | 0.000 | 0.000s |

### pliers

| Model | Correct / Total | Accuracy | Detection Coverage | Avg Time/Image |
|---|---:|---:|---:|---:|
| YOLO fine-tuned | 0 / 5 | 0.000 | 1.000 | 0.174s |
| Foundation | 0 / 5 | 0.000 | 0.000 | 0.000s |

### screwdriver

| Model | Correct / Total | Accuracy | Detection Coverage | Avg Time/Image |
|---|---:|---:|---:|---:|
| YOLO fine-tuned | 5 / 5 | 1.000 | 1.000 | 0.247s |
| Foundation | 0 / 5 | 0.000 | 0.000 | 0.000s |
