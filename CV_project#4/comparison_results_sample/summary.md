# Foundation vs YOLO Comparison

This evaluation uses folder names as image-level ground truth because bounding-box labels are not available.

- YOLO model: `C:\Users\lmhst\Downloads\yolov11_best.pt`
- Foundation model: `google/owlvit-base-patch32`
- Total images: 9

## Overall

| Model | Correct / Total | Accuracy | Detection Coverage | Avg Time/Image |
|---|---:|---:|---:|---:|
| YOLO fine-tuned | 9 / 9 | 1.000 | 1.000 | 0.256s |
| Foundation | 7 / 9 | 0.778 | 1.000 | 0.244s |

## By Class

### hammer

| Model | Correct / Total | Accuracy | Detection Coverage | Avg Time/Image |
|---|---:|---:|---:|---:|
| YOLO fine-tuned | 3 / 3 | 1.000 | 1.000 | 0.221s |
| Foundation | 2 / 3 | 0.667 | 1.000 | 0.290s |

### pliers

| Model | Correct / Total | Accuracy | Detection Coverage | Avg Time/Image |
|---|---:|---:|---:|---:|
| YOLO fine-tuned | 3 / 3 | 1.000 | 1.000 | 0.225s |
| Foundation | 2 / 3 | 0.667 | 1.000 | 0.266s |

### screwdriver

| Model | Correct / Total | Accuracy | Detection Coverage | Avg Time/Image |
|---|---:|---:|---:|---:|
| YOLO fine-tuned | 3 / 3 | 1.000 | 1.000 | 0.322s |
| Foundation | 3 / 3 | 1.000 | 1.000 | 0.175s |
