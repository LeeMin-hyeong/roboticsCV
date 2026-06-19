# Foundation vs YOLO Comparison

This evaluation uses folder names as image-level ground truth because bounding-box labels are not available.

- YOLO model: `C:\Users\lmhst\Downloads\yolov11_best.pt`
- Foundation model: `google/owlvit-base-patch32`
- Total images: 508

## Overall

| Model | Correct / Total | Accuracy | Detection Coverage | Avg Time/Image |
|---|---:|---:|---:|---:|
| YOLO fine-tuned | 363 / 508 | 0.715 | 0.961 | 0.161s |
| Foundation | 384 / 508 | 0.756 | 0.998 | 0.210s |

## By Class

### hammer

| Model | Correct / Total | Accuracy | Detection Coverage | Avg Time/Image |
|---|---:|---:|---:|---:|
| YOLO fine-tuned | 88 / 101 | 0.871 | 0.970 | 0.156s |
| Foundation | 62 / 101 | 0.614 | 1.000 | 0.204s |

### pliers

| Model | Correct / Total | Accuracy | Detection Coverage | Avg Time/Image |
|---|---:|---:|---:|---:|
| YOLO fine-tuned | 176 / 306 | 0.575 | 0.944 | 0.160s |
| Foundation | 234 / 306 | 0.765 | 0.997 | 0.211s |

### screwdriver

| Model | Correct / Total | Accuracy | Detection Coverage | Avg Time/Image |
|---|---:|---:|---:|---:|
| YOLO fine-tuned | 99 / 101 | 0.980 | 1.000 | 0.170s |
| Foundation | 88 / 101 | 0.871 | 1.000 | 0.212s |
