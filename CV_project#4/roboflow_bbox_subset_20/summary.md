# Roboflow COCO BBox Evaluation

- Images: 100
- Ground-truth boxes: 192
- Classes: drill, hammer, pliers, screwdriver, wrench

## Overall

| Model | mAP@0.5 | mAP@0.5:0.95 | Precision@0.5 | Recall@0.5 | Avg Time/Image |
|---|---:|---:|---:|---:|---:|
| YOLO fine-tuned | 0.910 | 0.686 | 0.176 | 0.948 | 0.036s |
| OWL-ViT | 0.339 | 0.215 | 0.108 | 0.552 | 0.053s |
| OWLv2 | 0.679 | 0.453 | 0.095 | 0.875 | 0.633s |
| Grounding DINO | 0.033 | 0.021 | 0.022 | 0.411 | 0.295s |

## AP@0.5 By Class

| Class | YOLO fine-tuned | OWL-ViT | OWLv2 | Grounding DINO |
|---|---:|---:|---:|---:|
| drill | 0.865 | 0.465 | 0.768 | 0.078 |
| hammer | 0.970 | 0.361 | 0.437 | 0.008 |
| pliers | 0.940 | 0.272 | 0.857 | 0.019 |
| screwdriver | 0.913 | 0.188 | 0.612 | 0.033 |
| wrench | 0.864 | 0.411 | 0.722 | 0.026 |