# Roboflow COCO BBox Evaluation

- Images: 10
- Ground-truth boxes: 14
- Classes: drill, hammer, pliers, screwdriver, wrench

## Overall

| Model | mAP@0.5 | mAP@0.5:0.95 | Precision@0.5 | Recall@0.5 | Avg Time/Image |
|---|---:|---:|---:|---:|---:|
| YOLO fine-tuned | 0.967 | 0.723 | 0.154 | 1.000 | 0.106s |
| OWL-ViT | 0.294 | 0.168 | 0.026 | 0.786 | 0.046s |
| OWLv2 | 0.889 | 0.719 | 0.028 | 1.000 | 0.186s |
| Grounding DINO | 0.053 | 0.018 | 0.002 | 0.786 | 0.745s |

## AP@0.5 By Class

| Class | YOLO fine-tuned | OWL-ViT | OWLv2 | Grounding DINO |
|---|---:|---:|---:|---:|
| drill | 1.000 | 0.225 | 0.581 | 0.030 |
| hammer | 1.000 | 0.625 | 1.000 | 0.002 |
| pliers | 1.000 | 0.222 | 1.000 | 0.216 |
| screwdriver | 0.833 | 0.167 | 1.000 | 0.017 |
| wrench | 1.000 | 0.229 | 0.867 | 0.002 |