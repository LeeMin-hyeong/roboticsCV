# Roboflow COCO BBox Evaluation

- Images: 157
- Ground-truth boxes: 295
- Classes: hammer, pliers, screwdriver, wrench

## Overall

| Model | mAP@0.5 | mAP@0.5:0.95 | Precision@0.5 | Recall@0.5 | Avg Time/Image |
|---|---:|---:|---:|---:|---:|
| YOLO fine-tuned | 0.857 | 0.587 | 0.380 | 0.892 | 0.030s |
| OWL-ViT | 0.398 | 0.307 | 0.222 | 0.624 | 0.119s |
| OWLv2 | 0.759 | 0.591 | 0.185 | 0.898 | 2.812s |
| Grounding DINO | 0.022 | 0.012 | 0.058 | 0.281 | 0.408s |

## AP@0.5 By Class

| Class | YOLO fine-tuned | OWL-ViT | OWLv2 | Grounding DINO |
|---|---:|---:|---:|---:|
| hammer | 0.938 | 0.290 | 0.631 | 0.025 |
| pliers | 0.904 | 0.367 | 0.884 | 0.015 |
| screwdriver | 0.785 | 0.408 | 0.821 | 0.020 |
| wrench | 0.802 | 0.526 | 0.702 | 0.025 |