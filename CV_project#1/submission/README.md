# CV_project#1 submission package

This folder contains the lightweight files intended for Git upload or final submission.

## Included

- `REPORT.md`: final Korean report in Markdown.
- `run_last_squeeze_coco_cv1.py`: final training/evaluation protocol script.
- `run_seed_repeat_last_squeeze_cv1.py`: seed-repeat statistical validation script.
- `run_final_vs_coco_on_baseline_voc3_cv1.py`: baseline-voc3 comparison script.
- `metrics/`: selected JSON metrics used by the report.
- `figures/`: selected report figures.

## Excluded From Git

The root `.gitignore` excludes downloaded VOC archives, generated datasets, YOLO run folders, and model checkpoints. These files are reproducible or too large/noisy for normal Git history.

If a checkpoint must be shared, use a release artifact or external storage rather than committing every experiment weight.
