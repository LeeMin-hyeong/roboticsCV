# COCO Pretrained YOLOv8n 대비 3-Class 검출 성능 향상 연구

학번: 20211435  
프로젝트: `CV_project#1`  
작성일: 2026-04-20

## 초록
본 연구는 VOC 기반 3-class(person, car, dog) 객체검출 과제에서 COCO pretrained `yolov8n.pt`를 기준선(B0)으로 두고, 하이퍼파라미터 및 데이터 구성 최적화를 통해 성능 향상을 달성하는 것을 목표로 한다. 핵심 전략은 `(1) 변수 영향 ablation`, `(2) 증강 강도 재설계`, `(3) 전체 레이어 미세조정`, `(4) 배경(negative) 샘플 포함`, `(5) seed 반복 및 통계 검정`이다. 최종 모델(last_squeeze 계열)은 5-seed 평균 기준 `mAP50-95=0.6626`으로 B0(`0.6245`) 대비 `+0.0381` 향상되었다. seed-level 검정(one-sided sign test, sign-flip permutation)은 모두 `p=0.03125`로 유의했으며, image-level paired-bootstrap에서는 `p=0.0612`로 엄격 기준에서 경계적 결과를 보였다. 본 보고서는 성능 향상 요인과 함께, 데이터 구성 차이가 결과 해석에 미치는 영향을 정량적으로 정리한다.

## 1. 서론
COCO pretrained 모델은 일반 객체 분포에서 강한 초기 성능을 보이지만, 소규모/도메인 편향 데이터셋에서는 과제 특이적 미세조정이 필요하다. 본 연구의 목적은 다음 두 가지다.

1. 3-class 도메인에서 COCO 기준선을 일관되게 상회하는 모델을 구축한다.
2. 향상이 통계적으로 유의한지 seed-level과 image-level 모두에서 검증한다.

## 2. 실험 설정

### 2.1 데이터셋 및 평가 프로토콜
- 타깃 클래스: `person`, `car`, `dog`
- 1차 지표: `mAP50-95`
- 보조 지표: `mAP50`, `mAP75`, 클래스별 AP, 4x4 confusion matrix(배경 포함)
- 비교 대상:
  - B0: COCO pretrained raw (`yolov8n.pt`)
  - Final: `last_squeeze` 최종 미세조정 모델
- 통계 검정:
  - seed-level: sign test, sign-flip permutation, bootstrap CI
  - image-level: paired-bootstrap

### 2.2 Baseline과 Final의 데이터 구성 차이
Baseline은 타깃 클래스가 포함된 양성 이미지 위주였고, Final은 배경(무라벨) 샘플을 학습에 명시적으로 포함했다.

| 구성 | train_images | empty_label_images | effective_train_samples | negative_ratio |
|---|---:|---:|---:|---:|
| baseline_voc3 | 150 | 0 | 150 | 0.00% |
| final_last_squeeze | 4086 | 1268 | 5349 | 31.03% |

![dataset_comp_bar](../figures/dataset_composition_baseline_vs_final_bar_cv1.png)

이 차이는 FP 억제 성능 및 임계값 운영 특성에 직접 영향을 주므로, 최종 성능 해석 시 `파라미터 효과`와 `데이터 분포 효과`를 함께 고려해야 한다.

### 2.3 이미지 편향과 완화 전략
Baseline과 Final의 핵심 차이는 단순 샘플 수 증가가 아니라, 학습 데이터의 편향 구조를 바꾼 점이다.

1. Baseline 편향
- 배경 결핍: `bg_ratio = 0.0` (empty-label 이미지 없음)
- 클래스 불균형: 인스턴스 비율 `person 72.35% / car 22.48% / dog 5.17%`
- 희소 클래스(특히 dog) 노출 부족: dog 포함 이미지 비율 `12.67%`

2. Final 편향 완화
- 배경 샘플 포함: unique 기준 `bg_ratio = 31.03%`로 배경 prior를 학습에 반영
- dog 반복 샘플링 적용: repeat_seq 기준 dog 인스턴스 비율 `21.53%`, dog 포함 이미지 비율 `31.48%`까지 상승
- 결과적으로 `배경 결핍`과 `dog 희소성`이라는 baseline의 두 가지 이미지 편향을 동시에 완화

| 시나리오 | images_total | bg_ratio | inst_person | inst_car | inst_dog | has_person | has_car | has_dog |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Baseline(voc3) | 150 | 0.000 | 0.724 | 0.225 | 0.052 | 0.673 | 0.353 | 0.127 |
| Final(unique) | 4086 | 0.310 | 0.714 | 0.216 | 0.070 | 0.502 | 0.182 | 0.103 |
| Final(repeat) | 5349 | 0.237 | 0.617 | 0.167 | 0.215 | 0.456 | 0.143 | 0.315 |

![dataset_bias_instance](../figures/dataset_bias_instance_ratio_cv1.png)

![dataset_bias_presence](../figures/dataset_bias_image_presence_cv1.png)

## 3. 방법론

### 3.1 변수 영향 분석(Ablation)
초기 단계에서 `epochs`, `imgsz`, `batch`, `lr0`, `freeze`를 중심으로 단일 변수 변경 실험을 수행하여 민감도를 파악했다.

![ablation_summary](../figures/ablation_sensitivity_summary_cv1.png)

### 3.2 최종 미세조정 설정
Final 설정은 다음과 같다.

- `optimizer=SGD`
- `epochs=18`, `imgsz=640`, `batch=16`
- `lr0=0.00035`, `lrf=0.02`, `momentum=0.937`, `weight_decay=0.00045`
- `box=7.8`, `cls=0.45`, `dfl=1.35`
- `mosaic=0.05`, `close_mosaic=1`, `translate=0.05`, `scale=0.35`
- `hsv_s=0.6`, `hsv_v=0.3`
- `freeze=0`, `patience=20`

설계 의도는 “강한 합성 증강 완화 + 전체 레이어 적응 + 안정적 학습률 스케줄”이다.

### 3.3 Baseline 대비 주요 하이퍼파라미터 변경
![hparams_table](../figures/baseline_vs_final_hparams_table_cv1.png)
![hparams_delta](../figures/baseline_vs_final_hparams_delta_cv1.png)

핵심 변경의 역할은 다음과 같다.

1. `optimizer auto -> SGD`: 학습 동작 고정, 재현성 개선
2. `freeze 5 -> 0`: 전체 네트워크 도메인 적응
3. `mosaic 1.0 -> 0.05`, `close_mosaic 10 -> 1`: 과도한 분포 왜곡 억제
4. `lr0 0.001 -> 0.00035`: pretrained 가중치의 급격한 붕괴 방지
5. `box/cls/dfl` 재균형: 위치 정밀도와 오탐 trade-off 조정

## 4. 실험 결과

### 4.1 기준선 대비 성능
B0(COCO raw)는 `mAP50=0.8250`, `mAP75=0.6813`, `mAP50-95=0.6245`였다.  
Final은 5-seed 평균 기준 `mAP50=0.8750`, `mAP75=0.7187`, `mAP50-95=0.6626`을 기록했다.

| 지표 | B0 | Final(5-seed mean) | Delta |
|---|---:|---:|---:|
| mAP50 | 0.8250 | 0.8750 | +0.0499 |
| mAP75 | 0.6813 | 0.7187 | +0.0374 |
| mAP50-95 | 0.6245 | 0.6626 | +0.0381 |

![metric_compare](../figures/last_squeeze_refine1_vs_b0_cv1.png)
![seed_delta](../figures/last_squeeze_seed_repeat_delta_cv1.png)

### 4.2 통계적 유의성
`mAP50-95` 기준:

- seed-level sign test(one-sided): `p=0.03125`
- seed-level sign-flip permutation(one-sided): `p=0.03125`
- seed bootstrap 95% CI: `[+0.0342, +0.0418]`
- image-level paired-bootstrap mean delta: `+0.0083`
- image-level paired-bootstrap 95% CI: `[-0.0022, +0.0202]`
- image-level paired-bootstrap(one-sided): `p=0.0612`

`mAP50` 보조 검정:

- seed-level mean delta: `+0.0499`
- image-level paired-bootstrap(one-sided): `p=0.0294`
- image-level 95% CI: `[-0.0004, +0.0226]` (경계적)


해석:
- seed-level에서는 일관된 개선이 확인된다.
- image-level 엄격 기준에서는 표본 변동성 때문에 유의성이 경계적이다.

### 4.3 Confusion Matrix 분석(4x4, 배경 포함)
![conf_counts](../figures/confusion_last_squeeze_vs_coco_4x4_counts_cv1.png)
![conf_norm](../figures/confusion_last_squeeze_vs_coco_4x4_normalized_cv1.png)

관찰된 주요 변화:

| 항목 | Baseline | Final | Delta |
|---|---:|---:|---:|
| person -> person (TP) | 160 | 163 | +3 |
| car -> car (TP) | 56 | 58 | +2 |
| dog -> dog (TP) | 13 | 15 | +2 |
| bg/other -> person (FP to person) | 106 | 88 | -18 |
| bg/other -> car (FP to car) | 25 | 43 | +18 |
| bg/other -> dog (FP to dog) | 2 | 6 | +4 |

즉, person FP는 감소했지만 car/dog 방향 FP가 증가하는 trade-off가 존재한다.

### 4.4 모델별 통계 유의성 충족 여부
판정 기준은 기본적으로 `alpha=0.05`이며, image-level 검정에서는 `95% CI가 0을 포함하지 않을 것`을 엄격 기준으로 추가했다.

| 모델/실험군 | 비교 기준 | 지표 | 검정 결과 | 판정 |
|---|---|---|---|---|
| `n_gentle_1` (COCO surpass 1차 후보) | COCO F1 | F1 delta | mean `-0.0299`, t-test(one-sided) `p=0.9994`, Wilcoxon `p=1.0` | 유의성 미충족 |
| `mstar_staged` (protocol) | B0(COCO raw) | mAP50-95 delta | bootstrap mean `-0.0020`, 95% CI `[-0.0076, +0.0039]`, one-sided `p=0.7556` | 유의성 미충족 |
| `mstar_staged` (protocol) | B1(기본 FT) | mAP50-95 delta | bootstrap mean `+0.0439`, 95% CI `[+0.0383, +0.0498]`, one-sided `p=0.0` | 유의성 충족(B1 대비) |
| `last_squeeze` 최종 | B0(COCO raw) | mAP50-95 delta (seed-level) | sign/permutation `p=0.03125`, bootstrap CI `[+0.0342, +0.0418]` | 유의성 충족(seed-level) |
| `last_squeeze` 최종 | B0(COCO raw) | mAP50-95 delta (image-level paired-bootstrap) | mean `+0.0083`, 95% CI `[-0.0022, +0.0202]`, one-sided `p=0.0612` | 엄격 기준 미충족 |
| `lg_02` 운영 2차 튜닝 | `lg_02` 운영 baseline | F1/FP 운영 개선 | F1 bootstrap 95% CI `[+0.0085, +0.0396]`, `p~0.0006` | 유의성 충족(운영 파라미터 수준) |

유의성 미충족 사례는 실제로 존재하며, 특히 다음 3가지는 명확히 “미충족”이다.

1. `n_gentle_1`의 COCO 직접 추월 주장(F1 기준)
2. `mstar_staged`의 B0 직접 우위 주장(mAP50-95 기준)
3. `last_squeeze`의 image-level 엄격 기준(mAP50-95 paired-bootstrap)

또한 초기 탐색군(`opt_*`, `aug_*`, `lg_*` 일부)은 단일 run 위주로 수행되어, p-value 기반 유의성 판정 자체를 적용하기 어렵다(반복/재표본 검정 미실시).

## 5. 시행착오 로그(최종 결과 외 전 과정)

### 5.1 단계별 시행착오 타임라인
아래 표는 “무엇을 시도했고, 왜 실패/부분성공이었는지”를 정량값과 함께 요약한 것이다.

| 단계 | 실험군 | 대표 지표 | 결과 | 판정 | 핵심 해석 |
|---|---|---|---:|---|---|
| P1 | baseline 구축 | mAP50-95 | 0.4219 | 기준선 | 양성 위주 소규모 데이터 기준선 |
| P2 | 1차 ablation | mAP50-95 | 0.4655 | 성공 | freeze/epochs 조정으로 초기 개선 |
| P3 | 초기 최적화 | mAP50-95 | 0.4708 | 성공 | lr/batch/freeze 탐색 이득은 제한적 |
| P4 | 강한 증강 탐색 | mAP50-95 | 0.4699 | 실패 | 과증강으로 분포 왜곡, 개선 정체 |
| P5 | 데이터 확장+증강 | mAP50-95 | 0.4822 | 부분성공 | 평균은 개선, 변동성은 큼 |
| P6 | 2-stage 프로토콜 | val mAP50-95 | 0.5783 | 부분성공 | val 고점, test 일반화 불확실 |
| P7 | 프로토콜 교정+last_squeeze | mAP50-95 | 0.6626 | 성공 | 공정 비교에서 B0 대비 +0.0381 |

![trial_timeline](../figures/trial_error_timeline_cv1.png)


### 5.2 대표 실패 케이스와 수정 조치
1. 이전 `best.pt` 연속 학습 이슈  
초기 일부 구간은 “같은 기준선에서 독립 비교”가 흐려질 수 있는 학습 경로가 섞여, 변수 영향 분리가 어려웠다. 이후 모든 비교를 `yolov8n.pt` 기준 독립 학습으로 재정렬했다.
2. 과한 증강 설정  
`mosaic/mixup` 강도를 높인 실험은 mAP50-95 개선이 미미하거나 역효과를 보였다(P4). 최종에서는 증강 강도를 낮추고(`mosaic=0.05`) 조기 종료(`close_mosaic=1`)로 전환했다.
3. val 고점 과신  
P7에서 val 지표가 높았지만, 후속 seed/test 검증(P8)에서 B0 대비 음수 델타가 관측되었다. 이로 인해 단일 split 고점을 채택하지 않고 seed 반복 검정을 필수화했다.
4. 운영 파라미터 단독 튜닝 한계  
임계값/운영점 조정만으로는 COCO 추월이 재현되지 않았다(P6). 이후 학습 레벨(데이터+하이퍼파라미터+프로토콜) 개선으로 방향을 전환했다.

### 5.3 시행착오가 최종 설계에 반영된 방식
최종 설계는 단일 아이디어가 아니라, 실패에서 확인된 제약을 차례로 반영한 결과다.

1. 비교 공정성: 동일 기준선과 독립 학습 프로토콜 고정
2. 일반화 우선: 단일 val 고점보다 5-seed 반복 결과를 채택
3. 데이터 분포 보정: negative 샘플 포함으로 배경 오탐 구조 학습
4. 증강 절제: 강한 합성 증강을 약화하고 후반 수렴 안정화

## 6. 논의

### 6.1 왜 성능이 상승했는가
성능 상승은 단일 요인보다는 다음의 복합 효과로 보는 것이 타당하다.

1. 학습 안정화: 낮은 `lr0`, 짧은 warmup, SGD 고정
2. 도메인 적응 강화: `freeze=0` 전체 미세조정
3. 데이터 분포 보완: 배경 negative 31% 포함으로 운영 환경 유사성 개선
4. 증강 현실화: 과도한 mosaic/기하 변환 완화

### 6.2 한계와 향후 과제
image-level paired-bootstrap에서 `p=0.0612`로 0.05를 소폭 상회했다. 엄격 기준까지 통과하려면 아래가 필요하다.

1. 테스트 표본 확대 또는 OOF 기반 재평가
2. seed 수 추가 확장(예: 10 seeds 이상)
3. FP가 증가한 car/dog 채널 중심 운영 threshold 재최적화

## 7. 결론
본 연구는 COCO pretrained YOLOv8n 대비 3-class 과제에서 최종 모델이 평균 성능을 꾸준히 상회함을 확인했다. seed-level 통계 검정에서는 유의한 개선이 관찰되었고, image-level 검정에서는 경계적 결과가 나타났다. 따라서 현재 결론은 “실무적으로 개선 신호가 분명하나, 엄격한 통계 기준 완전 충족을 위해 추가 표본 기반 검증이 필요”로 요약된다.
