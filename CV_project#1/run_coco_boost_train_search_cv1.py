from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from ultralytics import YOLO
import cv2

ROOT = Path(r"C:\Users\lmhst\git\roboticsCV\CV_project#1")
OUT = ROOT / "outputs"
RUNS = ROOT / "coco_boost_runs"

DATA_EVAL = ROOT / "voc3" / "data.yaml"
DATA_TRAIN = ROOT / "voc3_large" / "data.yaml"
COCO_MODEL = "yolov8n.pt"

NMS_IOU = 0.45
MAX_DET = 20
MATCH_IOU = 0.5
RAW_CONF = 0.10

GRID = {
    "person": [0.15, 0.20, 0.25, 0.30, 0.35, 0.40],
    "car": [0.15, 0.20, 0.25, 0.30, 0.35],
    "dog": [0.15, 0.20, 0.25, 0.30, 0.35],
}
BASE_THR = {"person": 0.40, "car": 0.35, "dog": 0.35}

CANDIDATES = [
    {
        "tag": "cb_r1",
        "epochs": 30,
        "batch": 16,
        "imgsz": 640,
        "lr0": 0.0012,
        "freeze": 0,
        "mosaic": 0.8,
        "mixup": 0.1,
        "close_mosaic": 3,
    },
    {
        "tag": "cb_r2",
        "epochs": 40,
        "batch": 16,
        "imgsz": 640,
        "lr0": 0.0010,
        "freeze": 0,
        "mosaic": 1.0,
        "mixup": 0.2,
        "close_mosaic": 2,
    },
]

CSV = OUT / "coco_boost_train_search_cv1.csv"
REPORT = OUT / "coco_boost_train_search_report_cv1.md"
MANIFEST = OUT / "coco_boost_train_search_manifest_cv1.json"


def iou(a,b):
    x1=max(a[0],b[0]); y1=max(a[1],b[1]); x2=min(a[2],b[2]); y2=min(a[3],b[3])
    iw=max(0.0,x2-x1); ih=max(0.0,y2-y1)
    inter=iw*ih
    if inter<=0: return 0.0
    aa=max(0.0,a[2]-a[0])*max(0.0,a[3]-a[1]); bb=max(0.0,b[2]-b[0])*max(0.0,b[3]-b[1])
    u=aa+bb-inter
    return inter/u if u>0 else 0.0


def sd(a,b):
    return a/b if b>0 else 0.0


def load_eval_data():
    cfg=yaml.safe_load(DATA_EVAL.read_text(encoding='utf-8'))
    names=cfg['names']
    if isinstance(names,dict): names=[names[i] for i in sorted(names.keys())]
    val=Path(cfg['val'])
    if not val.is_absolute(): val=(DATA_EVAL.parent/val).resolve()
    labels=val.parent.parent/'labels'/'val'
    imgs=sorted([p for p in val.iterdir() if p.suffix.lower() in {'.jpg','.jpeg','.png','.bmp'}])
    return names, imgs, labels


def load_gt(img, labels):
    im=cv2.imread(str(img)); h,w=im.shape[:2]
    lp=labels/f"{img.stem}.txt"
    out=[]
    if lp.exists():
        for ln in lp.read_text(encoding='utf-8').splitlines():
            ln=ln.strip()
            if not ln: continue
            p=ln.split(); c=int(float(p[0])); x,y,bw,bh=[float(v) for v in p[1:5]]
            out.append((c,[(x-bw/2)*w,(y-bh/2)*h,(x+bw/2)*w,(y+bh/2)*h]))
    return out


def cache_preds(model_path_or_name, names, imgs):
    m=YOLO(str(model_path_or_name))
    name2idx={n:i for i,n in enumerate(names)}
    rs=m.predict(source=[str(p) for p in imgs], conf=RAW_CONF, iou=NMS_IOU, max_det=300, imgsz=640, verbose=False, save=False, device=0)
    out={}
    for p,r in zip(imgs,rs):
        arr=[]
        if r.boxes is not None and len(r.boxes)>0:
            cls=r.boxes.cls.cpu().numpy().astype(int)
            conf=r.boxes.conf.cpu().numpy().astype(float)
            box=r.boxes.xyxy.cpu().numpy().astype(float)
            for c,s,b in zip(cls,conf,box):
                cname=m.names[int(c)]
                if cname not in name2idx: continue
                arr.append((name2idx[cname], float(s), b.tolist()))
        out[p.name]=arr
    return out


def eval_with_thr(names, imgs, labels, cache, thr):
    tp=fp=fn=0
    tpc=[0,0,0]; fpc=[0,0,0]; fnc=[0,0,0]
    bgp=0
    for img in imgs:
        gts=load_gt(img,labels)
        prs=[]
        for c,s,b in cache[img.name]:
            t=thr[names[c]]
            if s>=t: prs.append((c,s,b))
        prs.sort(key=lambda x:x[1], reverse=True)
        prs=prs[:MAX_DET]

        used=set()
        for gc,gb in gts:
            bj=-1; bi=0.0
            for j,(pc,ps,pb) in enumerate(prs):
                if j in used or pc!=gc: continue
                v=iou(gb,pb)
                if v>=MATCH_IOU and v>bi:
                    bi=v; bj=j
            if bj>=0:
                used.add(bj); tp+=1; tpc[gc]+=1
            else:
                fn+=1; fnc[gc]+=1
        for j,(pc,ps,pb) in enumerate(prs):
            if j not in used:
                fp+=1; fpc[pc]+=1
                if names[pc]=='person': bgp+=1

    p=sd(tp,tp+fp); r=sd(tp,tp+fn); f1=sd(2*p*r,p+r)
    fms=[]
    for i in range(3):
        pp=sd(tpc[i],tpc[i]+fpc[i]); rr=sd(tpc[i],tpc[i]+fnc[i]); ff=sd(2*pp*rr,pp+rr)
        fms.append(ff)
    return {'tp':tp,'fp':fp,'fn':fn,'f1':f1,'f1_macro':sum(fms)/3,'bg_to_person_fp':bgp}


def sweep_best(names, imgs, labels, cache):
    best=None
    for pt in GRID['person']:
        for ct in GRID['car']:
            for dt in GRID['dog']:
                thr={'person':pt,'car':ct,'dog':dt}
                m=eval_with_thr(names, imgs, labels, cache, thr)
                row={'thr_person':pt,'thr_car':ct,'thr_dog':dt,**m}
                if best is None or (row['f1']>best['f1']) or (abs(row['f1']-best['f1'])<1e-12 and row['fp']<best['fp']):
                    best=row
    return best


def train_candidate(cfg):
    tag=cfg['tag']
    exp_dir=RUNS/tag
    if exp_dir.exists():
        shutil.rmtree(exp_dir)
    model=YOLO(COCO_MODEL)
    model.train(
        data=str(DATA_TRAIN),
        model=COCO_MODEL,
        epochs=int(cfg['epochs']),
        imgsz=int(cfg['imgsz']),
        batch=int(cfg['batch']),
        lr0=float(cfg['lr0']),
        freeze=int(cfg['freeze']),
        mosaic=float(cfg['mosaic']),
        mixup=float(cfg['mixup']),
        close_mosaic=int(cfg['close_mosaic']),
        project=str(RUNS),
        name=tag,
        exist_ok=True,
        verbose=False,
        workers=0,
        seed=42,
        deterministic=True,
        resume=False,
    )
    best=exp_dir/'weights'/'best.pt'
    if not best.exists():
        best=exp_dir/'weights'/'last.pt'
    return best


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    RUNS.mkdir(parents=True, exist_ok=True)

    names, imgs, labels = load_eval_data()

    coco_cache = cache_preds(COCO_MODEL, names, imgs)
    coco_best = sweep_best(names, imgs, labels, coco_cache)
    coco_base = eval_with_thr(names, imgs, labels, coco_cache, BASE_THR)

    rows=[]
    best_global=None

    for cfg in CANDIDATES:
        print('[train]', cfg)
        pt=train_candidate(cfg)
        cache=cache_preds(pt, names, imgs)
        fin_base=eval_with_thr(names, imgs, labels, cache, BASE_THR)
        fin_best=sweep_best(names, imgs, labels, cache)

        row={
            'tag':cfg['tag'],'best_pt':str(pt),
            'train_epochs':cfg['epochs'],'train_lr0':cfg['lr0'],'train_freeze':cfg['freeze'],'train_mosaic':cfg['mosaic'],'train_mixup':cfg['mixup'],
            'fin_base_f1':fin_base['f1'],'fin_base_fp':fin_base['fp'],'fin_base_tp':fin_base['tp'],
            'fin_best_f1':fin_best['f1'],'fin_best_fp':fin_best['fp'],'fin_best_tp':fin_best['tp'],'fin_best_bgpersonfp':fin_best['bg_to_person_fp'],
            'fin_best_thr_person':fin_best['thr_person'],'fin_best_thr_car':fin_best['thr_car'],'fin_best_thr_dog':fin_best['thr_dog'],
            'delta_f1_vs_coco_best':fin_best['f1']-coco_best['f1'],
            'delta_tp_vs_coco_best':fin_best['tp']-coco_best['tp'],
            'delta_fp_vs_coco_best':fin_best['fp']-coco_best['fp'],
        }
        rows.append(row)
        if best_global is None or row['fin_best_f1']>best_global['fin_best_f1']:
            best_global=row

    pd.DataFrame(rows).to_csv(CSV,index=False,encoding='utf-8-sig')

    lines=[
        '# COCO 기준 성능 상향 - 재학습 탐색 결과',
        '',
        f"- COCO(best-thr sweep): TP={coco_best['tp']}, FP={coco_best['fp']}, F1={coco_best['f1']:.4f}, thr=({coco_best['thr_person']:.2f},{coco_best['thr_car']:.2f},{coco_best['thr_dog']:.2f})",
        f"- COCO(base-thr 0.40/0.35/0.35): TP={coco_base['tp']}, FP={coco_base['fp']}, F1={coco_base['f1']:.4f}",
        '',
        '## 후보 결과',
    ]
    for r in rows:
        lines += [
            f"- {r['tag']}: fin_best F1={r['fin_best_f1']:.4f} (TP={int(r['fin_best_tp'])}, FP={int(r['fin_best_fp'])}), thr=({r['fin_best_thr_person']:.2f},{r['fin_best_thr_car']:.2f},{r['fin_best_thr_dog']:.2f}), dF1_vs_COCO_best={r['delta_f1_vs_coco_best']:+.4f}",
        ]

    lines += [
        '',
        '## 결론',
        f"- best candidate: {best_global['tag']} (F1={best_global['fin_best_f1']:.4f})",
        f"- vs COCO(best): dF1={best_global['delta_f1_vs_coco_best']:+.4f}, dTP={int(best_global['delta_tp_vs_coco_best'])}, dFP={int(best_global['delta_fp_vs_coco_best'])}",
    ]
    REPORT.write_text('\n'.join(lines), encoding='utf-8')

    manifest={
        'csv':str(CSV), 'report':str(REPORT),
        'coco_best':coco_best, 'coco_base':coco_base,
        'best_candidate':best_global,
    }
    MANIFEST.write_text(json.dumps(manifest,indent=2),encoding='utf-8')

    print('[saved]',CSV)
    print('[saved]',REPORT)
    print('[saved]',MANIFEST)
    print('[coco_best]',coco_best)
    print('[best_candidate]',best_global)

if __name__=='__main__':
    main()
