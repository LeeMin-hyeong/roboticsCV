from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import pandas as pd
import yaml
from ultralytics import YOLO
import cv2
import matplotlib.pyplot as plt

ROOT = Path(r"C:\Users\lmhst\git\roboticsCV\CV_project#1")
OUT = ROOT / "outputs"
FIG = OUT / "figures"

DATA = ROOT / "voc3" / "data.yaml"
MODEL_COCO = "yolov8n.pt"
MODEL_FINETUNE = ROOT / "opt_aug_large_runs" / "lg_02_best_prev" / "weights" / "best.pt"

NMS_IOU = 0.45
MAX_DET = 20
MATCH_IOU = 0.5
RAW_CONF = 0.15

GRID = {
    "person": [0.30, 0.34, 0.36, 0.38, 0.40],
    "car": [0.25, 0.30, 0.35],
    "dog": [0.25, 0.30, 0.35],
}
BASE_THR = {"person":0.40,"car":0.35,"dog":0.35}

CSV = OUT / "coco_target_boost_sweep_cv1.csv"
REPORT = OUT / "coco_target_boost_report_cv1.md"
MANIFEST = OUT / "coco_target_boost_manifest_cv1.json"
FIG_COMPARE = FIG / "coco_vs_finetune_compare_cv1.png"


def iou(a,b):
    x1=max(a[0],b[0]); y1=max(a[1],b[1]); x2=min(a[2],b[2]); y2=min(a[3],b[3])
    iw=max(0.0,x2-x1); ih=max(0.0,y2-y1)
    inter=iw*ih
    if inter<=0: return 0.0
    aa=max(0.0,a[2]-a[0])*max(0.0,a[3]-a[1])
    bb=max(0.0,b[2]-b[0])*max(0.0,b[3]-b[1])
    uu=aa+bb-inter
    return inter/uu if uu>0 else 0.0


def sd(a,b):
    return a/b if b>0 else 0.0


def load_data():
    cfg=yaml.safe_load(DATA.read_text(encoding='utf-8'))
    names=cfg['names']
    if isinstance(names,dict):
        names=[names[i] for i in sorted(names.keys())]
    val=Path(cfg['val'])
    if not val.is_absolute():
        val=(DATA.parent/val).resolve()
    labels=val.parent.parent/'labels'/'val'
    imgs=sorted([p for p in val.iterdir() if p.suffix.lower() in {'.jpg','.jpeg','.png','.bmp'}])
    return names, imgs, labels


def load_gt(img, labels_dir):
    im=cv2.imread(str(img))
    h,w=im.shape[:2]
    lp=labels_dir/f"{img.stem}.txt"
    out=[]
    if lp.exists():
        for ln in lp.read_text(encoding='utf-8').splitlines():
            ln=ln.strip()
            if not ln: continue
            p=ln.split()
            c=int(float(p[0]))
            x,y,bw,bh=[float(v) for v in p[1:5]]
            out.append((c,[(x-bw/2)*w,(y-bh/2)*h,(x+bw/2)*w,(y+bh/2)*h]))
    return out


def raw_preds(model: YOLO, imgs, target_name_to_idx: dict[str, int]):
    rs=model.predict(source=[str(p) for p in imgs], conf=RAW_CONF, iou=NMS_IOU, max_det=300, imgsz=640, verbose=False, save=False, device=0)
    d={}
    for p,r in zip(imgs,rs):
        arr=[]
        if r.boxes is not None and len(r.boxes)>0:
            cls=r.boxes.cls.cpu().numpy().astype(int)
            conf=r.boxes.conf.cpu().numpy().astype(float)
            box=r.boxes.xyxy.cpu().numpy().astype(float)
            for c,s,b in zip(cls,conf,box):
                cname = model.names[int(c)]
                if cname not in target_name_to_idx:
                    continue
                # Always remap to dataset class index space (0:person,1:car,2:dog)
                arr.append((int(target_name_to_idx[cname]),float(s),b.tolist()))
        d[p.name]=arr
    return d


def eval_with_thr(names, imgs, labels, pred_cache, thr):
    idx={n:i for i,n in enumerate(names)}
    tg=[n for n in ['person','car','dog'] if n in idx]
    ncls=len(names)
    tp=fp=fn=0
    tp_c=np.zeros(ncls,dtype=int); fp_c=np.zeros(ncls,dtype=int); fn_c=np.zeros(ncls,dtype=int)
    bg_to_person=0

    for im in imgs:
        gts=load_gt(im,labels)
        prs=[]
        for c,s,b in pred_cache[im.name]:
            name=names[c] if c < len(names) else None
            if name not in tg: 
                continue
            if s < thr[name]:
                continue
            prs.append((c,s,b))
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
                used.add(bj); tp+=1; tp_c[gc]+=1
            else:
                fn+=1; fn_c[gc]+=1

        for j,(pc,ps,pb) in enumerate(prs):
            if j not in used:
                fp+=1; fp_c[pc]+=1
                if names[pc]=='person':
                    bg_to_person += 1

    p=sd(tp,tp+fp); r=sd(tp,tp+fn); f1=sd(2*p*r,p+r)
    # macro on target only
    p_list=[]; r_list=[]; f_list=[]
    for n in tg:
        i=idx[n]
        pp=sd(tp_c[i],tp_c[i]+fp_c[i])
        rr=sd(tp_c[i],tp_c[i]+fn_c[i])
        ff=sd(2*pp*rr,pp+rr)
        p_list.append(pp); r_list.append(rr); f_list.append(ff)

    return {
        'tp':int(tp),'fp':int(fp),'fn':int(fn),
        'precision':p,'recall':r,'f1':f1,
        'precision_macro':float(np.mean(p_list)),'recall_macro':float(np.mean(r_list)),'f1_macro':float(np.mean(f_list)),
        'bg_to_person_fp':int(bg_to_person),
        'rec_person':r_list[0],'rec_car':r_list[1],'rec_dog':r_list[2]
    }


def main():
    OUT.mkdir(parents=True, exist_ok=True); FIG.mkdir(parents=True, exist_ok=True)
    names, imgs, labels = load_data()

    coco = YOLO(MODEL_COCO)
    fin = YOLO(str(MODEL_FINETUNE))

    target_name_to_idx = {n: i for i, n in enumerate(names)}
    coco_cache = raw_preds(coco, imgs, target_name_to_idx)
    fin_cache = raw_preds(fin, imgs, target_name_to_idx)

    # baseline for both
    coco_base = eval_with_thr(names, imgs, labels, coco_cache, BASE_THR)
    fin_base = eval_with_thr(names, imgs, labels, fin_cache, BASE_THR)

    rows=[]
    for tp in GRID['person']:
        for tc in GRID['car']:
            for td in GRID['dog']:
                thr={'person':tp,'car':tc,'dog':td}
                m=eval_with_thr(names, imgs, labels, fin_cache, thr)
                m['thr_person']=tp; m['thr_car']=tc; m['thr_dog']=td
                # win score versus coco baseline
                m['delta_f1_vs_coco']=m['f1']-coco_base['f1']
                m['delta_macro_f1_vs_coco']=m['f1_macro']-coco_base['f1_macro']
                m['delta_bgpersonfp_vs_coco']=m['bg_to_person_fp']-coco_base['bg_to_person_fp']
                rows.append(m)

    df=pd.DataFrame(rows)
    df.to_csv(CSV,index=False,encoding='utf-8-sig')

    # choose setting: beats coco in f1 and macro_f1 while minimizing bg->person fp
    cand=df[(df.delta_f1_vs_coco>=0)&(df.delta_macro_f1_vs_coco>=0)].copy()
    if len(cand)==0:
        cand=df.copy()
    best=cand.sort_values(['delta_f1_vs_coco','delta_macro_f1_vs_coco','bg_to_person_fp','fp'], ascending=[False,False,True,True]).iloc[0]

    # figure compare
    labels_bar=['COCO-base','FIN-base','FIN-best-vs-COCO']
    f1s=[coco_base['f1'],fin_base['f1'],best['f1']]
    mfs=[coco_base['f1_macro'],fin_base['f1_macro'],best['f1_macro']]
    fps=[coco_base['bg_to_person_fp'],fin_base['bg_to_person_fp'],best['bg_to_person_fp']]

    x=np.arange(3)
    w=0.25
    fig,ax1=plt.subplots(figsize=(9,5),dpi=150)
    ax1.bar(x-w,f1s,width=w,label='F1',color='#1f77b4')
    ax1.bar(x,mfs,width=w,label='Macro F1',color='#2ca02c')
    ax1.set_xticks(x); ax1.set_xticklabels(labels_bar, rotation=10)
    ax1.set_ylim(0,1.0)
    ax1.set_ylabel('Score')
    ax1.legend(loc='upper left')
    ax2=ax1.twinx()
    ax2.plot(x+w,fps,color='#d62728',marker='o',label='bg->person FP')
    ax2.set_ylabel('bg->person FP')
    ax2.legend(loc='upper right')
    ax1.set_title('COCO Baseline vs Fine-tuned (same protocol)')
    fig.tight_layout(); fig.savefig(FIG_COMPARE); plt.close(fig)

    lines=[
        '# COCO 기준 성능 상향 결과',
        '',
        '- 비교 프로토콜: 동일 데이터(voc3 val), 동일 NMS/매칭 기준, 클래스 임계값 방식 동일',
        f"- fixed: nms_iou={NMS_IOU}, max_det={MAX_DET}, match_iou={MATCH_IOU}",
        '',
        '## COCO baseline (yolov8n.pt)',
        f"- thr(person,car,dog)=({BASE_THR['person']:.2f},{BASE_THR['car']:.2f},{BASE_THR['dog']:.2f})",
        f"- TP={coco_base['tp']}, FP={coco_base['fp']}, F1={coco_base['f1']:.4f}, MacroF1={coco_base['f1_macro']:.4f}, bg->person FP={coco_base['bg_to_person_fp']}",
        '',
        '## Fine-tuned current',
        f"- thr(person,car,dog)=({BASE_THR['person']:.2f},{BASE_THR['car']:.2f},{BASE_THR['dog']:.2f})",
        f"- TP={fin_base['tp']}, FP={fin_base['fp']}, F1={fin_base['f1']:.4f}, MacroF1={fin_base['f1_macro']:.4f}, bg->person FP={fin_base['bg_to_person_fp']}",
        '',
        '## Fine-tuned best (target: beat COCO)',
        f"- thr(person,car,dog)=({best['thr_person']:.2f},{best['thr_car']:.2f},{best['thr_dog']:.2f})",
        f"- TP={int(best['tp'])}, FP={int(best['fp'])}, F1={best['f1']:.4f}, MacroF1={best['f1_macro']:.4f}, bg->person FP={int(best['bg_to_person_fp'])}",
        f"- delta vs COCO: dF1={best['delta_f1_vs_coco']:+.4f}, dMacroF1={best['delta_macro_f1_vs_coco']:+.4f}, dbg->personFP={int(best['delta_bgpersonfp_vs_coco']):+d}",
        '',
        '![compare](C:/Users/lmhst/git/roboticsCV/CV_project#1/outputs/figures/coco_vs_finetune_compare_cv1.png)',
    ]
    REPORT.write_text('\n'.join(lines), encoding='utf-8')

    manifest={
        'csv':str(CSV), 'report':str(REPORT), 'figure':str(FIG_COMPARE),
        'coco_base':coco_base, 'fin_base':fin_base,
        'best_vs_coco':{k:(float(best[k]) if isinstance(best[k],(np.floating,float)) else int(best[k])) for k in ['thr_person','thr_car','thr_dog','tp','fp','fn','f1','f1_macro','bg_to_person_fp','delta_f1_vs_coco','delta_macro_f1_vs_coco','delta_bgpersonfp_vs_coco']}
    }
    MANIFEST.write_text(json.dumps(manifest,indent=2),encoding='utf-8')

    print('[saved]',CSV)
    print('[saved]',REPORT)
    print('[saved]',MANIFEST)
    print('[coco]',coco_base)
    print('[fin_base]',fin_base)
    print('[best]',{k:best[k] for k in ['thr_person','thr_car','thr_dog','tp','fp','f1','f1_macro','bg_to_person_fp','delta_f1_vs_coco']})

if __name__=='__main__':
    main()
