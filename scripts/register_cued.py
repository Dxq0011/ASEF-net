"""Register CUED night images to paired day reference by SIFT+RANSAC.

Outputs RGB-aligned low, copied high reference and common valid-mask PNGs;
does not perform any exposure correction. The same mask must be reused
across all benchmark methods.
"""
from __future__ import annotations
import argparse,csv,shutil
from pathlib import Path
import cv2
import numpy as np
from asef.data import EXTENSIONS


def register(low_path:Path,ref_path:Path,ratio:float=.75,reproj:float=3.0):
    low=cv2.imread(str(low_path)); ref=cv2.imread(str(ref_path))
    if low is None or ref is None:raise ValueError('Unrecognized image format')
    detector=cv2.SIFT_create()
    a,des_a=detector.detectAndCompute(cv2.cvtColor(low,cv2.COLOR_BGR2GRAY),None)
    b,des_b=detector.detectAndCompute(cv2.cvtColor(ref,cv2.COLOR_BGR2GRAY),None)
    if des_a is None or des_b is None: raise ValueError('Too few SIFT features')
    matches=cv2.BFMatcher(cv2.NORM_L2).knnMatch(des_a,des_b,k=2)
    good=[m for pair in matches if len(pair)==2 for m,n in [pair] if m.distance < ratio*n.distance]
    if len(good)<8:raise ValueError(f'Only {len(good)} reliable SIFT matches')
    src=np.float32([a[m.queryIdx].pt for m in good]).reshape(-1,1,2)
    dst=np.float32([b[m.trainIdx].pt for m in good]).reshape(-1,1,2)
    h,inliers=cv2.findHomography(src,dst,cv2.RANSAC,ransacReprojThreshold=reproj)
    if h is None:raise ValueError('RANSAC failed to find homography')
    height,width=ref.shape[:2]
    registered=cv2.warpPerspective(low,h,(width,height),flags=cv2.INTER_LINEAR)
    valid=cv2.warpPerspective(np.full(low.shape[:2],255,dtype=np.uint8),h,(width,height),flags=cv2.INTER_NEAREST)
    return registered,valid,len(good),int(inliers.sum())


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--low-dir',required=True)
    ap.add_argument('--high-dir',required=True)
    ap.add_argument('--out-dir',required=True)
    ap.add_argument('--reprojection-threshold',type=float,default=3.0)
    args=ap.parse_args()
    ld,hd,out=map(Path,[args.low_dir,args.high_dir,args.out_dir])
    for sub in ('low','high','valid_masks'): (out/sub).mkdir(parents=True,exist_ok=True)
    highs={p.stem:p for p in hd.iterdir() if p.suffix.lower() in EXTENSIONS}
    lows={p.stem:p for p in ld.iterdir() if p.suffix.lower() in EXTENSIONS}
    if highs.keys()!=lows.keys():raise ValueError('Input and reference image IDs do not match')
    report=[]
    for stem in sorted(lows):
        row={'image_id':stem,'status':'failed','matches':0,'inliers':0,'valid_mask_ratio':''}
        try:
            image,mask,matches,inliers=register(lows[stem],highs[stem],reproj=args.reprojection_threshold)
            cv2.imwrite(str(out/'low'/f'{stem}.png'),image)
            shutil.copy2(highs[stem],out/'high'/f'{stem}{highs[stem].suffix}')
            cv2.imwrite(str(out/'valid_masks'/f'{stem}.png'),mask)
            row.update(status='ok',matches=matches,inliers=inliers,valid_mask_ratio=float(mask.mean()/255.))
        except Exception as exc:row['error']=str(exc)
        report.append(row)
    with (out/'registration_report.csv').open('w',newline='',encoding='utf8') as f:
        fields=['image_id','status','matches','inliers','valid_mask_ratio','error']
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader();writer.writerows(report)
    successes=sum(r['status']=='ok' for r in report)
    print(f'Registered {successes}/{len(report)} image pairs; see registration_report.csv')
    if successes != len(report):raise SystemExit(2)

if __name__=='__main__':main()
