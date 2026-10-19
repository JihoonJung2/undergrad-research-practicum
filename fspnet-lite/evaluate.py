"""
평가 + 무게 측정. 결과를 results.csv 한 줄로 쌓아서 A~E를 표 하나로 비교.

  # 무게만 (파라미터, GFLOPs, FPS)
  python evaluate.py --profile_only --backbone tiny

  # 선생(공식 가중치) 평가 = 실험 A
  python evaluate.py --name A_teacher --backbone base --official FSPNet_best.pth --test_dirs .../CAMO .../COD10K

  # 학생 평가 = 실험 B~E
  python evaluate.py --name B_tiny --backbone tiny --ckpt runs/B/final.pth --test_dirs ...
"""
import argparse
import csv
import os
import time

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.flop_counter import FlopCounterMode

from common import TestSet
from fspnet_lite import FSPNet, BACKBONES, load_official_fspnet, count_params


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument('--name', default='run')
    p.add_argument('--backbone', default='tiny', choices=list(BACKBONES))
    p.add_argument('--official', default=None, help='공식 FSPNet 가중치')
    p.add_argument('--ckpt', default=None, help='train_kd.py로 학습한 가중치 (final.pth / best.pth / last.pt)')
    p.add_argument('--test_dirs', nargs='*', default=[])
    p.add_argument('--save_pred', default=None, help='예측 마스크 png 저장 폴더 (선택)')
    p.add_argument('--results', default='results.csv')
    p.add_argument('--profile_only', action='store_true')
    p.add_argument('--limit', type=int, default=None)
    p.add_argument('--no_amp', action='store_true')
    return p.parse_args()


def profile(model, dev):
    """파라미터를 부위별로, GFLOPs(입력 1장), FPS(배치 1)를 잰다."""
    groups = {'encoder': 0, 'nltem': 0, 'decoder': 0}
    for n, p in model.named_parameters():
        k = 'encoder' if n.startswith('encoder.') else ('nltem' if n.startswith(('group_converter', 'converter')) else 'decoder')
        groups[k] += p.numel()
    x = torch.randn(1, 3, 384, 384, device=dev)
    model.eval()
    with torch.no_grad():
        fc = FlopCounterMode(display=False)
        with fc:
            model(x)
        gflops = fc.get_total_flops() / 2 / 1e9  # FLOPs/2 = MACs, 논문들이 흔히 'FLOPs'로 보고하는 값
        fps = float('nan')
        if dev == 'cuda':
            for _ in range(10):
                model(x)
            torch.cuda.synchronize()
            t = time.time()
            for _ in range(50):
                model(x)
            torch.cuda.synchronize()
            fps = 50 / (time.time() - t)
    r = {k + '_M': round(v / 1e6, 2) for k, v in groups.items()}
    r['total_M'] = round(sum(groups.values()) / 1e6, 2)
    r['GMACs'] = round(gflops, 1)
    r['FPS_fp32_bs1'] = round(fps, 1)
    return r


@torch.no_grad()
def evaluate(model, root, dev, amp, limit=None, save_dir=None):
    import py_sod_metrics as M
    sm, em, wfm, mae = M.Smeasure(), M.Emeasure(), M.WeightedFmeasure(), M.MAE()
    ds = TestSet(root, limit=limit)
    loader = DataLoader(ds, batch_size=1, num_workers=2)
    model.eval()
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
    for img, gt, name in loader:
        gt = gt[0].numpy()
        with torch.autocast('cuda', enabled=amp):
            pred = model(img.to(dev))[3].float()
        p = F.interpolate(pred, gt.shape, mode='bilinear', align_corners=True)[0, 0].cpu().numpy()
        p = ((p - p.min()) / (p.max() - p.min() + 1e-8) * 255).astype(np.uint8)  # 공식 test.py와 동일
        for m in (sm, em, wfm, mae):
            m.step(pred=p, gt=gt)
        if save_dir:
            cv2.imwrite(os.path.join(save_dir, name[0]), p)
    e = em.get_results()['em']
    return {'Sm': sm.get_results()['sm'], 'wFm': wfm.get_results()['wfm'],
            'meanEm': e['curve'].mean(), 'MAE': mae.get_results()['mae'], 'n': len(ds)}


def main():
    a = get_args()
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    if a.official:  # 공개 선생 가중치는 converter 구조가 달라 'ln'으로 만든다
        model = load_official_fspnet(FSPNet('base', converter='ln'), a.official)
    else:
        model = FSPNet(a.backbone)
        if a.ckpt:
            sd = torch.load(a.ckpt, map_location='cpu', weights_only=False)
            model.load_state_dict(sd.get('model', sd))
    model = model.to(dev).eval()

    row = {'name': a.name, 'backbone': a.backbone, **profile(model, dev)}
    print('[무게]', row)
    if not a.profile_only:
        for d in a.test_dirs:
            ds_name = os.path.basename(os.path.normpath(d))
            save = os.path.join(a.save_pred, ds_name) if a.save_pred else None
            r = evaluate(model, d, dev, amp=(dev == 'cuda' and not a.no_amp), limit=a.limit, save_dir=save)
            print(f'[{ds_name}] ' + ' '.join(f'{k}={v:.4f}' if k != 'n' else f'n={v}' for k, v in r.items()))
            for k, v in r.items():
                if k != 'n':
                    row[f'{ds_name}_{k}'] = round(float(v), 4)

    new = not os.path.exists(a.results)
    keys = list(row.keys())
    if not new:  # 기존 헤더와 합치기
        with open(a.results) as f:
            old = list(csv.DictReader(f))
        keys = list(dict.fromkeys([*(old[0].keys() if old else []), *keys]))
        rows = old + [row]
        with open(a.results, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(rows)
    else:
        with open(a.results, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerow(row)
    print(f'→ {a.results}에 저장')


if __name__ == '__main__':
    main()
