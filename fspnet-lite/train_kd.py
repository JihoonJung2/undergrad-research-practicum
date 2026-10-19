"""
경량 FSPNet 학습 (단일 GPU, Colab용). 실험 B~E를 --kd 옵션 하나로 전환.

  B: --kd none   증류 없이 정답(GT)만
  C: --kd out    + 선생 최종 마스크 따라 하기
  D: --kd mid    + 선생 중간 예측(P0~P2)까지 따라 하기
  E: --kd ta     + 선생 NL-TEM 어텐션 맵 Ta까지 따라 하기

학생 loss = 공식 multi_bce(학생, GT)
          + lam_out x BCE(학생 최종, 선생 최종)                      (C, D, E)
          + lam_out x Σ w_i BCE(학생 P_i, 선생 P_i)   i=0..2         (D, E)
          + lam_ta  x KL(선생 Ta || 학생 Ta)                          (E)

세션이 끊겨도 --save_dir의 last.pt에서 자동으로 이어서 학습함.
"""
import argparse
import csv
import json
import math
import os
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from common import TrainSet, TestSet, multi_bce, ta_kl, FeatAdapter, feat_mse
from fspnet_lite import FSPNet, BACKBONES, load_deit_imagenet, load_official_fspnet, count_params


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument('--backbone', default='tiny', choices=list(BACKBONES))
    p.add_argument('--kd', default='none', choices=['none', 'out', 'mid', 'ta', 'feat'],
                   help="'feat' = mid + 디코더 중간 특징 증류(FitNets). 'ta'는 현재 공개 선생 가중치로는 불가")
    p.add_argument('--train_dir', required=True, nargs='+', help='Imgs/ GT/ 가 있는 폴더 (여러 개 가능)')
    p.add_argument('--val_dir', default=None, help='에폭마다 MAE를 볼 테스트셋 폴더 (예: CAMO)')
    p.add_argument('--deit', default=None, help='DeiT ImageNet 가중치 .pth (학생 인코더 초기화)')
    p.add_argument('--teacher', default=None, help='공식 FSPNet 학습 완료 가중치 .pth (--kd none이면 불필요)')
    p.add_argument('--save_dir', required=True)
    p.add_argument('--epochs', type=int, default=30)
    p.add_argument('--bs', type=int, default=8)
    p.add_argument('--lr', type=float, default=1e-4, help='디코더 lr (인코더는 x0.1, 공식과 동일)')
    p.add_argument('--lam_out', type=float, default=1.0)
    p.add_argument('--lam_ta', type=float, default=1.0)
    p.add_argument('--lam_feat', type=float, default=1.0, help='특징 증류 loss 비중')
    p.add_argument('--workers', type=int, default=2)
    p.add_argument('--no_amp', action='store_true')
    p.add_argument('--limit', type=int, default=None, help='디버그용: 학습 이미지 수 제한')
    p.add_argument('--val_limit', type=int, default=None)
    p.add_argument('--max_minutes', type=float, default=None, help='이 시간이 지나면 현재 에폭 끝나고 저장 후 종료')
    p.add_argument('--seed', type=int, default=0)
    return p.parse_args()


def val_collate(b):
    return torch.stack([x[0] for x in b]), [x[1] for x in b], [x[2] for x in b]


@torch.no_grad()
def quick_mae(model, loader, dev, amp):
    model.eval()
    maes = []
    for img, gt, _ in loader:
        with torch.autocast('cuda', enabled=amp):
            pred = model(img.to(dev))[3].float()
        for b in range(pred.shape[0]):
            g = gt[b]
            p = F.interpolate(pred[b:b + 1], g.shape, mode='bilinear', align_corners=True)[0, 0].cpu().numpy()
            p = (p - p.min()) / (p.max() - p.min() + 1e-8)
            maes.append(np.abs(p - g / 255.).mean())
    model.train()
    return float(np.mean(maes))


def main():
    a = get_args()
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    os.makedirs(a.save_dir, exist_ok=True)
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    amp = dev == 'cuda' and not a.no_amp
    if dev == 'cpu':
        print('[경고] GPU가 없음. Colab에서 런타임 > 런타임 유형 변경 > GPU 선택')
    with open(os.path.join(a.save_dir, 'args.json'), 'w') as f:
        json.dump(vars(a), f, indent=2, ensure_ascii=False)

    # ---- 학생 ----
    student = FSPNet(a.backbone)
    last = os.path.join(a.save_dir, 'last.pt')
    if a.deit and not os.path.exists(last):
        load_deit_imagenet(student.encoder, a.deit)
    student = student.to(dev).train()
    print(f"[학생] {a.backbone}: 전체 {count_params(student):.1f}M / 인코더 {count_params(student.encoder):.1f}M")

    # ---- 선생 ----
    teacher = None
    if a.kd != 'none':
        assert a.teacher, '--kd를 쓰려면 --teacher 가중치가 필요함'
        assert a.kd != 'ta', '공개 선생 가중치에는 NL-TEM이 없어 Ta 증류(E)는 불가. --kd none/out/mid만 사용'
        teacher = load_official_fspnet(FSPNet('base', converter='ln'), a.teacher).to(dev).eval()
        for p in teacher.parameters():
            p.requires_grad_(False)
        print(f"[선생] FSPNet-base 고정: {count_params(teacher):.1f}M")

    adapter = None
    if a.kd == 'feat':
        d = BACKBONES[a.backbone][0]
        adapter = FeatAdapter([d, d // 2, d // 4], [768, 384, 192]).to(dev).train()
        print(f"[변환기] 1x1 conv 3개: {count_params(adapter):.2f}M (학습에만 사용)")

    # ---- 데이터 ----
    train_set = TrainSet(a.train_dir, limit=a.limit)
    loader = DataLoader(train_set, batch_size=a.bs, shuffle=True, num_workers=a.workers,
                        pin_memory=True, drop_last=True, persistent_workers=a.workers > 0)
    val_loader = None
    if a.val_dir:
        val_loader = DataLoader(TestSet(a.val_dir, limit=a.val_limit), batch_size=8, num_workers=a.workers,
                                collate_fn=val_collate)

    # ---- 옵티마이저 (인코더 lr x0.1, 공식과 동일) + cosine ----
    enc = [p for n, p in student.named_parameters() if n.startswith('encoder.')]
    dec = [p for n, p in student.named_parameters() if not n.startswith('encoder.')]
    if adapter is not None:
        dec += list(adapter.parameters())
    opt = torch.optim.Adam([{'params': enc, 'lr': a.lr * 0.1}, {'params': dec, 'lr': a.lr}])
    base_lrs = [a.lr * 0.1, a.lr]
    total_steps = a.epochs * len(loader)
    scaler = torch.amp.GradScaler('cuda', enabled=amp)

    start_ep, step, best = 0, 0, float('inf')
    if os.path.exists(last):
        ck = torch.load(last, map_location='cpu', weights_only=False)
        student.load_state_dict(ck['model'])
        opt.load_state_dict(ck['opt'])
        scaler.load_state_dict(ck['scaler'])
        start_ep, step, best = ck['epoch'] + 1, ck['step'], ck.get('best', best)
        if adapter is not None and 'adapter' in ck:
            adapter.load_state_dict(ck['adapter'])
        print(f"[재개] {start_ep}에폭부터 이어서 학습")

    log_path = os.path.join(a.save_dir, 'log.csv')
    new_log = not os.path.exists(log_path)
    logf = open(log_path, 'a', newline='')
    writer = csv.writer(logf)
    if new_log:
        writer.writerow(['epoch', 'loss_total', 'loss_gt_main', 'loss_kd_out', 'loss_kd_aux', 'val_mae', 'sec'])

    t_start = time.time()
    for ep in range(start_ep, a.epochs):
        t0 = time.time()
        sums = np.zeros(4)
        n = 0
        for img, gt in loader:
            img, gt = img.to(dev, non_blocking=True), gt.to(dev, non_blocking=True)
            # cosine lr
            for g, b in zip(opt.param_groups, base_lrs):
                g['lr'] = b * 0.5 * (1 + math.cos(math.pi * step / total_steps))

            with torch.autocast('cuda', enabled=amp):
                feat = a.kd == 'feat'
                s_out = student(img, return_ta=(a.kd == 'ta'), logits=True, return_feats=feat)
                if teacher is not None:
                    with torch.no_grad():
                        t_out = teacher(img, return_ta=(a.kd == 'ta'), return_feats=feat)
                if feat:
                    s_out, s_feats = s_out
                    t_out, t_feats = t_out
                    s_adapted = adapter(s_feats)
            # BCE는 fp16에서 불안정 → loss는 float32로 계산
            s_preds, s_tas = (s_out if a.kd == 'ta' else (s_out, None))
            s_preds = [p.float() for p in s_preds]  # sigmoid 전 logit
            loss, main = multi_bce(s_preds, gt)
            bce_l = F.binary_cross_entropy_with_logits
            l_out = l_ta = torch.zeros((), device=dev)
            if teacher is not None:
                t_preds, t_tas = (t_out if a.kd == 'ta' else (t_out, None))
                t_preds = [p.float() for p in t_preds]
                l_out = bce_l(s_preds[3], t_preds[3])  # 선생 확률을 soft target으로
                if a.kd in ('mid', 'ta', 'feat'):
                    l_out = l_out + sum(bce_l(s_preds[i], t_preds[i]) * (2 ** i) / 16 for i in range(3))
                loss = loss + a.lam_out * l_out
                if a.kd == 'ta':
                    l_ta = ta_kl(s_tas, t_tas)
                    loss = loss + a.lam_ta * l_ta
                if a.kd == 'feat':
                    l_ta = feat_mse(s_adapted, t_feats)   # 로그의 4번째 칸(kd_aux)에 기록
                    loss = loss + a.lam_feat * l_ta

            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            step += 1
            sums += [loss.item(), main.item(), l_out.item(), l_ta.item()]
            n += 1
            if n % 50 == 0:
                print(f"  ep{ep} it{n}/{len(loader)} loss={sums[0] / n:.4f} gt={sums[1] / n:.4f} "
                      f"kd_out={sums[2] / n:.4f} kd_{'feat' if a.kd == 'feat' else 'ta'}={sums[3] / n:.4f}")

        val_mae = quick_mae(student, val_loader, dev, amp) if val_loader else float('nan')
        sec = time.time() - t0
        avg = sums / max(n, 1)
        writer.writerow([ep, *[f'{x:.5f}' for x in avg], f'{val_mae:.5f}', f'{sec:.0f}'])
        logf.flush()
        print(f"[ep {ep}] loss={avg[0]:.4f} val_MAE={val_mae:.4f} ({sec / 60:.1f}분)")

        if val_loader and val_mae < best:
            best = val_mae
            torch.save(student.state_dict(), os.path.join(a.save_dir, 'best.pth'))
        ck = {'model': student.state_dict(), 'opt': opt.state_dict(), 'scaler': scaler.state_dict(),
              'epoch': ep, 'step': step, 'best': best}
        if adapter is not None:
            ck['adapter'] = adapter.state_dict()
        torch.save(ck, last + '.tmp')
        os.replace(last + '.tmp', last)  # 저장 중 끊겨도 last.pt가 깨지지 않게

        if a.max_minutes and (time.time() - t_start) / 60 > a.max_minutes and ep < a.epochs - 1:
            print(f"[중단] {a.max_minutes}분 경과. 같은 명령을 다시 실행하면 {ep + 1}에폭부터 이어짐")
            break

    logf.close()
    done = torch.load(last, map_location='cpu', weights_only=False)['epoch'] == a.epochs - 1
    if done:
        torch.save(student.state_dict(), os.path.join(a.save_dir, 'final.pth'))
    print('학습 완료 (final.pth 저장)' if done else '일시 중단 (last.pt 저장됨, 다시 실행하면 이어짐)')


if __name__ == '__main__':
    main()
