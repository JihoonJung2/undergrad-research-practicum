"""데이터셋과 loss (공식 dataset.py / loss.py와 같은 규칙, 단일 GPU용으로 정리)"""
import os
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
IMG_EXT = ('.jpg', '.jpeg', '.png', '.bmp')


def _read_rgb(path):
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.


def _read_gt(path):
    return cv2.imread(path, cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.


def list_pairs(root):
    """root/Imgs/xxx.jpg 와 root/GT/xxx.png 짝 목록"""
    imgs = sorted(f for f in os.listdir(os.path.join(root, 'Imgs')) if f.lower().endswith(IMG_EXT))
    pairs = []
    for f in imgs:
        gt = os.path.join(root, 'GT', os.path.splitext(f)[0] + '.png')
        if os.path.exists(gt):
            pairs.append((os.path.join(root, 'Imgs', f), gt))
    return pairs


class TrainSet(Dataset):
    """공식과 같은 증강: 확률 1/4씩 좌우 / 상하 / 둘 다 뒤집기 / 그대로"""

    def __init__(self, roots, size=384, limit=None):
        self.pairs = [p for r in roots for p in list_pairs(r)]
        if limit:
            self.pairs = self.pairs[:limit]
        self.size = size
        print(f"[TrainSet] {len(self.pairs)}장")

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, i):
        ip, gp = self.pairs[i]
        img = cv2.resize(_read_rgb(ip), (self.size, self.size), interpolation=cv2.INTER_LINEAR)
        gt = cv2.resize(_read_gt(gp), (self.size, self.size), interpolation=cv2.INTER_LINEAR)
        r = np.random.rand()
        if r < 0.75:
            code = 1 if r < 0.25 else (0 if r < 0.5 else -1)
            img, gt = cv2.flip(img, code), cv2.flip(gt, code)
        img = (img - MEAN) / STD
        return torch.from_numpy(img.transpose(2, 0, 1).copy()), torch.from_numpy(gt[None].copy())


class TestSet(Dataset):
    """입력은 384로 줄이고, GT는 원본 크기 그대로 (평가는 원본 해상도에서)"""

    def __init__(self, root, size=384, limit=None):
        self.pairs = list_pairs(root)
        if limit:
            self.pairs = self.pairs[:limit]
        self.size = size

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, i):
        ip, gp = self.pairs[i]
        img = cv2.resize(_read_rgb(ip), (self.size, self.size), interpolation=cv2.INTER_LINEAR)
        img = (img - MEAN) / STD
        gt = (cv2.imread(gp, cv2.IMREAD_GRAYSCALE) > 127).astype(np.uint8) * 255
        return torch.from_numpy(img.transpose(2, 0, 1).copy()), gt, os.path.basename(gp)


def multi_bce(logits, target):
    """공식 loss: 최종 출력 BCE + 중간 출력 BCE x (1/16, 2/16, 4/16)
    sigmoid+BCE 대신 BCEWithLogits 사용 (수치적으로 안정, AMP에서 sigmoid 포화로 gradient가 죽는 문제 방지)"""
    bce = F.binary_cross_entropy_with_logits
    main = bce(logits[3], target)
    aux = sum(bce(logits[i], target) * (2 ** i) / 16 for i in range(3))
    return main + aux, main


def ta_kl(s_tas, t_tas, eps=1e-6):
    """Ta는 위치(576개)마다 노드 16개에 대한 확률분포(dim=1 softmax) → KL(선생 || 학생)의 평균"""
    loss = 0.
    for s, t in zip(s_tas, t_tas):
        s, t = s.float().clamp_min(eps), t.float().clamp_min(eps)
        loss = loss + (t * (t.log() - s.log())).sum(dim=1).mean()
    return loss / len(s_tas)


class FeatAdapter(torch.nn.Module):
    """특징 증류용 변환기: 학생 특징 채널(192/96/48)을 선생 채널(768/384/192)로 1x1 conv로 늘림.
    학습할 때만 쓰고 추론에선 버림."""

    def __init__(self, s_chs, t_chs):
        super().__init__()
        self.convs = torch.nn.ModuleList([torch.nn.Conv2d(s, t, 1) for s, t in zip(s_chs, t_chs)])

    def forward(self, s_feats):
        return [conv(f) for conv, f in zip(self.convs, s_feats)]


def feat_mse(s_feats_adapted, t_feats, eps=1e-5):
    """선생 특징을 채널별로 표준화(평균 0, 분산 1)한 뒤 MSE. 층마다 값 크기가 달라도 비슷한 비중이 되게."""
    loss = 0.
    for s, t in zip(s_feats_adapted, t_feats):
        s, t = s.float(), t.float()
        mean = t.mean(dim=(0, 2, 3), keepdim=True)
        std = t.std(dim=(0, 2, 3), keepdim=True)
        loss = loss + F.mse_loss(s, (t - mean) / (std + eps))
    return loss / len(t_feats)
