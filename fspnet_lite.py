"""
FSPNet 경량화 실험용 모델 (단일 파일, 최신 PyTorch에서 동작)

- 공식 FSPNet_model.py와 구조·파라미터 이름이 같아서 공식 학습 가중치를 그대로 불러올 수 있음 (선생 모델)
- 백본 폭(embed_dim)만 바꾸면 디코더 채널이 비율대로 같이 줄어듦 (학생 모델)
- forward(img, return_ta=True)이면 NL-TEM 어텐션 맵 Ta 12개도 같이 돌려줌 (증류 ③용)

공식 코드의 vit.py는 옛 timm API에 의존해서 최신 Colab에서 import가 깨지므로,
필요한 ViT 부분만 순수 PyTorch로 옮겨 옴 (파라미터 이름은 동일).
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ----------------------------------------------------------------------------
# ViT / DeiT 인코더 (공식 vit.py와 파라미터 이름 동일)
# ----------------------------------------------------------------------------
class PatchEmbed(nn.Module):
    def __init__(self, img_size=384, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        self.grid_size = (img_size // patch_size, img_size // patch_size)
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        return self.proj(x).flatten(2).transpose(1, 2)  # [B, N, C]


class Attention(nn.Module):
    def __init__(self, dim, num_heads, qkv_bias=True):
        super().__init__()
        self.num_heads = num_heads
        self.scale = (dim // num_heads) ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        x = F.scaled_dot_product_attention(q, k, v)  # softmax(qk^T * scale) v 와 동일
        return self.proj(x.transpose(1, 2).reshape(B, N, C))


class Mlp(nn.Module):
    def __init__(self, dim, hidden):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden, dim)

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))


class DropPath(nn.Module):
    def __init__(self, p=0.):
        super().__init__()
        self.p = p

    def forward(self, x):
        if self.p == 0. or not self.training:
            return x
        keep = 1 - self.p
        mask = x.new_empty((x.shape[0],) + (1,) * (x.ndim - 1)).bernoulli_(keep)
        return x * mask / keep


class Block(nn.Module):
    def __init__(self, dim, num_heads, drop_path=0.):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(dim, num_heads)
        self.drop_path = DropPath(drop_path)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = Mlp(dim, dim * 4)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class DistilledViT(nn.Module):
    """DeiT distilled ViT. 12개 블록의 출력을 전부 리스트로 돌려줌 (마지막 것만 norm 적용, 공식 코드와 동일)."""

    def __init__(self, img_size=384, embed_dim=768, depth=12, num_heads=12, drop_path_rate=0.1):
        super().__init__()
        self.patch_embed = PatchEmbed(img_size, 16, 3, embed_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.dist_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.patch_embed.num_patches + 2, embed_dim))
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.blocks = nn.ModuleList([Block(embed_dim, num_heads, dpr[i]) for i in range(depth)])
        self.norm = nn.LayerNorm(embed_dim)
        nn.init.trunc_normal_(self.pos_embed, std=.02)
        nn.init.trunc_normal_(self.cls_token, std=.02)
        nn.init.trunc_normal_(self.dist_token, std=.02)

    def forward(self, x):
        x = self.patch_embed(x)
        B = x.shape[0]
        x = torch.cat((self.cls_token.expand(B, -1, -1), self.dist_token.expand(B, -1, -1), x), dim=1)
        x = x + self.pos_embed
        feats = []
        for blk in self.blocks:
            x = blk(x)
            feats.append(x)
        feats[-1] = self.norm(x)
        return feats  # 12 x [B, 2+N, C]


def resize_pos_embed(posemb, grid_new, num_tokens=2):
    """224 입력용 위치 임베딩(14x14)을 384 입력용(24x24)으로 보간."""
    tok, grid = posemb[:, :num_tokens], posemb[0, num_tokens:]
    gs_old = int(math.sqrt(grid.shape[0]))
    grid = grid.reshape(1, gs_old, gs_old, -1).permute(0, 3, 1, 2)
    grid = F.interpolate(grid, size=grid_new, mode='bicubic', align_corners=False)
    grid = grid.permute(0, 2, 3, 1).reshape(1, grid_new[0] * grid_new[1], -1)
    return torch.cat([tok, grid], dim=1)


def load_deit_imagenet(encoder, path):
    """DeiT 공식 ImageNet 가중치(.pth)를 인코더에 로드. 입력 크기가 다르면 위치 임베딩을 보간."""
    sd = torch.load(path, map_location='cpu', weights_only=False)
    sd = sd.get('model', sd)
    if sd['pos_embed'].shape != encoder.pos_embed.shape:
        print(f"pos_embed 보간: {tuple(sd['pos_embed'].shape)} -> {tuple(encoder.pos_embed.shape)}")
        sd['pos_embed'] = resize_pos_embed(sd['pos_embed'], encoder.patch_embed.grid_size)
    msg = encoder.load_state_dict(sd, strict=False)
    missing = [k for k in msg.missing_keys]
    print(f"[DeiT 로드] missing={missing} / unexpected(head 등 무시)={len(msg.unexpected_keys)}개")


# ----------------------------------------------------------------------------
# NL-TEM (공식 코드 이름: Converter) — Ta를 꺼낼 수 있게만 수정
# ----------------------------------------------------------------------------
class GCN(nn.Module):
    def __init__(self, num_state, num_node, bias=False):
        super().__init__()
        self.conv1 = nn.Conv1d(num_node, num_node, kernel_size=1)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(num_state, num_state, kernel_size=1, bias=bias)

    def forward(self, x):
        h = self.conv1(x.permute(0, 2, 1)).permute(0, 2, 1)
        h = h - x
        return self.relu(self.conv2(h))


class Converter(nn.Module):
    def __init__(self, dim_in, dim_temp, img_size=384, mids=4):
        super().__init__()
        self.img_size, self.dim_in, self.dim_temp = img_size, dim_in, dim_temp
        self.num_n = mids * mids
        self.conv_fc = nn.Conv2d(dim_in * 2, dim_temp, kernel_size=1)
        for s in ('f1', 'f2'):
            setattr(self, f'norm_layer_{s}', nn.LayerNorm(dim_in))
            setattr(self, f'conv_{s}_Q', nn.Conv2d(dim_in, dim_temp, 1))
            setattr(self, f'conv_{s}_K', nn.Conv2d(dim_in, dim_temp, 1))
            setattr(self, f'ap_{s}', nn.AdaptiveAvgPool2d((mids + 2, mids + 2)))
            setattr(self, f'gcn_{s}', GCN(dim_temp, self.num_n))
            setattr(self, f'conv_{s}_extend', nn.Conv2d(dim_temp, dim_in, 1, bias=False))

    def _branch(self, s, f, fc_att):
        bs = f.shape[0]
        Q = getattr(self, f'conv_{s}_Q')(f).view(bs, self.dim_temp, -1)          # [B,c,576]
        K = getattr(self, f'conv_{s}_K')(f)                                       # [B,c,24,24]
        V = getattr(self, f'ap_{s}')(K * fc_att)[:, :, 1:-1, 1:-1].reshape(bs, self.dim_temp, -1)  # [B,c,16]
        ta = torch.softmax(torch.matmul(V.permute(0, 2, 1), K.reshape(bs, self.dim_temp, -1)), dim=1)  # [B,16,576]
        n_state = torch.matmul(Q, ta.permute(0, 2, 1))                            # [B,c,16]
        n_rel = getattr(self, f'gcn_{s}')(n_state)                                # [B,c,16]
        state = torch.matmul(n_rel, ta).view(bs, self.dim_temp, *f.shape[2:])     # [B,c,24,24]
        return f + getattr(self, f'conv_{s}_extend')(state), ta

    def forward(self, t1, t2):
        bs, _, chs = t1.shape
        g = self.img_size // 16
        f1 = self.norm_layer_f1(t1[:, 2:, :]).permute(0, 2, 1).reshape(bs, chs, g, g)
        f2 = self.norm_layer_f2(t2[:, 2:, :]).permute(0, 2, 1).reshape(bs, chs, g, g)
        fc = self.conv_fc(torch.cat((f1, f2), dim=1))
        fc_att = torch.softmax(fc, dim=1)[:, 1, :, :].unsqueeze(1)
        o1, ta1 = self._branch('f1', f1, fc_att)
        o2, ta2 = self._branch('f2', f2, fc_att)
        return [o1, o2], [ta1, ta2]


# ----------------------------------------------------------------------------
# FSD (공식 코드 이름: GroupFusion = AIM)
# ----------------------------------------------------------------------------
def cbr(i, o, k):
    return [nn.Conv2d(i, o, k, padding=k // 2, bias=False), nn.BatchNorm2d(o), nn.ReLU(inplace=True)]


class UpSampling2x(nn.Module):
    def __init__(self, in_chs, out_chs):
        super().__init__()
        self.up_module = nn.Sequential(*cbr(in_chs, out_chs * 4, 1), nn.PixelShuffle(2))

    def forward(self, x):
        return self.up_module(x)


class GroupFusion(nn.Module):
    def __init__(self, in_chs, out_chs, start=False):
        super().__init__()
        t = in_chs
        cin = in_chs if start else in_chs * 2
        self.gf1 = nn.Sequential(*cbr(cin, t, 1), *cbr(t, t, 3), *cbr(t, t, 3))
        self.gf2 = nn.Sequential(*cbr(t * 2, t, 1), *cbr(t, t, 3), *cbr(t, t, 3))
        self.up2x = UpSampling2x(t, out_chs)

    def forward(self, f_r, f_l):
        f12 = self.gf2(torch.cat((self.gf1(f_r), f_l), dim=1))
        return f12, self.up2x(f12)


class OutPut(nn.Module):
    def __init__(self, in_chs, scale=1):
        super().__init__()
        self.out = nn.Sequential(nn.Conv2d(in_chs, in_chs, 1, bias=False), nn.BatchNorm2d(in_chs),
                                 nn.ReLU(inplace=True), nn.Upsample(scale_factor=scale, mode='bilinear', align_corners=True),
                                 nn.Conv2d(in_chs, 1, 1))  # sigmoid는 밖에서 (fp16 포화 방지)

    def forward(self, x):
        return self.out(x).float()  # logit을 float32로 돌려줌


# ----------------------------------------------------------------------------
# 전체 모델
# ----------------------------------------------------------------------------
BACKBONES = {
    #        embed_dim, heads, DeiT ImageNet 가중치 (224 입력용, distilled)
    'base':  (768, 12, 'https://dl.fbaipublicfiles.com/deit/deit_base_distilled_patch16_384-d0272ac0.pth'),
    'small': (384, 6,  'https://dl.fbaipublicfiles.com/deit/deit_small_distilled_patch16_224-649709d9.pth'),
    'tiny':  (192, 3,  'https://dl.fbaipublicfiles.com/deit/deit_tiny_distilled_patch16_224-b40b3cf7.pth'),
}


class FSPNet(nn.Module):
    """
    backbone='base'이면 공식 FSPNet과 완전히 같은 구조 (채널 768/384/192/192).
    'small'/'tiny'면 채널이 비율대로 줄어듦 (tiny: 192/96/48/48).

    converter
      'nltem' : GitHub 코드의 NL-TEM 6개 (논문 구조, 학생용)
      'ln'    : LayerNorm만 거쳐 2D로 변환 (선생용)
                공개 학습 가중치는 GitHub 코드와 다른 초기 버전이라 NL-TEM 대신 'converter'(키 8개)가 있음.
                구조 후보 37개를 CAMO에서 비교한 결과, LayerNorm만 쓰는 형태가 가장 높았음 (Sm 0.90, 20장 기준)
    """

    def __init__(self, backbone='base', img_size=384, converter='nltem'):
        super().__init__()
        d, heads, _ = BACKBONES[backbone]
        self.backbone, self.img_size, self.conv_type = backbone, img_size, converter
        self.encoder = DistilledViT(img_size, d, 12, heads)
        c0, c1, c2 = d, d // 2, d // 4  # base: 768, 384, 192

        if converter == 'nltem':
            for i in range(6):
                setattr(self, f'group_converter_{i}', Converter(d, dim_temp=c1, img_size=img_size))
        elif converter == 'ln':
            self.converter = nn.Module()
            self.converter.norm_layer = nn.LayerNorm(d)
        else:
            raise ValueError(converter)

        self.gf1_1, self.gf1_2, self.gf1_3, self.gf1_4, self.gf1_5 = [GroupFusion(c0, c1) for _ in range(5)]
        self.gf1_6 = GroupFusion(c0, c1, start=True)
        self.gf2_1, self.gf2_2 = GroupFusion(c1, c2), GroupFusion(c1, c2)
        self.gf2_3 = GroupFusion(c1, c2, start=True)
        self.gf3_1 = GroupFusion(c2, c2)
        self.gf3_2 = GroupFusion(c2, c2, start=True)
        self.gf4_1 = GroupFusion(c2, c2, start=True)

        self.out1 = OutPut(c0, scale=16)
        self.out2 = OutPut(c1, scale=8)
        self.out3 = OutPut(c2, scale=4)
        self.out4 = OutPut(c2)

    def decode(self, f):
        f1_6_l, f2_6 = self.gf1_6(f[-1], f[-2])
        f1_5_l, f2_5 = self.gf1_5(torch.cat((f[-3], f1_6_l), 1), f[-4])
        f1_4_l, f2_4 = self.gf1_4(torch.cat((f[-5], f1_5_l), 1), f[-6])
        f1_3_l, f2_3 = self.gf1_3(torch.cat((f[-7], f1_4_l), 1), f[-8])
        f1_2_l, f2_2 = self.gf1_2(torch.cat((f[-9], f1_3_l), 1), f[-10])
        f1_1_o, f2_1 = self.gf1_1(torch.cat((f[-11], f1_2_l), 1), f[-12])
        f2_3_l, f3_3 = self.gf2_3(f2_6, f2_5)
        f2_2_l, f3_2 = self.gf2_2(torch.cat((f2_4, f2_3_l), 1), f2_3)
        f2_1_o, f3_1 = self.gf2_1(torch.cat((f2_2, f2_2_l), 1), f2_1)
        f3_2_l, f4_2 = self.gf3_2(f3_3, f3_2)
        f3_1_o, f4_1 = self.gf3_1(torch.cat((f3_2, f3_2_l), 1), f3_1)
        _, f5_1 = self.gf4_1(f4_2, f4_1)
        return f1_1_o, f2_1_o, f3_1_o, f5_1

    def forward(self, img, return_ta=False, logits=False, return_feats=False):
        """logits=True면 sigmoid 전 값을 돌려줌 (학습용, BCEWithLogits와 함께 사용)
        return_feats=True면 디코더 각 층 끝의 특징 3개(P0~P2를 만들기 직전)도 돌려줌 (특징 증류용)
          선생: [768,24,24] [384,48,48] [192,96,96] / 학생(tiny): [192,24,24] [96,48,48] [48,96,96]"""
        tokens = self.encoder(img)
        feats, tas = [], []
        if self.conv_type == 'nltem':
            for i in range(6):  # 인접 층 쌍 (0,1) (2,3) ... (10,11)
                o, t = getattr(self, f'group_converter_{i}')(tokens[2 * i], tokens[2 * i + 1])
                feats += o
                tas += t
        else:
            g = self.img_size // 16
            for t in tokens:
                B, _, C = t.shape
                feats.append(self.converter.norm_layer(t[:, 2:]).transpose(1, 2).reshape(B, C, g, g))
        g = self.decode(feats)
        preds = (self.out1(g[0]), self.out2(g[1]), self.out3(g[2]), self.out4(g[3]))  # 마지막이 최종 마스크
        if not logits:
            preds = tuple(torch.sigmoid(p) for p in preds)  # float32에서 sigmoid
        if return_feats:
            return preds, list(g[:3])
        return (preds, tas) if return_ta else preds


def load_official_fspnet(model, path):
    """공식 학습 완료 가중치(DDP로 저장돼 'module.' 접두사 있음) 로드. 선생은 FSPNet('base', converter='ln')."""
    sd = torch.load(path, map_location='cpu', weights_only=False)
    sd = sd.get('model', sd)
    sd = {(k[7:] if k.startswith('module.') else k): v for k, v in sd.items()}
    msg = model.load_state_dict(sd, strict=False)
    # converter의 attention 파라미터(token_atten, q, k, v, proj)는 'ln' 구조에서 쓰지 않음
    ignore = ('head', 'num_batches_tracked', 'converter.token_atten', 'converter.q.', 'converter.k.',
              'converter.v.', 'converter.proj.')
    missing = [k for k in msg.missing_keys if not any(s in k for s in ignore)]
    unexpected = [k for k in msg.unexpected_keys if not any(s in k for s in ignore)]
    print(f"[FSPNet 로드] missing={missing}  unexpected={unexpected}")
    assert not missing, "가중치 이름이 안 맞음. 위 missing 키를 확인할 것"
    return model


def count_params(module):
    return sum(p.numel() for p in module.parameters()) / 1e6
