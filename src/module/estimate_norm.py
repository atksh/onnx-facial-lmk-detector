from typing import Tuple

import torch

from .affine import affine, compose_affine, inverse_affine, trans_points2d
from .svd import svd2x2


@torch.jit.script
def src():
    return torch.tensor(
        [
            [
                [103.284, 100.23],
                [115.234, 99.98],
                [71.48, 138.014],
                [102.314, 178.1],
                [114.05, 179.404],
            ],
            [
                [90.062, 100.236],
                [131.136, 101.744],
                [79.354, 136.222],
                [90.354, 172.38],
                [128.492, 173.516],
            ],
            [
                [79.46, 102.276],
                [144.54, 102.276],
                [112.0, 136.986],
                [84.926, 174.02],
                [139.074, 174.02],
            ],
            [
                [93.69, 101.744],
                [134.764, 100.236],
                [145.474, 136.222],
                [96.334, 173.516],
                [134.472, 172.38],
            ],
            [
                [109.592, 99.98],
                [121.542, 100.23],
                [153.346, 138.014],
                [110.776, 179.404],
                [122.514, 178.1],
            ],
        ],
        dtype=torch.float32,
    )


# 224 -> 192
@torch.jit.script
def prepare_matrix(n: int):
    return torch.tensor(
        [[[0.57142857, 0.0, 32.0], [0.0, 0.57142857, 32.0]]],
        dtype=torch.float32,
    ).repeat((n, 1, 1))


# 192 -> 224
@torch.jit.script
def prepare_inverse_matrix(n: int):
    return torch.tensor(
        [[[1.75, -0.0, -56.0], [-0.0, 1.75, -56.0]]],
        dtype=torch.float32,
    ).repeat((n, 1, 1))


@torch.jit.script_if_tracing
def _umeyama(src, dst):
    N = src.shape[0]

    num = src.shape[1]
    dime = src.shape[2]

    # Compute mean of src and dst.
    src_mean = src.mean(dim=1, keepdim=True)
    dst_mean = dst.mean(dim=1, keepdim=True)

    # Subtract mean from src and dst.
    src_demean = src - src_mean
    dst_demean = dst - dst_mean

    # Eq. (38).
    A = torch.bmm(dst_demean.permute(0, 2, 1), src_demean) / num

    # Eq. (39).
    d = torch.ones((N, dime), dtype=torch.float)
    det_A = A[:, 0, 0] * A[:, 1, 1] - A[:, 0, 1] * A[:, 1, 0]
    d[:, dime - 1] = torch.sign(det_A)

    T = torch.eye(dime + 1, dtype=torch.float).unsqueeze(0).repeat(N, 1, 1)
    d_diag = torch.zeros((N, dime, dime), dtype=torch.float)
    d_diag[:, 0, 0] = d[:, 0]
    d_diag[:, 1, 1] = d[:, 1]

    U, S, V = svd2x2(A)

    T[:, :dime, :dime] = torch.bmm(torch.bmm(U, d_diag), V)

    scale = (
        1.0 / src_demean.var(dim=1, unbiased=False).sum(dim=-1) * (S * d).sum(dim=-1)
    )
    scale = scale.view(-1, 1, 1)

    # (bsz, dime, dime) x (bsz, dime, 1) -> (bsz, dime, 1)
    tmp = torch.bmm(T[:, :dime, :dime], src_mean.permute(0, 2, 1))
    # (bsz, dime, 1) x (bsz, dime, 1) -> (bsz, dime, 1)
    T[:, :dime, dime] = (dst_mean.permute(0, 2, 1) - scale * tmp)[:, :, 0]
    T[:, :dime, :dime] *= scale
    return T.float()


@torch.jit.script
def _estimate_norm(kpss: torch.Tensor):
    N = kpss.shape[0]
    kpss = kpss.view(N, 5, 2).float()
    lmk_tran = torch.cat([kpss, torch.ones(N, 5, 1, dtype=torch.float)], dim=2)
    tgt = src().unsqueeze(1).repeat(1, N, 1, 1)
    M0 = _umeyama(kpss, tgt[0])[:, 0:2]
    M1 = _umeyama(kpss, tgt[1])[:, 0:2]
    M2 = _umeyama(kpss, tgt[2])[:, 0:2]
    M3 = _umeyama(kpss, tgt[3])[:, 0:2]
    M4 = _umeyama(kpss, tgt[4])[:, 0:2]
    M = torch.stack([M0, M1, M2, M3, M4], dim=1)

    r0 = torch.bmm(M0, lmk_tran.permute(0, 2, 1)).permute(0, 2, 1)
    r1 = torch.bmm(M1, lmk_tran.permute(0, 2, 1)).permute(0, 2, 1)
    r2 = torch.bmm(M2, lmk_tran.permute(0, 2, 1)).permute(0, 2, 1)
    r3 = torch.bmm(M3, lmk_tran.permute(0, 2, 1)).permute(0, 2, 1)
    r4 = torch.bmm(M4, lmk_tran.permute(0, 2, 1)).permute(0, 2, 1)

    e0 = torch.sqrt(torch.sum((r0 - tgt[0]) ** 2, dim=2)).sum(dim=-1)
    e1 = torch.sqrt(torch.sum((r1 - tgt[1]) ** 2, dim=2)).sum(dim=-1)
    e2 = torch.sqrt(torch.sum((r2 - tgt[2]) ** 2, dim=2)).sum(dim=-1)
    e3 = torch.sqrt(torch.sum((r3 - tgt[3]) ** 2, dim=2)).sum(dim=-1)
    e4 = torch.sqrt(torch.sum((r4 - tgt[4]) ** 2, dim=2)).sum(dim=-1)

    e = torch.stack([e0, e1, e2, e3, e4], dim=1)
    idx = e.argmin(dim=1)
    a = torch.arange(N)
    M = M[a, idx]
    return M


class EstimateNorm(torch.nn.Module):
    def forward(
        self, xs, img
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        n = xs.shape[0]
        img = img.unsqueeze(0).permute(0, 3, 1, 2).repeat((n, 1, 1, 1))
        M = _estimate_norm(xs)

        tensor_imgs = affine(img, M, 224).float()
        imgs = tensor_imgs.permute(0, 2, 3, 1).to(dtype=torch.uint8)
        tensor_imgs = affine(tensor_imgs, prepare_matrix(n), 192)

        IM = inverse_affine(M)
        IM_composed = compose_affine(IM, prepare_inverse_matrix(n))
        return xs, IM_composed, imgs, tensor_imgs, M
