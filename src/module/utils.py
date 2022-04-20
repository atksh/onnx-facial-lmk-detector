import dataclasses as dc
from typing import Dict, List, Tuple

import torch
from typing_extensions import Final

from .affine import mesh, trans_points2d


@dc.dataclass
class Params:
    nms_threshold: float = 0.4
    score_threshold: float = 0.5
    strides: Tuple[int] = (8, 16, 32)
    num_anchors: int = 2
    input_size: int = 640
    max_num_boxes: int = 1000
    canvas_size: int = 1920


@torch.jit.script_if_tracing
def brg2rgb(img: torch.Tensor) -> torch.Tensor:
    """
    Convert BRG to RGB
    """
    return img[:, :, [2, 1, 0]]


@torch.jit.script_if_tracing
def nwhc2nchw(img: torch.Tensor) -> torch.Tensor:
    """
    Convert nWHC to nCHW
    """
    return img.permute(2, 0, 1)


@torch.jit.script_if_tracing
def to_tensor(img: torch.Tensor) -> torch.Tensor:
    """
    Preprocess image to tensor
    """
    img = brg2rgb(img)
    img = nwhc2nchw(img)
    img = img.unsqueeze(0)
    return img


@torch.jit.script_if_tracing
def dist2bbox(points, distance, n: int):
    preds = []
    for i in range(0, n, 2):
        px = points[:, i % 2] + distance[:, i]
        py = points[:, i % 2 + 1] + distance[:, i + 1]
        preds.append(px)
        preds.append(py)
    return torch.stack(preds, dim=-1)


@torch.jit.script_if_tracing
def _forward(
    x, y, z, ac, stride: int
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    scores = x.view(-1, 1, 1)
    bbox_preds = y * stride
    bbox_preds = torch.cat([-bbox_preds[:, [0, 1]], bbox_preds[:, [2, 3]]], dim=-1)
    kps_preds = z * stride
    bboxes = dist2bbox(ac, bbox_preds, 4)
    bboxes = bboxes.view(x.shape[0], 1, 4)
    kpss = dist2bbox(ac, kps_preds, 10)
    kpss = kpss.view(x.shape[0], 5, 2)
    return scores, bboxes, kpss


class Preprocess(torch.nn.Module):
    input_size: Final[int]
    canvas_size: Final[int]

    def __init__(self, params: Params):
        super().__init__()
        self.input_size = params.input_size
        self.canvas_size = params.canvas_size

    def forward(
        self, img: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        org_img = img.clone()
        org_size = torch.zeros(2, dtype=torch.long)
        org_size[0] = img.shape[0]
        org_size[1] = img.shape[1]
        size = [self.input_size, self.input_size]

        img = to_tensor(img)
        mu = torch.tensor([127.5, 127.5, 127.5]).view(1, 3, 1, 1)
        img = (img - mu) / 128.0

        canvas = torch.randn(
            (1, 3, self.canvas_size, self.canvas_size), dtype=torch.float32
        )
        canvas *= 0
        canvas[:, :, : org_size[0], : org_size[1]] = img
        max_size = torch.where(org_size[0] > org_size[1], org_size[0], org_size[1])
        img = canvas[:, :, :max_size, :max_size]
        img = torch.nn.functional.interpolate(
            img, size=size, mode="bilinear", align_corners=True
        )
        return img, org_size, org_img


@torch.jit.script_if_tracing
def get_ac(h: int, w: int, num_anchors: int, stride: int) -> torch.Tensor:
    h = h // stride
    w = w // stride
    anchor_center = mesh(h, w)[:, :, [1, 0]]
    anchor_center = (anchor_center * stride).view((-1, 2))
    anchor_center = torch.stack([anchor_center] * num_anchors, dim=1).view((-1, 2))
    return anchor_center


class Forward(torch.nn.Module):
    num_anchors: Final[int]
    input_size: Final[int]
    score_threshold: Final[float]
    nms_threshold: Final[float]
    strides: Final[Tuple[int]]
    fmc: Final[int]
    max_num_boxes: Final[int]

    def __init__(self, params: Params):
        super().__init__()
        self.num_anchors = params.num_anchors
        self.input_size = params.input_size
        self.strides = params.strides
        self.score_threshold = params.score_threshold
        self.nms_threshold = params.nms_threshold
        self.fmc = len(self.strides)
        self.max_num_boxes = params.max_num_boxes

    def forward(
        self, o1, o2, o3, o4, o5, o6, o7, o8, o9
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        ac0 = get_ac(
            self.input_size, self.input_size, self.num_anchors, self.strides[0]
        )
        ac1 = get_ac(
            self.input_size, self.input_size, self.num_anchors, self.strides[1]
        )
        ac2 = get_ac(
            self.input_size, self.input_size, self.num_anchors, self.strides[2]
        )
        scores0, bboxes0, kpss0 = _forward(o1, o4, o7, ac0, self.strides[0])
        scores1, bboxes1, kpss1 = _forward(o2, o5, o8, ac1, self.strides[1])
        scores2, bboxes2, kpss2 = _forward(o3, o6, o9, ac2, self.strides[2])
        scores = torch.cat([scores0, scores1, scores2], dim=0)
        bboxes = torch.cat([bboxes0, bboxes1, bboxes2], dim=0)
        kpss = torch.cat([kpss0, kpss1, kpss2], dim=0)

        a = torch.tensor(self.max_num_boxes)
        b = torch.tensor(self.nms_threshold)
        c = torch.tensor(self.score_threshold)
        bboxes = bboxes[:, :, [1, 0, 3, 2]].permute(1, 0, 2)
        scores = scores.permute(1, 2, 0)
        return scores, bboxes, kpss, a, b, c


class PostProcess(torch.nn.Module):
    input_size: Final[int]

    def __init__(self, params: Params) -> None:
        super().__init__()
        self.input_size = params.input_size

    def forward(
        self,
        org_size,
        scores,
        bboxes,
        kpss,
        selected_idx,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        bboxes = bboxes[:, :, [1, 0, 3, 2]].permute(1, 0, 2)
        scores = scores.permute(2, 0, 1)
        max_size = torch.where(org_size[0] > org_size[1], org_size[0], org_size[1])
        idx = selected_idx[:, 2]
        idx = torch.cat([torch.zeros(1).long(), idx], dim=0)
        scores = scores[idx].contiguous()
        bboxes = bboxes[idx].contiguous()
        kpss = kpss[idx].contiguous()
        bboxes[:, :, [0, 2]] = (bboxes[:, :, [0, 2]] * max_size / self.input_size).clip(
            0, org_size[1]
        )
        bboxes[:, :, [1, 3]] = (bboxes[:, :, [1, 3]] * max_size / self.input_size).clip(
            0, org_size[0]
        )
        kpss[:, :, 0] = (kpss[:, :, 0] * max_size.view(1, 1) / self.input_size).clip(
            0, org_size[1]
        )
        kpss[:, :, 1] = (kpss[:, :, 1] * max_size.view(1, 1) / self.input_size).clip(
            0, org_size[0]
        )
        scores = scores.view(-1)
        bboxes = bboxes.to(dtype=torch.long).view(-1, 4)
        kpss = kpss.to(dtype=torch.long)
        return scores, bboxes, kpss


class Output(torch.nn.Module):
    def forward(self, scores, bboxes, kpss, align_imgs, lmks, M):
        return scores[1:], bboxes[1:], kpss[1:], align_imgs[1:], lmks[1:], M[1:]


class PostLandmark(torch.nn.Module):
    def forward(self, IM, lmk):
        lmk = lmk.float()
        lmk = lmk.view(lmk.shape[0], -1, 2)
        lmk[:, :, 0:2] += 1
        lmk[:, :, 0:2] *= 192 // 2
        lmk = trans_points2d(lmk, IM)
        return lmk.long()
