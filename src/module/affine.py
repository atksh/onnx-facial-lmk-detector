import torch


@torch.jit.script
def rankup(M):
    M = torch.cat([M, torch.zeros(M.shape[0], 1, 3, dtype=M.dtype)], dim=1)
    M[:, 2, 2] = 1
    return M


@torch.jit.script
def inverse_affine(M):
    R = M[:, :2, :2]
    T = M[:, :2, 2]
    R_det = R[:, 0, 0] * R[:, 1, 1] - R[:, 0, 1] * R[:, 1, 0]
    sign = torch.sign(R_det)
    R_det_abs = torch.abs(R_det) + 1e-8
    R_inv = (
        torch.stack([R[:, 1, 1], -R[:, 0, 1], -R[:, 1, 0], R[:, 0, 0]], dim=-1).view(
            -1, 2, 2
        )
        / R_det_abs.view(-1, 1, 1)
    ) * sign.view(-1, 1, 1)
    R_inv_T = torch.bmm(R_inv, T.unsqueeze(-1))
    M_inv = torch.cat([R_inv, -R_inv_T], dim=2)
    return M_inv


@torch.jit.script
def mesh(height: int, width: int):
    h = torch.arange(height, dtype=torch.float32)
    w = torch.arange(width, dtype=torch.float32)
    grid_h = h.view(-1, 1).repeat(1, width)
    grid_w = w.view(1, -1).repeat(height, 1)
    ones = torch.ones((height, width), dtype=grid_h.dtype)
    return torch.stack([grid_h, grid_w, ones], dim=-1)


@torch.jit.script
def clip_xy(ref_xy, img_shape_x: int, img_shape_y: int):
    ref_x = torch.where((0 <= ref_xy[0]) & (ref_xy[0] < img_shape_x), ref_xy[0], -1)
    ref_y = torch.where((0 <= ref_xy[1]) & (ref_xy[1] < img_shape_y), ref_xy[1], -1)
    return torch.stack([ref_x, ref_y], dim=0)


@torch.jit.script
def affine(img, m, crop_size: int = 224):
    bsz, c, h, w = img.shape
    img = torch.cat([img, torch.zeros((bsz, c, 1, w), dtype=img.dtype)], dim=2)
    img = torch.cat([img, torch.zeros((bsz, c, h + 1, 1), dtype=img.dtype)], dim=3)

    m_inv = inverse_affine(m)
    m_inv = rankup(m_inv)
    xy_after = mesh(crop_size, crop_size)
    ref_xy = torch.einsum("hwd,bld->lbhw", xy_after, m_inv)[:2]

    # bilinear interpolation
    linear_upleft = ref_xy.long()
    linear_downleft = (
        linear_upleft + torch.tensor([1, 0], dtype=torch.long)[:, None, None, None]
    )
    linear_upright = (
        linear_upleft + torch.tensor([0, 1], dtype=torch.long)[:, None, None, None]
    )
    linear_downright = (
        linear_upleft + torch.tensor([1, 1], dtype=torch.long)[:, None, None, None]
    )

    # calculate weights
    upleft_diff = ref_xy - linear_upleft
    upleft_weight = (1 - upleft_diff[0]) * (1 - upleft_diff[1])
    downleft_weight = upleft_diff[0] * (1 - upleft_diff[1])
    upright_weight = (1 - upleft_diff[0]) * upleft_diff[1]
    downright_weight = upleft_diff[0] * upleft_diff[1]

    # clip coordinates
    linear_upleft = clip_xy(linear_upleft, w, h)
    linear_downleft = clip_xy(linear_downleft, w, h)
    linear_upright = clip_xy(linear_upright, w, h)
    linear_downright = clip_xy(linear_downright, w, h)

    # slicing with weight
    idx = (
        torch.arange(bsz, dtype=torch.long)
        .view(-1, 1, 1)
        .repeat((1, crop_size, crop_size))
    )
    img = img.permute(0, 3, 2, 1).float()
    linear_with_weight = (
        upleft_weight[:, :, :, None] * img[idx, linear_upleft[0], linear_upleft[1]]
    )
    linear_with_weight = linear_with_weight + (
        downleft_weight[:, :, :, None]
        * img[idx, linear_downleft[0], linear_downleft[1]]
    )
    linear_with_weight = linear_with_weight + (
        upright_weight[:, :, :, None] * img[idx, linear_upright[0], linear_upright[1]]
    )
    linear_with_weight = linear_with_weight + (
        downright_weight[:, :, :, None]
        * img[idx, linear_downright[0], linear_downright[1]]
    )
    img_linear = linear_with_weight.permute(0, 3, 2, 1)
    img_linear = img_linear.clip(0, 255).to(dtype=img.dtype)
    return img_linear


@torch.jit.script
def trans_points2d(pts, M):
    # M: (b, 2, 3)
    # pts: (b, c, 2)
    M = rankup(M)
    new_pts = torch.cat([pts, torch.ones_like(pts[:, :, [0]])], dim=-1)
    return torch.bmm(M, new_pts.permute(0, 2, 1))[:, :2].permute(0, 2, 1)


@torch.jit.script
def compose_affine(M1, M2):
    M1 = rankup(M1)
    M2 = rankup(M2)
    M = torch.bmm(M1, M2)
    return M[:, :2]
