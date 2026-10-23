"""Differentiable convex polygon intersection for oriented boxes."""
import torch


def corners(boxes):
    xy, wh, a = boxes[..., :2], boxes[..., 2:4], boxes[..., 4]
    local = boxes.new_tensor([[-.5, -.5], [.5, -.5], [.5, .5], [-.5, .5]]) * wh.unsqueeze(-2)
    c, s = a.cos(), a.sin()
    x = local[..., 0] * c.unsqueeze(-1) - local[..., 1] * s.unsqueeze(-1)
    y = local[..., 0] * s.unsqueeze(-1) + local[..., 1] * c.unsqueeze(-1)
    return torch.stack((x, y), -1) + xy.unsqueeze(-2)


def cross(a, b):
    return a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]


def inside(points, rect):
    origin = rect[..., :1, :]
    ex, ey = rect[..., 1:2, :] - origin, rect[..., 3:4, :] - origin
    delta = points - origin
    tx = (delta * ex).sum(-1) / ex.square().sum(-1).clamp_min(1e-30)
    ty = (delta * ey).sum(-1) / ey.square().sum(-1).clamp_min(1e-30)
    eps = 1e-10 if points.dtype == torch.float64 else 1e-6
    return (tx >= -eps) & (tx <= 1 + eps) & (ty >= -eps) & (ty <= 1 + eps)


def aligned_iou(boxes1, boxes2):
    """Same shape (...,5) inputs -> (...) aligned IoU, including broadcast pairs."""
    if boxes1.shape != boxes2.shape:
        raise ValueError('aligned_iou requires identical input shapes')
    shape = boxes1.shape[:-1]
    if boxes1.numel() == 0:
        return boxes1.new_zeros(shape)
    b1, b2 = boxes1.reshape(-1, 5), boxes2.reshape(-1, 5)
    a, b = corners(b1), corners(b2)
    # Four by four segment intersections; never divide by zero.
    p, q = a.unsqueeze(-2), b.unsqueeze(-3)
    r = (a.roll(-1, -2) - a).unsqueeze(-2)
    s = (b.roll(-1, -2) - b).unsqueeze(-3)
    den = cross(r, s)
    eps = torch.finfo(den.dtype).eps * 16
    parallel = den.abs() <= eps * (r.norm(dim=-1) * s.norm(dim=-1)).clamp_min(1e-30)
    safe = torch.where(parallel, torch.ones_like(den), den)
    t, u = cross(q-p, s)/safe, cross(q-p, r)/safe
    valid = (~parallel) & (t >= 0) & (t <= 1) & (u >= 0) & (u <= 1)
    inter = p + t.unsqueeze(-1) * r
    vertices = torch.cat((a, b, inter.flatten(1, 2)), 1)
    mask = torch.cat((inside(a, b), inside(b, a), valid.flatten(1, 2)), 1)
    count = mask.sum(-1)
    centroid = (vertices * mask.unsqueeze(-1)).sum(1) / count.clamp_min(1).unsqueeze(-1)
    centered = vertices - centroid.unsqueeze(1)
    # Sorting indices deliberately carry no gradient; gathered vertices do.
    angles = torch.atan2(centered[..., 1], centered[..., 0]).detach()
    angles = torch.where(mask, angles, torch.full_like(angles, float('inf')))
    order = angles.argsort(-1)
    verts = centered.gather(1, order.unsqueeze(-1).expand(-1, -1, 2))
    index = torch.arange(24, device=verts.device).unsqueeze(0).expand(len(verts), -1)
    next_index = torch.remainder(index + 1, count.clamp_min(1).unsqueeze(-1))
    nxt = verts.gather(1, next_index.unsqueeze(-1).expand(-1, -1, 2))
    area = (cross(verts, nxt) * (index < count.unsqueeze(-1))).sum(-1).abs() * .5
    area1 = b1[:, 2] * b1[:, 3]
    area2 = b2[:, 2] * b2[:, 3]
    return (area / (area1 + area2 - area).clamp_min(1e-30)).clamp(0, 1).reshape(shape)
