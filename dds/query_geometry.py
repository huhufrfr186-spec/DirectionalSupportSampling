"""Directional support sampling: query geometry."""
import torch
from torch import nn
from torch.nn import functional as F
from .geometry import projected_shape, excess_variance, demand


class QueryGeometryHead(nn.Module):
    def __init__(self, hidden_dim, bins, ridge=1e-8):
        super().__init__()
        bins = torch.as_tensor(bins, dtype=torch.float32)
        if bins.ndim != 1 or bins.numel() < 2 or not torch.all(bins[1:] > bins[:-1]):
            raise ValueError('Residual bins must be strictly increasing')
        self.register_buffer('bins', bins)
        self.ridge = ridge
        self.center_delta = nn.Linear(hidden_dim, 2 * bins.numel())
        self.cholesky = nn.Linear(hidden_dim, 3)
        self.angle_delta = nn.Linear(hidden_dim, 1)
        nn.init.zeros_(self.center_delta.weight)
        nn.init.zeros_(self.center_delta.bias)
        nn.init.zeros_(self.cholesky.weight)
        # Initial side length 0.05 in the common normalized square.
        diagonal = .05 / (12 ** .5)
        bias = torch.log(torch.expm1(torch.tensor(diagonal))).item()
        nn.init.constant_(self.cholesky.bias, bias)
        self.cholesky.bias.data[1] = 0
        nn.init.zeros_(self.angle_delta.weight)
        nn.init.zeros_(self.angle_delta.bias)

    def forward(self, queries, reference_center, previous=None):
        logits = self.center_delta(queries).float().reshape(*queries.shape[:-1], 2, -1)
        raw = self.cholesky(queries).float()
        delta = 4 * self.angle_delta(queries).float().squeeze(-1)
        # Neural projections can use AMP; shape matrices, distributions and
        # their normalization must retain FP32 for tiny oriented targets.
        with torch.autocast(device_type=queries.device.type, enabled=False):
            return self.decode_geometry(logits, raw, delta, reference_center.float(), previous)

    def decode_geometry(self, logits, raw, delta, reference_center, previous):
        if previous is not None:
            logits = previous['center_logits'].float() + logits
        extra, residual = excess_variance(logits, self.bins)
        center = reference_center + residual
        a, b, c = F.softplus(raw[..., 0]) + 1e-6, raw[..., 1], F.softplus(raw[..., 2]) + 1e-6
        zero = torch.zeros_like(a)
        L = torch.stack((a, zero, b, c), -1).reshape(*a.shape, 2, 2)
        H = L @ L.transpose(-1, -2)
        phase = torch.stack((delta.cos(), delta.sin()), -1)
        if previous is not None:
            old = previous['phase'].float()
            phase = torch.stack((phase[..., 0] * old[..., 0] - phase[..., 1] * old[..., 1],
                                 phase[..., 1] * old[..., 0] + phase[..., 0] * old[..., 1]), -1)
        phase = F.normalize(phase, dim=-1)
        theta, sizes, sigma = projected_shape(H, phase)
        covariance = torch.diag_embed(extra)
        relative_demand = demand(covariance, sigma, self.ridge)
        consistency = (H - sigma).square().sum((-2, -1)) / H.diagonal(dim1=-2, dim2=-1).sum(-1).square().clamp_min(1e-20)
        return dict(center=center, center_logits=logits, shape_raw=H, phase=phase,
                    sizes=sizes, angle=theta, shape=sigma, localization_covariance=covariance,
                    demand=relative_demand, consistency=consistency,
                    boxes=torch.cat((center, sizes, theta[..., None]), -1))


def interpolated_center_loss(logits, residual, bins):
    """Adjacent-bin interpolation supervision for equation (6)."""
    if torch.any((residual < bins[0]) | (residual > bins[-1])):
        raise ValueError('Matched center residual lies outside the configured bins')
    left = torch.searchsorted(bins, residual.contiguous(), right=True).sub(1).clamp(0, len(bins) - 2)
    weight = (residual - bins[left]) / (bins[left + 1] - bins[left])
    logp = logits.log_softmax(-1)
    low = logp.gather(-1, left[..., None]).squeeze(-1)
    high = logp.gather(-1, (left + 1)[..., None]).squeeze(-1)
    return -((1 - weight) * low + weight * high).mean()
