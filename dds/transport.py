"""Differentiable positive-density triangular sampler on the unit square.

The density is bilinear within each grid cell. Integrals and their inverse
quadratics use the same representation, avoiding unrelated forward/backward
lookup grids. This implements existing conditional-CDF transport mathematics.
"""
import torch
from torch import nn
from torch.nn import functional as F


def integrate_grid(values):
    vertical = (values[:-1] + values[1:]) * (0.5 / (values.shape[0] - 1))
    marginal = vertical.sum(0)
    return ((marginal[:-1] + marginal[1:]) *
            (0.5 / (values.shape[1] - 1))).sum()



class ConditionalCDF:
    def __init__(self, density):
        if density.ndim != 2 or min(density.shape) < 2:
            raise ValueError("Expected a two-dimensional density node grid")
        if not bool(torch.isfinite(density).all() and (density > 0).all()):
            raise ValueError("Density must be finite and strictly positive")
        self.rho = density / integrate_grid(density)
        self.h, self.w = density.shape
        self.dx, self.dy = 1 / (self.w - 1), 1 / (self.h - 1)
        vertical = (self.rho[:-1] + self.rho[1:]) * (self.dy / 2)
        self.cy = torch.cat((torch.zeros_like(self.rho[:1]),
                             vertical.cumsum(0)), 0)
        self.m = self.cy[-1]
        horizontal = (self.m[:-1] + self.m[1:]) * (self.dx / 2)
        self.cx = torch.cat((self.m.new_zeros(1), horizontal.cumsum(0)), 0)

    def _cells(self, points):
        if not bool(((points >= 0) & (points <= 1)).all()):
            raise ValueError("Points outside unit square must be explicitly masked")
        x, y = points[..., 0], points[..., 1]
        ix = (x / self.dx).floor().long().clamp(0, self.w - 2)
        iy = (y / self.dy).floor().long().clamp(0, self.h - 2)
        tx, ty = x / self.dx - ix, y / self.dy - iy
        return ix, iy, tx, ty

    def forward(self, points, return_jacobian=False):
        shape = points.shape
        p = points.reshape(-1, 2)
        ix, iy, tx, ty = self._cells(p)
        m0, dm = self.m[ix], self.m[ix + 1] - self.m[ix]
        mx = m0 + dm * tx
        u = self.cx[ix] + self.dx * (m0 * tx + 0.5 * dm * tx.square())
        r0 = self.rho[iy, ix] * (1 - tx) + self.rho[iy, ix + 1] * tx
        r1 = self.rho[iy + 1, ix] * (1 - tx) + self.rho[iy + 1, ix + 1] * tx
        prefix = self.cy[iy, ix] * (1 - tx) + self.cy[iy, ix + 1] * tx
        ay = prefix + self.dy * (r0 * ty + 0.5 * (r1 - r0) * ty.square())
        v = ay / mx
        result = torch.stack((u, v), -1).reshape(shape)
        if not return_jacobian:
            return result
        dr0 = (self.rho[iy, ix + 1] - self.rho[iy, ix]) / self.dx
        dr1 = (self.rho[iy + 1, ix + 1] - self.rho[iy + 1, ix]) / self.dx
        day_dx = ((self.cy[iy, ix + 1] - self.cy[iy, ix]) / self.dx +
                  self.dy * (dr0 * ty + 0.5 * (dr1 - dr0) * ty.square()))
        dv_dx = (day_dx * mx - ay * dm / self.dx) / mx.square()
        dv_dy = (r0 + (r1 - r0) * ty) / mx
        jacobian = torch.stack((mx, torch.zeros_like(mx), dv_dx, dv_dy), -1)
        return result, jacobian.reshape(*shape[:-1], 2, 2)

    @staticmethod
    def _quadratic_fraction(a, delta, integral):
        # Solve a*t + delta*t^2/2 = integral without cancellation.
        discriminant = (a.square() + 2 * delta * integral).clamp_min(0)
        return 2 * integral / (a + discriminant.sqrt())

    def inverse(self, points):
        if not bool(((points >= 0) & (points <= 1)).all()):
            raise ValueError("Inverse input outside unit square")
        shape = points.shape
        z = points.reshape(-1, 2)
        u, v = z[:, 0].contiguous(), z[:, 1]
        ix = torch.searchsorted(self.cx.contiguous(), u, right=True) - 1
        ix = ix.clamp(0, self.w - 2)
        a = self.m[ix]
        delta = self.m[ix + 1] - a
        tx = self._quadratic_fraction(a, delta, (u - self.cx[ix]) / self.dx)
        tx = tx.clamp(0, 1)
        x = (ix + tx) * self.dx
        mx = a + delta * tx
        y_integrals = ((1 - tx[:, None]) * self.cy[:, ix].T +
                       tx[:, None] * self.cy[:, ix + 1].T)
        conditional = y_integrals / mx[:, None]
        iy = torch.searchsorted(conditional.contiguous(),
                                v[:, None].contiguous(), right=True)[:, 0] - 1
        iy = iy.clamp(0, self.h - 2)
        row = torch.arange(len(z), device=z.device)
        r0 = self.rho[iy, ix] * (1 - tx) + self.rho[iy, ix + 1] * tx
        r1 = self.rho[iy + 1, ix] * (1 - tx) + self.rho[iy + 1, ix + 1] * tx
        target = (v * mx - y_integrals[row, iy]) / self.dy
        ty = self._quadratic_fraction(r0, r1 - r0, target).clamp(0, 1)
        y = (iy + ty) * self.dy
        return torch.stack((x, y), -1).reshape(shape)

    def backward_grid(self, height, width):
        # align_corners=True and normalized endpoint coordinates are a contract.
        yy, xx = torch.meshgrid(torch.linspace(0, 1, height, dtype=self.rho.dtype,
                                              device=self.rho.device),
                                torch.linspace(0, 1, width, dtype=self.rho.dtype,
                                               device=self.rho.device), indexing="ij")
        points = torch.stack((xx, yy), -1).reshape(-1,2)
        result = []
        for chunk in points.split(16384):
            if torch.is_grad_enabled() and self.rho.requires_grad:
                from torch.utils.checkpoint import checkpoint
                inverse = checkpoint(lambda p,rho:ConditionalCDF(rho).inverse(p),
                                     chunk,self.rho,use_reentrant=False)
            else:
                inverse = self.inverse(chunk)
            result.append(inverse)
        return torch.cat(result).reshape(height,width,2)

    def resample(self, image, size):
        grid = self.backward_grid(*size).to(dtype=image.dtype)
        return F.grid_sample(image, grid[None].expand(image.shape[0], -1, -1, -1) * 2 - 1,
                             mode="bilinear", padding_mode="zeros", align_corners=True)
