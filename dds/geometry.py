"""Object geometry, excess dispersion, and directional allocation."""
import torch
from .transport import integrate_grid


def rotation(theta):
    c, s = theta.cos(), theta.sin()
    return torch.stack((c, -s, s, c), -1).reshape(*theta.shape, 2, 2)


def bounded_density(logits, floor=.15):
    if not 0 < floor < 1:
        raise ValueError('density_floor must be between zero and one')
    q = floor + (1-floor)*logits.sigmoid()
    return q/integrate_grid(q)


def projected_shape(H, phase):
    theta = torch.atan2(phase[..., 1], phase[..., 0])/4
    R = rotation(theta)
    diagonal = (R.transpose(-1,-2) @ H @ R).diagonal(dim1=-2,dim2=-1)
    sizes = (12*diagonal.clamp_min(1e-16)).sqrt()
    sigma = R @ torch.diag_embed(diagonal) @ R.transpose(-1,-2)
    return theta, sizes, sigma


def excess_variance(logits, bins):
    p = logits.softmax(-1)
    mean = (p*bins).sum(-1)
    variance = (p*(bins-mean[...,None]).square()).sum(-1)
    j = torch.searchsorted(bins, mean.contiguous(), right=True).sub(1).clamp(0,len(bins)-2)
    minimum = (mean-bins[j])*(bins[j+1]-mean)
    return (variance-minimum).clamp_min(0), mean


def demand(C, sigma, ridge=1e-8):
    with torch.autocast(device_type=sigma.device.type, enabled=False):
        dtype = torch.float64 if sigma.dtype == torch.float64 else torch.float32
        shape, covariance = sigma.to(dtype), C.to(dtype)
        shape = shape+ridge*torch.eye(2, device=shape.device, dtype=dtype)
        return torch.linalg.solve(shape, covariance).diagonal(dim1=-2,dim2=-1).sum(-1)


def metrics(transport, boxes, canvas=1024, grid=5):
    if not len(boxes):
        empty = boxes.new_empty(0)
        return {key:empty for key in ['thickness','nonaffinity','area','svd','visible']}
    B = rotation(boxes[:,4]) @ torch.diag_embed(boxes[:,2:4])
    axis = torch.linspace(-.5,.5,grid,device=boxes.device,dtype=boxes.dtype)
    y,x = torch.meshgrid(axis,axis,indexing='ij')
    xi = torch.stack((x,y),-1).reshape(-1,2)
    points = boxes[:,None,:2] + torch.einsum('nij,kj->nki',B,xi)
    valid = ((points>=0)&(points<=1)).all(-1)
    mapped,J = transport.forward(points.clamp(0,1),return_jacobian=True)
    center,J0 = transport.forward(boxes[:,:2].clamp(0,1),return_jacobian=True)
    A, A0 = J @ B[:,None], J0 @ B
    determinant = torch.linalg.det(A).abs()
    thickness = canvas*determinant/A.norm(dim=-2).amax(-1).clamp_min(1e-16)
    thickness = thickness.masked_fill(~valid,float('inf')).amin(-1)
    residual = mapped-center[:,None]-torch.einsum('nij,kj->nki',A0,xi)
    relative = torch.linalg.solve(A0[:,None],residual[...,None]).squeeze(-1).abs().amax(-1)
    nonaffinity = relative.masked_fill(~valid,0).amax(-1)
    area = (canvas*determinant.sqrt()).masked_fill(~valid,float('inf')).amin(-1)
    # Stable 2x2 minimum singular value, without eigenvector derivatives.
    largest = torch.linalg.matrix_norm(A,ord=2)
    svd = (canvas*determinant/largest.clamp_min(1e-16)).masked_fill(~valid,float('inf')).amin(-1)
    return dict(thickness=thickness, nonaffinity=nonaffinity, area=area,
                svd=svd, visible=valid.any(-1))


def allocation(transport, boxes, r=None, canvas=1024, grid=5,
               mode='thickness', use_deformation=True, tau=16., eta=.1,
               beta=1., kappa=1., lambda_e=1., epsilon=1e-8):
    if mode not in ('thickness','area','svd'):
        raise ValueError(f'Unknown allocation mode: {mode}')
    m = metrics(transport,boxes,canvas,grid)
    if not len(boxes):
        return transport.rho.sum()*0,m
    if r is None:
        r = boxes.new_zeros(len(boxes))
    bounded = (r/(1+r)).detach()
    threshold, weight = tau*(1+beta*bounded),1+kappa*bounded
    support = m[mode]
    if mode == 'area':
        aspect = boxes[:,2:4].amax(-1)/boxes[:,2:4].amin(-1).clamp_min(1e-16)
        threshold = threshold*aspect.sqrt()
    penalty = torch.log((threshold+epsilon)/(support+epsilon)).clamp_min(0).square()
    if use_deformation:
        penalty = penalty+lambda_e*(m['nonaffinity']-eta).clamp_min(0).square()
    valid = m['visible']
    return (weight[valid]*penalty[valid]).sum()/weight[valid].sum().clamp_min(epsilon),m
