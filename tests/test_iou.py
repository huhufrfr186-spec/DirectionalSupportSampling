import numpy as np
import torch
from dds.rotated_iou import aligned_iou
from dds.losses import pairwise_iou
from dds.evaluation import box_polygon,overlap


def test_iou_matches_polygon_reference_and_has_gradients():
    rng = np.random.default_rng(42)
    a = torch.tensor(np.c_[rng.uniform(.3,.7,(32,2)),rng.uniform(.05,.4,(32,2)),rng.uniform(-2,2,32)],dtype=torch.float64,requires_grad=True)
    b = a.detach().clone();b[:,:2]+=.01;b[:,4]+=.1
    result = aligned_iou(a,b)
    reference = torch.tensor([overlap(box_polygon(x),box_polygon(y)) for x,y in zip(a.detach().numpy(),b.numpy())])
    assert torch.allclose(result,reference.double(),atol=2e-7)
    (1-result).sum().backward()
    assert torch.isfinite(a.grad).all() and a.grad.abs().sum()>0


def test_identical_disjoint_and_pairwise_equal_length():
    a = torch.tensor([[0.,0.,1.,1.,0.],[3.,3.,1.,1.,.4]])
    assert torch.allclose(aligned_iou(a,a),torch.ones(2))
    matrix = pairwise_iou(a,a)
    assert matrix.shape==(2,2)
    assert torch.allclose(matrix,torch.eye(2),atol=1e-6)
