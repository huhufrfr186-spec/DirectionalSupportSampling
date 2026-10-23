import torch
from dds.transport import ConditionalCDF,integrate_grid
from dds.geometry import bounded_density,metrics,allocation,excess_variance,demand,rotation,projected_shape


def test_transport_identity_and_inverse():
    torch.manual_seed(5)
    points = torch.cat([torch.rand(100,2,dtype=torch.float64),torch.tensor([[0.,0.],[1.,1.],[0.,1.],[1.,0.]],dtype=torch.float64)])
    uniform = ConditionalCDF(torch.ones(9,9,dtype=torch.float64))
    assert torch.allclose(uniform.forward(points),points,atol=1e-12)
    logits = torch.randn(9,9,dtype=torch.float64)
    rho = bounded_density(logits)
    assert abs(integrate_grid(rho)-1)<1e-12
    assert rho.min()>=.15 and rho.max()<=1/.15
    mapping = ConditionalCDF(rho)
    assert torch.allclose(mapping.inverse(mapping.forward(points)),points,atol=1e-10)


def test_transport_jacobian_and_density():
    torch.manual_seed(2)
    mapping = ConditionalCDF(bounded_density(torch.randn(9,9,dtype=torch.float64)))
    points = torch.tensor([[.213,.341],[.62,.76]],dtype=torch.float64)
    _,jacobian = mapping.forward(points,True)
    actual = torch.stack([torch.autograd.functional.jacobian(mapping.forward,p) for p in points])
    assert torch.allclose(jacobian,actual,atol=1e-9)
    for p,J in zip(points,jacobian):
        ix,iy,tx,ty = mapping._cells(p)
        rho = ((1-tx)*(1-ty)*mapping.rho[iy,ix]+tx*(1-ty)*mapping.rho[iy,ix+1]
               +(1-tx)*ty*mapping.rho[iy+1,ix]+tx*ty*mapping.rho[iy+1,ix+1])
        assert torch.allclose(torch.linalg.det(J),rho,atol=1e-10)


def test_inverse_and_allocation_gradients():
    torch.manual_seed(11)
    logits = (torch.randn(9,9,dtype=torch.float64)*.1).requires_grad_()
    point = torch.tensor([[.217,.358]],dtype=torch.float64)
    assert torch.autograd.gradcheck(lambda g:ConditionalCDF(bounded_density(g)).inverse(point),
                                    (logits,),atol=2e-5,rtol=1e-3)
    mapping = ConditionalCDF(bounded_density(logits))
    boxes = torch.tensor([[.5,.5,.2,.03,.3]],dtype=torch.float64)
    loss,_ = allocation(mapping,boxes)
    loss.backward()
    assert torch.isfinite(logits.grad).all() and logits.grad.abs().sum()>0


def test_uniform_object_geometry_and_area_matching():
    mapping = ConditionalCDF(torch.ones(9,9,dtype=torch.float64))
    boxes = torch.tensor([[.5,.5,.2,.03,.3],[.4,.4,.1,.05,-.6]],dtype=torch.float64)
    m = metrics(mapping,boxes,canvas=100)
    assert torch.allclose(m['thickness'],100*boxes[:,2:4].amin(-1),atol=1e-10)
    assert m['nonaffinity'].max()<1e-12
    a,_ = allocation(mapping,boxes,canvas=100,mode='area',epsilon=0)
    d,_ = allocation(mapping,boxes,canvas=100,mode='thickness',epsilon=0)
    assert torch.allclose(a,d,atol=1e-10)


def test_excess_variance_and_coordinate_invariance():
    bins = torch.tensor([-1.,-.1,.2,1.],dtype=torch.float64)
    p = torch.tensor([0.,.7,.3,0.],dtype=torch.float64)
    extra,_ = excess_variance(p.log(),bins)
    assert extra.abs()<1e-12
    sigma = torch.tensor([[.01,.002],[.002,.02]],dtype=torch.float64)
    C = torch.diag(torch.tensor([.001,.002],dtype=torch.float64))
    J = torch.tensor([[2.,.5],[.1,.7]],dtype=torch.float64)
    assert torch.allclose(demand(C,sigma,ridge=0),demand(J@C@J.T,J@sigma@J.T,ridge=0),atol=1e-12)


def test_quarter_turn_box_equivalence():
    from dds.rotated_iou import aligned_iou
    H = torch.tensor([[[.01,.002],[.002,.02]]],dtype=torch.float64)
    theta = torch.tensor([.23],dtype=torch.float64)
    phase = torch.stack(((4*theta).cos(),(4*theta).sin()),-1)
    angle,sizes,sigma = projected_shape(H,phase)
    a = torch.cat((torch.full((1,2),.5),sizes,angle[:,None]),-1)
    b = a.clone();b[:,2:4] = a[:,[3,2]];b[:,4]+=torch.pi/2
    assert torch.allclose(aligned_iou(a,b),torch.ones(1,dtype=torch.float64),atol=1e-10)
