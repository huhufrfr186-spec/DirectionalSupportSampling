"""Shared final Hungarian assignment and equations (18)--(20)."""
import torch
from torch import nn
from torch.nn import functional as F
from scipy.optimize import linear_sum_assignment
from .rotated_iou import aligned_iou, corners
from .geometry import rotation, allocation
from .query_geometry import interpolated_center_loss
from .semantic_alignment import semantic_alignment


def pairwise_iou(a,b,chunk=32):
    if not len(a) or not len(b):
        return a.new_zeros(len(a),len(b))
    return torch.cat([aligned_iou(x[:,None].expand(-1,len(b),5),
                                  b[None].expand(len(x),-1,5)) for x in a.split(chunk)])


def chamfer_cost(a,b):
    distance = (corners(a)[:,None,:,None]-corners(b)[None,:,None]).square().sum(-1)
    return distance.amin(-1).mean(-1)+distance.amin(-2).mean(-1)


def focal_loss(logits,target,alpha=.75,gamma=2.):
    logits,target = logits.float(),target.float()
    p = logits.sigmoid()
    pt = p*target+(1-p)*(1-target)
    weights = alpha*target+(1-alpha)*(1-target)
    return F.binary_cross_entropy_with_logits(logits,target,reduction='none')*(1-pt).pow(gamma)*weights


def query_classification_loss(logits,boxes,binary,targets,count):
    values = focal_loss(logits,binary)
    with torch.no_grad():
        valid = torch.ones_like(logits[...,0])
        for image,target in enumerate(targets):
            ignored = target.get('ignored_boxes')
            if ignored is not None and len(ignored):
                overlaps = pairwise_iou(boxes[image].float(),ignored.float()).amax(-1)
                # A matched foreground label always retains its supervision.
                valid[image] = ((overlaps<.5)|binary[image].bool().any(-1)).to(valid)
    return (values*valid[...,None]).sum()/count


@torch.no_grad()
def match(outputs,targets,weights):
    assignments = []
    for logits,boxes,target in zip(outputs['pred_logits'],outputs['pred_boxes'],targets):
        truth, labels = target['boxes'].to(boxes),target['labels'].to(boxes.device)
        if not len(truth):
            empty = torch.empty(0,device=boxes.device,dtype=torch.long)
            assignments.append((empty,empty))
            continue
        p = logits.float().sigmoid()[:,labels]
        positive = .25*(1-p).square()*(-p.clamp_min(1e-8).log())
        negative = .75*p.square()*(-(1-p).clamp_min(1e-8).log())
        cost = weights['class']*(positive-negative)
        cost = cost+weights['iou']*(1-pairwise_iou(boxes.float(),truth.float()))
        cost = cost+weights['vertex']*chamfer_cost(boxes.float(),truth.float())
        rows,columns = linear_sum_assignment(cost.cpu().numpy())
        assignments.append((torch.as_tensor(rows,device=boxes.device),
                            torch.as_tensor(columns,device=boxes.device)))
    return assignments


def state_losses(state,targets,assignments,bins):
    logits = state['query_logits'].float()
    binary = torch.zeros_like(logits)
    selected,truth,refs,distributions,shapes,phases,consistency = [],[],[],[],[],[],[]
    for batch,(rows,cols) in enumerate(assignments):
        if not len(rows):
            continue
        target = targets[batch]
        binary[batch,rows,target['labels'][cols]] = 1
        truth.append(target['boxes'][cols].float())
        for collection,name in [(selected,'boxes'),(refs,'reference_center'),
            (distributions,'center_logits'),(shapes,'shape'),(phases,'phase'),
            (consistency,'consistency')]:
            collection.append(state[name][batch,rows].float())
    count = max(sum(len(r) for r,_ in assignments),1)
    zero = state['boxes'].sum()*0
    losses = dict(classification=query_classification_loss(logits,state['boxes'],binary,targets,count),
                  center=zero,shape=zero,phase=zero,box=zero,consistency=zero)
    if not selected:
        return losses
    predicted,target = torch.cat(selected),torch.cat(truth)
    R = rotation(target[:,4])
    target_shape = R @ torch.diag_embed(target[:,2:4].square()/12) @ R.transpose(-1,-2)
    scale = target_shape.diagonal(dim1=-2,dim2=-1).sum(-1).square().clamp_min(1e-20)
    losses['shape'] = ((torch.cat(shapes)-target_shape).square().sum((-2,-1))/scale).mean()
    angle = 4*target[:,4]
    target_phase = torch.stack((angle.cos(),angle.sin()),-1)
    losses['phase'] = (1-(torch.cat(phases)*target_phase).sum(-1)).mean()
    losses['center'] = interpolated_center_loss(torch.cat(distributions),target[:,:2]-torch.cat(refs),bins)
    losses['box'] = (1-aligned_iou(predicted,target)).mean()
    losses['consistency'] = torch.cat(consistency).mean()
    return losses


def center_heatmap_loss(logits,targets):
    binary = torch.zeros_like(logits)
    visible = torch.ones_like(logits)
    _,_,height,width = logits.shape
    yy,xx = torch.meshgrid(torch.arange(height,device=logits.device),
                          torch.arange(width,device=logits.device),indexing='ij')
    for image,target in enumerate(targets):
        for ignored in target.get('ignored_boxes',[]):
            # Mask the oriented ignored region in the auxiliary center map.
            dx,dy = (xx+.5)/width-ignored[0],(yy+.5)/height-ignored[1]
            c,s = ignored[4].cos(),ignored[4].sin()
            inside = ((c*dx+s*dy).abs()<=ignored[2]/2)&((-s*dx+c*dy).abs()<=ignored[3]/2)
            visible[image,:,inside] = 0
        for box,label in zip(target['boxes'],target['labels']):
            cx = (box[0]*width).floor().clamp(0,width-1)
            cy = (box[1]*height).floor().clamp(0,height-1)
            base = (box[2]*width*box[3]*height).sqrt()
            radius = (2*torch.log1p(base)).floor().clamp_min(1)
            sigma = (2*radius+1)/6
            kernel = (-((xx-cx).square()+(yy-cy).square())/(2*sigma.square())).exp()
            kernel = kernel*((xx-cx).abs()<=radius)*((yy-cy).abs()<=radius)
            binary[image,label] = torch.maximum(binary[image,label],kernel)
    visible = torch.where(binary>0,torch.ones_like(visible),visible)
    return (focal_loss(logits,binary)*visible).sum()/binary.sum().clamp_min(1)


class DetectionCriterion(nn.Module):
    def __init__(self,bins,weights,allocation_options):
        super().__init__()
        self.register_buffer('bins',bins.detach().clone())
        self.weights,self.options = weights,allocation_options

    def forward(self,outputs,targets,teacher=None):
        assignments = match(outputs,targets,self.weights['matching'])
        losses = {}
        states = outputs['context_geometry_layers']+outputs['geometry_layers']
        for index,state in enumerate(states):
            values = state_losses(state,targets,assignments,self.bins)
            for name,value in values.items():
                losses[f'layer{index}_{name}'] = value*self.weights[name]
        if 'initial_prediction' in outputs:
            initial = outputs['initial_prediction']
            binary = torch.zeros_like(initial['pred_logits'])
            predicted,truth = [],[]
            for batch,(rows,cols) in enumerate(assignments):
                binary[batch,rows,targets[batch]['labels'][cols]] = 1
                predicted.append(initial['pred_boxes'][batch,rows])
                truth.append(targets[batch]['boxes'][cols])
            count = max(sum(len(r) for r,_ in assignments),1)
            losses['initial_classification'] = query_classification_loss(initial['pred_logits'],
                initial['pred_boxes'],binary,targets,count)
            if sum(len(p) for p in predicted):
                a,b = torch.cat(predicted),torch.cat(truth)
                losses['initial_geometry'] = self.weights['initialization']*(
                    F.l1_loss(a[:,:4],b[:,:4])+ (1-aligned_iou(a,b)).mean())
        if 'encoder_aux_pred' in outputs:
            losses['center_heatmap'] = self.weights['heatmap']*center_heatmap_loss(outputs['encoder_aux_pred'],targets)
        allocated = []
        if outputs['allocation_mode'] != 'uniform':
            coarse = outputs['context_geometry_layers'][-1]
            for image,(rows,cols) in enumerate(assignments):
                if len(rows):
                    value,_ = allocation(outputs['transports'][image],targets[image]['boxes'][cols],
                        coarse['demand'][image,rows],canvas=outputs['canvas'],
                        mode=outputs['allocation_mode'],**self.options)
                    allocated.append(value)
        losses['allocation'] = self.weights['allocation']*(torch.stack(allocated).mean()
            if allocated else outputs['density'].sum()*0)
        if self.weights['semantic']:
            if teacher is None:
                raise ValueError('Semantic training requires frozen image teacher features')
            losses['semantic'] = self.weights['semantic']*sum(semantic_alignment(outputs['semantic'],targets,teacher).values())
        return losses
