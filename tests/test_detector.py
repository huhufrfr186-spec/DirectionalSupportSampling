import torch
from dds.factory import load_config,build_model,build_criterion,load_checkpoint


def small_config():
    cfg = load_config('configs/codrone.yaml')
    cfg['model'].update(canvas=64,density_nodes=9,hidden_dim=32,ffn_dim=64,queries=8,layers=2,heads=4)
    cfg['loss']['semantic'] = 0
    return cfg


def test_full_detector_update_and_checkpoint(tmp_path):
    torch.set_num_threads(2)
    torch.manual_seed(42)
    cfg = small_config()
    model = build_model(cfg,torch.randn(12,16))
    criterion = build_criterion(cfg,model)
    optimizer = torch.optim.AdamW(model.parameters(),lr=1e-5)
    image = torch.rand(1,3,64,64)
    targets = [dict(boxes=torch.tensor([[.5,.5,.2,.1,.2]]),labels=torch.tensor([0]),ignored_boxes=torch.empty(0,5))]
    model.train()
    result = model(image)
    losses = criterion(result,targets)
    loss = sum(losses.values())
    assert torch.isfinite(loss)
    loss.backward()
    gradients = [p.grad for p in model.parameters() if p.grad is not None]
    assert gradients and all(torch.isfinite(g).all() for g in gradients)
    assert model.router.predictor[-1].weight.grad.abs().sum()>0
    before = model.router.predictor[-1].weight.detach().clone()
    optimizer.step()
    assert not torch.equal(before,model.router.predictor[-1].weight)
    model.eval()
    with torch.no_grad():
        prediction = model(image)['pred_boxes']
    path = tmp_path/'weights.pth'
    torch.save(dict(config=cfg,model=model.state_dict()),path)
    restored,_,_ = load_checkpoint(path)
    restored.eval()
    with torch.no_grad():
        assert torch.equal(prediction,restored(image)['pred_boxes'])


def test_empty_target_and_detached_query_maps():
    from dds.density_router import stopped_query_maps
    torch.set_num_threads(2)
    cfg = small_config()
    model = build_model(cfg,torch.randn(12,16)).train()
    output = model(torch.rand(1,3,64,64))
    maps = stopped_query_maps(output['context_geometry_layers'][-1],9)
    assert not maps.requires_grad
    target = [dict(boxes=torch.empty(0,5),labels=torch.empty(0,dtype=torch.long),ignored_boxes=torch.empty(0,5))]
    loss = sum(build_criterion(cfg,model)(output,target).values())
    loss.backward()
    assert torch.isfinite(loss)


def test_invalid_border_anchors_cannot_initialize_queries():
    from dds.networks.decoder import RTDETRTransformer
    decoder = RTDETRTransformer(num_classes=1,hidden_dim=32,num_queries=8,
        feat_channels=[32],feat_strides=[8],num_levels=1,nhead=4,
        num_decoder_layers=1,dim_feedforward=64,num_denoising=0)
    torch.nn.init.zeros_(decoder.enc_score_head.weight)
    torch.nn.init.zeros_(decoder.enc_score_head.bias)
    memory = torch.zeros(1,128*128,32)
    _,reference,boxes,logits = decoder._get_decoder_input(memory,[(128,128)])
    assert torch.isfinite(reference).all() and torch.isfinite(boxes).all()
