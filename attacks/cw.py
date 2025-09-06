import torch, torch.nn.functional as F
def cw_l2_attack(x, y, model, kappa=0.0, steps=200, lr=5e-2):
    delta=torch.zeros_like(x,requires_grad=True); opt=torch.optim.Adam([delta],lr=lr)
    for _ in range(steps):
        adv=x+delta; logits=model(adv)
        real=logits.gather(1,y.view(-1,1)).squeeze(1)
        other=logits.topk(2,dim=1).values[:,1]
        f=torch.clamp(real - other + kappa, min=0)
        l2=(delta**2).view(delta.size(0),-1).sum(1).sqrt()
        loss=(1e-2*f + l2).mean()
        opt.zero_grad(); loss.backward(); opt.step()
    return (x+delta).detach()
