import torch, torch.nn.functional as F
IMAGENET_STD  = torch.tensor([0.229, 0.224, 0.225]).view(1,3,1,1)
def pgd_attack_normalized(x, y, model, eps_pixel=8/255, alpha_pixel=2/255, iters=20, clamp_fn=None):
    eps = eps_pixel / IMAGENET_STD.to(x.device)
    alpha = alpha_pixel / IMAGENET_STD.to(x.device)
    x_adv = x.clone().detach() + torch.empty_like(x).uniform_(-1.0,1.0) * eps
    x_adv.requires_grad_(True)
    for _ in range(iters):
        loss = F.cross_entropy(model(x_adv), y)
        loss.backward()
        with torch.no_grad():
            x_adv += alpha * x_adv.grad.sign()
            x_adv = torch.min(torch.max(x_adv, x - eps), x + eps)
            if clamp_fn is not None: x_adv = clamp_fn(x_adv)
        x_adv.requires_grad_(True)
    return x_adv.detach()
