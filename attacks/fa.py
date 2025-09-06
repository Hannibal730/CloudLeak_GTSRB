import torch, torch.nn.functional as F
def feature_adversary(x, guide, feature_fn, max_iter=200, step=1.0):
    x_adv=x.clone().detach().requires_grad_(True); opt=torch.optim.SGD([x_adv],lr=step)
    for _ in range(max_iter):
        f_x=feature_fn(x_adv); f_g=feature_fn(guide).detach()
        loss=F.mse_loss(f_x,f_g); opt.zero_grad(); loss.backward(); opt.step()
    return x_adv.detach()
