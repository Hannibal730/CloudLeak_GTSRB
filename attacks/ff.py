# attacks/ff.py (수정)
import torch, torch.nn.functional as F

def featurefool_lbfgs(x, guide, anchor, feature_fn, alpha_triplet=0.5, lambda_triplet=1e-3, max_iter=200, clamp_fn=None):
    x_adv = x.clone().detach().requires_grad_(True)

    # ★ guide/anchor 특징은 한 번만 계산해 캐시
    with torch.no_grad():
        f_pos = feature_fn(guide).detach()
        f_neg = feature_fn(anchor).detach()

    opt = torch.optim.LBFGS([x_adv], lr=1.0, max_iter=max_iter, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad(set_to_none=True)
        f_x = feature_fn(x_adv)  # 여기는 grad 필요
        loss = F.mse_loss(f_x, f_pos) - alpha_triplet * F.mse_loss(f_x, f_neg)
        loss = loss + lambda_triplet * (x_adv - x).pow(2).mean()
        loss.backward()
        return loss

    opt.step(closure)
    with torch.no_grad():
        if clamp_fn is not None:
            x_adv[:] = clamp_fn(x_adv)
    return x_adv.detach()
