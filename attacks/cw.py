# attacks/cw.py
import torch
import torch.nn.functional as F

@torch.no_grad()
def _max_except_y(logits: torch.Tensor, y: torch.Tensor):
    tmp = logits.clone()
    tmp.scatter_(1, y.view(-1, 1), float("-inf"))
    other_max, _ = tmp.max(dim=1)
    return other_max

def cw_margin_loss(logits: torch.Tensor, y: torch.Tensor, kappa: float = 0.0):
    real = logits.gather(1, y.view(-1, 1)).squeeze(1)
    other = _max_except_y(logits, y)
    return torch.relu(real - other + kappa).mean()

def cw_l2_attack(
    x: torch.Tensor,
    y: torch.Tensor,
    model: torch.nn.Module,
    c: float = 1e-2,
    steps: int = 200,
    lr: float = 5e-3,
    kappa: float = 0.0,
    clamp_fn=None,
    min_steps: int = 50,
    early_stop: bool = True
):
    delta = torch.zeros_like(x, requires_grad=True)
    opt = torch.optim.Adam([delta], lr=lr)

    best_adv = x.clone()
    best_obj = torch.full((x.size(0),), float("inf"), device=x.device)

    for t in range(steps):
        x_adv = x + delta
        logits = model(x_adv)

        l2 = (delta.pow(2).flatten(1).sum(dim=1))
        margin = cw_margin_loss(logits, y, kappa)
        obj = l2.mean() + c * margin

        opt.zero_grad(set_to_none=True)
        obj.backward()
        opt.step()

        with torch.no_grad():
            if clamp_fn is not None:
                x_adv = clamp_fn(x + delta)
                delta.data.copy_(x_adv - x)

            logits = model(x + delta)
            l2_now = (delta.pow(2).flatten(1).sum(dim=1))
            margin_now = cw_margin_loss(logits, y, kappa).detach()
            obj_now = l2_now + c * margin_now
            upd = obj_now < best_obj
            best_obj[upd] = obj_now[upd]
            best_adv[upd] = (x + delta)[upd]

            if early_stop and (t + 1) >= min_steps:
                pred = logits.argmax(1)
                if (pred != y).all() and margin_now.item() == 0.0:
                    break

    return best_adv.detach()
