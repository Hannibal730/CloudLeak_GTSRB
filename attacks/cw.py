# attacks/cw.py
import torch
import torch.nn.functional as F

IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1,3,1,1)

def cw_margin_loss(logits, y, kappa=0.0):
    """
    Untargeted CW margin: max( z_y - max_{i != y} z_i + kappa, 0 )
    값을 '작게' 만들수록 y가 top1이 아니게 됨.
    """
    b = logits.size(0)
    real = logits.gather(1, y.view(-1,1)).squeeze(1)
    # y 위치를 -inf로 만들어 '다른 클래스' 최댓값만 선택
    mask = torch.zeros_like(logits)
    mask.scatter_(1, y.view(-1,1), float('-inf'))
    other, _ = (logits + mask).max(1)
    return torch.relu(real - other + kappa).mean()

def cw_linf_attack(
    x, y, model,
    eps_pixel=8/255,         # L∞ 예산 (픽셀 스케일)
    alpha_pixel=2/255,       # 스텝 크기 (픽셀 스케일)
    iters=50,
    kappa=0.0,
    clamp_fn=None,
    min_iters=10             # 너무 일찍 멈추지 않게 최소 스텝 보장
):
    """
    CW(L∞): CW margin loss로 업데이트, 매 스텝 L∞-ball 투영 + box clamp
    - y: '멀어지고 싶은' 레이블 (여기선 학생의 현재 예측을 사용)
    """
    device = x.device
    std   = IMAGENET_STD.to(device)
    eps   = eps_pixel   / std
    alpha = alpha_pixel / std

    x_adv = x.clone().detach()
    for t in range(iters):
        x_adv.requires_grad_(True)
        logits = model(x_adv)
        loss = cw_margin_loss(logits, y, kappa=kappa)
        grad = torch.autograd.grad(loss, x_adv)[0]

        with torch.no_grad():
            x_adv = x_adv + alpha * grad.sign()   # CW loss로 ascent
            # 원본 around L∞ projection
            x_adv = torch.min(torch.max(x_adv, x - eps), x + eps)
            if clamp_fn is not None:
                x_adv = clamp_fn(x_adv)

        # 조기 종료: 충분히 y에서 벗어났으면 종료 (min_iters 뒤부터)
        if t+1 >= min_iters:
            with torch.no_grad():
                pred = logits.argmax(1)
                if (pred != y).all():
                    break

    return x_adv.detach()
