# attacks/fa.py
import torch
import torch.nn.functional as F

# 정규화 좌표계에서 픽셀 스케일 ε를 채널별로 변환할 때 사용
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

def feature_adversary(
    x,
    guide,
    feature_fn,
    max_iter: int = 200,
    step: float = 1.0,
    eps_pixel: float = 8/255,    # (C) 픽셀 기준 ε (예: 8/255)
    clamp_fn=None,               # (A) box 제약 함수(정규화 좌표계용)
):
    """
    Feature Adversary (FA)
    - 마지막 은닉 특징에서 guide에 근접하도록 유도 (MSE)
    - (A)(C) 원본 x 기준 L∞(ε) 제약 + box clamp 적용
      * eps_pixel: 픽셀 스케일(0~1)에서의 ε. 내부에서 /std 하여 정규화 좌표로 변환.
    """
    device = x.device
    eps = eps_pixel / IMAGENET_STD.to(device)

    x_adv = x.clone().detach().requires_grad_(True)
    opt = torch.optim.SGD([x_adv], lr=step)

    for _ in range(max_iter):
        f_x = feature_fn(x_adv)
        with torch.no_grad():
            f_g = feature_fn(guide).detach()
        loss = F.mse_loss(f_x, f_g)

        opt.zero_grad()
        loss.backward()
        opt.step()

        # box clamp
        with torch.no_grad():
            if clamp_fn is not None:
                x_adv[:] = clamp_fn(x_adv)

    return x_adv.detach()