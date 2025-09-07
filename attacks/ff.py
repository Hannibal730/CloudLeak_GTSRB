# attacks/ff.py
import torch
import torch.nn.functional as F
from typing import List, Optional

@torch.no_grad()
def _pairwise_sqdist_mean(FK: torch.Tensor) -> float:
    """
    FK: [N, D] (해당 클래스의 DeepID 특징들)
    반환: i != j 쌍에 대한 평균 제곱거리
    """
    N = FK.size(0)
    if N <= 1:
        return 0.0
    # cdist^2를 쓰면 수식 그대로 구현 가능
    # 주의: 대각 성분(0)은 제외
    Dmat = torch.cdist(FK, FK, p=2).pow(2)   # [N,N]
    off = (Dmat.sum() - torch.diag(Dmat).sum()) / float(N * N - N)
    return float(off.item())

def _compute_margin_per_sample(
    class_feat_banks: Optional[List[Optional[torch.Tensor]]],
    alpha: float,
    device: torch.device
) -> torch.Tensor:
    """
    식(11)에 따라 표본별 M을 계산.
    - class_feat_banks: 길이 B의 리스트. 각 원소는 [N_c, D] 텐서(가이드 클래스의 DeepID 특징들) 또는 None
    - 반환: [B] 텐서, 각 표본의 M
    """
    if class_feat_banks is None:
        # 전 표본 동일 폴백: M = α
        return torch.full((1,), alpha, device=device)
    Ms = []
    for bank in class_feat_banks:
        if bank is None or (bank.dim() != 2) or (bank.size(0) < 2):
            Ms.append(alpha)  # 표본 부족 → 식(11)에서 평균항 0 취급
        else:
            # 평균 제곱거리 계산
            m = alpha - _pairwise_sqdist_mean(bank)
            Ms.append(m)
    return torch.tensor(Ms, device=device, dtype=torch.float32)

def featurefool_lbfgs(
    x: torch.Tensor,                   # [B,C,H,W] 원본(=negative)
    guide: torch.Tensor,               # [B,C,H,W] 가이드(=positive)
    anchor: torch.Tensor,              # [B,C,H,W] 논문: negative = source x (runner에서 x를 넘김)
    feature_fn,                        # φ_K(·)를 반환하는 함수: x -> [B,D] 또는 [B,*,D] → 평탄화해 사용
    class_feat_banks: Optional[List[Optional[torch.Tensor]]] = None,
    alpha: float = 0.5,                # 식(11)의 α (논문 권장: 0.5)
    lambda_reg: float = 1e-3,          # 식(9)의 λ (가능한 한 작게; 논문 권고)
    max_iter: int = 200,
    clamp_fn=None,                     # 정규화 좌표계 [0,1] box clamp 함수(미분 비사용; step 후 투영)
    line_search_fn: str = "strong_wolfe"
) -> torch.Tensor:
    """
    CloudLeak FeatureFool (논문 정합 구현)
    식(8) box-제약 L-BFGS 최적화 문제를 식(9)로 재정식화, 식(10) triplet hinge 사용.
    마진 M은 식(11)로 클래스 내 분산 기반 동적 산출.

    Args:
        class_feat_banks:
            길이 B의 리스트. 각 원소는 해당 표본의 "가이드 클래스" DeepID 특징들을 쌓은 [N_c, D] 텐서.
            (runner에서 클래스별 feature bank를 관리해 표본별로 전달)
            값이 None이거나 N_c<2면 M=α 폴백.
    """
    device = x.device
    B = x.size(0)

    # 최적화 변수
    x_adv = x.clone().detach().requires_grad_(True)

    # guide/anchor 특징 캐시 (LBFGS closure가 여러 번 불리므로)
    with torch.no_grad():
        f_pos = feature_fn(guide).detach()    # [B, D*] -> 평탄화 예정
        f_neg = feature_fn(anchor).detach()   # [B, D*]

        def _flatten(feat: torch.Tensor) -> torch.Tensor:
            if feat.dim() > 2:
                return feat.view(feat.size(0), -1)
            return feat
        f_pos = _flatten(f_pos)
        f_neg = _flatten(f_neg)

    # 표본별 마진 M (식(11))
    M_vec = _compute_margin_per_sample(class_feat_banks, alpha, device)
    if M_vec.numel() == 1 and B > 1:
        # 전 표본 동일 폴백(모두 α): [B]로 브로드캐스트
        M_vec = M_vec.expand(B)

    # L-BFGS
    opt = torch.optim.LBFGS([x_adv], lr=1.0, max_iter=max_iter, line_search_fn=line_search_fn)

    def closure():
        opt.zero_grad(set_to_none=True)
        f_x = feature_fn(x_adv)               # [B, D*]
        if f_x.dim() > 2:
            f_x = f_x.view(f_x.size(0), -1)

        # 식(10): D = L2^2 거리로 구현
        d_pos = (f_x - f_pos).pow(2).sum(dim=1)   # [B]
        d_neg = (f_x - f_neg).pow(2).sum(dim=1)   # [B]
        triplet = torch.relu(d_pos - d_neg + M_vec).mean()

        # 식(9): d(x',x)=||x'-x||_2^2
        img_reg = (x_adv - x).pow(2).mean()

        loss = triplet + lambda_reg * img_reg
        loss.backward()
        return loss

    opt.step(closure)

    # box-constraint 투영 (정규화 좌표에서 [0,1])
    with torch.no_grad():
        if clamp_fn is not None:
            # 주의: 일부 clamp_fn이 .data를 직접 조작한다면, 최적화 후 한 번만 호출되도록 유지
            x_clamped = clamp_fn(x_adv)
            # clamp_fn이 in-place가 아니라면 값 복사
            if x_clamped is not x_adv:
                x_adv.copy_(x_clamped)

    return x_adv.detach()
