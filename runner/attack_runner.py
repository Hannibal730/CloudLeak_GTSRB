# runner/attack_runner.py
import os, time, json, random
from typing import Dict, List, Optional

import torch, torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, TensorDataset
from tqdm import tqdm
from PIL import Image

# ==== 프로젝트 내부 모듈 ====
from models.vgg19_deepid import VGG19DeepID
from utils.data import (
    load_gtsrb_imagefolder, split_qpool_dtest_from_test,
    ImagePathDataset, get_transforms
)
from utils.logger import QueryLogger
from utils.metrics import compute_ate, save_metrics

from attacks.rm import query_indices_rm, identity_attack
from attacks.pgd import pgd_attack_normalized
from attacks.cw import cw_l2_attack
from attacks.fa import feature_adversary
from attacks.ff import featurefool_lbfgs

from oracle.resnet import get_resnet50  # 로컬 오라클(ResNet50 on GTSRB)

# ==== 예산(논문 Fig.6) ====
BUDGET_MAP = {"A": 430, "B": 1290, "C": 2150}

# ==== 정규화 파라미터 ====
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

def normalized_clamp_(x, device):
    """정규화 좌표계에서 [0,1] 박스 제약."""
    mean = torch.tensor(IMAGENET_MEAN, device=device).view(1, 3, 1, 1)
    std  = torch.tensor(IMAGENET_STD,  device=device).view(1, 3, 1, 1)
    lo = (0.0 - mean) / std
    hi = (1.0 - mean) / std
    x.data = torch.max(torch.min(x, hi), lo)
    return x

def softmax_probs(logits):
    return torch.softmax(logits, dim=1)

# ==== LC(학생 기준) 선택 ====
def least_confidence_indices_student(student, pool_loader, k, device):
    """
    LC(Least-Confidence) 기반 선택 (학생 확신도 사용).
    - 오라클 사전질의 비용 없이 학생의 확신도를 대리로 활용.
    """
    student.eval()
    scores = []
    base = 0
    with torch.no_grad():
        for x, _ in pool_loader:
            x = x.to(device)
            probs = softmax_probs(student(x))
            maxp, _ = probs.max(1)
            for i, p in enumerate(maxp):
                scores.append((1.0 - p.item(), base + i))  # 확신도 낮을수록 우선
            base += x.size(0)
    scores.sort(reverse=True)
    return [j for _, j in scores[:k]]

# ==== 안전한 학생 호출 래퍼 (which 지원/미지원 모두 호환) ====
def make_student_forward(student):
    """
    return_features=True 호출 시:
      - 모델이 which="deepid" 인자를 지원하면 사용
      - 미지원이면 (기존 구현이 반환하는 특징을) 그대로 사용
    """
    def _forward(x_tensor, return_features=False):
        if not return_features:
            return student(x_tensor)
        try:
            return student(x_tensor, return_features=True, which="deepid")
        except TypeError:
            return student(x_tensor, return_features=True)
    return _forward

# ==== DeepID 특징 추출 유틸 ====
@torch.no_grad()
def extract_feat(student_forward, x_tensor):
    """
    DeepID 특징(또는 폴백 특징)을 2D [B,D]로 반환.
    """
    student_forward  # noqa
    logits, feat = student_forward(x_tensor, return_features=True)
    if feat.dim() > 2:
        feat = feat.view(feat.size(0), -1)
    return feat

def run_experiment(
    data_dir: str,
    save_dir: str,
    attack: str,                # 'ff' | 'fa' | 'pgd' | 'cw' | 'rm'
    budget_key: str,            # 'A' | 'B' | 'C'
    oracle_weights: Optional[str],
    seed: int = 0,
    qpool_n: int = 6000,
    dtest_n: int = 2000,
    device=None,

    # ==== 실험 프로토콜(논문 분위기에 맞춘 기본값) ====
    selector: str = "lc",       # 'lc' | 'rs'
    seed_q: int = 86,           # 시드 질의 수
    update_every: int = 512,    # 업데이트 주기 (완화: 과도 상승 방지)
    epochs_per_update: int = 1,
    batch_update: int = 64
):
    """
    CloudLeak 정합 러너:
    - LC 기반 쿼리 선택(기본)
    - 오라클: 로컬 ResNet50 (weights 로드)
    - 학생(전이): VGG19-DeepID, 합성곱 동결, FC만 학습
    - 공격: FF(논문식), FA(가이드 정합), PGD(L∞), CW(L2+κ)
    - FF: 식(8)~(11) 정합. per-class feature bank로 식(11)의 M을 표본별 산출.
    """
    random.seed(seed); torch.manual_seed(seed)
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(save_dir, exist_ok=True)

    # ==== 데이터 로드 ====
    train_ds, test_ds = load_gtsrb_imagefolder(data_dir, size=224)
    n_classes = len(train_ds.classes)
    q_paths, (d_paths, d_labels) = split_qpool_dtest_from_test(
        test_ds, qpool_n=qpool_n, dtest_n=dtest_n, seed=seed
    )

    # ==== 오라클 로드 ====
    oracle = get_resnet50(num_classes=n_classes, pretrained=True).to(device)
    if oracle_weights and os.path.exists(oracle_weights):
        try:
            state = torch.load(oracle_weights, map_location=device, weights_only=True)
        except TypeError:
            state = torch.load(oracle_weights, map_location=device)
        oracle.load_state_dict(state)
    oracle.eval()

    # ==== 학생(전이) 모델 ====
    student = VGG19DeepID(num_classes=n_classes, freeze_features=True).to(device)
    student_forward = make_student_forward(student)
    params = [p for p in student.parameters() if p.requires_grad]
    optim  = torch.optim.Adam(params, lr=1e-3)
    crit   = nn.CrossEntropyLoss()

    # ==== 풀/테스트 로더 ====
    pool_ds = ImagePathDataset(q_paths, transform=get_transforms(224))
    pool_loader = DataLoader(pool_ds, batch_size=64, shuffle=False, num_workers=2)

    class DTestDataset(Dataset):
        def __init__(self, paths, labels):
            self.paths = paths; self.labels = labels; self.tfm = get_transforms(224)
        def __len__(self): return len(self.paths)
        def __getitem__(self, idx):
            return self.tfm(Image.open(self.paths[idx]).convert("RGB")), self.labels[idx]
    dtest_loader = DataLoader(DTestDataset(d_paths, d_labels), batch_size=64, shuffle=False, num_workers=2)

    # ==== 설정/로그/버퍼 ====
    cfg_path = os.path.join("configs", f"{attack}.json")
    cfg = json.load(open(cfg_path, "r", encoding="utf-8")) if os.path.exists(cfg_path) else {}
    budget = BUDGET_MAP[budget_key]
    remaining = list(range(len(q_paths)))
    out_dir = os.path.join(save_dir, attack, budget_key)
    qlog = QueryLogger(out_dir)

    Ds_x, Ds_y = [], []           # 누적 학습 버퍼(오라클 라벨)
    guides: Dict[int, str] = {}   # class_id -> path (시드에서만 구축)

    # ---- FF용 per-class feature bank (DeepID 특징 누적; 식(11)) ----
    #  class_bank[c]: torch.Tensor [Nc, D] (CPU에 저장; 필요 시 배치마다 GPU로 올림)
    class_bank: Dict[int, torch.Tensor] = {}
    BANK_CAP = int(cfg.get("bank_cap", 50))  # 클래스별 최대 저장 개수(메모리 보호)
    ALPHA = float(cfg.get("alpha", 0.5))     # 식(11)의 α (논문 권장 0.5)

    # ==== 특징함수 (논문: DeepID, 정규화 X) ====
    def feature_fn(x_tensor):
        student.eval()
        logits, feat = student_forward(x_tensor, return_features=True)
        if feat.dim() > 2:
            feat = feat.view(feat.size(0), -1)
        return feat

    # ==== 시드 질의 ====
    step = 0
    if seed_q > 0:
        k = min(seed_q, budget, len(remaining))
        seed_idx = query_indices_rm(remaining, k, random)  # 시드는 랜덤

        xs = [get_transforms(224)(Image.open(q_paths[i]).convert("RGB")) for i in seed_idx]
        x_batch = torch.stack(xs, 0).to(device)

        # 오라클 질의(시드)
        with torch.no_grad():
            o_probs = torch.softmax(oracle(x_batch), dim=1)
            o_top1 = o_probs.argmax(1)

        # 로그
        for j, bi in enumerate(seed_idx):
            top5_p, top5_i = o_probs[j].topk(5)
            top5 = [{"label": int(top5_i[t].item()), "prob": float(top5_p[t].item())} for t in range(5)]
            qlog.log(j, bi, "seed", {}, int(o_top1[j].item()), float(o_probs[j].max().item()), top5, 0.0)

        # 가이드 구축 (라벨만 규칙: 해당 클래스 '첫 등장' 1장)
        for j, bi in enumerate(seed_idx):
            c = int(o_top1[j].item())
            if c not in guides:
                guides[c] = q_paths[bi]

        # 클래스 뱅크에 시드 특징 누적 (식(11)용)
        with torch.no_grad():
            seed_feats = extract_feat(student_forward, x_batch).cpu()  # [k, D] on CPU
        for j, bi in enumerate(seed_idx):
            c = int(o_top1[j].item())
            if c not in class_bank:
                class_bank[c] = seed_feats[j:j+1].clone()
            else:
                cb = class_bank[c]
                if cb.size(0) < BANK_CAP:
                    class_bank[c] = torch.cat([cb, seed_feats[j:j+1]], dim=0)

        # 버퍼 적재
        Ds_x.append(x_batch.detach().cpu())
        Ds_y.append(o_top1.detach().cpu())

        # remaining 갱신
        for bi in seed_idx:
            if bi in remaining: remaining.remove(bi)
        step = len(seed_idx)

        # 초기 미세학습
        X = torch.cat(Ds_x, 0); Y = torch.cat(Ds_y, 0)
        dl = DataLoader(TensorDataset(X, Y), batch_size=batch_update, shuffle=True)
        student.train()
        for _ in range(epochs_per_update):
            for xb, yb in dl:
                xb, yb = xb.to(device), yb.to(device)
                logits = student(xb)
                loss = crit(logits, yb)
                optim.zero_grad(set_to_none=True); loss.backward(); optim.step()

    pbar = tqdm(total=budget, desc=f"{attack.upper()}-{budget_key}")
    pbar.update(step)

    # ==== 본 루프 ====
    while step < budget and len(remaining) > 0:
        k = min(32, budget - step)

        # LC/RS 선택
        if selector == "rs":
            batch_idx = query_indices_rm(remaining, k, random)
        else:  # 'lc'
            all_ranked = least_confidence_indices_student(
                student,
                DataLoader(pool_ds, batch_size=64, shuffle=False, num_workers=2),
                len(remaining), device
            )
            batch_idx = [i for i in all_ranked if i in remaining][:k] or query_indices_rm(remaining, k, random)

        # 입력/가이드 배치
        xs = [get_transforms(224)(Image.open(q_paths[bi]).convert("RGB")) for bi in batch_idx]
        x_batch = torch.stack(xs, 0).to(device)

        with torch.no_grad():
            s_logits, _ = student_forward(x_batch, return_features=True)
            s_cls = s_logits.argmax(1)  # 학생 예측 클래스(가이드 선택 용도)

        guides_x = []
        guide_classes: List[int] = []
        for j, cls in enumerate(s_cls.tolist()):
            gp = guides.get(cls, None)
            if gp is None:
                guides_x.append(x_batch[j:j+1])
                guide_classes.append(cls)
            else:
                guides_x.append(get_transforms(224)(Image.open(gp).convert("RGB")).unsqueeze(0).to(device))
                guide_classes.append(cls)
        guide_batch = torch.cat(guides_x, 0)

        # ===== 합성 =====
        t0 = time.time()
        if attack == "rm":
            x_query = identity_attack(x_batch)
            params = {}
        elif attack == "pgd":
            eps   = float(cfg.get("epsilon", 8/255))
            alpha = float(cfg.get("alpha",   2/255))
            iters = int(cfg.get("iters",     20))
            with torch.no_grad():
                pred = s_logits.argmax(1)
            x_query = pgd_attack_normalized(
                x_batch, pred, student,
                eps_pixel=eps, alpha_pixel=alpha, iters=iters,
                clamp_fn=lambda z: normalized_clamp_(z, device)
            )
            params = {"epsilon": eps, "alpha": alpha, "iters": iters}
        elif attack == "cw":
            c     = float(cfg.get("c",     1e-2))
            steps = int(cfg.get("steps",   200))
            lr    = float(cfg.get("lr",    5e-3))
            kappa = float(cfg.get("kappa", 0.0))
            with torch.no_grad():
                pred = s_logits.argmax(1)  # untargeted: '멀어질' 레이블
            x_query = cw_l2_attack(
                x_batch, pred, student,
                c=c, steps=steps, lr=lr, kappa=kappa,
                clamp_fn=lambda z: normalized_clamp_(z, device),
                min_steps=50, early_stop=True
            )
            params = {"c": c, "steps": steps, "lr": lr, "kappa": kappa}
        elif attack == "fa":
            # 정합 우선: ε-볼 투영은 fa.py에서 제거(또는 미사용), box-clamp만
            max_iter = int(cfg.get("max_iter", 120))
            step_sz  = float(cfg.get("step", 1/255))
            x_query = feature_adversary(
                x_batch, guide_batch, feature_fn,
                max_iter=max_iter, step=step_sz,
                clamp_fn=lambda z: normalized_clamp_(z, device)
            )
            params = {"max_iter": max_iter, "step": step_sz}
        elif attack == "ff":
            # ---- FF: 식(8)~(11) 정합 ----
            max_iter   = int(cfg.get("max_iter", 200))
            lambda_reg = float(cfg.get("lambda_reg", 1e-3))
            alpha      = float(cfg.get("alpha", ALPHA))  # 식(11) α (기본 0.5)
            anchor_batch = x_batch                       # negative=원본 x
            FF_CHUNK = int(cfg.get("ff_chunk", 8))      # OOM 방지용 마이크로배치

            # per-sample class_feat_bank: guide_classes 기준으로 클래스별 뱅크를 모아 전달
            #  - 저장은 CPU에 했지만, ff 내부 계산을 위해 여기서 GPU로 옮겨줌
            per_sample_bank: List[Optional[torch.Tensor]] = []
            for cls_id in guide_classes:
                if cls_id in class_bank and class_bank[cls_id].size(0) >= 2:
                    per_sample_bank.append(class_bank[cls_id].to(device))
                else:
                    per_sample_bank.append(None)

            outs = []
            for b0 in range(0, x_batch.size(0), FF_CHUNK):
                xb = x_batch[b0:b0+FF_CHUNK]
                gb = guide_batch[b0:b0+FF_CHUNK]
                ab = anchor_batch[b0:b0+FF_CHUNK]
                # 이 배치에 해당하는 per-sample bank만 슬라이스
                banks_slice = per_sample_bank[b0:b0+FF_CHUNK]
                out_b = featurefool_lbfgs(
                    xb, gb, ab, feature_fn,
                    class_feat_banks=banks_slice,
                    alpha=alpha, lambda_reg=lambda_reg, max_iter=max_iter,
                    clamp_fn=lambda z: normalized_clamp_(z, device),
                    line_search_fn="strong_wolfe"
                )
                outs.append(out_b)
            x_query = torch.cat(outs, 0)
            params = {"max_iter": max_iter, "lambda_reg": lambda_reg, "alpha": alpha, "ff_chunk": FF_CHUNK}
        else:
            raise ValueError("unknown attack")
        elapsed_ms = (time.time() - t0) * 1000.0

        # ===== 오라클 질의 =====
        with torch.no_grad():
            o_probs = torch.softmax(oracle(x_query), dim=1)
            o_top1_p, o_top1 = o_probs.max(1)
            top5_p, top5_i = o_probs.topk(5, dim=1)

        # ===== 로그/버퍼 =====
        for j, bi in enumerate(batch_idx):
            top5 = [{"label": int(top5_i[j, k].item()), "prob": float(top5_p[j, k].item())} for k in range(5)]
            qlog.log(step + j, bi, attack, params, int(o_top1[j].item()), float(o_top1_p[j].item()), top5, elapsed_ms)

        # 누적 학습 버퍼
        Ds_x.append(x_query.detach().cpu())
        Ds_y.append(o_top1.detach().cpu())

        # ---- FF용 클래스 뱅크 업데이트 (오라클 라벨 기준) ----
        #  새로 질의한 x_query의 DeepID 특징을 추출하여 해당 클래스 은행에 누적
        if attack in ("ff", "fa", "pgd", "cw", "rm"):
            with torch.no_grad():
                feats_q = extract_feat(student_forward, x_query).cpu()  # [B, D] on CPU
            for j in range(o_top1.size(0)):
                c = int(o_top1[j].item())
                if c not in class_bank:
                    class_bank[c] = feats_q[j:j+1].clone()
                else:
                    cb = class_bank[c]
                    if cb.size(0) < BANK_CAP:
                        class_bank[c] = torch.cat([cb, feats_q[j:j+1]], dim=0)

        # 주기적 미세학습
        if sum(t.shape[0] for t in Ds_x) >= update_every or step + len(batch_idx) >= budget:
            X = torch.cat(Ds_x, 0); Y = torch.cat(Ds_y, 0)
            dl = DataLoader(TensorDataset(X, Y), batch_size=batch_update, shuffle=True)
            student.train()
            for _ in range(epochs_per_update):
                for xb, yb in dl:
                    xb, yb = xb.to(device), yb.to(device)
                    logits = student(xb)
                    loss = crit(logits, yb)
                    optim.zero_grad(set_to_none=True); loss.backward(); optim.step()

        # remaining 갱신
        for bi in batch_idx:
            if bi in remaining: remaining.remove(bi)
        step += len(batch_idx)
        pbar.update(len(batch_idx))

    pbar.close()

    # ==== 평가 ====
    student.eval(); gt_correct = total = 0
    with torch.no_grad():
        for x, y in dtest_loader:
            x = x.to(device)
            y = torch.as_tensor(y, dtype=torch.long, device=device)
            pred = student(x).argmax(1)
            gt_correct += (pred == y).sum().item()
            total += y.numel()
    acc = gt_correct / max(1, total)

    s_preds, o_preds = [], []
    with torch.no_grad():
        for x, y in dtest_loader:
            x = x.to(device)
            s_preds += student(x).argmax(1).cpu().tolist()
            o_preds += oracle(x).argmax(1).cpu().tolist()
    ate = compute_ate(s_preds, o_preds)

    # ==== 저장 ====
    os.makedirs(out_dir, exist_ok=True)
    torch.save(student.state_dict(), os.path.join(out_dir, "student.pth"))

    metrics = {
        "attack": attack,
        "budget": budget_key,
        "selector": selector,
        "seed": seed,
        "update_every": update_every,
        "epochs_per_update": epochs_per_update,
        "accuracy": acc,
        "ATE": ate,
        "seed_queries": seed_q,
        "total_queries": step
    }
    # FF에서 쓰인 마이크로배치 기록(있을 때만)
    if attack == "ff" and "FF_CHUNK" in locals():
        metrics["ff_chunk"] = int(cfg.get("ff_chunk", 8))
    # FF 식(11) 관련 파라미터 기록
    if attack == "ff":
        metrics["alpha_ff"] = float(cfg.get("alpha", 0.5))
        metrics["bank_cap"] = int(cfg.get("bank_cap", 50))

    save_metrics(out_dir, metrics)
    return metrics
