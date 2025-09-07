# runner/attack_runner.py
import os, time, json, random
import torch, torch.nn as nn
from torch.utils.data import DataLoader, Dataset, TensorDataset
from tqdm import tqdm

from models.vgg19_deepid import VGG19DeepID
from utils.data import load_gtsrb_imagefolder, split_qpool_dtest_from_test, ImagePathDataset, get_transforms
from utils.logger import QueryLogger
from utils.metrics import compute_ate, save_metrics
from attacks.rm import query_indices_rm, identity_attack
from attacks.pgd import pgd_attack_normalized
from attacks.cw import cw_linf_attack
from attacks.fa import feature_adversary
from attacks.ff import featurefool_lbfgs
from oracle.resnet import get_resnet50

BUDGET_MAP = {"A": 430, "B": 1290, "C": 2150}
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

def normalized_clamp_(x, device):
    mean = torch.tensor(IMAGENET_MEAN, device=device).view(1, 3, 1, 1)
    std  = torch.tensor(IMAGENET_STD,  device=device).view(1, 3, 1, 1)
    lo = (0.0 - mean) / std
    hi = (1.0 - mean) / std
    x.data = torch.max(torch.min(x, hi), lo)
    return x

def softmax_probs(logits):
    return torch.softmax(logits, dim=1)

def least_confidence_indices(student, pool_loader, k, device):
    student.eval()
    scores = []
    base = 0
    with torch.no_grad():
        for x, _ in pool_loader:
            x = x.to(device)
            probs = softmax_probs(student(x))
            maxp, _ = probs.max(1)
            for i, p in enumerate(maxp):
                scores.append((1.0 - p.item(), base + i))
            base += x.size(0)
    scores.sort(reverse=True)
    return [j for _, j in scores[:k]]

def run_experiment(
    data_dir, save_dir, attack, budget_key, oracle_weights,
    seed=0, qpool_n=6000, dtest_n=2000, device=None,
    seed_q=200, update_every=128, epochs_per_update=3, batch_update=64
):
    """
    (A) FA/FF에 원본 기준 L∞ ε 제약 + box clamp 적용
    (B) FF의 anchor는 '다른 클래스' 가이드로 선택 (시드에서만 만든 가이드/피처 은행 사용)
    (C) FA도 동일 ε 제약 사용
    """
    random.seed(seed); torch.manual_seed(seed)
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(save_dir, exist_ok=True)

    # 데이터
    train_ds, test_ds = load_gtsrb_imagefolder(data_dir, size=224)
    q_paths, (d_paths, d_labels) = split_qpool_dtest_from_test(test_ds, qpool_n=qpool_n, dtest_n=dtest_n, seed=seed)

    # 오라클
    oracle = get_resnet50(num_classes=len(train_ds.classes), pretrained=True).to(device)
    if oracle_weights and os.path.exists(oracle_weights):
        oracle.load_state_dict(torch.load(oracle_weights, map_location=device, weights_only=True))
    oracle.eval()

    # 학생(전이) 모델
    student = VGG19DeepID(num_classes=len(train_ds.classes), freeze_features=True).to(device)
    params = [p for p in student.parameters() if p.requires_grad]
    optim  = torch.optim.Adam(params, lr=1e-3)
    crit   = nn.CrossEntropyLoss()

    # 풀/테스트 로더
    pool_ds = ImagePathDataset(q_paths, transform=get_transforms(224))
    pool_loader = DataLoader(pool_ds, batch_size=64, shuffle=False, num_workers=2)

    from PIL import Image
    class DTestDataset(Dataset):
        def __init__(self, paths, labels):
            self.paths = paths
            self.labels = labels
            self.tfm = get_transforms(224)
        def __len__(self): return len(self.paths)
        def __getitem__(self, idx):
            return self.tfm(Image.open(self.paths[idx]).convert("RGB")), self.labels[idx]
    dtest_loader = DataLoader(DTestDataset(d_paths, d_labels), batch_size=64, shuffle=False, num_workers=2)

    # 구성/로그/버퍼
    cfg_path = os.path.join("configs", f"{attack}.json")
    cfg = json.load(open(cfg_path, "r", encoding="utf-8")) if os.path.exists(cfg_path) else {}
    budget = BUDGET_MAP[budget_key]
    remaining = list(range(len(q_paths)))
    out_dir = os.path.join(save_dir, attack, budget_key)
    qlog = QueryLogger(out_dir)

    Ds_x, Ds_y = [], []           # 누적 버퍼
    guides = {}                   # class_id -> path (시드에서만 구축)
    guides_bank = {}              # class_id -> image tensor [1,C,H,W] (device)
    guides_feat_bank = {}         # class_id -> feature tensor [D] (device)
    ff_chunk_used = None          # 메트릭 기록용

    # 특징 함수 (x_adv로 grad가 흘러야 하므로 no_grad 금지)
    def feature_fn(x_tensor):
        student.eval()
        logits, feat = student(x_tensor, return_features=True)
        return feat

    # ===== 시드 질의 (예산 내) =====
    step = 0
    if seed_q > 0:
        k = min(seed_q, budget, len(remaining))
        seed_idx = query_indices_rm(remaining, k, random)

        xs = [get_transforms(224)(Image.open(q_paths[i]).convert("RGB")) for i in seed_idx]
        x_batch = torch.stack(xs, 0).to(device)

        # 시드는 합성 없이 그대로 질의
        with torch.no_grad():
            o_probs = torch.softmax(oracle(x_batch), dim=1)
            o_top1 = o_probs.argmax(1)

        # 로그
        for j, bi in enumerate(seed_idx):
            top5_p, top5_i = o_probs[j].topk(5)
            top5 = [{"label": int(top5_i[t].item()), "prob": float(top5_p[t].item())} for t in range(5)]
            qlog.log(j, bi, "seed", {}, int(o_top1[j].item()), float(o_probs[j].max().item()), top5, 0.0)

        # (B) 시드에서 클래스별 가이드 선정 (최대 확신)
        for j, bi in enumerate(seed_idx):
            c = int(o_top1[j].item())
            conf = float(o_probs[j, c].item())
            p = q_paths[bi]
            if c not in guides or conf > guides[c][0]:
                guides[c] = (conf, p)
        guides = {c: p for c, (conf, p) in guides.items()}  # path만 남김

        # (B) 가이드 이미지/피처 은행 구성 (학생 특징 공간)
        tfm = get_transforms(224)
        student.eval()
        with torch.no_grad():
            for c, p in guides.items():
                gimg = tfm(Image.open(p).convert("RGB")).unsqueeze(0).to(device)
                logits, gfeat = student(gimg, return_features=True)
                guides_bank[c] = gimg
                guides_feat_bank[c] = gfeat.squeeze(0)  # [D]

        # 버퍼 적재
        Ds_x.append(x_batch.detach().cpu())
        Ds_y.append(o_top1.detach().cpu())

        # 인덱스 갱신
        for bi in seed_idx:
            if bi in remaining:
                remaining.remove(bi)
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
                optim.zero_grad(set_to_none=True)
                loss.backward()
                optim.step()

    pbar = tqdm(total=budget, desc=f"{attack.upper()}-{budget_key}")
    pbar.update(step)

    # ===== 본 루프 =====
    while step < budget and len(remaining) > 0:
        # 배치 선택(LC; 없으면 RM)
        if attack == "rm":
            k = min(32, budget - step)
            batch_idx = query_indices_rm(remaining, k, random)
        else:
            k = min(32, budget - step)
            all_ranked = least_confidence_indices(
                student,
                DataLoader(pool_ds, batch_size=64, shuffle=False, num_workers=2),
                len(remaining), device
            )
            batch_idx = [i for i in all_ranked if i in remaining][:k] or query_indices_rm(remaining, k, random)

        xs = [get_transforms(224)(Image.open(q_paths[bi]).convert("RGB")) for bi in batch_idx]
        x_batch = torch.stack(xs, 0).to(device)

        # 가이드 클래스는 '학생 예측'으로 결정(추가 오라클 호출 없음)
        with torch.no_grad():
            s_logits, s_feat = student(x_batch, return_features=True)  # s_feat: [B,D]
            s_cls = s_logits.argmax(1)  # [B]

        # 배치별 guide(양성) 구성
        guides_x = []
        for j, cls in enumerate(s_cls.tolist()):
            gp = guides.get(cls, None)
            if gp is None:
                guides_x.append(x_batch[j:j+1])  # 해당 클래스 가이드가 없으면 자기 자신
            else:
                guides_x.append(get_transforms(224)(Image.open(gp).convert("RGB")).unsqueeze(0).to(device))
        guide_batch = torch.cat(guides_x, 0)  # [B,C,H,W]

        # (B) FF용 anchor(부정) 배치 구성: '다른 클래스' 가이드 중 최근접
        def build_anchor_batch():
            anchors = []
            for j, t in enumerate(s_cls.tolist()):
                f = s_feat[j]  # [D]
                best_c, best_d = None, None
                for c, gf in guides_feat_bank.items():
                    if c == t:
                        continue
                    d = torch.norm(f - gf, p=2)
                    if (best_d is None) or (d < best_d):
                        best_c, best_d = c, d
                if best_c is None:
                    anchors.append(x_batch[j:j+1])  # fallback
                else:
                    anchors.append(guides_bank[best_c])
            return torch.cat(anchors, 0)
        anchor_batch = build_anchor_batch()  # [B,C,H,W] (FF에서만 사용)

        # 합성
        t0 = time.time()
        
        
        
        if attack == "rm":
            x_query = identity_attack(x_batch)
            params = {}
            
            
            
        elif attack == "pgd":
            eps   = float(cfg.get("epsilon", 8/255))
            alpha = float(cfg.get("alpha", 2/255))
            iters = int(cfg.get("iters", 20))
            with torch.no_grad():
                pred = s_logits.argmax(1)
            x_query = pgd_attack_normalized(
                x_batch, pred, student,
                eps_pixel=eps, alpha_pixel=alpha, iters=iters,
                clamp_fn=lambda z: normalized_clamp_(z, device)
            )
            params = {"epsilon": eps, "alpha": alpha, "iters": iters}
            

            
        elif attack == "cw":
            # configs/cw.json에서 읽어오기
            eps   = float(cfg.get("epsilon"))
            alpha = float(cfg.get("alpha"))
            iters = int(cfg.get("iters"))
            kappa = float(cfg.get("kappa"))
            with torch.no_grad():
                # 학생 현재 예측을 '멀어질' 레이블로 사용 (untargeted)
                pred = s_logits.argmax(1)
            from attacks.cw import cw_linf_attack
            x_query = cw_linf_attack(
                x_batch, pred, student,
                eps_pixel=eps, alpha_pixel=alpha, iters=iters, kappa=kappa,
                clamp_fn=lambda z: normalized_clamp_(z, device),
                min_iters=10
            )
            params = {"epsilon": eps, "alpha": alpha, "iters": iters, "kappa": kappa}
                    
                    
            
        elif attack == "fa":
            max_iter = int(cfg.get("max_iter", 200))
            eps      = float(cfg.get("epsilon", 8/255))  # (C)
            x_query = feature_adversary(
                x_batch, guide_batch, feature_fn,
                max_iter=max_iter, step=1.0,
                eps_pixel=eps,                        # (C)
                clamp_fn=lambda z: normalized_clamp_(z, device)
            )
            params = {"max_iter": max_iter, "epsilon": eps}
            
            
            
        elif attack == "ff":
            max_iter = int(cfg.get("max_iter", 200))
            a        = float(cfg.get("alpha_triplet", 0.2))
            lam      = float(cfg.get("lambda_triplet", 1e-3))
            eps      = float(cfg.get("epsilon", 8/255))  # (A)
            # 마이크로배치 (OOM 방지)
            FF_CHUNK = 8
            ff_chunk_used = FF_CHUNK
            outs = []
            for b0 in range(0, x_batch.size(0), FF_CHUNK):
                xb = x_batch[b0:b0+FF_CHUNK]
                gb = guide_batch[b0:b0+FF_CHUNK]
                ab = anchor_batch[b0:b0+FF_CHUNK]
                out_b = featurefool_lbfgs(
                    xb, gb, ab, feature_fn,
                    alpha_triplet=a, lambda_triplet=lam,
                    margin=None,                     # 필요 시 margin>0로 켜서 더 보수적으로
                    max_iter=max_iter,
                    eps_pixel=eps,                   # (A)
                    clamp_fn=lambda z: normalized_clamp_(z, device)
                )
                outs.append(out_b)
            x_query = torch.cat(outs, 0)
            params = {"max_iter": max_iter, "alpha_triplet": a, "lambda_triplet": lam,
                      "epsilon": eps, "ff_chunk": FF_CHUNK}
        
        
        
        else:
            raise ValueError("unknown attack")
        elapsed_ms = (time.time() - t0) * 1000.0

        # 오라클 질의 (이것만 비용)
        with torch.no_grad():
            o_probs = torch.softmax(oracle(x_query), dim=1)
            o_top1_p, o_top1 = o_probs.max(1)
            top5_p, top5_i = o_probs.topk(5, dim=1)

        # 로그
        for j, bi in enumerate(batch_idx):
            top5 = [{"label": int(top5_i[j, k].item()), "prob": float(top5_p[j, k].item())} for k in range(5)]
            qlog.log(step + j, bi, attack, params, int(o_top1[j].item()), float(o_top1_p[j].item()), top5, elapsed_ms)

        # 버퍼 적재 + 주기적 미세학습
        Ds_x.append(x_query.detach().cpu())
        Ds_y.append(o_top1.detach().cpu())
        if sum(t.shape[0] for t in Ds_x) >= update_every or step + len(batch_idx) >= budget:
            X = torch.cat(Ds_x, 0); Y = torch.cat(Ds_y, 0)
            dl = DataLoader(TensorDataset(X, Y), batch_size=batch_update, shuffle=True)
            student.train()
            for _ in range(epochs_per_update):
                for xb, yb in dl:
                    xb, yb = xb.to(device), yb.to(device)
                    logits = student(xb)
                    loss = crit(logits, yb)
                    optim.zero_grad(set_to_none=True)
                    loss.backward()
                    optim.step()

        # 인덱스 갱신
        for bi in batch_idx:
            if bi in remaining:
                remaining.remove(bi)
        step += len(batch_idx)
        pbar.update(len(batch_idx))

    pbar.close()

    # ===== 평가 =====
    class DTestEval(Dataset):
        def __init__(self, paths, labels):
            self.paths = paths
            self.labels = labels
            self.tfm = get_transforms(224)
        def __len__(self): return len(self.paths)
        def __getitem__(self, idx):
            from PIL import Image
            return self.tfm(Image.open(self.paths[idx]).convert("RGB")), self.labels[idx]
    dtest_loader = DataLoader(DTestEval(d_paths, d_labels), batch_size=64, shuffle=False, num_workers=2)

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

    # 저장
    os.makedirs(out_dir, exist_ok=True)

    torch.save(student.state_dict(), os.path.join(out_dir, "student.pth"))

    metrics = {
        "attack": attack,
        "budget": budget_key,
        "seed": seed, # 재현/추적을 위한 메타데이터

        "seed_queries": seed_q,
        "total_queries": step,
        "update_every": update_every,
        "epochs_per_update": epochs_per_update,

        # 성능 지표
        "accuracy": acc,
        "ATE": ate
    }

    # FF에서만 의미 있는 청크 크기(다른 공격이면 None)
    if ff_chunk_used is not None:
        metrics["ff_chunk"] = ff_chunk_used

    save_metrics(out_dir, metrics)
    
    return {"accuracy": acc, "ATE": ate, "seed_queries": seed_q, "total_queries": step}

