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
from attacks.cw import cw_l2_attack
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
    논문 프로토콜 정렬:
    - 가이드는 시드 질의(예산 내)에서만 구축
    - 배치 내 가이드 클래스는 '학생 예측'으로 선택 (추가 오라클 호출 X)
    - 쿼리/라벨링은 오직 x_query에 대해서만 수행 (로그/예산에 반영)
    """
    random.seed(seed); torch.manual_seed(seed)
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(save_dir, exist_ok=True)

    # 데이터 불러오기
    train_ds, test_ds = load_gtsrb_imagefolder(data_dir, size=224)
    q_paths, (d_paths, d_labels) = split_qpool_dtest_from_test(test_ds, qpool_n=qpool_n, dtest_n=dtest_n, seed=seed)

    # 오라클 설정
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

    # ===== 가이드 저장소 (시드에서만 구축) =====
    guides = {}  # class_id -> path (시드 내 최확신 샘플)

    # 구성/로그/버퍼
    cfg_path = os.path.join("configs", f"{attack}.json")
    cfg = json.load(open(cfg_path, "r", encoding="utf-8")) if os.path.exists(cfg_path) else {}
    budget = BUDGET_MAP[budget_key]
    remaining = list(range(len(q_paths)))
    selected = []
    out_dir = os.path.join(save_dir, attack, budget_key)
    qlog = QueryLogger(out_dir)
    # FF에서 실제 사용한 마이크로배치 크기 기록용 (다른 공격이면 None)
    ff_chunk_used = None


    Ds_x = []  # 누적 입력(정규화 좌표)
    Ds_y = []  # 오라클 하드 라벨(top1)

    # 특징 추출 함수 (x_adv 쪽으로 grad 흘러야 하므로 no_grad 금지)
    def feature_fn(x_tensor):
        student.eval()
        logits, feat = student(x_tensor, return_features=True)
        return feat

    # ===== 시드 질의 X0 (예산 내) =====
    step = 0
    if seed_q > 0:
        k = min(seed_q, budget, len(remaining))
        seed_idx = query_indices_rm(remaining, k, random)

        xs = [get_transforms(224)(Image.open(q_paths[i]).convert("RGB")) for i in seed_idx]
        x_batch = torch.stack(xs, 0).to(device)

        # 시드 샘플은 '있는 그대로'를 쿼리로 본다(합성 없음) → 라벨링
        with torch.no_grad():
            o_probs = torch.softmax(oracle(x_batch), dim=1)
            o_top1 = o_probs.argmax(1)

        # 시드 질의 로깅
        for j, bi in enumerate(seed_idx):
            top5_p, top5_i = o_probs[j].topk(5)
            top5 = [{"label": int(top5_i[t].item()), "prob": float(top5_p[t].item())} for t in range(5)]
            qlog.log(j, bi, "seed", {}, int(o_top1[j].item()), float(o_probs[j].max().item()), top5, 0.0)

        # === 시드에서만 클래스별 가이드 구축 ===
        # 같은 클래스가 여러 개면 '확신도 최대'를 가이드로 고정
        for j, bi in enumerate(seed_idx):
            c = int(o_top1[j].item())
            conf = float(o_probs[j, c].item())
            p = q_paths[bi]
            if c not in guides or conf > guides[c][0]:
                guides[c] = (conf, p)
        # 경량화를 위해 path만 남김
        guides = {c: p for c, (conf, p) in guides.items()}

        # 버퍼 적재
        Ds_x.append(x_batch.detach().cpu())
        Ds_y.append(o_top1.detach().cpu())

        # 인덱스 갱신
        for bi in seed_idx:
            if bi in remaining:
                remaining.remove(bi)
                selected.append(bi)
        step = len(seed_idx)

        # 초기 미세학습(헤드만, 하드 CE)
        X = torch.cat(Ds_x, 0)
        Y = torch.cat(Ds_y, 0)
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
        # 배치 선택 (LC; 없으면 RM)
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

        # === 가이드 클래스는 '학생 예측'으로 결정 (추가 오라클 호출 없음) ===
        with torch.no_grad():
            s_cls = student(x_batch).argmax(1)  # [B]
        guides_x = []
        for j, cls in enumerate(s_cls.tolist()):
            gp = guides.get(cls, None)
            if gp is None:
                # 해당 클래스의 가이드가 아직 없다면, 현재 샘플 자체를 가이드로 사용
                guides_x.append(x_batch[j:j+1])
            else:
                guides_x.append(
                    get_transforms(224)(Image.open(gp).convert("RGB")).unsqueeze(0).to(device)
                )
        guide_batch = torch.cat(guides_x, 0)

        # 공격/합성
        t0 = time.time()
        if attack == "rm":
            x_query = identity_attack(x_batch)
            params = {}
        elif attack == "pgd":
            eps   = float(cfg.get("epsilon", 8/255))
            alpha = float(cfg.get("alpha", 2/255))
            iters = int(cfg.get("iters", 20))
            with torch.no_grad():
                pred = student(x_batch).argmax(1)
            x_query = pgd_attack_normalized(
                x_batch, pred, student,
                eps_pixel=eps, alpha_pixel=alpha, iters=iters,
                clamp_fn=lambda z: normalized_clamp_(z, device)
            )
            params = {"epsilon": eps, "alpha": alpha, "iters": iters}
        elif attack == "cw":
            kappa = float(cfg.get("kappa", 0.0))
            steps = int(cfg.get("max_steps", 200))
            lr    = float(cfg.get("lr", 5e-2))
            with torch.no_grad():
                pred = student(x_batch).argmax(1)
            x_query = cw_l2_attack(x_batch, pred, student, kappa=kappa, steps=steps, lr=lr)
            x_query = normalized_clamp_(x_query, device)
            params = {"kappa": kappa, "steps": steps, "lr": lr}
        elif attack == "fa":
            max_iter = int(cfg.get("max_iter", 200))
            x_query = feature_adversary(x_batch, guide_batch, feature_fn, max_iter=max_iter)
            x_query = normalized_clamp_(x_query, device)
            params = {"max_iter": max_iter}
        elif attack == "ff":
            max_iter = int(cfg.get("max_iter", 200))
            a  = float(cfg.get("alpha_triplet", 0.5))
            lam = float(cfg.get("lambda_triplet", 1e-3))

            # 마이크로배치로 FF 실행(메모리 절약)
            FF_CHUNK = 8  # 8GB GPU면 8 권장, OOM이면 4/2로 축소
            ff_chunk_used = FF_CHUNK
            outs = []
            for b0 in range(0, x_batch.size(0), FF_CHUNK):
                xb = x_batch[b0:b0+FF_CHUNK]
                gb = guide_batch[b0:b0+FF_CHUNK]
                ab = x_batch[b0:b0+FF_CHUNK]  # anchor = 원본
                out_b = featurefool_lbfgs(
                    xb, gb, ab, feature_fn,
                    alpha_triplet=a, lambda_triplet=lam, max_iter=max_iter,
                    clamp_fn=lambda z: normalized_clamp_(z, device)
                )
                outs.append(out_b)
            x_query = torch.cat(outs, 0)
            params = {"max_iter": max_iter, "alpha_triplet": a, "lambda_triplet": lam, "ff_chunk": FF_CHUNK}
        else:
            raise ValueError("unknown attack")
        elapsed_ms = (time.time() - t0) * 1000.0

        # === 오라클 라벨 획득 (이 쿼리만이 비용으로 집계됨) ===
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
            X = torch.cat(Ds_x, 0)
            Y = torch.cat(Ds_y, 0)
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
                selected.append(bi)
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

    s_preds = []; o_preds = []
    with torch.no_grad():
        for x, y in dtest_loader:
            x = x.to(device)
            s_preds += student(x).argmax(1).cpu().tolist()
            o_preds += oracle(x).argmax(1).cpu().tolist()
    ate = compute_ate(s_preds, o_preds)

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
