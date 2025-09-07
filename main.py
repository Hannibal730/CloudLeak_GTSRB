import argparse, json, torch
from runner.attack_runner import run_experiment
if __name__ == "__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--save_dir", default="runs")
    ap.add_argument("--attack", required=True, choices=["rm","pgd","cw","fa","ff"])
    ap.add_argument("--budget", required=True, choices=["A","B","C"])
    ap.add_argument("--oracle_weights", default="models_ckpt/resnet50_gtsrb.pth")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--qpool", type=int, default=6000)
    ap.add_argument("--dtest", type=int, default=2000)
    ap.add_argument("--seed_q", type=int, default=70)
    ap.add_argument("--update_every", type=int, default=128)
    ap.add_argument("--epochs_per_update", type=int, default=1)
    ap.add_argument("--batch_update", type=int, default=64)
    args=ap.parse_args()
    res = run_experiment(
        data_dir=args.data_dir, save_dir=args.save_dir, attack=args.attack, budget_key=args.budget,
        oracle_weights=args.oracle_weights, seed=args.seed,
        qpool_n=args.qpool, dtest_n=args.dtest,
        seed_q=args.seed_q, update_every=args.update_every, epochs_per_update=args.epochs_per_update,
        batch_update=args.batch_update, device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    )
    print(json.dumps(res, indent=2, ensure_ascii=False))
