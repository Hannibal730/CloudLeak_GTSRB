# CloudLeak GTSRB (PyTorch 2.5.1)
- GTSRB / 오라클 ResNet50 / 대체 VGG19-DeepID
- 공격: RM, PGD(L∞), CW(L2), FA, FF(Triplet@last-hidden + LBFGS 근사)
- LC(Least-Confidence) 액티브 러닝, 예산 A/B/C = 430/1290/2150, A ⊂ B ⊂ C
- 출력: `runs/{attack}/{budget}/` (queries.csv, metrics.json)
실행법은 README 최하단 또는 main.py 도움말 참조.



python -m oracle.train_oracle   --data_dir ./data/GTSRB   --epochs 2   --out models_ckpt/resnet50_gtsrb.pth

python scripts/evaluate.py --data_dir ./data/GTSRB --ckpt runs/ff/C/student.pth 

python main.py --data_dir ./data/GTSRB --attack ff --budget A   --oracle_weights models_ckpt/resnet50_gtsrb.pth   --seed_q 70 --update_every 128 --epochs_per_update 1

python main.py --data_dir ./data/GTSRB --attack ff --budget B   --oracle_weights models_ckpt/resnet50_gtsrb.pth   --seed_q 70 --update_every 128 --epochs_per_update 1

python main.py --data_dir ./data/GTSRB --attack ff --budget C   --oracle_weights models_ckpt/resnet50_gtsrb.pth   --seed_q 70 --update_every 128 --epochs_per_update 1


fill models_ckpt, data