import json, os
def compute_ate(student_preds, oracle_preds):
    mismatches=sum(1 for s,o in zip(student_preds, oracle_preds) if s!=o)
    return mismatches/max(1,len(student_preds))
def save_metrics(save_dir: str, metrics: dict):
    os.makedirs(save_dir, exist_ok=True)
    with open(os.path.join(save_dir,"metrics.json"),"w",encoding="utf-8") as f:
        json.dump(metrics,f,indent=2,ensure_ascii=False)
