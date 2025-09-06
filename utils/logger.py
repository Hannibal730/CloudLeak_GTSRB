import os, csv, json
class QueryLogger:
    def __init__(self, save_dir: str):
        os.makedirs(save_dir, exist_ok=True)
        self.csv_path=os.path.join(save_dir,"queries.csv")
        if not os.path.exists(self.csv_path):
            with open(self.csv_path,"w",newline="",encoding="utf-8") as f:
                w=csv.writer(f); w.writerow(["step","pool_index","attack","params","oracle_top1","oracle_p_top1","oracle_top5_json","elapsed_ms"])
    def log(self, step, pool_index, attack, params, oracle_top1, p_top1, top5, elapsed_ms):
        with open(self.csv_path,"a",newline="",encoding="utf-8") as f:
            w=csv.writer(f); w.writerow([step,pool_index,attack,json.dumps(params,ensure_ascii=False),int(oracle_top1),f"{p_top1:.6f}",json.dumps(top5),f"{elapsed_ms:.2f}"])
