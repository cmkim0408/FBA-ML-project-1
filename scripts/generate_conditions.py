#!/usr/bin/env python3
import pandas as pd, random, os

models = [f.split(".xml")[0] for f in os.listdir("data") if f.endswith(".xml")]
carbon_sources = ["EX_glc__D_e", "EX_ac_e", "EX_glyc_e"]
oxygen_bounds  = [(-20, -5), (-10, -1), (0, 0)]
n_per_model = 200   # 모델당 200조건

conds = []
for model in models:
    for i in range(n_per_model):
        csrc = random.choice(carbon_sources)
        o2_lb, o2_ub = random.choice(oxygen_bounds)
        conds.append({
            "organism": model,
            "cond_id": f"{model}_c{i:03d}",
            "carbon_source": csrc,
            "O2_lb": o2_lb,
            "O2_ub": o2_ub
        })

os.makedirs("data", exist_ok=True)
pd.DataFrame(conds).to_csv("data/conditions.tsv", sep="\t", index=False)
print(f"✅ {len(conds)} conditions saved → data/conditions.tsv")
