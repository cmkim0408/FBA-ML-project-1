#!/usr/bin/env python3
import cobra, pandas as pd, os
from cobra.flux_analysis import flux_variability_analysis
from tqdm import tqdm

# test_mode=True면 각 모델당 1개 조건만 실행
test_mode = True
conds = pd.read_csv("data/conditions.tsv", sep="\t")
os.makedirs("data/fva_results", exist_ok=True)

for model_name in conds["organism"].unique():
    model_path = f"data/{model_name}.xml"
    model = cobra.io.read_sbml_model(model_path)
    subset = conds[conds.organism == model_name]
    if test_mode:
        subset = subset.head(1)  # 각 모델당 1개 조건만
    results = []

    print(f"\n▶ Running FVA for {model_name} ({len(subset)} conditions)")
    for _, r in tqdm(subset.iterrows(), total=len(subset)):
        m = model.copy()
        for ex in m.exchanges:
            ex.lower_bound = 0
        if r["carbon_source"] in m.reactions:
            m.reactions.get_by_id(r["carbon_source"]).lower_bound = -10
        if "EX_o2_e" in m.reactions:
            m.reactions.EX_o2_e.bounds = (r["O2_lb"], r["O2_ub"])
        sol = flux_variability_analysis(m, fraction_of_optimum=0.9)
        sol["cond_id"] = r["cond_id"]
        results.append(sol)

    out_path = f"data/fva_results/{model_name}_test.parquet"
    pd.concat(results).to_parquet(out_path)
    print(f"✅ Saved → {out_path}")
