#!/usr/bin/env python3
import cobra, pandas as pd
from cobra.flux_analysis import flux_variability_analysis

# 테스트 조건 로드
conds = pd.read_csv("data/conditions_test.tsv", sep="\t").iloc[0]
model_path = f"data/{conds.organism}.xml"
model = cobra.io.read_sbml_model(model_path)

# 조건 적용
if conds.carbon_source in model.reactions:
    model.reactions.get_by_id(conds.carbon_source).lower_bound = -10
if "EX_o2_e" in model.reactions:
    model.reactions.EX_o2_e.bounds = (conds.O2_lb, conds.O2_ub)

# FBA & FVA
sol = model.optimize()
print(f"✅ Biomass flux: {sol.objective_value:.4f}")

fva = flux_variability_analysis(model, fraction_of_optimum=0.9)
print(fva.head(10))  # 상위 10개 반응만 표시
fva.to_csv("outputs/test_FVA_result.tsv", sep="\t")
print("✅ Saved outputs/test_FVA_result.tsv")
