#!/usr/bin/env python3
import cobra, pandas as pd, os, traceback
from cobra.flux_analysis import flux_variability_analysis
from tqdm import tqdm

# 모든 조건 실행 모드
test_mode = False
conds = pd.read_csv("data/conditions.tsv", sep="\t")
os.makedirs("data/fva_results", exist_ok=True)

# 공통 무기질 설정 (non-growing 모델 안정화용)
basic_nutrients = {
    "EX_nh4_e": -10, "EX_pi_e": -10, "EX_so4_e": -10,
    "EX_fe3_e": -0.01, "EX_na1_e": -1000, "EX_k_e": -1000,
    "EX_ca2_e": -1000, "EX_mg2_e": -1000, "EX_cl_e": -1000,
    "EX_cobalt2_e": -1000
}

for model_name in conds["organism"].unique():
    model_path = f"data/{model_name}.xml"
    model = cobra.io.read_sbml_model(model_path)
    subset = conds[conds.organism == model_name]
    if test_mode:
        subset = subset.head(1)
    results = []

    print(f"\n▶ Running FVA for {model_name} ({len(subset)} conditions)")
    for _, r in tqdm(subset.iterrows(), total=len(subset)):
        m = model.copy()
        # 안전한 bound 초기화
        for ex in m.exchanges:
            try:
                ex.lower_bound = min(0, ex.upper_bound)
            except Exception:
                pass
        # 공통 영양소 추가
        for ex_id, lb in basic_nutrients.items():
            if ex_id in m.reactions:
                m.reactions.get_by_id(ex_id).lower_bound = lb
        # 기질/산소 적용
        if r["carbon_source"] in m.reactions:
            m.reactions.get_by_id(r["carbon_source"]).lower_bound = -10
        if "EX_o2_e" in m.reactions:
            m.reactions.EX_o2_e.bounds = (r["O2_lb"], r["O2_ub"])

        try:
            sol = flux_variability_analysis(m, fraction_of_optimum=0.9)
            sol["cond_id"] = r["cond_id"]
            results.append(sol)
        except Exception as e:
            err = f"⚠️  Skipped {r['cond_id']} ({str(e)})"
            print(err)
            with open("outputs/fva_skipped.log", "a") as f:
                f.write(err + "\n")
            continue

    if results:
        out_path = f"data/fva_results/{model_name}.parquet"
        pd.concat(results).to_parquet(out_path)
        print(f"✅ Saved → {out_path}")
    else:
        print(f"⚠️  No valid FVA results for {model_name}.")
