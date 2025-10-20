#!/usr/bin/env python3
import cobra, pandas as pd, os, traceback
from cobra.flux_analysis import flux_variability_analysis
from tqdm import tqdm

# 전체 조건 실행
test_mode = False
conds = pd.read_csv("data/conditions.tsv", sep="\t")
os.makedirs("data/fva_results", exist_ok=True)
os.makedirs("outputs", exist_ok=True)

# 공통 무기질 세트 (비성장 모델 방지)
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
        try:
            m = model.copy()

            # bound 초기화
            for ex in m.exchanges:
                try:
                    ex.lower_bound = min(0, ex.upper_bound)
                except Exception:
                    pass

            # 필수 무기질 추가
            for ex_id, lb in basic_nutrients.items():
                if ex_id in m.reactions:
                    m.reactions.get_by_id(ex_id).lower_bound = lb

            # 탄소원 / 산소 적용
            if r["carbon_source"] in m.reactions:
                m.reactions.get_by_id(r["carbon_source"]).lower_bound = -10
            if "EX_o2_e" in m.reactions:
                m.reactions.EX_o2_e.bounds = (r["O2_lb"], r["O2_ub"])

            # FVA 수행
            sol = flux_variability_analysis(m, fraction_of_optimum=0.9)
            sol["cond_id"] = r["cond_id"]
            results.append(sol)

        except Exception as e:
            err_msg = f"[{model_name}:{r['cond_id']}] ❌ {type(e).__name__}: {e}"
            print(err_msg)
            with open("outputs/fva_error_log.txt", "a") as f:
                f.write(err_msg + "\n")
            continue  # → 다음 조건으로 넘어감

    if results:
        out_path = f"data/fva_results/{model_name}.parquet"
        pd.concat(results).to_parquet(out_path)
        print(f"✅ Saved → {out_path}")
    else:
        print(f"⚠️  No valid FVA results for {model_name}.")
