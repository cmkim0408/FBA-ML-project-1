#!/usr/bin/env python3
import cobra, pandas as pd, os, traceback, time, signal, sys
from cobra.flux_analysis import flux_variability_analysis
from tqdm import tqdm
from multiprocessing import Process, Queue

# 전체 조건 실행
test_mode = False
conds = pd.read_csv("data/conditions.tsv", sep="\t")
os.makedirs("data/fva_results", exist_ok=True)
os.makedirs("outputs", exist_ok=True)

basic_nutrients = {
    "EX_nh4_e": -10, "EX_pi_e": -10, "EX_so4_e": -10,
    "EX_fe3_e": -0.01, "EX_na1_e": -1000, "EX_k_e": -1000,
    "EX_ca2_e": -1000, "EX_mg2_e": -1000, "EX_cl_e": -1000,
    "EX_cobalt2_e": -1000
}

def run_fva_worker(model_xml, cond, q):
    """독립 프로세스에서 FVA 수행"""
    try:
        m = cobra.io.read_sbml_model(model_xml)
        for ex in m.exchanges:
            try:
                ex.lower_bound = min(0, ex.upper_bound)
            except Exception:
                pass
        for ex_id, lb in basic_nutrients.items():
            if ex_id in m.reactions:
                m.reactions.get_by_id(ex_id).lower_bound = lb
        if cond["carbon_source"] in m.reactions:
            m.reactions.get_by_id(cond["carbon_source"]).lower_bound = -10
        if "EX_o2_e" in m.reactions:
            m.reactions.EX_o2_e.bounds = (cond["O2_lb"], cond["O2_ub"])
        sol = flux_variability_analysis(m, fraction_of_optimum=0.9)
        sol["cond_id"] = cond["cond_id"]
        q.put(sol)
    except Exception as e:
        q.put(e)

def run_fva_with_timeout(model_xml, cond, timeout=30):
    """timeout 기능 포함된 안전 실행"""
    q = Queue()
    p = Process(target=run_fva_worker, args=(model_xml, cond, q))
    p.start()
    p.join(timeout)
    if p.is_alive():
        p.terminate()
        p.join()
        raise TimeoutError("FVA timeout")
    result = q.get() if not q.empty() else None
    if isinstance(result, Exception):
        raise result
    return result

for model_name in conds["organism"].unique():
    model_path = f"data/{model_name}.xml"
    subset = conds[conds.organism == model_name]
    if test_mode:
        subset = subset.head(1)
    results = []

    print(f"\n▶ Running FVA for {model_name} ({len(subset)} conditions)")
    for _, r in tqdm(subset.iterrows(), total=len(subset)):
        try:
            res = run_fva_with_timeout(model_path, r, timeout=45)
            results.append(res)
        except Exception as e:
            err = f"[{model_name}:{r['cond_id']}] ❌ {type(e).__name__}: {e}"
            print(err)
            with open("outputs/fva_error_log.txt", "a") as f:
                f.write(err + "\n")
            continue

    if results:
        out_path = f"data/fva_results/{model_name}.parquet"
        pd.concat(results).to_parquet(out_path)
        print(f"✅ Saved → {out_path}")
    else:
        print(f"⚠️  No valid FVA results for {model_name}.")
