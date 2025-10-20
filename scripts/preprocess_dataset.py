#!/usr/bin/env python3
import pandas as pd, glob, os

# 병합할 parquet 파일 경로
files = glob.glob("data/fva_results/*.parquet")
if not files:
    raise FileNotFoundError("❌ No FVA result files found in data/fva_results/")

print(f"✅ {len(files)} parquet files found. Merging...")

# 파일별 데이터 읽기
dfs = []
for f in files:
    try:
        df = pd.read_parquet(f)
        df["source_file"] = os.path.basename(f)
        dfs.append(df)
    except Exception as e:
        print(f"⚠️  Skipped {f}: {e}")

# 병합
merged = pd.concat(dfs, ignore_index=True)
out_path = "data/merged_dataset.parquet"
merged.to_parquet(out_path)
print(f"✅ Merged dataset saved → {out_path}")
print(f"🧾 Total rows: {merged.shape[0]}, columns: {merged.shape[1]}")
