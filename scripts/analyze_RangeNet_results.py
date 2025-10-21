#!/usr/bin/env python3
import torch, torch.nn as nn, pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import r2_score, mean_squared_error
import numpy as np, os

# 데이터 및 모델 로드
df = pd.read_parquet("data/merged_dataset.parquet")
model_files = [f for f in os.listdir("outputs") if f.endswith(".pt")]
if not model_files:
    raise FileNotFoundError("❌ 학습된 모델(FVA_RangeNet.pt)이 없습니다.")
MODEL_PATH = os.path.join("outputs", model_files[0])

# 입력 데이터 구성 (train_FVA_RangeNet.py 와 동일하게)
Y = df[["minimum", "maximum"]].mean(axis=1).values.reshape(-1,1)
X = pd.get_dummies(df[["source_file"]], drop_first=True)
X_tensor = torch.tensor(X.values).float()

# 모델 구조 (train_FVA_RangeNet.py 과 동일해야 함)
model = nn.Sequential(nn.Linear(X.shape[1],256), nn.ReLU(),
                      nn.Linear(256,128), nn.ReLU(),
                      nn.Linear(128,1))
model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
model.eval()

# 예측
with torch.no_grad():
    preds = model(X_tensor).numpy()

# 평가 지표
r2 = r2_score(Y, preds)
import numpy as np
rmse = np.sqrt(mean_squared_error(Y, preds))
print(f"✅ R2={r2:.4f}, RMSE={rmse:.4f}")

# 산점도
plt.figure(figsize=(6,6))
sns.scatterplot(x=Y.flatten(), y=preds.flatten(), alpha=0.3, s=10)
plt.xlabel("True FVA mean flux")
plt.ylabel("Predicted flux (RangeNet)")
plt.title(f"FVA-RangeNet Performance\nR2={r2:.3f}, RMSE={rmse:.3f}")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("outputs/FVA_RangeNet_scatter.png", dpi=300)
plt.show()

# 분포 비교
plt.figure(figsize=(6,4))
sns.kdeplot(Y.flatten(), label="True", shade=True)
sns.kdeplot(preds.flatten(), label="Predicted", shade=True)
plt.xlabel("Flux value")
plt.title("Distribution Comparison (True vs Predicted)")
plt.legend()
plt.tight_layout()
plt.savefig("outputs/FVA_RangeNet_distribution.png", dpi=300)
plt.show()

print("✅ Figures saved → outputs/FVA_RangeNet_scatter.png, FVA_RangeNet_distribution.png")
