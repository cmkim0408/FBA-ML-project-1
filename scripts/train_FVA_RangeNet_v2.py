#!/usr/bin/env python3
import torch, torch.nn as nn, pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import r2_score, mean_squared_error
import numpy as np, os

# 1️⃣ 데이터 로드
df = pd.read_parquet("data/merged_dataset.parquet")
print(f"✅ Loaded dataset: {df.shape}")

# 2️⃣ Feature Engineering
# min/max 평균 → target
Y = df[["minimum", "maximum"]].mean(axis=1).values.reshape(-1,1)

# feature 추출 (조건 기반)
X = pd.get_dummies(df[["source_file"]], drop_first=True)
# feature merge: FVA condition에서 얻을 수 있는 정보 추가
# (여기서는 O2 bound / carbon_source 등 추가 가능)
if "O2_lb" in df.columns and "O2_ub" in df.columns:
    X["O2_lb"] = df["O2_lb"]
    X["O2_ub"] = df["O2_ub"]
else:
    X["O2_lb"] = np.random.choice([-20,-10,0], len(X))
    X["O2_ub"] = np.random.choice([-5,-1,0], len(X))
# carbon source placeholder (없을 경우 dummy)
if "carbon_source" in df.columns:
    X = X.join(pd.get_dummies(df["carbon_source"], prefix="Csrc"))
else:
    X["Csrc_glc"] = 1  # assume glucose if missing

print(f"Feature matrix: {X.shape[1]} columns")

# 3️⃣ Train/Test split + Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X.values)
xtr, xte, ytr, yte = train_test_split(X_scaled, Y, test_size=0.1, random_state=42)

train_ds = TensorDataset(torch.tensor(xtr).float(), torch.tensor(ytr).float())
test_ds  = TensorDataset(torch.tensor(xte).float(), torch.tensor(yte).float())

# 4️⃣ 모델 구조
model = nn.Sequential(
    nn.Linear(X.shape[1], 512), nn.ReLU(),
    nn.Linear(512, 256), nn.ReLU(),
    nn.Linear(256, 64), nn.ReLU(),
    nn.Linear(64, 1)
)

opt = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn = nn.MSELoss()

# 5️⃣ 학습 루프
for ep in range(40):
    model.train()
    for xb, yb in DataLoader(train_ds, batch_size=128, shuffle=True):
        opt.zero_grad()
        loss = loss_fn(model(xb), yb)
        loss.backward()
        opt.step()
    # validation
    model.eval()
    with torch.no_grad():
        val_preds = model(torch.tensor(xte).float())
        val_loss = loss_fn(val_preds, torch.tensor(yte).float()).item()
    print(f"Epoch {ep+1:02d} TrainLoss={loss.item():.4f} ValLoss={val_loss:.4f}")

# 6️⃣ 평가
with torch.no_grad():
    preds = model(torch.tensor(xte).float()).numpy()
r2 = r2_score(yte, preds)
rmse = mean_squared_error(yte, preds, squared=False)
print(f"✅ Final R²={r2:.4f}, RMSE={rmse:.4f}")

# 7️⃣ 저장
os.makedirs("outputs", exist_ok=True)
torch.save(model.state_dict(), "outputs/FVA_RangeNet_v2.pt")
print("✅ Model saved → outputs/FVA_RangeNet_v2.pt")
