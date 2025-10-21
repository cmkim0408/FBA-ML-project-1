#!/usr/bin/env python3
"""
Safe re-training of FVA_RangeNet_v2
→ epoch마다 checkpoint 저장, RMSE 계산 시 호환성 보장
"""

import torch, torch.nn as nn, pandas as pd, numpy as np, os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import r2_score, mean_squared_error

print("✅ Loading dataset ...")
df = pd.read_parquet("data/merged_dataset.parquet")
Y = df[["minimum", "maximum"]].mean(axis=1).values.reshape(-1,1)

# feature 확장
X = pd.get_dummies(df[["source_file"]], drop_first=True)
X["O2_lb"] = np.random.choice([-20,-10,0], len(X))
X["O2_ub"] = np.random.choice([-5,-1,0], len(X))
X["Csrc_glc"] = 1

# scaling + split
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X.values)
xtr, xte, ytr, yte = train_test_split(X_scaled, Y, test_size=0.1, random_state=42)

train_ds = TensorDataset(torch.tensor(xtr).float(), torch.tensor(ytr).float())
test_ds  = TensorDataset(torch.tensor(xte).float(), torch.tensor(yte).float())

# model
model = nn.Sequential(
    nn.Linear(X.shape[1], 512), nn.ReLU(),
    nn.Linear(512, 256), nn.ReLU(),
    nn.Linear(256, 64), nn.ReLU(),
    nn.Linear(64, 1)
)
opt = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn = nn.MSELoss()
os.makedirs("outputs", exist_ok=True)

# training loop
print("✅ Starting training ...")
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
    torch.save(model.state_dict(), f"outputs/FVA_RangeNet_v2_ep{ep+1}.pt")

print("✅ Training finished. Final checkpoint saved.")

with torch.no_grad():
    preds = model(torch.tensor(xte).float()).numpy()
r2 = r2_score(yte, preds)
rmse = np.sqrt(mean_squared_error(yte, preds))
print(f"✅ Final R²={r2:.4f}, RMSE={rmse:.4f}")

torch.save(model.state_dict(), "outputs/FVA_RangeNet_v2.pt")
print("✅ Model saved → outputs/FVA_RangeNet_v2.pt")
