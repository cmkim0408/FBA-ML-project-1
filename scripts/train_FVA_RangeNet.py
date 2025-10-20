#!/usr/bin/env python3
import torch, torch.nn as nn, pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

# 데이터 로드
df = pd.read_parquet("data/merged_dataset.parquet")
print("✅ Loaded:", df.shape)

# 예시: min/max 평균값을 target으로 학습 (필요시 변경)
Y = df[["minimum", "maximum"]].mean(axis=1).values.reshape(-1,1)

# 간단히 cond_id 와 source_file 을 숫자형으로 인코딩
X = pd.get_dummies(df[["source_file"]], drop_first=True)
xtr, xte, ytr, yte = train_test_split(X.values, Y, test_size=0.1, random_state=42)

train_ds = TensorDataset(torch.tensor(xtr).float(), torch.tensor(ytr).float())
test_ds  = TensorDataset(torch.tensor(xte).float(), torch.tensor(yte).float())

model = nn.Sequential(nn.Linear(X.shape[1],256), nn.ReLU(),
                      nn.Linear(256,128), nn.ReLU(),
                      nn.Linear(128,1))
opt = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn = nn.MSELoss()

for ep in range(30):
    for xb, yb in DataLoader(train_ds,64,shuffle=True):
        opt.zero_grad(); loss=loss_fn(model(xb), yb); loss.backward(); opt.step()
    with torch.no_grad():
        val = loss_fn(model(torch.tensor(xte).float()), torch.tensor(yte).float()).item()
    print(f"Epoch {ep+1:02d} TrainLoss={loss.item():.4f} ValLoss={val:.4f}")

torch.save(model.state_dict(),"outputs/FVA_RangeNet.pt")
print("✅ Model saved → outputs/FVA_RangeNet.pt")
