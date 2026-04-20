import pandas as pd
import numpy as np

df = pd.read_csv(r"D:\chandra\IEX_WEATHER_PREPROCESSED.csv")


df["time"] = pd.to_datetime(df["time"])
df = df.sort_values("time").set_index("time")

target = "mcp"


df["lag_1"] = df[target].shift(1)
df["lag_4"] = df[target].shift(4)
df["lag_96"] = df[target].shift(96)
df["lag_192"] = df[target].shift(192)


df["hour"] = df.index.hour
df["day"] = df.index.day
df["month"] = df.index.month
df["weekday"] = df.index.weekday
df["is_weekend"] = (df["weekday"] >= 5).astype(int)

df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)


drop_cols = [
    "purchase_bid",
    "sell_bid",
    "mcv",
    "scheduled_volume",
    "state_bengalure",
    "state_chennai",
    "state_delhi",
    "state_hyderabad",
    "state_kolkata",
    "state_mumbai"
]

df = df.drop(columns=drop_cols, errors="ignore")


df = df.dropna()

print("FINAL SHAPE:", df.shape)

target = "mcp"
features = [c for c in df.columns if c != target]

print("Feature count:", len(features))
print(features)

df = df.copy()


df = df.drop(columns=["start_time"], errors="ignore")

target = "mcp"

features = [c for c in df.columns if c != target]

split = int(len(df) * 0.8)

train_df = df.iloc[:split]
test_df  = df.iloc[split:]

X_train = train_df[features]
y_train = train_df[target]

X_test = test_df[features]
y_test = test_df[target]

from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error
import numpy as np

model = XGBRegressor(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)

model.fit(X_train, y_train)

pred_train = model.predict(X_train)
pred_test  = model.predict(X_test)

train_mape = np.mean(np.abs((y_train - pred_train) / y_train)) * 100
test_mape  = np.mean(np.abs((y_test - pred_test) / y_test)) * 100

print("TRAIN MAPE:", train_mape)
print("TEST MAPE :", test_mape)


import numpy as np
import pandas as pd

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def metrics(name, y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    r2 = r2_score(y_true, y_pred)
    return [name, mae, rmse, mape, r2]



lr = LinearRegression()

rf = RandomForestRegressor(
    n_estimators=200,
    max_depth=15,
    random_state=42,
    n_jobs=-1
)

xgb = XGBRegressor(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)


lr.fit(X_train, y_train)
rf.fit(X_train, y_train)
xgb.fit(X_train, y_train)



pred_lr = lr.predict(X_test)
pred_rf = rf.predict(X_test)
pred_xgb = xgb.predict(X_test)


results = []

results.append(metrics("Linear Regression", y_test, pred_lr))
results.append(metrics("Random Forest", y_test, pred_rf))
results.append(metrics("XGBoost", y_test, pred_xgb))

comparison_df = pd.DataFrame(results, columns=["Model", "MAE", "RMSE", "MAPE", "R2"])


comparison_df = comparison_df.sort_values("RMSE")

print(comparison_df)

best_model_name = comparison_df.iloc[0]["Model"]
print("BEST MODEL:", best_model_name)


import json

feature_list = features  # your final feature list

with open("features.json", "w") as f:
    json.dump(feature_list, f)

print("Features saved")

from xgboost import XGBRegressor

final_model = XGBRegressor(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)

final_model.fit(X_train, y_train)

print("Model trained")

import joblib

joblib.dump(final_model, "model.pkl")

print("Model saved successfully")

meta = {
    "target": "mcp",
    "model": "XGBoost",
    "train_size": len(X_train),
    "test_size": len(X_test)
}

with open("meta.json", "w") as f:
    json.dump(meta, f)

print("Metadata saved")

import numpy as np
import pandas as pd
import joblib
import json

# load model
model = joblib.load("model.pkl")

# load features
with open("features.json", "r") as f:
    features = json.load(f)
    
    
import seaborn as sns

plt.figure(figsize=(10,6))

sns.histplot(df["mcp"], bins=50, kde=True)

plt.title("Distribution of Market Clearing Price (MCP)", fontsize=14)
plt.xlabel("MCP (Rs/MWh)", fontsize=12)
plt.ylabel("Density", fontsize=12)

plt.grid(True)

plt.tight_layout()
plt.show()

import matplotlib.pyplot as plt

forecast = [9289.39, 7357.06, 6512.74, 6064.33, 5990.59, 5687.24, 5521.74, 5494.77,
            5359.88, 5303.59, 5261.76, 5260.25, 5070.81, 5134.41, 4971.55, 4686.73]

plt.figure(figsize=(10,4))
plt.plot(forecast)
plt.title("96-Step MCP Forecast (24 Hours)")
plt.xlabel("Time Steps (15-min intervals)")
plt.ylabel("MCP Price (₹/MWh)")
plt.grid(True)
plt.show()