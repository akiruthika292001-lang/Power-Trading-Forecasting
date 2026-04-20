import os

files = os.listdir(r"D:\chandra")
print(files)

import pandas as pd

weather = pd.read_excel(r"D:\chandra\Weather_data.xlsx")

print(weather.shape)
print(weather.head())

weather['time'] = pd.to_datetime(weather['time'])

weather = weather.sort_values('time')

print(weather['time'].dtype)
print(weather.head())

print(weather['state'].value_counts())

weather_all = []

for city, df_city in weather.groupby("state"):
    
    df_city = df_city.sort_values("time")
    
    df_city = df_city.set_index("time")
    
    
    df_15 = df_city.resample("15T").interpolate(method="time")

    df_15["state"] = city
    
    weather_all.append(df_15)

weather_15min = pd.concat(weather_all)

weather_15min = weather_15min.reset_index()

print(weather_15min.shape)
print(weather_15min.head())

weather_clean = weather_15min.copy()

weather_india = weather_clean.groupby("time").mean(numeric_only=True).reset_index()

print(weather_india.shape)
print(weather_india.head())

weather_india = weather_15min.groupby("time").mean(numeric_only=True).reset_index()
print(weather_india.shape)
weather_15min.to_csv(r"D:\chandra\weather_15min.csv", index=False)
weather_india.to_csv(r"D:\chandra\weather_india.csv", index=False)


import pandas as pd

iex = pd.read_excel(r"D:\chandra\IEX_DATA.xlsx")

iex['Date'] = pd.to_datetime(iex['Date'], dayfirst=True)

iex.head()

iex['start_time'] = iex['Time Block'].str.split('-').str[0].str.strip()

iex.head()

iex['time'] = pd.to_datetime(
    iex['Date'].dt.strftime('%Y-%m-%d') + ' ' + iex['start_time']
)

iex = iex.sort_values('time')

print(iex[['Date','Time Block','time']].head(10))

print(iex['time'].duplicated().sum())
iex.to_csv(r"D:\chandra\IEX_CLEAN_FINAL.csv", index=False)

import pandas as pd

iex = pd.read_csv(r"D:\chandra\IEX_CLEAN_FINAL.csv")
weather = pd.read_csv(r"D:\chandra\weather_15min.csv")

iex['time'] = pd.to_datetime(iex['time'])
weather['time'] = pd.to_datetime(weather['time'])

print(iex.shape, weather.shape)

iex['time'] = pd.to_datetime(iex['time'], dayfirst=True, errors='coerce')
weather['time'] = pd.to_datetime(weather['time'], errors='coerce')
print(iex['time'].isna().sum())
print(weather['time'].isna().sum())

iex = iex.sort_values("time")
weather = weather.sort_values("time")

final_df = pd.merge_asof(
    iex,
    weather,
    on="time",
    direction="nearest"
)

print(final_df.shape)
print(final_df.head())

print(final_df.isnull().sum())
final_df.to_csv(r"D:\chandra\IEX_WEATHER_MERGED.csv", index=False)

import pandas as pd

df = pd.read_csv(r"D:\chandra\IEX_WEATHER_MERGED.csv")

print(df.shape)
print(df.head())

df['time'] = pd.to_datetime(df['time'])
print(df['time'].dtype)
print(df.columns)

df = df.sort_values("time").reset_index(drop=True)

print(df.head())

df = df.set_index("time")

print(df.index)

time_diff = df.index.to_series().diff().value_counts()
print(time_diff)

print("Duplicate timestamps:", df.index.duplicated().sum())

full_range = pd.date_range(start=df.index.min(), end=df.index.max(), freq="15min")
missing = len(full_range) - len(df)

print("Expected rows:", len(full_range))
print("Actual rows:", len(df))
print("Missing timestamps:", missing)

full_range = pd.date_range(
    start=df.index.min(),
    end=df.index.max(),
    freq="15min"
)

df = df.reindex(full_range)
df.index.name = "time"

print(df.isnull().sum())
print(df.shape)

market_cols = [
    "Purchase Bid (MW)",
    "Sell Bid (MW)",
    "MCV (MW)",
    "Final Scheduled Volume (MW)",
    "MCP (Rs/MWh) *"
]

df[market_cols] = df[market_cols].ffill()

weather_cols = [
    "temperature_2m (Â°C)",
    "relative_humidity_2m (%)",
    "precipitation (mm)",
    "wind_speed_10m (km/h)",
    "cloud_cover (%)",
    "dew_point_2m (Â°C)",
    "wind_gusts_10m (km/h)",
    "surface_pressure (hPa)",
    "evapotranspiration (mm)",
    "visibility (m)"
]

df[weather_cols] = df[weather_cols].interpolate(method="time")

df = df.ffill()
df = df.bfill()

print(df.isnull().sum().sum())   # should be 0
print(df.shape)

df.columns = (
    df.columns
    .str.strip()
    .str.lower()
    .str.replace(r"[^\w\s]", "", regex=True)
    .str.replace(r"\s+", "_", regex=True)
)

print(df.columns)

print("Total missing values:", df.isnull().sum().sum())
print(df.isnull().sum().sort_values(ascending=False).head(10))

print("Duplicate rows:", df.duplicated().sum())
print("Index uniqueness:", df.index.is_unique)
print("Shape:", df.shape)

print(df.dtypes)
df.columns = df.columns.str.replace("âc", "c", regex=False)
df.columns = df.columns.str.replace("__", "_", regex=False)
df.columns = df.columns.str.strip("_")

print(df.columns)

df = df.rename(columns={

    "mcp_rsmwh_": "mcp",

    "purchase_bid_mw": "purchase_bid",
    "sell_bid_mw": "sell_bid",
    "mcv_mw": "mcv",
    "final_scheduled_volume_mw": "scheduled_volume",

    "temperature_2m_c": "temp",
    "relative_humidity_2m_": "humidity",
    "precipitation_mm": "precipitation",
    "wind_speed_10m_kmh": "wind_speed",
    "cloud_cover_": "cloud_cover",
    "dew_point_2m_c": "dew_point",
    "wind_gusts_10m_kmh": "wind_gust",
    "surface_pressure_hpa": "pressure",
    "evapotranspiration_mm": "evapotranspiration",
    "visibility_m": "visibility"
})

print(df.columns)

df = df.rename(columns={
    "mcp_rsmwh": "mcp"
})

print([col for col in df.columns if "mcp" in col])
df = pd.get_dummies(df, columns=["state"], drop_first=True)


df.shape
df.info()
df.describe()

print("Start:", df.index.min())
print("End:", df.index.max())
print("Total records:", len(df))
print("Expected 15-min blocks per day:", 96)

import matplotlib.pyplot as plt

plt.figure(figsize=(12,5))
plt.plot(df.index, df["mcp"])
plt.title("MCP Price Over Time (15-min Resolution)")
plt.xlabel("Time")
plt.ylabel("MCP (Rs/MWh)")
plt.show()

daily_mcp = df["mcp"].resample("D").mean()

plt.figure(figsize=(12,5))
plt.plot(daily_mcp)
plt.title("Daily Average MCP Trend")
plt.show()

weekly_mcp = df["mcp"].resample("W").mean()

plt.figure(figsize=(12,5))
plt.plot(weekly_mcp)
plt.title("Weekly MCP Trend")
plt.show()

df["hour"] = df.index.hour

hourly_pattern = df.groupby("hour")["mcp"].mean()

plt.figure(figsize=(10,5))
plt.plot(hourly_pattern)
plt.title("Hourly MCP Pattern (Daily Cycle)")
plt.xlabel("Hour")
plt.ylabel("Average MCP")
plt.show()

import seaborn as sns

plt.figure(figsize=(12,8))
sns.heatmap(df.corr(numeric_only=True), annot=False, cmap="coolwarm")
plt.title("Feature Correlation Matrix")
plt.show()

plt.scatter(df["temp"], df["mcp"], alpha=0.3)
plt.title("Temperature vs MCP")
plt.xlabel("Temperature")
plt.ylabel("MCP")
plt.show()

plt.figure(figsize=(10,5))
sns.histplot(df["mcp"], bins=50, kde=True)
plt.title("MCP Distribution")
plt.show()

df["hour"] = df.index.hour

peak = df.groupby("hour")["mcp"].mean().sort_values(ascending=False)
print(peak.head(10))

df.to_csv(r"D:\chandra\IEX_WEATHER_PREPROCESSED.csv", index=True)
df.to_parquet(r"D:\chandra\IEX_WEATHER_PREPROCESSED.parquet")
test = pd.read_csv(r"D:\chandra\IEX_WEATHER_PREPROCESSED.csv")
print(test.shape)
print(test.head())

import pandas as pd

df = pd.read_csv(r"D:\chandra\IEX_WEATHER_PREPROCESSED.csv")

print("Shape:", df.shape)
print("Missing values:", df.isnull().sum().sum())
print("Duplicate rows:", df.duplicated().sum())

df = pd.read_csv(r"D:\chandra\IEX_WEATHER_PREPROCESSED.csv")

