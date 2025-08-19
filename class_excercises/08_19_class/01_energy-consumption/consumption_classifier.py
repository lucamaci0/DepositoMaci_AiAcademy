import os
import pandas as pd

datasets_dir = "other/Archivio Datasets/02_Lesson"
dataset = "AEP_hourly.csv"
dataset_dir = os.path.join(datasets_dir,dataset)

df = pd.read_csv(dataset_dir, parse_dates=["Datetime"])
print(df.head())

df["hour"] = df["Datetime"].dt.hour
# df["dayofweek"] = df["Datetime"].dt.dayofweek
df["dayofyear"] = df["Datetime"].dt.dayofyear
df["week"] = df["Datetime"].dt.isocalendar().week
df["month"] = df["Datetime"].dt.month
df["year"] = df["Datetime"].dt.year

df["daily_avg"] = (
    df.groupby([df["year"], df["dayofyear"]])["AEP_MW"]
      .transform("mean")
)
df["weekly_avg"] = (
    df.groupby([df["year"], df["week"]])["AEP_MW"]
      .transform("mean")
)
df["monthly_avg"] = (
    df.groupby([df["year"], df["month"]])["AEP_MW"]
      .transform("mean")
)
df["yearly_avg"] = (
    df.groupby("year")["AEP_MW"]
      .transform("mean")
)

df["target_daily"] = (df["AEP_MW"] > df["daily_avg"]).astype(int)
df["target_weekly"] = (df["AEP_MW"] > df["weekly_avg"]).astype(int)
df["target_monthly"] = (df["AEP_MW"] > df["monthly_avg"]).astype(int)
df["target_yearly"] = (df["AEP_MW"] > df["yearly_avg"]).astype(int)

print(df.head(50))

