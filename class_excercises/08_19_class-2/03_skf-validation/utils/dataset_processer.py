import os
import pandas as pd
import matplotlib.pyplot as plt

def process_dataset_datetime(dataset_path: str, datetime_col_name: str, plot_consumptions: bool = False) -> pd.DataFrame:

  # Load the dataset

  df = pd.read_csv(dataset_path, parse_dates=[datetime_col_name])
  df = df.sort_values("Datetime").reset_index(drop=True)

  # Derive better calendar features from the Datetime column

  df["hour"] = df["Datetime"].dt.hour
  df["dayofweek"] = df["Datetime"].dt.dayofweek
  df["dayofyear"] = df["Datetime"].dt.dayofyear
  df["week"] = df["Datetime"].dt.isocalendar().week
  df["month"] = df["Datetime"].dt.month
  df["year"] = df["Datetime"].dt.year

  # Compute periodic averages

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

  # Create target variables

  df["target_daily"] = (df["AEP_MW"] > df["daily_avg"]).astype(int)
  df["target_weekly"] = (df["AEP_MW"] > df["weekly_avg"]).astype(int)
  df["target_monthly"] = (df["AEP_MW"] > df["monthly_avg"]).astype(int)
  df["target_yearly"] = (df["AEP_MW"] > df["yearly_avg"]).astype(int)

   # Flavour plot

  if plot_consumptions:

    df_filtered = df[df["Datetime"] >= "2018-01-01"]

    plt.figure(figsize=(15,6))

    # Blue line: actual consumption
    plt.plot(df_filtered["Datetime"], df_filtered["AEP_MW"], color="blue", label="Hourly Consumption", linewidth=0.4)

    # Red line: yearly average (step-like constant within each year)
    plt.plot(df_filtered["Datetime"], df_filtered["monthly_avg"], color="red", label="Monthly Average", linewidth=1)

    plt.xlabel("Datetime")
    plt.ylabel("Consumption (MW)")
    plt.title("Hourly Consumption vs Monthly Average past 2018")
    plt.legend()
    plt.show() # TODO: fix this. It's blocking code until the figure is manually closed. Why?

  return df