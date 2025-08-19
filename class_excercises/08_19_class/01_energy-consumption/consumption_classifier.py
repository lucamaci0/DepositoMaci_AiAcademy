import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report

#####################################################

# Setup before running

datasets_dir = "other/Archivio Datasets/02_Lesson"
dataset = "AEP_hourly.csv"

should_plot_consumptions = False
should_plot_decision_tree = True

# Choose decision-tree target feature:
# "target_daily", "target_weekly", "target_monthly", "target_yearly"
target = "target_daily"

#####################################################

dataset_dir = os.path.join(datasets_dir,dataset)

df = pd.read_csv(dataset_dir, parse_dates=["Datetime"])
df = df.sort_values("Datetime").reset_index(drop=True)


# Derive better calendar features from the Datetime column

df["hour"] = df["Datetime"].dt.hour
df["dayofweek"] = df["Datetime"].dt.dayofweek
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


# Creation of target variables

df["target_daily"] = (df["AEP_MW"] > df["daily_avg"]).astype(int)
df["target_weekly"] = (df["AEP_MW"] > df["weekly_avg"]).astype(int)
df["target_monthly"] = (df["AEP_MW"] > df["monthly_avg"]).astype(int)
df["target_yearly"] = (df["AEP_MW"] > df["yearly_avg"]).astype(int)

 # Flavour plot

if should_plot_consumptions:

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


# Different approach: using a decision tree

### Choosing training and target features

X = df[["hour", "dayofweek", "month", "year"]]
y = df[target]

### Split del dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, stratify=y, test_size=0.2, random_state=42
)

dec_tree = DecisionTreeClassifier(max_depth=4, random_state=42)
dec_tree.fit(X_train, y_train)
y_pred_tree = dec_tree.predict(X_test)

print("Decision Tree: (close figure to continue)")

if should_plot_decision_tree:

  plt.figure(figsize=(40, 4), dpi=80)
  tree.plot_tree(
    dec_tree,
    filled=True,
    feature_names=X.columns.astype(str).tolist(),
    class_names=[str(c) for c in dec_tree.classes_],
    rounded=True,
    fontsize=7,
    proportion=True,
    precision=2
  )
  plt.tight_layout()
  plt.show()

print(classification_report(y_test, y_pred_tree, digits=3))
