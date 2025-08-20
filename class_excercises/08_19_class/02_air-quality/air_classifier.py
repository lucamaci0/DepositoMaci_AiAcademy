import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score


# Dataset loading and cleaning

datasets_dir = "other/Archivio Datasets/02_Lesson"
dataset_name = "AirQualityUCI.csv"
dataset_path = os.path.join(datasets_dir, dataset_name)

df = pd.read_csv(
    dataset_path,
    sep=";",
    decimal=",",
    dtype={"Date": "string", "Time": "string"}
)

# Replace faulty "-200" entries with "NA"
df.replace(to_replace=[-200, "-200"], value=pd.NA, inplace=True)

# Drop the extra empty columns created by trailing ";;"
df = df.dropna(axis=1, how="all")

# Drop rows that are entirely empty (the lines made only of semicolons)
df = df.dropna(how="any")

# How I found out about the faulty "-200" value
"""
for col in df.columns:
  print(f"Column {col} has lowest unique values:")
  print(df[col].dropna().sort_values().unique()[:10])
"""

# Extract features from Date

df["hour"] = pd.to_datetime(df["Time"], format="%H.%M.%S", errors="raise").dt.hour
df["Date"] = pd.to_datetime(df["Date"], format="%d/%m/%Y", dayfirst=True, errors="raise")
df["dayofweek"] = df["Date"].dt.dayofweek
df["dayofyear"] = df["Date"].dt.dayofyear
df["week"]      = df["Date"].dt.isocalendar().week
df["month"]     = df["Date"].dt.month
df["year"]      = df["Date"].dt.year

chemical_cols = ['CO(GT)', 'PT08.S1(CO)', 'NMHC(GT)', 'C6H6(GT)',
       'PT08.S2(NMHC)', 'NOx(GT)', 'PT08.S3(NOx)', 'NO2(GT)', 'PT08.S4(NO2)',
       'PT08.S5(O3)']

print(df.columns)


# Compute periodic averages and target values for each chemical

for chem in chemical_cols:

  df[f"{chem}_daily_avg"] = (
      df.groupby([df["year"], df["dayofyear"]])[chem]
        .transform("mean")
  )
  df[f"{chem}_target_daily"] = (df[chem] > df[f"{chem}_daily_avg"]).astype(int)
  
  df[f"{chem}_weekly_avg"] = (
      df.groupby([df["year"], df["week"]])[chem]
        .transform("mean")
  )
  df[f"{chem}_target_weekly"] = (df[chem] > df[f"{chem}_weekly_avg"]).astype(int)

  df[f"{chem}_monthly_avg"] = (
      df.groupby([df["year"], df["month"]])[chem]
        .transform("mean")
  )
  df[f"{chem}_target_monthly"] = (df[chem] > df[f"{chem}_monthly_avg"]).astype(int)
  
  df[f"{chem}_yearly_avg"] = (
      df.groupby("year")[chem]
        .transform("mean")
  )
  df[f"{chem}_target_yearly"] = (df[chem] > df[f"{chem}_yearly_avg"]).astype(int)


#############################################
# Choose features and target:
#############################################

chem = 'CO(GT)'
training_features = ["hour", "dayofweek", "month", "year", 'T', 'RH', 'AH', chem]
classification_target = f"{chem}_target_daily"

X = df[training_features]
y = df[classification_target]

### Split del dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, stratify=y, test_size=0.2, random_state=42
)


#############################################
### Model: Decision Tree
#############################################

dec_tree = DecisionTreeClassifier(max_depth=4, random_state=42)
dec_tree.fit(X_train, y_train)
y_pred_tree = dec_tree.predict(X_test)

print("Decision Tree (close figure to continue):")

plt.figure(figsize=(40, 4), dpi=80)
plot_tree(
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
plt.show() # TODO: fix this. It's blocking code until the figure is manually closed. Why?

print(classification_report(y_test, y_pred_tree, digits=3))

f1_tree = float(f1_score(y_test, y_pred_tree, average="macro"))
print(f"Decision Tree F1-score: {f1_tree}\n")


#############################################
### Model: Logistic Regression
#############################################

clf = LogisticRegression(max_iter=1000)
clf.fit(X_train, y_train)
y_pred_logistic = clf.predict(X_test)

print(f"Logistic Regression Coefficients:")
coefs = clf.coef_[0]
for i in range(len(coefs)):
  print(f"{training_features[i]}: {coefs[i]}")

print(classification_report(y_test, y_pred_logistic, digits=3))

f1_logistic = float(f1_score(y_test, y_pred_logistic, average="macro"))
print(f"Decision Logistic Regression F1-score: {f1_logistic}")
