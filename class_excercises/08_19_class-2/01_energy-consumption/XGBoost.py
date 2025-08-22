import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import xgboost as xgb
from utils.dataset_processer import process_dataset_datetime
from sklearn.metrics import classification_report, f1_score


datasets_dir = "other/Archivio Datasets/02_Lesson"
dataset_name = "AEP_hourly.csv"
dataset_path = os.path.join(datasets_dir, dataset_name)

datetime_col_name = "Datetime"

df = process_dataset_datetime(dataset_path, datetime_col_name)
print(df.head)


### Choosing training and target features
X = df[["hour", "dayofweek", "month", "year"]]
y = df["target_daily"]


### Split del dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, stratify=y, test_size=0.2, random_state=42
)


### Il Modello: XGBoost

XGBoost = xgb.XGBClassifier( eval_metric="logloss")
XGBoost.fit(X_train, y_train)
y_pred = XGBoost.predict(X_test)

print("XGBoost: ")
print(classification_report(y_test, y_pred, digits=3))

f1_tree = float(f1_score(y_test, y_pred, average="macro"))
print(f"XGBoost F1-score: {f1_tree}")