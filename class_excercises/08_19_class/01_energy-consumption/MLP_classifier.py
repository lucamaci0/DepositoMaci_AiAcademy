import os
import matplotlib.pyplot as plt
from utils.dataset_processer import process_dataset_datetime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, f1_score


datasets_dir = "other/Archivio Datasets/02_Lesson"
dataset_name = "AEP_hourly.csv"
datetime_col_name = "Datetime"

dataset_path = os.path.join(datasets_dir, dataset_name)
df = process_dataset_datetime(dataset_path, datetime_col_name)
print(df.head)


# Choose MLP-Classifier features and target:

training_features = ["hour", "dayofweek", "month", "year"]
classification_target = "target_daily"

X = df[training_features]
y = df[classification_target]

### Split del dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, stratify=y, test_size=0.2, random_state=42
)


### Modello 2 – MLPClassifier

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

mlp = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=300, random_state=42)
mlp.fit(X_train_scaled, y_train)
y_pred_mlp = mlp.predict(X_test_scaled)

print("Neural Network:")
print(classification_report(y_test, y_pred_mlp, digits=3))

f1_mlp = float(f1_score(y_test, y_pred_mlp, average="macro"))
print(f"Decision Tree MLP-Classifier: {f1_mlp}")
