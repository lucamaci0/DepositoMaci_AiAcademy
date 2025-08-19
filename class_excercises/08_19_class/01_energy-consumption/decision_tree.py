import os
import matplotlib.pyplot as plt
from dataset_processer import process_dataset_datetime
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import classification_report, f1_score


datasets_dir = "other/Archivio Datasets/02_Lesson"
dataset_name = "AEP_hourly.csv"
dataset_path = os.path.join(datasets_dir, dataset_name)

df = process_dataset_datetime(dataset_path)
print(df.head)


# Choose Decision-Tree features and target:

training_features = ["hour", "dayofweek", "month", "year"]
classification_target = "target_daily"

X = df[training_features]
y = df[classification_target]

### Split del dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, stratify=y, test_size=0.2, random_state=42
)


### Il Modello: Decision Tree

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
print(f"Decision Tree F1-score: {f1_tree}")
