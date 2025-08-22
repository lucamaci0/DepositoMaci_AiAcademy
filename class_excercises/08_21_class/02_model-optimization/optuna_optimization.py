# ============================================================
# Classificazione "alto/basso consumo" su AEP_hourly
# Modelli: Logistic Regression, Random Forest, XGBoost
# - Pipeline per ogni modello
# - Normalizzazione (solo dove utile)
# - Train/Test split stratificato
# - Tuning iperparametri con Optuna (ROC-AUC, 5-fold)
# - Report finale su test
# ============================================================

import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, classification_report

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

import optuna # TODO: understand why this isn't working


# ------------------------------------------------------------
# 1) Caricamento dati e feature engineering minimale
# ------------------------------------------------------------
# CSV Kaggle AEP_hourly: colonne ["Datetime", "AEP_MW"]

datasets_dir = "other/Archivio Datasets/02_Lesson"
dataset_name = "AEP_hourly.csv"

dataset_path = os.path.join(datasets_dir, dataset_name)
df = pd.read_csv("AEP_hourly.csv", parse_dates=["Datetime"])

# Feature temporali semplici (tutte numeriche)
df["hour"] = df["Datetime"].dt.hour
df["dayofweek"] = df["Datetime"].dt.dayofweek
df["month"] = df["Datetime"].dt.month

# Target binario: 1 se consumo sopra la mediana globale
df["target"] = (df["AEP_MW"] > df["AEP_MW"].median()).astype(int)

# Selezione feature / target
FEATURES = ["hour", "dayofweek", "month"]
X = df[FEATURES].copy()
y = df["target"].astype(int).copy()


# ------------------------------------------------------------
# 2) Train/Test split stratificato
# ------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.20,
    stratify=y,
    random_state=42
)

# K-fold per valutazione coerente (shuffle + seed fisso)
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Rapporto classi per XGBoost (utile se sbilanciato)
pos = y_train.sum()
neg = len(y_train) - pos
scale_pos_weight_default = (neg / pos) if pos > 0 else 1.0


# ------------------------------------------------------------
# 3) Utility: cross-val ROC-AUC per pipeline
# ------------------------------------------------------------
def mean_cv_auc(model: Pipeline, X, y, cv) -> float:
    """Ritorna la media della ROC-AUC su K-fold stratificati."""
    scores = cross_val_score(model, X, y, cv=cv, scoring="roc_auc", n_jobs=-1)
    return float(np.mean(scores))


# ------------------------------------------------------------
# 4) Tuning con Optuna: LOGISTIC REGRESSION
#    - Richiede scaling (StandardScaler)
# ------------------------------------------------------------
def tune_logreg(n_trials: int = 25):
    # Pipeline: scaler -> logistic regression
    # (tutte le feature sono numeriche: niente ColumnTransformer)
    def objective(trial):
        # Spazio iperparametri "piccolo ma sensato"
        C = trial.suggest_float("C", 1e-3, 1e2, log=True)
        solver = trial.suggest_categorical("solver", ["lbfgs", "liblinear"])
        class_weight = trial.suggest_categorical("class_weight", [None, "balanced"])
        max_iter = trial.suggest_int("max_iter", 200, 1000, step=200)

        pipe = Pipeline(steps=[
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(
                C=C,
                solver=solver,
                penalty="l2",
                class_weight=class_weight,
                max_iter=max_iter,
                random_state=42,
                n_jobs=None if solver == "liblinear" else -1
            ))
        ])

        return mean_cv_auc(pipe, X_train, y_train, cv)

    study = optuna.create_study(direction="maximize", study_name="logreg_auc")
    study.optimize(objective, n_trials=n_trials)
    return study


# ------------------------------------------------------------
# 5) Tuning con Optuna: RANDOM FOREST
#    - Gli alberi non richiedono scaling
# ------------------------------------------------------------
def tune_rf(n_trials: int = 25):
    def objective(trial):
        n_estimators = trial.suggest_int("n_estimators", 200, 800, step=100)
        max_depth = trial.suggest_int("max_depth", 4, 18)
        min_samples_split = trial.suggest_int("min_samples_split", 2, 20)
        min_samples_leaf = trial.suggest_int("min_samples_leaf", 1, 10)
        max_features = trial.suggest_categorical("max_features", ["sqrt", "log2", None])
        class_weight = trial.suggest_categorical("class_weight", [None, "balanced", "balanced_subsample"])

        pipe = Pipeline(steps=[
            ("clf", RandomForestClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                min_samples_split=min_samples_split,
                min_samples_leaf=min_samples_leaf,
                max_features=max_features,
                class_weight=class_weight,
                n_jobs=-1,
                random_state=42
            ))
        ])

        return mean_cv_auc(pipe, X_train, y_train, cv)

    study = optuna.create_study(direction="maximize", study_name="rf_auc")
    study.optimize(objective, n_trials=n_trials)
    return study


# ------------------------------------------------------------
# 6) Tuning con Optuna: XGBOOST
#    - Niente scaling; tree_method 'hist' per velocità su CPU
# ------------------------------------------------------------
def tune_xgb(n_trials: int = 30):
    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 300, 900, step=100),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.2, log=True),
            "subsample": trial.suggest_float("subsample", 0.7, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.7, 1.0),
            "min_child_weight": trial.suggest_float("min_child_weight", 1.0, 8.0),
            "gamma": trial.suggest_float("gamma", 0.0, 3.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-6, 1e-1, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 5.0, log=True),
            # gest. sbilanciamento intorno al valore stimato
            "scale_pos_weight": trial.suggest_float(
                "scale_pos_weight",
                max(0.5, scale_pos_weight_default * 0.5),
                max(1.0, scale_pos_weight_default * 1.5)
            ),
            # fissi
            "objective": "binary:logistic",
            "eval_metric": "auc",
            "tree_method": "hist",
            "random_state": 42,
            "n_jobs": -1
        }

        pipe = Pipeline(steps=[
            ("clf", XGBClassifier(**params))
        ])

        return mean_cv_auc(pipe, X_train, y_train, cv)

    study = optuna.create_study(direction="maximize", study_name="xgb_auc",
                                pruner=optuna.pruners.MedianPruner(n_warmup_steps=8))
    study.optimize(objective, n_trials=n_trials)
    return study


# ------------------------------------------------------------
# 7) Esegui tuning per tutti i modelli
# ------------------------------------------------------------
print(">> Tuning Logistic Regression...")
study_lr = tune_logreg(n_trials=25)
print("Best AUC (LogReg CV):", round(study_lr.best_value, 4))
print("Best params (LogReg):", study_lr.best_params)

print("\n>> Tuning Random Forest...")
study_rf = tune_rf(n_trials=25)
print("Best AUC (RF CV):", round(study_rf.best_value, 4))
print("Best params (RF):", study_rf.best_params)

print("\n>> Tuning XGBoost...")
study_xgb = tune_xgb(n_trials=30)
print("Best AUC (XGB CV):", round(study_xgb.best_value, 4))
print("Best params (XGB):", study_xgb.best_params)


# ------------------------------------------------------------
# 8) Fit finale sul TRAIN e valutazione su TEST
#    - Usa i migliori iperparametri trovati da Optuna
# ------------------------------------------------------------
def fit_and_report(model_name: str, best_params: dict):
    """Crea la pipeline finale, fa fit su TRAIN e valuta su TEST."""
    if model_name == "logreg":
        pipe = Pipeline(steps=[
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(
                C=best_params["C"],
                solver=best_params["solver"],
                penalty="l2",
                class_weight=best_params["class_weight"],
                max_iter=best_params["max_iter"],
                random_state=42,
                n_jobs=None if best_params["solver"] == "liblinear" else -1
            ))
        ])

    elif model_name == "rf":
        pipe = Pipeline(steps=[
            ("clf", RandomForestClassifier(
                n_estimators=best_params["n_estimators"],
                max_depth=best_params["max_depth"],
                min_samples_split=best_params["min_samples_split"],
                min_samples_leaf=best_params["min_samples_leaf"],
                max_features=best_params["max_features"],
                class_weight=best_params["class_weight"],
                n_jobs=-1,
                random_state=42
            ))
        ])

    elif model_name == "xgb":
        # Inseriamo i parametri scelti da Optuna e fissiamo quelli di base
        xgb_params = best_params.copy()
        xgb_params.update({
            "objective": "binary:logistic",
            "eval_metric": "auc",
            "tree_method": "hist",
            "random_state": 42,
            "n_jobs": -1
        })
        pipe = Pipeline(steps=[
            ("clf", XGBClassifier(**xgb_params))
        ])

    else:
        raise ValueError("Modello non riconosciuto")

    # Fit su TRAIN
    pipe.fit(X_train, y_train)

    # Valutazione su TEST (ROC-AUC su probabilità, report su classi 0/1 con soglia 0.5)
    proba = pipe.predict_proba(X_test)[:, 1]
    preds = (proba >= 0.5).astype(int)

    auc = roc_auc_score(y_test, proba)
    print(f"\n==== {model_name.upper()} | Test ROC-AUC: {auc:.4f}")
    print(classification_report(y_test, preds, digits=3))

    return pipe


final_lr = fit_and_report("logreg", study_lr.best_params)
final_rf = fit_and_report("rf", study_rf.best_params)
final_xgb = fit_and_report("xgb", study_xgb.best_params)

# (Opzionale) salva il modello migliore con joblib
# import joblib
# joblib.dump(final_xgb, "best_model.joblib")
