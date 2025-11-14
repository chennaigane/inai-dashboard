# frontend_streamlit/components/auto_ml.py
from __future__ import annotations
import pandas as pd
import numpy as np
from typing import Tuple, Dict, Any
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score, confusion_matrix,
    r2_score, mean_absolute_error, mean_squared_error
)
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

def infer_task(y: pd.Series) -> str:
    # If target is numeric & >10 unique values → regression else classification
    if pd.api.types.is_numeric_dtype(y):
        return "regression" if y.nunique(dropna=True) > 10 else "classification"
    return "classification"

def train_predict(df: pd.DataFrame, target: str) -> Dict[str, Any]:
    df = df.dropna(subset=[target]).copy()
    y = df[target]
    X = df.drop(columns=[target])

    task = infer_task(y)
    num_cols = X.select_dtypes(include="number").columns.tolist()
    cat_cols = [c for c in X.columns if c not in num_cols]

    pre = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(with_mean=False), num_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
        ],
        remainder="drop"
    )

    if task == "classification":
        model = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
    else:
        model = RandomForestRegressor(n_estimators=300, random_state=42, n_jobs=-1)

    pipe = Pipeline([("prep", pre), ("model", model)])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=None if task=="regression" else y)

    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)

    metrics = {}
    if task == "classification":
        proba = None
        try:
            proba = pipe.predict_proba(X_test)[:, 1]
        except Exception:
            pass
        metrics["accuracy"] = float(accuracy_score(y_test, y_pred))
        metrics["f1_macro"] = float(f1_score(y_test, y_pred, average="macro"))
        if proba is not None and len(np.unique(y_test)) == 2:
            metrics["roc_auc"] = float(roc_auc_score(y_test, proba))
        metrics["confusion_matrix"] = confusion_matrix(y_test, y_pred).tolist()
    else:
        metrics["r2"] = float(r2_score(y_test, y_pred))
        metrics["mae"] = float(mean_absolute_error(y_test, y_pred))
        metrics["rmse"] = float(mean_squared_error(y_test, y_pred, squared=False))

    # Feature importances (best-effort)
    importances = None
    try:
        # get feature names after preprocessing
        num_feats = num_cols
        cat_feats = list(pipe.named_steps["prep"].named_transformers_["cat"].get_feature_names_out(cat_cols))
        feat_names = num_feats + cat_feats
        importances = getattr(pipe.named_steps["model"], "feature_importances_", None)
        if importances is not None:
            importances = sorted(
                [{"feature": f, "importance": float(i)} for f, i in zip(feat_names, importances)],
                key=lambda x: x["importance"], reverse=True
            )[:25]
    except Exception:
        pass

    return {
        "task": task,
        "metrics": metrics,
        "importances": importances,
        "model": pipe,  # keep in memory; don’t serialize here
        "holdout": {"y_true": y_test.tolist(), "y_pred": y_pred.tolist()},
    }
