# components/predictor.py
import pandas as pd
import numpy as np
from components.predictor import train_predict
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (
    accuracy_score, f1_score, mean_absolute_error, mean_squared_error, r2_score,
    confusion_matrix
)

from __future__ import annotations
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from typing import Dict, Any, Optional, Tuple, List

def train_predict(df: pd.DataFrame, target_col: str):
    """
    Automatically detect whether classification or regression is needed,
    train a simple Random Forest model, and return metrics + insights.
    """
    result = {
        "task": None,
        "metrics": {},
        "importances": [],
        "holdout": {}
    }

    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not in dataset.")

    # --- Preprocessing ---
    df = df.dropna(subset=[target_col])
    y = df[target_col]
    X = df.drop(columns=[target_col])

    # Encode categorical variables
    for c in X.columns:
        if X[c].dtype == "object" or str(X[c].dtype).startswith("category"):
            X[c] = LabelEncoder().fit_transform(X[c].astype(str))
    if y.dtype == "object" or str(y.dtype).startswith("category"):
        y = LabelEncoder().fit_transform(y.astype(str))

    # Detect task type
    result["task"] = "classification" if len(np.unique(y)) < 20 else "regression"

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Scale numerical data
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Choose model
    if result["task"] == "classification":
        model = RandomForestClassifier(random_state=42, n_estimators=150)
    else:
        model = RandomForestRegressor(random_state=42, n_estimators=150)

    # Train and predict
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # --- Evaluation ---
    if result["task"] == "classification":
        result["metrics"] = {
            "accuracy": float(accuracy_score(y_test, y_pred)),
            "f1_score": float(f1_score(y_test, y_pred, average="weighted")),
            "confusion_matrix": confusion_matrix(y_test, y_pred).tolist()
        }
    else:
        result["metrics"] = {
            "MAE": float(mean_absolute_error(y_test, y_pred)),
            "RMSE": float(mean_squared_error(y_test, y_pred, squared=False)),
            "R2": float(r2_score(y_test, y_pred))
        }

    # --- Feature Importances ---
    try:
        imp = pd.DataFrame({
            "feature": df.drop(columns=[target_col]).columns,
            "importance": model.feature_importances_
        }).sort_values(by="importance", ascending=False)
        result["importances"] = imp.to_dict("records")
    except Exception:
        result["importances"] = []

    # --- Holdout info for visualization ---
    result["holdout"]["y_true"] = y_test.tolist()
    result["holdout"]["y_pred"] = y_pred.tolist()

    return result

# components/predictor.py
from __future__ import annotations
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from typing import Dict, Any, Optional, Tuple, List

# ---- Sklearn (tabular AutoML) ----
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score,
    mean_absolute_error, mean_squared_error, r2_score
)
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer

from sklearn.linear_model import LogisticRegression, Ridge, LinearRegression
from sklearn.ensemble import (
    RandomForestClassifier, GradientBoostingClassifier,
    RandomForestRegressor, GradientBoostingRegressor
)

# ---- Forecasting (time series) ----
try:
    from statsmodels.tsa.holtwinters import ExponentialSmoothing
    HAS_HW = True
except Exception:
    HAS_HW = False

try:
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    HAS_SARIMAX = True
except Exception:
    HAS_SARIMAX = False

import plotly.express as px
import plotly.graph_objects as go


# ============================ UTILITIES ============================

def _split_features_target(df: pd.DataFrame, target: str) -> Tuple[pd.DataFrame, pd.Series]:
    df = df.copy()
    df = df.dropna(subset=[target])
    y = df[target]
    X = df.drop(columns=[target])
    return X, y


def _guess_problem_type(y: pd.Series) -> str:
    """Heuristic: <= 20 unique values → classification, else regression."""
    if y.dtype.kind in "if":  # numeric
        if y.nunique() <= 20:
            return "classification"
        return "regression"
    # non-numeric → classification
    return "classification"


def _build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    num_cols = X.select_dtypes(include=["number"]).columns.tolist()
    cat_cols = [c for c in X.columns if c not in num_cols]

    num_pipe = Pipeline(steps=[
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler())
    ])

    cat_pipe = Pipeline(steps=[
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])

    pre = ColumnTransformer(
        transformers=[
            ("num", num_pipe, num_cols),
            ("cat", cat_pipe, cat_cols),
        ],
        remainder="drop",
        n_jobs=None
    )
    return pre


def _evaluate_regression(y_true, y_pred) -> Dict[str, float]:
    return {
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": float(mean_squared_error(y_true, y_pred, squared=False)),
        "R2": float(r2_score(y_true, y_pred))
    }


def _evaluate_classification(y_true, y_pred, proba=None) -> Dict[str, float]:
    out = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1_weighted": float(f1_score(y_true, y_pred, average="weighted")),
    }
    # add ROC-AUC if binary and proba available
    try:
        if proba is not None and len(np.unique(y_true)) == 2:
            out["roc_auc"] = float(roc_auc_score(y_true, proba[:, 1]))
    except Exception:
        pass
    return out


def _plot_holdout_scatter(y_true, y_pred, title="Pred vs True"):
    dfp = pd.DataFrame({"y_true": y_true, "y_pred": y_pred})
    fig = px.scatter(dfp, x="y_true", y="y_pred", title=title)
    fig.add_trace(go.Line(x=[dfp.y_true.min(), dfp.y_true.max()],
                          y=[dfp.y_true.min(), dfp.y_true.max()],
                          name="Ideal", line=dict(color="gray", dash="dash")))
    fig.update_layout(template="plotly_white")
    return fig


# ============================ TABULAR AUTOML ============================

def train_auto_model(
    df: pd.DataFrame,
    target: str,
    test_size: float = 0.2,
    random_state: int = 42
) -> Tuple[Pipeline, Dict[str, Any]]:
    """
    AutoML-style trainer: picks best model among a small set for classification/regression.
    Returns (fitted_pipeline, metrics dict).
    """
    if target not in df.columns:
        raise ValueError(f"Target '{target}' not found.")

    X, y = _split_features_target(df, target)
    problem = _guess_problem_type(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y if problem=="classification" else None
    )

    pre = _build_preprocessor(X)

    candidates = []
    if problem == "classification":
        candidates = [
            ("LogReg", LogisticRegression(max_iter=200)),
            ("RF", RandomForestClassifier(n_estimators=200, random_state=random_state)),
            ("GBC", GradientBoostingClassifier(random_state=random_state)),
        ]
        metric_key = "f1_weighted"
        best_score = -1
    else:
        candidates = [
            ("Linear", LinearRegression()),
            ("Ridge", Ridge(alpha=1.0, random_state=random_state)),
            ("RF", RandomForestRegressor(n_estimators=200, random_state=random_state)),
            ("GBR", GradientBoostingRegressor(random_state=random_state)),
        ]
        metric_key = "R2"
        best_score = -1e9

    best_pipe, best_metrics, best_name = None, {}, "N/A"
    y_pred_best, y_proba_best = None, None

    for name, model in candidates:
        pipe = Pipeline(steps=[("pre", pre), ("model", model)])
        pipe.fit(X_train, y_train)

        # Predictions
        y_pred = pipe.predict(X_test)
        proba = None
        if problem == "classification":
            try:
                proba = pipe.predict_proba(X_test)
            except Exception:
                proba = None
            metrics = _evaluate_classification(y_test, y_pred, proba)
        else:
            metrics = _evaluate_regression(y_test, y_pred)

        score = metrics.get(metric_key, -1e9)
        if score > best_score:
            best_score = score
            best_pipe = pipe
            best_metrics = metrics
            best_name = name
            y_pred_best = y_pred
            y_proba_best = proba

    # Additional artifacts
    best_metrics["problem"] = problem
    best_metrics["best_model"] = best_name
    artifacts: Dict[str, Any] = {"metrics": best_metrics}

    # Feature importances (tree models)
    try:
        model = best_pipe.named_steps["model"]
        if hasattr(model, "feature_importances_"):
            # derive feature names from preprocessor
            pre: ColumnTransformer = best_pipe.named_steps["pre"]
            num_cols = pre.transformers_[0][2]
            cat_pipe: Pipeline = pre.transformers_[1][1]
            cat_cols = cat_pipe.named_steps["onehot"].get_feature_names_out(pre.transformers_[1][2])
            feat_names = list(num_cols) + list(cat_cols)
            importances = list(model.feature_importances_)
            # Align lengths
            feats = feat_names[:len(importances)]
            artifacts["importances"] = [
                {"feature": f, "importance": float(i)} for f, i in zip(feats, importances)
            ]
        else:
            artifacts["importances"] = []
    except Exception:
        artifacts["importances"] = []

    # Holdout plot
    if best_metrics["problem"] == "regression":
        artifacts["holdout_fig"] = _plot_holdout_scatter(y_test, y_pred_best)
    else:
        # return a small df for confusion matrix if caller wants to render
        artifacts["holdout_df"] = pd.DataFrame({"y_true": y_test, "y_pred": y_pred_best})

    return best_pipe, artifacts


# ============================ FORECASTING ============================

def _infer_freq(idx: pd.DatetimeIndex) -> Optional[str]:
    try:
        return pd.infer_freq(idx)
    except Exception:
        return None

def forecast_series(
    df: pd.DataFrame,
    date_col: str,
    target_col: str,
    periods: int = 12
) -> Dict[str, Any]:
    """
    Forecast future values of target_col using date_col.
    Uses Holt-Winters ExponentialSmoothing by default, falls back / upgrades to SARIMAX if available.
    Returns dict with 'forecast_df' and 'fig'.
    """
    if date_col not in df.columns or target_col not in df.columns:
        raise ValueError("date_col or target_col not found in the dataframe.")

    data = df[[date_col, target_col]].dropna().copy()
    data[date_col] = pd.to_datetime(data[date_col], errors="coerce")
    data = data.dropna(subset=[date_col]).sort_values(date_col)
    data = data.set_index(date_col)

    y = pd.to_numeric(data[target_col], errors="coerce").dropna()
    if y.empty:
        raise ValueError("No numeric data to forecast after cleaning.")

    # Ensure a DatetimeIndex with a frequency
    if not isinstance(y.index, pd.DatetimeIndex):
        raise ValueError("date_col must be convertible to datetime.")
    freq = _infer_freq(y.index)
    if freq is None:
        # try to coerce to a regular frequency by resampling daily
        y = y.asfreq("D")
        freq = "D"

    # Fit model
    fitted = None
    try:
        if HAS_SARIMAX and len(y) >= 10:
            # simple SARIMAX(1,1,1) – reasonable default
            model = SARIMAX(y, order=(1,1,1), enforce_stationarity=False, enforce_invertibility=False)
            fitted = model.fit(disp=False)
            fc = fitted.get_forecast(steps=periods)
            pred = fc.predicted_mean
            conf = fc.conf_int(alpha=0.2)  # 80% band
            lower = conf.iloc[:, 0]
            upper = conf.iloc[:, 1]
        elif HAS_HW and len(y) >= 10:
            # Holt-Winters with season length guess
            season_len = {"D": 7, "W": 52, "M": 12, "Q": 4, "H": 24}.get(freq[0], 12)
            model = ExponentialSmoothing(y, trend="add", seasonal="add", seasonal_periods=season_len)
            fitted = model.fit()
            pred = fitted.forecast(periods)
            lower = pred - 1.28 * pred.std()  # ~80% band proxy
            upper = pred + 1.28 * pred.std()
        else:
            # naive last-value forecast
            last = y.iloc[-1]
            idx = pd.date_range(y.index[-1], periods=periods+1, freq=freq)[1:]
            pred = pd.Series([last] * periods, index=idx)
            lower = pred * 0.95
            upper = pred * 1.05

    except Exception:
        # Robust fallback: naive
        last = y.iloc[-1]
        idx = pd.date_range(y.index[-1], periods=periods+1, freq=freq)[1:]
        pred = pd.Series([last] * periods, index=idx)
        lower = pred * 0.95
        upper = pred * 1.05

    hist = y.copy()
    fc_df = pd.DataFrame({
        "ds": list(hist.index) + list(pred.index),
        "y": list(hist.values) + [None]*len(pred),
        "yhat": [None]*len(hist) + list(pred.values),
        "yhat_lower": [None]*len(hist) + list(lower.values),
        "yhat_upper": [None]*len(hist) + list(upper.values),
        "split": ["history"]*len(hist) + ["forecast"]*len(pred),
    })

    # Plot
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=hist.index, y=hist.values, mode="lines", name="History"))
    fig.add_trace(go.Scatter(x=pred.index, y=pred.values, mode="lines", name="Forecast"))
    fig.add_trace(go.Scatter(
        x=list(pred.index) + list(pred.index[::-1]),
        y=list(upper.values) + list(lower.values[::-1]),
        fill="toself",
        name="Conf. band",
        line=dict(color="rgba(99,110,250,0.15)"),
        showlegend=True
    ))
    fig.update_layout(
        title=f"Forecast of {target_col}",
        template="plotly_white",
        xaxis_title=date_col, yaxis_title=target_col,
        margin=dict(l=8, r=8, t=48, b=8)
    )

    return {"forecast_df": fc_df, "fig": fig}

