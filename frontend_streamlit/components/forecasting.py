# frontend_streamlit/components/forecasting.py
"""
Minimal forecasting stub — safe to import. Replace with full logic later.
"""

import pandas as pd
import numpy as np
from typing import Tuple, Dict

def monthly_aggregate(df: pd.DataFrame, date_col: str, value_col: str) -> pd.DataFrame:
    tmp = df.copy()
    tmp[date_col] = pd.to_datetime(tmp[date_col], errors="coerce")
    tmp = tmp.dropna(subset=[date_col])
    monthly = tmp.set_index(date_col).resample("M")[value_col].sum().reset_index()
    return monthly

def linear_forecast(monthly_df: pd.DataFrame, months_ahead: int = 6) -> Tuple[pd.DataFrame, Dict]:
    if monthly_df is None or monthly_df.empty:
        return pd.DataFrame(), {"error":"no data"}
    # very small fallback: repeat last value as forecast
    monthly_df = monthly_df.sort_values(monthly_df.columns[0])
    last = monthly_df.iloc[-1, 1] if monthly_df.shape[1] >= 2 else 0
    future_dates = [monthly_df.iloc[-1,0] + pd.DateOffset(months=i) for i in range(1, months_ahead+1)]
    fut_vals = [last for _ in future_dates]
    out = pd.concat([monthly_df, pd.DataFrame({monthly_df.columns[0]: future_dates, monthly_df.columns[1]: fut_vals})], ignore_index=True)
    info = {"method":"naive_repeat_last"}
    return out, info

def plot_forecast(forecast_df: pd.DataFrame, title: str = "Forecast"):
    # simple text fallback if plotly is not installed
    try:
        import plotly.express as px
        fig = px.line(forecast_df, x=forecast_df.columns[0], y=forecast_df.columns[1], title=title)
        return fig
    except Exception:
        return None
import pandas as pd


def monthly_aggregate(df, date_col, value_col):
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    if df[date_col].isna().all():
        raise ValueError(f"All values in '{date_col}' are missing or invalid dates.")
    df['Month'] = df[date_col].dt.to_period('M')
    return df.groupby('Month')[value_col].sum().reset_index()

def linear_forecast(monthly, months_ahead=6):
    meta = {"future_months": months_ahead}
    return monthly, meta

def plot_forecast(df, title="Forecast"):
    import plotly.express as px
    return px.line(df, x=df.columns[0], y=df.columns[1], title=title)

if df[date_col].isna().all():
    raise ValueError(f"All values in '{date_col}' are missing or invalid dates.")
    df['Month'] = df[date_col].dt.to_period('M')
    return df.groupby('Month')[value_col].sum().reset_index()
def linear_forecast(monthly, months_ahead=6):
    meta = {"future_months": months_ahead}
    return monthly, meta


if df[date_col].isna().all():
    raise ValueError(f"All values in {date_col} are missing or unparseable as dates.")

def linear_forecast(monthly, months_ahead=6):
    # Dummy stub -- return monthly and meta info
    meta = {"future_months": months_ahead}
    return monthly, meta

def plot_forecast(df, title="Forecast"):
    import plotly.express as px
    return px.line(df, x=df.columns[0], y=df.columns[1], title=title)

# WRONG (in global/module level)
df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
if df[date_col].isna().all():
    raise ValueError(f"All values in '{date_col}' are missing or invalid dates.")

import pandas as pd

def monthly_aggregate(df, date_col, value_col):
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    if df[date_col].isna().all():
        raise ValueError(f"All values in '{date_col}' are missing or invalid dates.")
    df['Month'] = df[date_col].dt.to_period('M')
    return df.groupby('Month')[value_col].sum().reset_index()

def linear_forecast(monthly, months_ahead=6):
    meta = {"future_months": months_ahead}
    return monthly, meta

def plot_forecast(df, title="Forecast"):
    import plotly.express as px
    return px.line(df, x=df.columns[0], y=df.columns[1], title=title)
    return df.groupby('Month')[value_col].sum().reset_index()
def linear_forecast(monthly, months_ahead=6):
    meta = {"future_months": months_ahead}
    return monthly, meta
def plot_forecast(df, title="Forecast"):
    import plotly.express as px
    return px.line(df, x=df.columns[0], y=df.columns[1], title=title)

import pandas as pd

def monthly_aggregate(df, date_col, value_col):
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    if df[date_col].isna().all():
        raise ValueError(f"All values in '{date_col}' are missing or invalid dates.")
    df['Month'] = df[date_col].dt.to_period('M')
    return df.groupby('Month')[value_col].sum().reset_index()

def linear_forecast(monthly, months_ahead=6):
    meta = {"future_months": months_ahead}
    return monthly, meta

def plot_forecast(df, title="Forecast"):
    import plotly.express as px
    return px.line(df, x=df.columns[0], y=df.columns[1], title=title)
    return df.groupby('Month')[value_col].sum().reset_index()
def linear_forecast(monthly, months_ahead=6):
    meta = {"future_months": months_ahead}
    return monthly, meta
def plot_forecast(df, title="Forecast"):
    import plotly.express as px
    return px.line(df, x=df.columns[0], y=df.columns[1], title=title)
    out["_final_anomaly_flag"] = out[["_flag_neg_values", "_flag_many_nans"]].any(axis=1)
    out = out.drop(columns=["_nan_ratio"])
    return out
