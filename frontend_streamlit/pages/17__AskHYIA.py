def plan_to_result(df: pd.DataFrame, plan: dict) -> pd.DataFrame:
    """
    Convert a parsed JSON plan into a pandas DataFrame result.
    Always returns a DataFrame (even for single KPI values).
    """
    intent = (plan.get("intent") or "").lower()
    cols = plan.get("columns", {})
    dim  = cols.get("dimension")
    meas = cols.get("measure")
    date = cols.get("date")
    agg  = (plan.get("aggregation") or "").lower() or None
    filters = plan.get("filters", []) or []
    topn = plan.get("topn")
    chart = plan.get("chart", "table")

    work = apply_filters(df, filters)

    # timeseries -> ensure date parsing
    if date and date in work.columns:
        work[date] = pd.to_datetime(work[date], errors="coerce")

    # Aggregation-style intents
    if intent in ("aggregate", "topn", "timeseries", "kpi") and meas:
        # pick group-by key
        group_key = None
        if intent == "timeseries" and date:
            work["_bucket"] = work[date].dt.to_period("M").dt.to_timestamp()
            group_key = "_bucket"
        elif dim and dim in work.columns:
            group_key = dim

        agg_fn = agg if agg in ("sum", "mean", "count", "max", "min") else "sum"

        # If there's a group key, do groupby -> series -> reset_index
        if group_key is not None:
            # ensure numeric conversion for measure column when needed
            if agg_fn != "count":
                # convert measure to numeric for safe aggregation
                work[meas] = pd.to_numeric(work[meas], errors="coerce")
            grouped = work.groupby(group_key, dropna=False)[meas]
            if agg_fn == "count":
                res_ser = grouped.count()
            else:
                res_ser = grouped.agg(agg_fn)
            res = res_ser.reset_index()
            # Rename columns to be predictable: [group_key, <agg>_<measure>]
            res.columns = [group_key, f"{agg_fn}_{meas}"]
            # Apply top-n if requested
            if topn and isinstance(topn, int) and topn > 0:
                res = res.sort_values(by=res.columns[-1], ascending=False).head(topn)
            return res.reset_index(drop=True)

        # No group key -> scalar KPI (single value)
        else:
            if agg_fn == "count":
                value = int(len(work))
            else:
                value = pd.to_numeric(work[meas], errors="coerce").agg(agg_fn)
                # If result is a numpy scalar, keep as Python scalar for nicer display
                try:
                    if hasattr(value, "item"):
                        value = value.item()
                except Exception:
                    pass
            # return a small dataframe with metric/value columns
            return pd.DataFrame([{"metric": f"{agg_fn}_{meas}", "value": value}])

    # describe -> descriptive table
    if intent == "describe":
        return work.describe(include="all").T.reset_index().rename(columns={"index": "column"})

    # fallback: a sample table
    return work.head(200).reset_index(drop=True)
