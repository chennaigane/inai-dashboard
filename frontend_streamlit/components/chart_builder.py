import streamlit as st, pandas as pd, plotly.express as px

def render_chart(df:pd.DataFrame, kind:str, x:str|None, y:list[str]|None, color:str|None=None, agg:str="sum"):
    if df.empty: 
        st.warning("No data.")
        return
    if x and y:
        if agg and len(y)==1:
            df = df.groupby(x, as_index=False)[y[0]].agg(agg)
    if kind=="bar": fig = px.bar(df, x=x, y=y, color=color)
    elif kind=="line": fig = px.line(df, x=x, y=y, color=color)
    elif kind=="area": fig = px.area(df, x=x, y=y, color=color)
    elif kind=="pie": fig = px.pie(df, names=x, values=y[0] if y else None)
    elif kind=="table": st.dataframe(df, use_container_width=True); return
    elif kind=="kpi":
        val = df[y[0]].sum() if y else df.iloc[:,0].sum()
        st.metric(label=y[0] if y else "Value", value=f"{val:,.2f}")
        return
    else:
        st.info("Unsupported chart type.")
        return
    st.plotly_chart(fig, use_container_width=True)
