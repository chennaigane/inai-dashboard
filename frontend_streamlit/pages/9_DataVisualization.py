import streamlit as st
import pandas as pd
import plotly.express as px


# Load data from session or use demo data
if "main_data" in st.session_state:
    df = st.session_state["main_data"]
else:
    # Demo
    df = pd.DataFrame({
        "Category": ["A", "B", "C", "D"],
        "Value1": [10, 30, 20, 40],
        "Value2": [5, 15, 10, 20],
        "Type": ["X", "Y", "X", "Y"]
    })

# Define chart types available
chart_types = ["Bar", "Line", "Pie", "Scatter", "Box"]

# Create a layout with 2 rows, 2 columns (4 quadrants)
row1_col1, row1_col2 = st.columns(2)
row2_col1, row2_col2 = st.columns(2)

def render_chart(column, slot):
    st.markdown(f"#### Chart {slot}")
    chart_type = st.selectbox(f"Chart Type {slot}", chart_types, key=f"ct{slot}")
    x_axis = st.selectbox(f"X Axis {slot}", df.columns, key=f"x{slot}")
    y_axis = st.selectbox(f"Y Axis {slot}", df.columns)


# Always have a non-empty dataframe for testing
if "main_data" in st.session_state and not st.session_state["main_data"].empty:
    df = st.session_state["main_data"]
else:
    df = pd.DataFrame({
        "Category": ["A", "B", "C", "D"],
        "Value1": [10, 30, 20, 40],
        "Value2": [5, 15, 10, 20],
        "Type": ["X", "Y", "X", "Y"]
    })

st.title("Data Visualization: 4-at-a-Time Comparison")
chart_types = ["Bar", "Line", "Pie", "Scatter", "Box"]

row1_col1, row1_col2 = st.columns(2)
row2_col1, row2_col2 = st.columns(2)

def render_chart(slot):
    st.markdown(f"#### Chart {slot}")
    chart_type = st.selectbox(f"Chart Type {slot}", chart_types, key=f"ct{slot}")
    x_axis = st.selectbox(f"X Axis {slot}", df.columns, key=f"x{slot}0")
    if chart_type != "Pie":
        y_axis = st.selectbox(f"Y Axis {slot}", df.columns, key=f"y{slot}1")
    else:
        y_axis = None
    color = st.selectbox(f"Color {slot}", ["None"] + list(df.columns), key=f"color{slot}2")

    fig = None
    try:
        if chart_type == "Bar":
            fig = px.bar(df, x=x_axis, y=y_axis, color=color if color != "None" else None)
        elif chart_type == "Line":
            fig = px.line(df, x=x_axis, y=y_axis, color=color if color != "None" else None)
        elif chart_type == "Scatter":
            fig = px.scatter(df, x=x_axis, y=y_axis, color=color if color != "None" else None)
        elif chart_type == "Pie":
            fig = px.pie(df, names=x_axis, values=y_axis if y_axis else df.columns[1],
                         color=color if color != "None" else None)
        elif chart_type == "Box":
            fig = px.box(df, x=x_axis, y=y_axis, color=color if color != "None" else None)
    except Exception as e:
        st.error(f"Chart failed to render: {e}")

    if fig:
        st.plotly_chart(fig, use_container_width=True, key=f"plot_{slot}")
    else:
        st.info("Please select valid columns for your chart.")

with row1_col1:
    render_chart(1)
with row1_col2:
    render_chart(2)
with row2_col1:
    render_chart(3)
with row2_col2:
    render_chart(4)

import streamlit as st
import pandas as pd
import plotly.express as px

if "main_data" in st.session_state and not st.session_state["main_data"].empty:
    df = st.session_state["main_data"]
else:
    df = pd.DataFrame({
        "Category": ["A", "B", "C", "D"],
        "Value1": [10, 30, 20, 40],
        "Value2": [5, 15, 10, 20],
        "Type": ["X", "Y", "X", "Y"]
    })

def render_chart(slot):
    st.markdown(f"#### Chart {slot}")
    chart_types = ["Bar", "Line", "Pie", "Scatter", "Box"]
    chart_type = st.selectbox(f"Chart Type {slot}", chart_types, key=f"ct{slot}")
    x_axis = st.selectbox(f"X Axis {slot}", df.columns, key=f"x{slot}0")
    if chart_type != "Pie":
        y_axis = st.selectbox(f"Y Axis {slot}", df.columns, key=f"y{slot}1")
    else:
        y_axis = None
    color = st.selectbox(f"Color {slot}", ["None"] + list(df.columns), key=f"color{slot}2")

    # [Chart creation logic as before, with unique key for plotly_chart]
    # e.g.: st.plotly_chart(fig, use_container_width=True, key=f"plot_{slot}")


