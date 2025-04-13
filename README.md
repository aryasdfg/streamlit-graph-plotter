import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

st.set_page_config(layout="wide", page_title="Smart Graph Plotter")

st.title("📊 Smart Data Visualizer & Graph Plotter")

# Layout: 2 columns
left, right = st.columns([2, 3])

with left:
    uploaded_file = st.file_uploader("📁 Upload your data file (CSV or Excel)", type=["csv", "xlsx"])

if uploaded_file:
    # Read file
    if uploaded_file.name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)

    numeric_cols = df.select_dtypes(include=["float64", "int64"]).columns.tolist()
    category_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    all_cols = df.columns.tolist()

    with right:
        st.subheader("📄 Data Preview")
        st.dataframe(df, use_container_width=True)

    with left:
        st.subheader("🔍 Auto Axis Suggestion")
        
        suggested_x = df[numeric_cols].nunique().idxmax() if numeric_cols else None
        suggested_y = df[numeric_cols].var().idxmax() if numeric_cols else None

        st.markdown(f"**📌 Suggested X-axis:** `{suggested_x}`")
        st.markdown(f"**📌 Suggested Y-axis:** `{suggested_y}`")

        x_col = st.selectbox("Select X-axis", options=numeric_cols, index=numeric_cols.index(suggested_x) if suggested_x else 0)
        y_col = st.selectbox("Select Y-axis", options=[col for col in numeric_cols if col != x_col], index=numeric_cols.index(suggested_y) if suggested_y and suggested_y != x_col else 0)

        hue_col = st.selectbox("Optional: Group lines by", options=["None"] + category_cols)
        hue_col = None if hue_col == "None" else hue_col

        smooth = st.checkbox("🧮 Apply Curve Fitting", value=True)
        degree = st.slider("Polynomial Degree (if fitting applied)", 1, 5, 2)
        dark = st.checkbox("🌙 Dark Mode")

        title = st.text_input("Plot Title", f"{y_col} vs {x_col}")
        xlabel = st.text_input("X-axis Label", x_col)
        ylabel = st.text_input("Y-axis Label", y_col)

        if st.button("📈 Generate Plot"):
            sns.set_style("darkgrid" if dark else "whitegrid")
            fig, ax = plt.subplots(figsize=(10, 6))
            if hue_col:
                for group in df[hue_col].unique():
                    sub = df[df[hue_col] == group]
                    sns.lineplot(data=sub, x=x_col, y=y_col, label=str(group), ax=ax, marker="o")
                    if smooth:
                        X = sub[[x_col]].values
                        y = sub[y_col].values
                        poly = PolynomialFeatures(degree=degree)
                        X_poly = poly.fit_transform(X)
                        model = LinearRegression().fit(X_poly, y)
                        x_fit = np.linspace(X.min(), X.max(), 200).reshape(-1, 1)
                        y_fit = model.predict(poly.transform(x_fit))
                        ax.plot(x_fit, y_fit, linestyle="--", label=f"{group} (fit)")
            else:
                sns.lineplot(data=df, x=x_col, y=y_col, ax=ax, marker="o", label=f"{y_col}")
                if smooth:
                    X = df[[x_col]].values
                    y = df[y_col].values
                    poly = PolynomialFeatures(degree=degree)
                    X_poly = poly.fit_transform(X)
                    model = LinearRegression().fit(X_poly, y)
                    x_fit = np.linspace(X.min(), X.max(), 200).reshape(-1, 1)
                    y_fit = model.predict(poly.transform(x_fit))
                    ax.plot(x_fit, y_fit, linestyle="--", label="Fit")

            ax.set_title(title, fontsize=16)
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            ax.grid(True)
            ax.legend()
            st.pyplot(fig)
