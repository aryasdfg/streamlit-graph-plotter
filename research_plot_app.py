import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from io import BytesIO

st.set_page_config(layout="wide")
st.title("📊 Advanced Research Graph Plotter")

uploaded_file = st.file_uploader("📁 Upload your CSV file", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write("📄 Data Preview:")
    st.dataframe(df.head())

    numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()

    if len(numeric_cols) < 2:
        st.error("❌ At least two numeric columns are required.")
    else:
        st.sidebar.header("🔧 Plot Configuration")

        x_col = st.sidebar.selectbox("X-axis", numeric_cols)
        y_cols = st.sidebar.multiselect("Y-axis (you can select multiple)", [col for col in numeric_cols if col != x_col], default=[numeric_cols[1]])

        hue_col = None
        if categorical_cols:
            hue_col = st.sidebar.selectbox("Group by (optional)", ["None"] + categorical_cols)
            if hue_col == "None":
                hue_col = None

        st.sidebar.header("🎨 Style & Curve Fitting")
        style = st.sidebar.selectbox("Seaborn Style", ["whitegrid", "darkgrid", "white", "dark", "ticks"])
        marker = st.sidebar.selectbox("Marker Style", ['o', 's', '^', 'D', '*', '+', 'x'], index=0)
        palette = st.sidebar.selectbox("Color Palette", ["deep", "muted", "bright", "dark", "colorblind"])
        dark_mode = st.sidebar.checkbox("🌙 Enable Dark Mode", value=False)
        smooth = st.sidebar.checkbox("🧮 Apply Curve Fit", value=False)
        degree = st.sidebar.slider("Polynomial Degree (for Curve Fit)", 1, 5, 2)

        title = st.sidebar.text_input("Plot Title", "Research Plot")
        xlabel = st.sidebar.text_input("X-axis Label", x_col)
        ylabel = st.sidebar.text_input("Y-axis Label", ", ".join(y_cols))

        sns.set_theme(style=style, palette=palette)

        fig, ax = plt.subplots(figsize=(12, 6))
        if dark_mode:
            plt.style.use("dark_background")

        for y_col in y_cols:
            if hue_col:
                for group in df[hue_col].unique():
                    sub_data = df[df[hue_col] == group]
                    sns.lineplot(data=sub_data, x=x_col, y=y_col, label=f"{y_col} - {group}", marker=marker, ax=ax)

                    if smooth:
                        X = sub_data[[x_col]].values
                        y = sub_data[y_col].values
                        poly = PolynomialFeatures(degree=degree)
                        X_poly = poly.fit_transform(X)
                        model = LinearRegression().fit(X_poly, y)
                        X_fit = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
                        y_fit = model.predict(poly.transform(X_fit))
                        ax.plot(X_fit, y_fit, linestyle='--', label=f"{y_col} - {group} (fit)")
            else:
                sns.lineplot(data=df, x=x_col, y=y_col, label=y_col, marker=marker, ax=ax)

                if smooth:
                    X = df[[x_col]].values
                    y = df[y_col].values
                    poly = PolynomialFeatures(degree=degree)
                    X_poly = poly.fit_transform(X)
                    model = LinearRegression().fit(X_poly, y)
                    X_fit = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
                    y_fit = model.predict(poly.transform(X_fit))
                    ax.plot(X_fit, y_fit, linestyle='--', label=f"{y_col} (fit)")

        ax.set_title(title, fontsize=16)
        ax.set_xlabel(xlabel, fontsize=14)
        ax.set_ylabel(ylabel, fontsize=14)
        ax.grid(True)
        ax.legend()
        sns.despine()

        st.pyplot(fig)

        # Export Options
        st.subheader("📤 Export Plot")
        img_buf = BytesIO()
        fig.savefig(img_buf, format="png", dpi=300)
        st.download_button("📷 Download PNG", img_buf.getvalue(), file_name="plot.png", mime="image/png")

        pdf_buf = BytesIO()
        fig.savefig(pdf_buf, format="pdf", dpi=300)
        st.download_button("📄 Download PDF", pdf_buf.getvalue(), file_name="plot.pdf", mime="application/pdf")

        st.success("✅ Graph is ready for thesis or publication!")
