import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from io import StringIO
import docx
import PyPDF2

st.set_page_config(layout="wide", page_title="Universal Data Plotter")
st.title("📊 Universal Data Plotter and Analyzer")

# Function to read various file types
def read_uploaded_file(uploaded_file):
    if uploaded_file.name.endswith(".csv"):
        return pd.read_csv(uploaded_file)
    elif uploaded_file.name.endswith(".xlsx") or uploaded_file.name.endswith(".xls"):
        return pd.read_excel(uploaded_file)
    elif uploaded_file.name.endswith(".txt"):
        stringio = StringIO(uploaded_file.getvalue().decode("utf-8"))
        text = stringio.read()
        return pd.DataFrame({"Lines": text.splitlines()})
    elif uploaded_file.name.endswith(".pdf"):
        reader = PyPDF2.PdfReader(uploaded_file)
        text = ""
        for page in reader.pages:
            text += page.extract_text() + "\n"
        return pd.DataFrame({"PDF Text": text.splitlines()})
    elif uploaded_file.name.endswith(".docx"):
        doc = docx.Document(uploaded_file)
        paragraphs = [para.text for para in doc.paragraphs if para.text.strip() != ""]
        return pd.DataFrame({"Paragraphs": paragraphs})
    else:
        return None

# Layout
left, right = st.columns([2, 3])

with left:
    uploaded_file = st.file_uploader("📁 Upload CSV, Excel, PDF, Word, or Text file", type=["csv", "xlsx", "xls", "txt", "pdf", "docx"])

if uploaded_file:
    df = read_uploaded_file(uploaded_file)
    if df is not None:
        with right:
            st.subheader("📄 Data Preview")
            st.dataframe(df, use_container_width=True)

        if df.select_dtypes(include=["number"]).shape[1] >= 1:
            numeric_cols = df.select_dtypes(include=["float64", "int64"]).columns.tolist()
            category_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()

            with left:
                st.subheader("🧭 Plot Configuration")

                graph_type = st.selectbox("Select Graph Type", ["Line Plot", "Bar Chart", "Pie Chart"])
                x_col = st.selectbox("Select X-axis", options=numeric_cols + category_cols)
                y_col = st.selectbox("Select Y-axis", options=[col for col in numeric_cols if col != x_col])
                hue_col = st.selectbox("Optional: Group/Color By", options=["None"] + category_cols)
                hue_col = None if hue_col == "None" else hue_col
                apply_smooth = st.checkbox("Smooth Curve (Line Graph only)", value=False)
                dark_mode = st.checkbox("🌙 Dark Mode")

                title = st.text_input("Graph Title", f"{graph_type} of {y_col} vs {x_col}")

                if st.button("📈 Generate Plot"):
                    sns.set_style("darkgrid" if dark_mode else "whitegrid")
                    fig, ax = plt.subplots(figsize=(10, 6))

                    if graph_type == "Line Plot":
                        if hue_col:
                            sns.lineplot(data=df, x=x_col, y=y_col, hue=hue_col, marker="o", ax=ax)
                        else:
                            sns.lineplot(data=df, x=x_col, y=y_col, marker="o", ax=ax)
                        if apply_smooth:
                            X = df[[x_col]].values
                            y = df[y_col].values
                            poly = PolynomialFeatures(degree=2)
                            X_poly = poly.fit_transform(X)
                            model = LinearRegression().fit(X_poly, y)
                            x_fit = np.linspace(X.min(), X.max(), 200).reshape(-1, 1)
                            y_fit = model.predict(poly.transform(x_fit))
                            ax.plot(x_fit, y_fit, linestyle="--")

                    elif graph_type == "Bar Chart":
                        if hue_col:
                            sns.barplot(data=df, x=x_col, y=y_col, hue=hue_col, ax=ax)
                        else:
                            sns.barplot(data=df, x=x_col, y=y_col, ax=ax)

                    elif graph_type == "Pie Chart":
                        fig, ax = plt.subplots()
                        data_for_pie = df.groupby(x_col)[y_col].sum()
                        ax.pie(data_for_pie, labels=data_for_pie.index, autopct="%1.1f%%")

                    ax.set_title(title)
                    if graph_type != "Pie Chart":
                        ax.set_xlabel(x_col)
                        ax.set_ylabel(y_col)
                        ax.legend()
                        ax.grid(True)

                    st.pyplot(fig)
    else:
        st.error("❌ Unsupported or empty file. Please check the format and try again.")
