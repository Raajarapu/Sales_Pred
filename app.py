import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# App title
st.title("📊 Sales Data EDA App")

# Upload CSV
uploaded_file = st.file_uploader("Upload your Sales CSV file", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.success("CSV uploaded successfully!")

    # Show raw data
    st.subheader("Dataset Preview")
    st.dataframe(df.head())

    # Show summary
    st.subheader("Summary Statistics")
    st.write(df.describe())

    # Column selection for plotting
    st.subheader("Visualizations")
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()

    if len(numeric_cols) >= 2:
        x_axis = st.selectbox("Choose X-axis", options=numeric_cols)
        y_axis = st.selectbox("Choose Y-axis", options=numeric_cols)

        fig, ax = plt.subplots()
        ax.scatter(df[x_axis], df[y_axis], alpha=0.6)
        ax.set_xlabel(x_axis)
        ax.set_ylabel(y_axis)
        ax.set_title(f"{x_axis} vs {y_axis}")
        st.pyplot(fig)
    else:
        st.warning("Not enough numeric columns for scatter plot.")

else:
    st.info("Please upload a CSV file to start.")
