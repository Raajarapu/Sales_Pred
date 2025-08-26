import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

st.title("📊 Sales Prediction App")

# File uploader
uploaded_file = st.file_uploader("Upload your Sales CSV file", type=["csv"])

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    st.subheader("📂 Dataset Preview")
    st.write(data.head())

    # Select features and target
    all_columns = data.columns.tolist()
    target_col = st.selectbox("Select Target Column (Y)", all_columns)
    feature_cols = st.multiselect("Select Feature Columns (X)", [c for c in all_columns if c != target_col])

    if target_col and feature_cols:
        X = data[feature_cols]
        y = data[target_col]

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Train model
        model = LinearRegression()
        model.fit(X_train, y_train)

        # Predictions
        y_pred = model.predict(X_test)

        st.subheader("📈 Model Evaluation")
        st.write(f"Mean Squared Error: {mean_squared_error(y_test, y_pred):.2f}")
        st.write(f"R² Score: {r2_score(y_test, y_pred):.2f}")

        # Scatter plot
        fig, ax = plt.subplots()
        ax.scatter(y_test, y_pred)
        ax.set_xlabel("Actual Sales")
        ax.set_ylabel("Predicted Sales")
        ax.set_title("Actual vs Predicted")
        st.pyplot(fig)

        # Try new prediction
        st.subheader("🔮 Make a New Prediction")
        input_data = []
        for col in feature_cols:
            value = st.number_input(f"Enter {col}", float(data[col].min()), float(data[col].max()))
            input_data.append(value)

        if st.button("Predict Sales"):
            result = model.predict([input_data])[0]
            st.success(f"Predicted Sales: {result:.2f}")
