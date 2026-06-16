import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt

st.title("📈 Stock Price Prediction System")

uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])

if uploaded_file is not None:

    df = pd.read_csv(uploaded_file)

    st.subheader("Dataset Preview")
    st.write(df.head())

    X = df[['Open', 'High', 'Low', 'Volume']]
    y = df['Close']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = LinearRegression()
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    score = r2_score(y_test, predictions)

    st.subheader("Model Accuracy")
    st.write(f"R² Score: {score:.4f}")

    st.subheader("Actual vs Predicted")

    fig, ax = plt.subplots()

    ax.plot(y_test.values, label="Actual")
    ax.plot(predictions, label="Predicted")

    ax.legend()

    st.pyplot(fig)
