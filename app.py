import streamlit as st
import joblib
import pandas as pd
import matplotlib.pyplot as plt

# Load model
model = joblib.load("model.pkl")

# Page config
st.set_page_config(page_title="House Price Predictor", layout="centered")

# Title
st.title("🏠 House Price Predictor")
st.write("Enter details to predict house price")

# Sidebar
st.sidebar.header("Input Details")

area = st.sidebar.number_input("Area", value=1000)
rooms = st.sidebar.number_input("Rooms", value=2)

# Button
if st.button("Predict Price 💰"):
    new_data = pd.DataFrame({
        "area": [area],
        "rooms": [rooms]
    })

    result = model.predict(new_data)

    st.success(f"Predicted Price: {result[0]}")

    # Graph
    st.subheader("📊 Visualization")

    # Sample data (same as training)
    df = pd.read_csv("data.csv")

    plt.scatter(df["area"], df["price"])
    plt.xlabel("Area")
    plt.ylabel("Price")

    # Plot predicted point
    plt.scatter(area, result[0])

    st.pyplot(plt)