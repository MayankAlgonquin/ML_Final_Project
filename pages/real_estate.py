import streamlit as st
import pandas as pd
import pickle
import matplotlib.pyplot as plt

st.title("🏡 Real Estate Price Predictor (Random Forest)")

# Load model
@st.cache_resource
def load_model():
    return pickle.load(open("../models/rf_model_real_estate.pkl", "rb"))

model = load_model()

st.subheader("Enter Property Details")

# Inputs
year_sold = st.number_input("Year Sold", 2000, 2025, 2013)
property_tax = st.number_input("Property Tax", 0, 10000, 200)
insurance = st.number_input("Insurance", 0, 500, 80)
beds = st.number_input("Bedrooms", 1, 10, 3)
baths = st.number_input("Bathrooms", 1, 10, 2)
sqft = st.number_input("Square Feet", 200, 10000, 1000)
year_built = st.number_input("Year Built", 1900, 2025, 2000)
lot_size = st.number_input("Lot Size", 0, 10000, 500)
basement = st.selectbox("Basement", [0, 1])
popular = st.selectbox("Popular Area", [0, 1])
recession = st.selectbox("Recession", [0, 1])
property_age = st.number_input("Property Age", 0, 100, 10)
property_type_condo = st.selectbox("Is Condo?", [0, 1])

# IMPORTANT: match training order EXACTLY
feature_order = [
    "year_sold", "property_tax", "insurance", "beds", "baths",
    "sqft", "year_built", "lot_size", "basement", "popular",
    "recession", "property_age", "property_type_Condo"
]

if st.button("Predict Price"):
    input_dict = {
        "year_sold": year_sold,
        "property_tax": property_tax,
        "insurance": insurance,
        "beds": beds,
        "baths": baths,
        "sqft": sqft,
        "year_built": year_built,
        "lot_size": lot_size,
        "basement": basement,
        "popular": popular,
        "recession": recession,
        "property_age": property_age,
        "property_type_Condo": property_type_condo
    }

    input_df = pd.DataFrame([input_dict])[feature_order]

    prediction = model.predict(input_df)[0]

    st.success(f"💰 Estimated Price: ${prediction:,.0f}")
    st.subheader("📊 Feature Importance")

    if hasattr(model, "feature_importances_"):
        importance_df = pd.DataFrame({
            "feature": feature_order,
            "importance": model.feature_importances_
        }).sort_values(by="importance", ascending=False)

        st.bar_chart(importance_df.set_index("feature"))
        st.subheader("📈 Sensitivity (+/-10%)")

    fig, ax = plt.subplots()

    top_features = importance_df.head(5)["feature"]

    for feature in top_features:
        base = input_df.copy()
        original = base[feature].values[0]

        base[feature] = original * 1.1
        high = model.predict(base)[0]

        base[feature] = original * 0.9
        low = model.predict(base)[0]

        ax.plot(["-10%", "Base", "+10%"],
                [low, prediction, high],
                label=feature)

    ax.set_ylabel("Price")
    ax.legend()
    st.pyplot(fig)