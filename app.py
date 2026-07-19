import streamlit as st
import pandas as pd
import pickle

# 1. Load the pre-trained pipeline (Model + Preprocessing)
with open('RidgeModel.pkl', 'rb') as f:
    model = pickle.load(f)

# 2. UI Layout
st.title("Welcome to Bangalore House Price Predictor")
st.write("Want to predict the price of a new House in Bangalore? Try filling the details below:")

col1, col2 = st.columns(2)

with col1:
    # Use the locations list from your training data if you saved it in data.pkl
    with open('data.pkl', 'rb') as f:
        locations = pickle.load(f)
    location = st.selectbox("Select the Location:", locations)
    bath = st.number_input("Enter Number of Bathrooms:", min_value=1, step=1)

with col2:
    bhk = st.number_input("Enter BHK:", min_value=1, step=1)
    sqft = st.number_input("Enter Total Square Feet:", min_value=100.0)

# 3. Simple Prediction
if st.button("Predict Price"):
    # Create a DataFrame with the exact column names expected by your pipeline
    input_data = pd.DataFrame([[location, sqft, bath, bhk]],
                              columns=['location', 'total_sqft', 'bath', 'bhk'])

    # The pipeline handles scaling and encoding automatically
    prediction = model.predict(input_data)[0]

    st.success(f"The estimated price is: {prediction:.2f} Lakhs")