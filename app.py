import streamlit as st
import pandas as pd
import joblib  

# 1. Load the model and the full cleaned dataframe
try:
    # Loading using joblib and the .joblib file extensions
    model = joblib.load('RidgeModel.joblib')
    df = joblib.load('data.joblib')
except Exception as e:
    st.error(f"Error loading model or data: {e}")
    st.stop()

# 2. Page Title
st.title("Welcome to Bangalore House Price Predictor")

# 3. Dynamic Inputs
col1, col2 = st.columns(2)

with col1:
    # Pulling unique locations dynamically from the dataframe
    location = st.selectbox("Select the Location:", sorted(df['location'].unique()))
    bath = st.number_input("Enter Number of Bathrooms:", min_value=1, step=1)

with col2:
    bhk = st.number_input("Enter BHK:", min_value=1, step=1)
    sqft = st.number_input("Enter Total Square Feet:", min_value=100.0)

# 4. Prediction Logic
if st.button("Predict Price"):
    try:
        # Create input DataFrame with columns matching your training data
        input_data = pd.DataFrame(
            [[location, sqft, bath, bhk]],
            columns=['location', 'total_sqft', 'bath', 'bhk']
        )

        # Predict
        prediction = model.predict(input_data)

        # Display result
        st.success(f"The estimated price is: {prediction[0]:,.2f} Lakhs")

    except Exception as e:
        st.error(f"Error during prediction: {e}")