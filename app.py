import streamlit as st
import pickle
import pandas as pd

# 1. Page Configuration
st.set_page_config(
    page_title="Bangalore House Price Predictor",
    page_icon="🏠",
    layout="centered"
)

# 2. Load the trained model and data
@st.cache_resource
def load_assets():
    model = pickle.load(open('RidgeModel.pkl', 'rb'))
    df = pickle.load(open('data.pkl', 'rb'))
    return model, df

model, df = load_assets()

# 3. App Header
st.markdown("<h1 style='text-align: center;'>Welcome to Bangalore House Price Predictor</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: gray;'>Want to predict the price of a new House in Bangalore? Try filling the details below:</p>", unsafe_allow_html=True)
st.write("")

# 4. Extract unique locations for the dropdown
locations = sorted(df['location'].unique())

# 5. Create a clean 2-column layout for input fields
col1, col2 = st.columns(2)

with col1:
    location = st.selectbox('Select the Location:', locations)
    bath = st.number_input('Enter Number of Bathrooms:', min_value=1, max_value=20, value=2, step=1)

with col2:
    bhk = st.number_input('Enter BHK:', min_value=1, max_value=15, value=2, step=1)
    total_sqft = st.number_input('Enter Total Square Feet:', min_value=300.0, max_value=50000.0, value=1200.0, step=50.0)

st.write("")
st.write("")

# 6. Prediction Button & Logic
if st.button('Predict Price', use_container_width=True):
    # Create a DataFrame matching the training feature names and types
    input_data = pd.DataFrame({
        'location': [location],
        'total_sqft': [float(total_sqft)],
        'bath': [int(bath)],
        'bhk': [int(bhk)]
    })
    
    try:
        # Predict using the loaded pipeline/model
        prediction = model.predict(input_data)[0]
        
        # Display the result styled cleanly matching the reference image format
        st.markdown(f"<h3 style='text-align: center; color: #2e7d32;'>Prediction: ₹{prediction:,.2f}</h3>", unsafe_allow_html=True)
    
    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")