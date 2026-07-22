import streamlit as st
import pickle
import numpy as np

# 1. Load the model and dataframe using correct file names and binary read mode
try:
    pipe = pickle.load(open('RidgeModel.pkl', 'rb'))
    df = pickle.load(open('data.pkl', 'rb'))
except Exception as e:
    st.error(f"Error loading model or data: {e}")
    st.stop()

# 2. Page Title
st.title("Laptop Price Predictor")

# 3. Dynamic Inputs for Laptop Specifications
company = st.selectbox('Brand', df['Company'].unique())
type = st.selectbox('Type', df['TypeName'].unique())
ram = st.selectbox('RAM (in GB)', [2, 4, 6, 8, 12, 16, 24, 32, 64])
weight = st.number_input('Weight of the Laptop (kg)')
touchscreen = st.selectbox('Touchscreen', ['No', 'Yes'])
ips = st.selectbox('IPS', ['No', 'Yes'])
screen_size = st.slider('Screen size in inches', 10.0, 18.0, 13.0)
resolution = st.selectbox('Screen Resolution',
                          ['1920x1080', '1366x768', '1600x900', '3840x2160', '3200x1800', '2880x1800', '2560x1600',
                           '2560x1440', '2304x1440'])
cpu = st.selectbox('CPU', df['Cpu_Brand'].unique())
hdd = st.selectbox('HDD (in GB)', [0, 128, 256, 512, 1024, 2048])
ssd = st.selectbox('SSD (in GB)', [0, 8, 128, 256, 512, 1024])
gpu = st.selectbox('GPU', df['Gpu_Brand'].unique())
os = st.selectbox('OS', df['OS'].unique())

# 4. Prediction Logic
if st.button('Predict Price'):
    try:
        # Format binary inputs
        ts_val = 1 if touchscreen == 'Yes' else 0
        ips_val = 1 if ips == 'Yes' else 0

        # Calculate PPI from resolution and screen size
        X_res = int(resolution.split('x')[0])
        Y_res = int(resolution.split('x')[1])
        ppi = ((X_res ** 2) + (Y_res ** 2)) ** 0.5 / screen_size

        # Create query array matching your model pipeline expectations
        query = np.array([company, type, ram, weight, ts_val, ips_val, ppi, cpu, hdd, ssd, gpu, os], dtype=object)
        query = query.reshape(1, 12)

        # Predict price (reversing log transformation if applicable)
        predicted_price = int(np.exp(pipe.predict(query)[0]))

        # Display result
        st.success(f"The estimated price of this configuration is: €{predicted_price:,}")

    except Exception as e:
        st.error(f"Error during prediction: {e}")