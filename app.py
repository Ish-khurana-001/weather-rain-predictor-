import streamlit as st
import joblib
import pandas as pd

# Load model (run weather_forecast.py and you will get pkl file)
model = joblib.load('weather_logreg_model.pkl')

st.title('🌦️ Weather Rain Prediction')
st.write("Enter today's weather data to predict if it will rain tomorrow.")

# Sidebar for input (basic ui for deployment)
with st.sidebar:
    st.header('Input Weather Data')
    humidity = st.slider('Humidity (%)', 0, 100, 50)
    pressure = st.slider('Pressure (hPa)', 980, 1040, 1013)
    min_temp = st.number_input('Min Temperature (°C)', value=15.0)
    max_temp = st.number_input('Max Temperature (°C)', value=25.0)
    wind_gust_speed = st.slider('Wind Gust Speed (km/h)', 0, 100, 30)
    wind_gust_dir = st.selectbox('Wind Gust Direction', [
        'N', 'NNE', 'NE', 'ENE', 'E', 'ESE', 'SE', 'SSE',
        'S', 'SSW', 'SW', 'WSW', 'W', 'WNW', 'NW', 'NNW', 'NA'
    ])

# Feature engineering for input
temp_range = max_temp - min_temp
wind_gust_speed_scaled = wind_gust_speed / 100  # Assuming 100 is max as in training

input_df = pd.DataFrame({
    'Humidity': [humidity],
    'Pressure': [pressure],
    'TempRange': [temp_range],
    'WindGustSpeedScaled': [wind_gust_speed_scaled],
    'WindGustDir': [wind_gust_dir]
})

if st.button('Predict Rain Tomorrow'):
    prediction = model.predict(input_df)[0]
    proba = model.predict_proba(input_df)[0][1]
    if prediction == 1:
        st.success(f"🌧️ Rain is likely tomorrow! (Probability: {proba:.2%})")
    else:
        st.info(f"☀️ No rain expected tomorrow. (Probability: {proba:.2%})")
