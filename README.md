# AI-Based-Multilingual-Weather-Forecasting-System-Using-Transformers

1. AI-Based Weather Forecast Web App

A real-time weather forecasting system that uses machine learning to predict temperature, humidity, wind speed, and the likelihood of rain for the next 1 to 24 hours. Built with Streamlit, this app delivers live forecasts using data from OpenWeatherMap and trained ML models.

2. Live Demo

Coming Soon – deploy on Streamlit Cloud after uploading this repo.

3. Features

Predicts weather for any city in India, Forecast ranges: next 1, 6, 12, 24 hours 

Outputs:
🌡️ Temperature (°C)
💧 Humidity (%)
🍃 Wind Speed (km/h)
🌧️ Will It Rain (Yes/No)

Real-time data from OpenWeatherMap API Powered by scikit-learn Random Forest + engineered features Cycle-aware model with is_daytime, sin_hour, cos_hour

4. Models Used

RandomForestRegressor for Temperature, Humidity, Wind Speed
 
RandomForestClassifier for rain yes/no

Models trained on historical data from 35 Indian cities

5. How to Use

Local Setup

Clone the repo:

git clone https://github.com/yourusername/weather-forecast-app.git
cd weather-forecast-app

Install dependencies:

pip install -r requirements.txt

Run the app:

streamlit run streamlit_app.py

Replace your_api_key_here in streamlit_app.py with your OpenWeatherMap API key.

6. Deploy on Streamlit Cloud

Upload this repo to GitHub

Go to https://streamlit.io/cloud

Click "New App" → select this repo

Set streamlit_app.py as the entry point

Click Deploy 

7. Folder Structure

weather-forecast-app/
├── streamlit_app.py
├── requirements.txt
├── run_forecast_pipeline.py
├── models/
│   ├── temp_model_with_cycle.pkl
│   ├── humidity_model_with_cycle.pkl
│   ├── wind_model_with_cycle.pkl
│   └── rain_model_with_cycle_balanced.pkl
├── temperature_scaler.pkl
├── humidity_scaler.pkl
├── wind_speed_scaler.pkl
└── README.md

8. Example Forecast Output

 Weather Forecast for Ulhasnagar — Next 6 Hour(s)
============================================================
      Time (IST)  🌡️ Temp (°C)  💧 Humidity (%)  🍃 Wind (km/h) 🌧️ Rain
2025-06-07 10:37         31.38           71.57          24.57      No
2025-06-07 11:37         32.10           71.51          24.28      No
2025-06-07 12:37         31.40           70.88          31.88      No
2025-06-07 13:37         30.98           77.99          28.19      No
2025-06-07 14:37         31.05           75.15          26.65      No
2025-06-07 15:37         30.50           73.68          28.68      No

9. License

This project is open-source for academic and research use.

10. Project Author

Harshal Rajendra Badgujar – AI-based Weather Forecasting System
