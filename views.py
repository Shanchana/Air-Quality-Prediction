from django.shortcuts import render
import pandas as pd
import openmeteo_requests
import requests_cache
from retry_requests import retry
from datetime import datetime, timedelta
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
import warnings
warnings.filterwarnings("ignore")



# Create your views here.

def home(request):
    return render(request, 'Home.html')

# Define a function to create sequences for LSTM
def create_sequences(data, sequence_length=24):
    X, y = [], []
    for i in range(len(data) - sequence_length):
        X.append(data[i:i+sequence_length])
        y.append(data[i+sequence_length])
    return np.array(X), np.array(y)

# pollutant data preprocessing
def pollutant_data_preprocess(df):
     # 1. First convert to datetime and keep it as datetime (don't convert back to string)
    df['date'] = pd.to_datetime(df['date'], format='%Y/%m/%d')
    
    # 2. Now sort
    df = df.sort_values('date')
    
    # 3. If you need to display dates in a specific format (only after all processing is done)
    df['date'] = df['date'].dt.strftime('%d-%m-%Y')

    df.set_index('date' , inplace = True) # set date as index

    # Convert object columns to numeric with error handling
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')  # Converts to NaN for non-convertible values
    
    # Interpolate missing values
    df = df.interpolate().bfill()

    return df

def get_weather_data():
    # Setup the Open-Meteo API client with cache and retry on error
    cache_session = requests_cache.CachedSession('.cache', expire_after=3600)
    retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
    openmeteo = openmeteo_requests.Client(session=retry_session)
    
    # Get the start and end dates for the previous month until today
    end_date = datetime.now().date()  # Today's date
    start_date = (end_date.replace(day=1) - timedelta(days=1)).replace(day=1)  # First day of the previous month
    
    # Update with Coimbatore coordinates and required weather variables
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": 11.0168,     # Latitude for Coimbatore
        "longitude": 76.9558,    # Longitude for Coimbatore
        "hourly": "temperature_2m,relative_humidity_2m,dew_point_2m,pressure_msl,wind_speed_10m,precipitation",
        "start_date": start_date.isoformat(),
        "end_date": end_date.isoformat(),
    }
    
    # Retrieve the data
    responses = openmeteo.weather_api(url, params=params)
    
    # Process the first location (or iterate for multiple locations)
    response = responses[0]

    # Process hourly data. The order of variables needs to be the same as requested.
    hourly = response.Hourly()
    hourly_temperature_2m = hourly.Variables(0).ValuesAsNumpy()
    hourly_relative_humidity_2m = hourly.Variables(1).ValuesAsNumpy()
    hourly_dew_point_2m = hourly.Variables(2).ValuesAsNumpy()
    hourly_pressure_msl = hourly.Variables(3).ValuesAsNumpy()
    hourly_wind_speed_10m = hourly.Variables(4).ValuesAsNumpy()
    hourly_precipitation = hourly.Variables(5).ValuesAsNumpy()
    
    hourly_data = {
        "date": pd.date_range(
            start=pd.to_datetime(hourly.Time(), unit="s", utc=True),
            end=pd.to_datetime(hourly.TimeEnd(), unit="s", utc=True),
            freq=pd.Timedelta(seconds=hourly.Interval()),
            inclusive="left"
        )
    }
    hourly_data["temperature_2m"] = hourly_temperature_2m
    hourly_data["relative_humidity_2m"] = hourly_relative_humidity_2m
    hourly_data["dew_point_2m"] = hourly_dew_point_2m
    hourly_data["pressure_msl"] = hourly_pressure_msl
    hourly_data["wind_speed_10m"] = hourly_wind_speed_10m
    hourly_data["precipitation"] = hourly_precipitation

    df_hourly = pd.DataFrame(data=hourly_data)

    # 1. First convert dates to datetime (keeping original format as YYYY/M/D)
    df_hourly['date'] = pd.to_datetime(df_hourly['date'], format='%Y/%m/%d')
    
    # 2. Sort the data chronologically
    df_hourly = df_hourly.sort_values('date')
    
    # 3. Set date as index and resample
    df_hourly.set_index('date', inplace=True)
    daily_weather = df_hourly.resample('D').mean()
    
    # 4. Only if you need the dates in DD-MM-YYYY format after all processing:
    daily_weather.index = daily_weather.index.strftime('%d-%m-%Y')

    daily_weather = daily_weather.reset_index()

    return daily_weather

def mainfunction(request):
    df = pd.read_csv(r"C:\Users\sanja\Downloads\sidco-kurichi, coimbatore-air-quality.csv")

    df = pollutant_data_preprocess(df)

    daily_weather = get_weather_data()

    # Merge pollutant data with weather data
    combined_data = pd.merge(df, daily_weather, on='date', how='inner')

    combined_data.columns = combined_data.columns.str.strip() # strip the spaces in the name

    df = combined_data.apply(lambda x: np.log1p(x) if pd.api.types.is_numeric_dtype(x) else x)

    target_columns = ['pm25', 'pm10', 'o3', 'no2', 'so2', 'co', 
                  'temperature_2m', 'relative_humidity_2m', 'dew_point_2m', 
                  'pressure_msl', 'wind_speed_10m', 'precipitation']
    

    # Model training and prediction
    predictions = {}
    
    for col in target_columns:
        # Prepare sequences for each column
        data = df[col].values
        X, y = create_sequences(data)
    
        # Reshape for LSTM input (samples, timesteps, features)
        X = X.reshape((X.shape[0], X.shape[1], 1))
    
        # Build the LSTM model
        model = Sequential()
        model.add(LSTM(50, activation='relu', input_shape=(X.shape[1], X.shape[2])))
        model.add(Dense(1))
        model.compile(optimizer='adam', loss='mse')
    
        # Train the model
        model.fit(X, y, epochs=10, batch_size=1, verbose=0)
    
        # Prepare the last 24 hours of data for prediction
        last_sequence = data[-24:].reshape((1, 24, 1))
    
        # Predict the next 24 hours
        yhat = model.predict(last_sequence, verbose=0)
    
        # Store predictions without reverse scaling, as log transformation was already applied
        predictions[col] = yhat.flatten()

    for key, pred in predictions.items():
        predictions[key] = np.expm1(pred)  # Convert each prediction array back to the original scale

    aqi_values = []
    aqi_values.append(predictions.get('pm25', [0])[0])
    aqi_values.append(predictions.get('pm10', [0])[0])
    aqi_values.append(predictions.get('o3', [0])[0])
    aqi_values.append(predictions.get('no2', [0])[0])
    aqi_values.append(predictions.get('so2', [0])[0])
    aqi_values.append(predictions.get('co', [0])[0])

    # Weather data from predictions dictionary
    temperature = predictions.get('temperature_2m', [0])[0]
    humidity = predictions.get('relative_humidity_2m', [0])[0]
    dew_point = predictions.get('dew_point_2m', [0])[0]
    pressure = predictions.get('pressure_msl', [0])[0]
    wind_speed = predictions.get('wind_speed_10m', [0])[0]
    precipitation = predictions.get('precipitation', [0])[0]

    # Calculate weather influence as a scaling factor
    weather_influence = (
        1 + 0.001 * (temperature - 25)          # Adjust around moderate level
        + 0.002 * (humidity / 100)              # Influence from humidity
        - 0.001 * (dew_point / 10)              # Influence from dew point
        + 0.0005 * (pressure / 1000)            # Influence from pressure
        - 0.001 * (wind_speed / 5)              # Influence from wind speed
        - 0.002 * precipitation                 # Influence from precipitation
    )

    # Calculate total AQI considering weather influence
    total_aqi = sum(aqi_values) * weather_influence

    return render(request , 'mainpage.html' , {'aqi' : int(total_aqi) } )





    
    
    
