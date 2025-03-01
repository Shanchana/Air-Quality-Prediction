# Air-Quality-Prediction

## Overview
This project is an air quality prediction model that utilizes Long Short-Term Memory (LSTM) networks to forecast pollution levels based on historical data. By integrating the World Air Quality Index (WAQI) API, the model incorporates real-time pollutant and meteorological factors to provide accurate and localized air quality predictions. The goal is to support public health initiatives and environmental monitoring efforts.

## Features
- **LSTM-based Prediction Model**: Uses deep learning to forecast air quality indices.
- **WAQI API Integration**: Fetches real-time air pollution and meteorological data.
- **Multi-factor Analysis**: Considers pollutants (e.g., PM2.5, PM10, NO2, SO2) and weather conditions.
- **Localized Forecasts**: Provides city or region-specific air quality predictions.
- **Public Health and Environmental Monitoring Support**: Helps in decision-making for pollution control and awareness campaigns.


## Dataset
The model is trained on historical air quality data retrieved from WAQI and supplemented with meteorological data. Data preprocessing includes:
- Handling missing values
- Normalization
- Time-series feature engineering

## Model Architecture
The LSTM model is designed with:
- Multiple stacked LSTM layers
- Dropout for regularization
- Fully connected output layer

## Results
The model is evaluated using:
- RMSE (Root Mean Square Error)
- MAE (Mean Absolute Error)
- Visualization of predicted vs. actual pollution levels

## Contributions
Feel free to contribute by improving model performance, adding new features, or optimizing data processing. Fork the repo and submit a pull request!


## Contact
For any queries, feel free to reach out at shanchana2317@gmail.com or open an issue in the repository.

---
Happy coding! 🚀

