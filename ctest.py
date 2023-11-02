# Import necessary libraries
import streamlit as st
import pandas as pd
import os
import shutil
import numpy as np
import matplotlib.pyplot as plt
from sqlalchemy import create_engine
import statsmodels.api as sm
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.metrics import mean_absolute_error, mean_squared_error
import time
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.application import MIMEApplication
from io import BytesIO
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter

# Function for ARIMA modeling with customizable parameters
def arima_model(data, p, d, q):
    model = sm.tsa.ARIMA(data, order=(p, d, q))
    results = model.fit()
    return results

# Create a Streamlit app
st.title("Time-Series Data Forecasting App")

# Create a temporary directory to store uploaded files
UPLOADS_DIR = "uploads"
if not os.path.exists(UPLOADS_DIR):
    os.makedirs(UPLOADS_DIR)

# Function to import data from an Excel file
def import_data_from_excel(file_path):
    if file_path is not None:
        try:
            df = pd.read_excel(file_path, engine='openpyxl')  # Specify the engine
            return df
        except Exception as e:
            st.error(f"An error occurred while reading the Excel file: {e}")
    else:
        return None

# Function to import data from a CSV file
def import_data_from_csv(file_path):
    if file_path is not None:
        try:
            df = pd.read_csv(file_path)  # Specify the engine if necessary
            return df
        except Exception as e:
            st.error(f"An error occurred while reading the CSV file: {e}")
    else:
        return None

# Function to import data from a local file
def import_data_from_file(file_path):
    if file_path is not None:
        file_extension = file_path.name.split('.')[-1].lower()
        if file_extension in ['xls', 'xlsx']:
            return import_data_from_excel(file_path)  # Use the Excel function
        elif file_extension == 'csv':
            return import_data_from_csv(file_path)  # Use the new CSV function
        else:
            # Handle other file formats or raise an error
            raise ValueError("Unsupported file format")
    else:
        return None

# Add an "Upload Button" to allow users to upload Excel spreadsheets and CSV files
st.sidebar.header("Data Import and Integration")
uploaded_file = st.file_uploader("Upload Data File", type=["xls", "xlsx", "csv"])

# Data source selection
data_source = st.sidebar.selectbox("Select Data Source", ["Local File", "Database", "API"])

# Function to save uploaded files securely
def save_uploaded_file(uploaded_file):
    if uploaded_file is not None:
        file_path = os.path.join(UPLOADS_DIR, uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.read())
        return file_path
    else:
        return None

# Function to import data from a database
def import_data_from_database(database_url, table_name):
    engine = create_engine(database_url)
    query = f"SELECT * FROM {table_name}"
    df = pd.read_sql(query, engine)
    return df

# Function to import data from an API (You would need to implement this based on the API you are using)
def import_data_from_api(api_url):
    # Implement API data retrieval logic here
    pass

# Display instructions for data source selection
if data_source == "Local File":
    st.sidebar.info("Please upload an Excel file or a CSV file.")
elif data_source == "Database":
    st.sidebar.info("Enter the database URL and table name.")
else:  # API
    st.sidebar.info("Enter the API URL.")

# Function to handle missing values
def handle_missing_values(data, method):
    if method == "Drop Rows with Missing Values":
        data = data.dropna()
    elif method == "Fill with Mean":
        data = data.fillna(data.mean())
    elif method == "Fill with Median":
        data = data.fillna(data.median())
    elif method == "Fill with Forward Fill":
        data = data.ffill()
    elif method == "Fill with Backward Fill":
        data = data.bfill()
    return data

# Function to smooth the data
def smooth_data(data, window_size):
    if window_size > 1:
        data['Smoothed'] = data['Value'].rolling(window=window_size).mean()
        return data
    else:
        return data

# Function to remove outliers using Z-score
#def remove_outliers(data, z_threshold):
    #z_scores = np.abs((data['Value'] - data['Value'].mean()) / data['Value'].std())
    #data_filtered = data[z_scores < z_threshold]
    #return data_filtered

# Function to remove outliers using Z-score
def remove_outliers(data, z_threshold, column_name='Value'):
    if column_name in data.columns:
        z_scores = np.abs((data[column_name] - data[column_name].mean()) / data[column_name].std())
        data_filtered = data[z_scores < z_threshold]
        return data_filtered
    else:
        raise KeyError(f"Column '{column_name}' not found in the DataFrame.")


# Function to calculate MAE and MSE
def calculate_metrics(true_values, predicted_values):
    mae = mean_absolute_error(true_values, predicted_values)
    mse = mean_squared_error(true_values, predicted_values)
    return mae, mse

# Function for generating forecasts
#def generate_forecast(model_results, steps):
    #forecast = model_results.get_forecast(steps=steps)
    #forecast_mean = forecast.predicted_mean
    #forecast_conf_int = forecast.conf_int()
    #return forecast_mean, forecast_conf_int

# Function for generating forecasts
def generate_forecast(model_results, steps):
    # Use the forecast method instead of get_forecast
    forecast = model_results.forecast(steps)
    forecast_mean = forecast
    # You may also obtain confidence intervals if supported by the model
    forecast_conf_int = model_results.get_forecast(steps).conf_int()  # Modify this line if needed
    return forecast_mean, forecast_conf_int


# Function for time-series decomposition
def decompose_time_series(data, decomposition_type):
    if decomposition_type == "Additive":
        decomposition = seasonal_decompose(data, model="additive")
    elif decomposition_type == "Multiplicative":
        decomposition = seasonal_decompose(data, model="multiplicative")

    trend = decomposition.trend
    seasonal = decomposition.seasonal
    residual = decomposition.resid

    return trend, seasonal, residual

# Data Preprocessing Section
if st.checkbox("Data Preprocessing"):
    st.header("Data Preprocessing")

    # Handle missing values
    if uploaded_file:
        data_filtered = import_data_from_file(uploaded_file)
        st.subheader("Handle Missing Values")
        missing_method = st.selectbox("Select Missing Value Handling Method", ["None", "Drop Rows with Missing Values", "Fill with Mean", "Fill with Median", "Fill with Forward Fill", "Fill with Backward Fill"])

        if missing_method != "None":
            data_filtered = handle_missing_values(data_filtered, missing_method)
            st.write("Data after Handling Missing Values:")
            st.write(data_filtered)
    else:
        st.warning("Please upload your data file first.")
    
    # Smooth the data
    st.subheader("Smooth Data")
    if 'data_filtered' in locals():
        smooth_window_size = st.slider("Select Window Size for Smoothing (1 for no smoothing)", min_value=1, max_value=len(data_filtered))
        data_filtered = smooth_data(data_filtered, smooth_window_size)
        st.write("Data after Smoothing:")
        st.write(data_filtered)
        
        # Remove outliers
        st.subheader("Remove Outliers (Z-score)")
        z_threshold = st.slider("Select Z-score Threshold for Outlier Removal", min_value=0.1, max_value=10.0, step=0.1)
        #data_filtered = remove_outliers(data_filtered, z_threshold)
        data_filtered = remove_outliers(data_filtered, z_threshold, column_name='Value')

        st.write("Data after Outlier Removal:")
        st.write(data_filtered)
    else:
        st.warning("Please upload historical data and filter it first.")

# Model Selection Section
st.header("Model Selection")
st.subheader("Choose Forecasting Model")

# Define available forecasting models
models = {
    "ARIMA": "ARIMA (AutoRegressive Integrated Moving Average)",
    "Exponential Smoothing": "Exponential Smoothing",
    # Add other models as needed
}

selected_model = st.selectbox("Select a Forecasting Model", list(models.keys()))

if selected_model == "ARIMA":
    st.subheader("ARIMA Model")
    order_p = st.slider("Select AR (AutoRegressive) Order (p)", min_value=0, max_value=10, step=1)
    order_d = st.slider("Select I (Integrated) Order (d)", min_value=0, max_value=2, step=1)
    order_q = st.slider("Select MA (Moving Average) Order (q)", min_value=0, max_value=10, step=1)

if selected_model == "Exponential Smoothing":
    st.subheader("Exponential Smoothing Model")
    trend_type = st.selectbox("Select Trend Type", ["add", "additive", "multiplicative"])
    seasonal_type = st.selectbox("Select Seasonal Type", ["add", "additive", "multiplicative"])
    seasonal_period = st.number_input("Enter Seasonal Period (e.g., 12 for monthly data)", min_value=1, value=12)

# Forecasting Section
if st.button("Generate Forecast"):
    if 'data_filtered' in locals():
        # Forecasting logic here based on the selected model
        if selected_model == "ARIMA":
            model_results = arima_model(data_filtered['Value'], order_p, order_d, order_q)
            forecast_mean, forecast_conf_int = generate_forecast(model_results, steps=12)
        elif selected_model == "Exponential Smoothing":
            model = ExponentialSmoothing(data_filtered['Value'], trend=trend_type, seasonal=seasonal_type, seasonal_periods=seasonal_period)
            model_results = model.fit()
            forecast_mean, forecast_conf_int = generate_forecast(model_results, steps=12)

        # Create a DataFrame with the forecast
        forecast_df = pd.DataFrame({'Date': pd.date_range(start=data_filtered.index[-1], periods=len(forecast_mean)), 'Forecast': forecast_mean})

        st.subheader("Forecasted Data")
        st.write(forecast_df)

        # Calculate metrics (MAE and MSE)
        true_values = data_filtered['Value'][-12:]
        predicted_values = forecast_mean
        
        # Check if true_values and predicted_values have the same length
        if len(true_values) != len(predicted_values):
            st.error("The lengths of true_values and predicted_values must be the same.")
        else:
            # Calculate metrics (MAE and MSE)
            mae = mean_absolute_error(true_values, predicted_values)
            mse = mean_squared_error(true_values, predicted_values)
            st.subheader("Model Evaluation")
            st.write(f"Mean Absolute Error (MAE): {mae}")
            st.write(f"Mean Squared Error (MSE): {mse}")

        # Generate a line plot for the forecast
        st.subheader("Forecast Plot")
        plt.figure(figsize=(10, 5))
        plt.plot(data_filtered.index, data_filtered['Value'], label='Actual Data', marker='o')
        plt.plot(forecast_df['Date'], forecast_df['Forecast'], label='Forecast', linestyle='--', marker='o')
        plt.fill_between(forecast_df['Date'], forecast_conf_int.iloc[:, 0], forecast_conf_int.iloc[:, 1], color='pink', alpha=0.4)
        plt.xlabel("Date")
        plt.ylabel("Value")
        plt.title("Actual Data vs. Forecast")
        plt.legend()
        st.pyplot(plt)
    else:
        st.warning("Please upload historical data and preprocess it before generating a forecast.")
