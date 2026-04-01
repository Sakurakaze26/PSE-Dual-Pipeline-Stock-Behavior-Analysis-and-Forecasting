# PSE Dual-Pipeline Stock Behavior Analysis and Forecasting

## Overview
This project presents a dual-pipeline machine learning approach to understanding and predicting short-term stock price behavior in the **Philippine Stock Exchange (PSE)**. It utilizes over 480,000 daily trading records from 303 publicly listed companies, spanning from December 2011 to March 2021. 

The analysis explores two distinct perspectives:
1.  **Temporal Forecasting:** Capture sequential trends and predict future `Change%` values using chronological data.
2.  **Non-Temporal Supervised Learning:** Reframes the problem as a row-level regression task, treating each stock-day record independently.

## Dataset Information
* **Records:** 481,921
* **Companies:** 303
* **Date Range:** December 2011 – March 24, 2021
* **Target Variable:** `Change%` (Daily percentage change from the previous closing price)
* **Features:** Stock Name, Code, Date, Open, High, Low, Price, and Volume
* **Data Source:** Web-scraped historical data from Investing.com

## Project Structure
The notebook is organized into several phases to handle the dual-track modeling setup:
* **Data Cleaning:** Sorting price history by date, handling missing entries (notably in the Volume column), and ensuring temporal consistency.
* **Feature Engineering:** Addition of technical indicators such as moving averages (MA_3, MA_10), volatility metrics, and lag-based features (Lag_Change_1).
* **Modeling Pipelines:**
    * **Forecasting Track:** Implements `LSTM` (Long Short-Term Memory) and Facebook's `Prophet` models.
    * **Regression Track:** Utilizes `XGBoost`, `Random Forest`, and `Linear Regression`.

## Key Results
The following models were evaluated on Root Mean Squared Error (RMSE), Mean Absolute Error (MAE), and Directional Accuracy (predicting if the price goes up or down):

| Model | RMSE | MAE | Directional Accuracy |
| :--- | :--- | :--- | :--- |
| **LSTM** | 1.8279 | 1.3487 | 63.17% |
| **XGBoost** | 1.8148 | 1.3306 | 62.11% |
| **Prophet** | 1.9806 | 1.4186 | 58.59% |

* **LSTM** proved highly effective as a sequence-based model, successfully capturing short-term trends signals.
* **XGBoost** performed strongly by effectively modeling nonlinear relationships and interactions using engineered features.
* **Prophet** struggled with short-term fluctuations, as it is primarily designed for capturing long-term trends and seasonality rather than noisy daily market volatility.

## Requirements
To run this notebook, you will need:
* Python 3.x
* Pandas & NumPy
* Scikit-Learn
* TensorFlow (for LSTM)
* Prophet
* XGBoost
* Matplotlib & Seaborn (for visualization)
