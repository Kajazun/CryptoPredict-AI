
https://github.com/user-attachments/assets/7b2633b2-4b6a-4eb8-b735-4ec85121dc92

https://github.com/user-attachments/assets/4aab4cd6-ca4b-459e-8e74-c2b050035db6
AI Crypto Forecaster: Next-Day Price Prediction
This project is a practical implementation of a Graduation Thesis titled: "Cryptocurrency price prediction for the next day using neural networks." The application serves as a functional Full-stack Web Dashboard that translates complex deep learning models into accessible financial insights.

Project Demo
https://github.com/user-attachments/assets/a6e94b1e-9e6f-46e0-a224-ca156661177a


Project Overview
The primary objective of this system is to analyze historical market trends and provide automated price forecasts for a 24-hour horizon. It focuses on bridging the gap between theoretical Recurrent Neural Networks (RNN) and real-time market software.

Key Features
Deep Learning Forecasts: Employs Gated Recurrent Unit (GRU) architectures specifically optimized for sequential time-series data.

Live Data Integration: Utilizes the Yahoo Finance API (yfinance) for up-to-the-minute market updates.

Decision Support: Provides automated BUY or SELL recommendations based on predicted vs. current market value.

Data Visualization: Includes interactive candlestick charts developed with Plotly.js for detailed technical analysis.

Sentiment Indicators: Features a live news feed to monitor external market drivers.

Technical Stack
Backend: Python, Flask

Machine Learning: TensorFlow, Keras, Scikit-learn

Data Science: Pandas, NumPy

Frontend: Tailwind CSS, Plotly.js

Data Source: Yahoo Finance RSS

Methodology
The prediction pipeline is structured according to the research findings of the original thesis:

Data Acquisition: Retrieval of the last 30 days of OHLC (Open, High, Low, Close) data.

Preprocessing: Feature scaling using MinMaxScaler to ensure neural network convergence.

Model Execution: Data is processed through a trained GRU-based RNN model.

Post-processing: Inverse transformation of normalized values into human-readable price predictions.

Local Deployment
Clone the Repository

Bash
git clone https://github.com/Kajazun/CryptoPredict-AI.git
cd CryptoPredict-AI
Environment Setup

Bash
python -m venv venv
# Activation for Windows:
venv\Scripts\activate
# Activation for macOS/Linux:
source venv/bin/activate
Dependency Installation

Bash
pip install -r requirements.txt
Application Launch

Bash
python app.py
The dashboard will be available at: http://127.0.0.1:5000

Academic Context
This software represents the final deliverable of my undergraduate studies. It investigates the efficacy of Gated Recurrent Units in identifying patterns within high-volatility financial environments.
