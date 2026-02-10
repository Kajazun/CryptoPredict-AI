AI Crypto Forecaster: Next-Day Price Prediction
This project is the practical evolution of my Graduation Thesis, titled: "Cryptocurrency price prediction for the next day using neural networks." It transforms academic research into a functional Full-stack Web Dashboard.

Project Overview
The core objective of this application is to analyze historical cryptocurrency data and provide price forecasts for the next 24 hours. It bridges the gap between deep learning theory and real-world financial tools.

Key Features:
Deep Learning Prediction: Utilizes GRU (Gated Recurrent Unit) networks, optimized for time-series forecasting.

Real-time Data: Fetches live market data using the Yahoo Finance API (yfinance).

Automated Signals: Generates BUY or SELL recommendations based on the model's output vs. current price.

Interactive Visualization: Dynamic candlestick charts built with Plotly.js.

Live News Feed: Real-time crypto news integration for fundamental analysis.

Tech Stack
Backend: Python / Flask

AI/Machine Learning: TensorFlow, Keras, Scikit-learn

Data Processing: Pandas, NumPy

Frontend: Tailwind CSS (Modern Glassmorphism UI), Plotly.js

API: Yahoo Finance RSS

Methodology
Based on the research conducted in my thesis, the model:

Processes the last 30 days of historical data (Open, High, Low, Close).

Normalizes data using MinMaxScaler for stable neural network training.

Feeds the sequence into a GRU-based RNN architecture.

Inverts the transformation to provide a human-readable price forecast.

How to Run Locally
1.Clone the repository:
https://github.com/Kajazun/CryptoPredict-AI.git
cd CryptoPredict-AI

2.Create a Virtual Environment:
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

3.Install Dependencies:
pip install -r requirements.txt

4.Launch the App:
python app.py
Access the dashboard at http://127.0.0.1:5000

Academic Background
This project represents the final stage of my undergraduate studies, focusing on how Recurrent Neural Networks (RNNs) can identify patterns in highly volatile markets like Cryptocurrency.