import os
import numpy as np
import pandas as pd
import yfinance as yf
import joblib
import json
import plotly
import plotly.graph_objects as go
import feedparser
import datetime as dt  
from flask import Flask, render_template, jsonify
from tensorflow.keras.models import load_model

app = Flask(__name__)
CRYPTO_LIST = ['BTC', 'ETH', 'LTC', 'DOGE']
MODELS = {}
SCALERS = {}

MAE_SCORES = {'BTC': 0.025, 'ETH': 0.031, "LTC": 6.261831211693168, 'DOGE': 0.01061369335631047}

# 1. Մոդելների բեռնում
print("⏳ Մոդելները բեռնվում են...")
for coin in CRYPTO_LIST:
    model_path = f'models/{coin}_model.h5'
    scaler_path = f'models/{coin}_scaler.pkl'
    if os.path.exists(model_path):
        MODELS[coin] = load_model(model_path, compile=False)
        SCALERS[coin] = joblib.load(scaler_path)
        print(f"✅ {coin} Ready")


@app.route('/')
def index():
    return render_template('index.html', cryptos=CRYPTO_LIST)


@app.route('/predict/<coin>')
def predict(coin):
    coin = coin.upper()
    if coin not in MODELS:
        return jsonify({'error': 'Model not found'}), 404

    df = yf.download(f"{coin}-USD", period='40d', interval='1d', progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    relevant_data = df[['Open', 'Close', 'High', 'Low']].tail(30).values
    scaled_data = SCALERS[coin].transform(relevant_data)

    prediction = MODELS[coin].predict(np.array([scaled_data]), verbose=0)
    dummy = np.zeros((1, 4))
    dummy[0, 1] = prediction[0, 0]
    final = SCALERS[coin].inverse_transform(dummy)[0, 1]

    error_rate = MAE_SCORES.get(coin, 0.05)
    confidence = (1 - error_rate) * 100

    return jsonify({
        'coin': coin,
        'prediction': round(float(final), 2),
        'confidence': round(confidence, 1)
    })


@app.route('/analysis')
def analysis():
    try:
        coin = "BTC"
        df = yf.download(f"{coin}-USD", period='60d', interval='1d', progress=False)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        last_price = df['Close'].iloc[-1]
        signal = {"type": "NEUTRAL", "color": "gray"}
        final_pred = None

        # Գրաֆիկի ստեղծում
        fig = go.Figure(data=[go.Candlestick(
            x=df.index, open=df['Open'], high=df['High'],
            low=df['Low'], close=df['Close'], name='Market'
        )])

        if coin in MODELS:
            recent_data = df[['Open', 'Close', 'High', 'Low']].tail(30).values
            scaled_data = SCALERS[coin].transform(recent_data)
            prediction = MODELS[coin].predict(np.array([scaled_data]), verbose=0)

            dummy = np.zeros((1, 4))
            dummy[0, 1] = prediction[0, 0]
            final_pred = SCALERS[coin].inverse_transform(dummy)[0, 1]

            # Ազդանշանի որոշում
            if final_pred > last_price:
                signal = {"type": "BUY", "color": "#10b981"}
            else:
                signal = {"type": "SELL", "color": "#ef4444"}

            # Ավելացնում ենք կանխատեսված կետը
            tomorrow = df.index[-1] + dt.timedelta(days=1)
            fig.add_trace(go.Scatter(
                x=[tomorrow], y=[final_pred],
                mode='markers',
                marker=dict(color='#fbbf24', size=15, symbol='star', line=dict(width=2, color='white')),
                name='AI Forecast'
            ))

        fig.update_layout(
            template='plotly_dark',
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            height=500,
            yaxis=dict(tickformat='~s', side='right', gridcolor='#1e293b'),
            xaxis=dict(gridcolor='#1e293b'),
            xaxis_rangeslider_visible=False,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            margin=dict(t=50, r=10, b=40, l=40)
        )

        graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

        # Նորություններ
        news_feed = feedparser.parse("https://news.google.com/rss/search?q=bitcoin+crypto&hl=en-US")
        news = [{'title': n.title, 'link': n.link, 'date': n.published[:16]} for n in news_feed.entries[:4]]

        return render_template('analysis.html', graphJSON=graphJSON, news=news,
                               signal=signal, last_price=round(last_price, 2))
    except Exception as e:
        return f"Error: {e}"


@app.route('/about')
def about():
    return render_template('about.html')


if __name__ == '__main__':
    app.run(port=5000, debug=False, use_reloader=False)