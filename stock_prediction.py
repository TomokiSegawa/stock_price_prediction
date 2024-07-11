import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import japanize_matplotlib
import traceback
import requests
from pandas.tseries.offsets import BDay
from bs4 import BeautifulSoup
from statsmodels.tsa.arima.model import ARIMA
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.optimizers import Adam

def is_valid_stock_code(code):
    return len(code) == 4 and code.isdigit()

def get_stock_data(code):
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365 * 2)  # 2年分のデータを取得
    try:
        stock = yf.Ticker(f"{code}.T")
        df = stock.history(start=start_date, end=end_date)
        if df.empty:
            raise ValueError("データが取得できませんでした。")
        return stock, df
    except Exception as e:
        st.error(f"Yahoo Finance APIからのデータ取得に失敗しました: {str(e)}")
        return None, None

def get_company_name(code):
    try:
        stock = yf.Ticker(f"{code}.T")
        company_name = stock.info.get('longName') or stock.info.get('shortName')
        if company_name:
            return company_name

        url = f"https://www.google.com/search?q={code}+株式会社"
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(url, headers=headers)
        soup = BeautifulSoup(response.text, 'html.parser')
        
        title = soup.find('h3', class_='r')
        if title:
            return title.text.split('-')[0].strip()
        
        return f"企業 {code}"
    except Exception as e:
        st.warning(f"企業名の取得に失敗しました: {str(e)}")
        return f"企業 {code}"

def create_sequences(data, seq_length):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:(i + seq_length), 0])
        y.append(data[i + seq_length, 0])
    return np.array(X), np.array(y)

def build_lstm_model(input_shape):
    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=input_shape))
    model.add(LSTM(50))
    model.add(Dense(1))
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')
    return model

def predict_hybrid(df, forecast_period=10):
    # ARIMA model
    arima_model = ARIMA(df['Close'], order=(1, 1, 1))
    arima_results = arima_model.fit()
    arima_forecast = arima_results.forecast(steps=forecast_period)
    
    # LSTM model
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(df['Close'].values.reshape(-1, 1))
    
    seq_length = 60
    X, y = create_sequences(scaled_data, seq_length)
    
    split = int(0.8 * len(X))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    
    lstm_model = build_lstm_model((seq_length, 1))
    lstm_model.fit(X_train, y_train, epochs=50, batch_size=32, verbose=0)
    
    last_sequence = scaled_data[-seq_length:]
    lstm_forecast = []
    
    for _ in range(forecast_period):
        next_pred = lstm_model.predict(last_sequence.reshape(1, seq_length, 1))
        lstm_forecast.append(next_pred[0, 0])
        last_sequence = np.roll(last_sequence, -1)
        last_sequence[-1] = next_pred
    
    lstm_forecast = scaler.inverse_transform(np.array(lstm_forecast).reshape(-1, 1)).flatten()
    
    # Combine ARIMA and LSTM forecasts
    hybrid_forecast = (arima_forecast + lstm_forecast) / 2
    
    last_date = df.index[-1]
    future_dates = pd.date_range(start=last_date + BDay(1), periods=forecast_period, freq='B')
    forecast = pd.DataFrame({'ds': future_dates, 'yhat': hybrid_forecast})
    
    return forecast

def create_stock_chart(df, forecast, company_name, code):
    plt.figure(figsize=(12, 6))
    plt.plot(df.index, df['Close'], label='過去の株価')
    plt.plot(forecast['ds'], forecast['yhat'], label='予測株価', linestyle='--')
    plt.title(f'{company_name}（{code}）の株価チャート（過去2年間と今後の予測）')
    plt.xlabel('日付')
    plt.ylabel('株価')
    plt.legend()
    return plt

def create_prediction_table(df, forecast):
    last_close = df['Close'].iloc[-1]
    
    table_data = {
        '日付': forecast['ds'].dt.strftime('%Y-%m-%d'),
        '始値': [f'{round(last_close):,}'] + [f'{round(v):,}' for v in forecast['yhat'][:-1]],
        '終値': [f'{round(v):,}' for v in forecast['yhat']],
        '値差': [f'{round(forecast["yhat"].iloc[0] - last_close):,}'] + 
                [f'{round(forecast["yhat"].iloc[i] - forecast["yhat"].iloc[i-1]):,}' for i in range(1, len(forecast))],
        '騰落率': [f'{((forecast["yhat"].iloc[0] - last_close) / last_close * 100):.1f}%'] + 
                 [f'{((forecast["yhat"].iloc[i] - forecast["yhat"].iloc[i-1]) / forecast["yhat"].iloc[i-1] * 100):.1f}%' for i in range(1, len(forecast))]
    }
    prediction_df = pd.DataFrame(table_data)
    return prediction_df.set_index('日付').T

def main():
    st.set_page_config(layout="wide")
    st.title('株価予測アプリ（ハイブリッドモデル版）')

    stock_code = st.text_input('4桁の株式コードを入力してください:')

    if stock_code:
        if is_valid_stock_code(stock_code):
            try:
                stock, df = get_stock_data(stock_code)
                if stock is None or df is None:
                    st.error("データの取得に失敗しました。別の銘柄コードを試すか、しばらく待ってから再度お試しください。")
                    return

                company_name = get_company_name(stock_code)
                forecast = predict_hybrid(df)

                st.subheader(f'【{company_name}（{stock_code}）の株価チャート】')
                chart = create_stock_chart(df, forecast, company_name, stock_code)
                st.pyplot(chart)

                st.subheader(f'【{company_name}（{stock_code}）の株価予測表】')
                prediction_table = create_prediction_table(df, forecast)
                st.table(prediction_table)

            except Exception as e:
                st.error(f'エラーが発生しました: {str(e)}')
                st.error(f'詳細なエラー情報:\n{traceback.format_exc()}')
        else:
            st.error('無効な株式コードです。4桁の数字を入力してください。')

if __name__ == '__main__':
    main()