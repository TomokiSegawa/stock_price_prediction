import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import japanize_matplotlib
import traceback
import requests
from pandas.tseries.offsets import BDay
from bs4 import BeautifulSoup
from sklearn.preprocessing import MinMaxScaler

# TensorFlowのインポートを修正
import tensorflow as tf
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
        # まず、yfinanceから企業名を取得
        stock = yf.Ticker(f"{code}.T")
        company_name = stock.info.get('longName') or stock.info.get('shortName')
        if company_name:
            return company_name

        # yfinanceで取得できない場合、Google検索を試みる
        url = f"https://www.google.com/search?q={code}+株式会社"
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(url, headers=headers)
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Google検索結果の最初のタイトルを取得
        title = soup.find('h3', class_='r')
        if title:
            return title.text.split('-')[0].strip()
        
        return f"企業 {code}"
    except Exception as e:
        st.warning(f"企業名の取得に失敗しました: {str(e)}")
        return f"企業 {code}"

def prepare_data_for_lstm(df):
    data = df[['Close']].values
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)
    return scaled_data, scaler

def create_sequences(data, seq_length):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:(i + seq_length), 0])
        y.append(data[i + seq_length, 0])
    return np.array(X), np.array(y)

def build_lstm_model(input_shape):
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=input_shape))
    model.add(LSTM(units=50))
    model.add(Dense(1))
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')
    return model

def predict_stock_price(df):
    scaled_data, scaler = prepare_data_for_lstm(df)
    
    seq_length = 60
    X, y = create_sequences(scaled_data, seq_length)
    
    split = int(0.8 * len(X))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    
    model = build_lstm_model((seq_length, 1))
    model.fit(X_train, y_train, epochs=50, batch_size=32, verbose=0)
    
    last_sequence = scaled_data[-seq_length:]
    next_10_days = []
    
    for _ in range(10):
        next_pred = model.predict(last_sequence.reshape(1, seq_length, 1))
        next_10_days.append(next_pred[0, 0])
        last_sequence = np.roll(last_sequence, -1)
        last_sequence[-1] = next_pred
    
    next_10_days = scaler.inverse_transform(np.array(next_10_days).reshape(-1, 1))
    
    last_date = df.index[-1]
    future_dates = pd.date_range(start=last_date + BDay(1), periods=10, freq='B')
    forecast = pd.DataFrame({'ds': future_dates, 'yhat': next_10_days.flatten()})
    
    return forecast

def create_stock_chart(df, forecast, company_name, code):
    plt.figure(figsize=(12, 6))
    sns.lineplot(data=df, x=df.index, y='Close', label='過去の株価')
    sns.lineplot(x=forecast['ds'], y=forecast['yhat'], label='予測株価', linestyle='--')
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
    st.title('株価予測アプリ（改良版）')

    stock_code = st.text_input('4桁の株式コードを入力してください:')

    if stock_code:
        if is_valid_stock_code(stock_code):
            try:
                stock, df = get_stock_data(stock_code)
                if stock is None or df is None:
                    st.error("データの取得に失敗しました。別の銘柄コードを試すか、しばらく待ってから再度お試しください。")
                    return

                company_name = get_company_name(stock_code)
                forecast = predict_stock_price(df)

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