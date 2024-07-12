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
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.optimizers import Adam
import xgboost as xgb
from sklearn.model_selection import train_test_split
import time
import sys
import os

def is_valid_stock_code(code):
    return len(code) == 4 and code.isdigit()

def get_stock_data(code, max_retries=3, retry_delay=5):
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365 * 2)  # 2年分のデータを取得
    
    for attempt in range(max_retries):
        try:
            stock = yf.Ticker(f"{code}.T")
            df = stock.history(start=start_date, end=end_date, timeout=10)
            if df.empty:
                raise ValueError("データが取得できませんでした。")
            return stock, df
        except requests.exceptions.RequestException as e:
            if attempt < max_retries - 1:
                st.warning(f"データ取得に失敗しました。{retry_delay}秒後に再試行します。(試行回数: {attempt + 1})")
                time.sleep(retry_delay)
            else:
                st.error(f"Yahoo Finance APIからのデータ取得に失敗しました: {str(e)}")
                return None, None
        except Exception as e:
            st.error(f"予期せぬエラーが発生しました: {str(e)}")
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
        X.append(data[i:(i + seq_length), :])
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
    # データの準備
    data = df[['Close', 'Open', 'High', 'Low', 'Volume']].values
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)

    # LSTMモデルの学習
    seq_length = 60
    X, y = create_sequences(scaled_data, seq_length)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    lstm_model = build_lstm_model((seq_length, 5))
    lstm_model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test), verbose=0)

    # XGBoostモデルの学習
    xgb_data = df[['Open', 'High', 'Low', 'Volume']].values
    xgb_target = df['Close'].values
    xgb_model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100)
    xgb_model.fit(xgb_data, xgb_target)

    # 予測
    last_sequence = scaled_data[-seq_length:]
    lstm_forecast = []
    xgb_forecast = []

    try:
        for _ in range(forecast_period):
            # LSTMの予測
            lstm_input = last_sequence.reshape(1, seq_length, 5)
            lstm_pred = lstm_model.predict(lstm_input, verbose=0)
            
            # XGBoostの予測
            xgb_input = last_sequence[-1, 1:].reshape(1, -1)
            xgb_pred = xgb_model.predict(xgb_input)
            
            hybrid_pred = (lstm_pred[0, 0] + xgb_pred[0]) / 2
            lstm_forecast.append(hybrid_pred)
            xgb_forecast.append(hybrid_pred)

            new_row = np.array([hybrid_pred] + list(last_sequence[-1, 1:]))
            last_sequence = np.roll(last_sequence, -1, axis=0)
            last_sequence[-1] = new_row

    except Exception as e:
        st.error(f"予測中にエラーが発生しました: {str(e)}")
        st.error(traceback.format_exc())
        return None

    # スケール変換を戻す
    forecast = scaler.inverse_transform(np.column_stack((lstm_forecast, last_sequence[-len(lstm_forecast):, 1:])))[:, 0]

    last_date = df.index[-1]
    future_dates = pd.date_range(start=last_date + BDay(1), periods=forecast_period, freq='B')
    forecast_df = pd.DataFrame({'ds': future_dates, 'yhat': forecast})

    return forecast_df


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
    # エラー出力をファイルにリダイレクト
    sys.stderr = open('error_log.txt', 'w')
    
    st.set_page_config(layout="wide")
    st.title('株価予測アプリ')

    stock_code = st.text_input('4桁の株式コードを入力してください:')

    if stock_code:
        if is_valid_stock_code(stock_code):
            try:
                with st.spinner('データを取得中...'):
                    stock, df = get_stock_data(stock_code)
                if stock is None or df is None:
                    st.error("データの取得に失敗しました。別の銘柄コードを試すか、しばらく待ってから再度お試しください。")
                    return

                company_name = get_company_name(stock_code)
                
                with st.spinner('予測を計算中...'):
                    forecast = predict_hybrid(df)
                
                if forecast is None:
                    st.error("予測の計算中にエラーが発生しました。")
                    return

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

    # エラーログを表示（開発時のデバッグ用）
    if os.path.exists('error_log.txt'):
        with open('error_log.txt', 'r') as f:
            st.text(f.read())

if __name__ == '__main__':
    main()