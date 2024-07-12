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
from prophet import Prophet

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

def predict_hybrid(df, forecast_period=10):
    # ARIMA model
    arima_model = ARIMA(df['Close'], order=(1, 1, 1))
    arima_results = arima_model.fit()
    arima_forecast = arima_results.forecast(steps=forecast_period)
    
    # Prophet model
    prophet_df = df.reset_index()[['Date', 'Close']]
    prophet_df.columns = ['ds', 'y']
    prophet_df['ds'] = prophet_df['ds'].dt.tz_localize(None)  # タイムゾーン情報を削除
    prophet_model = Prophet(daily_seasonality=True)
    prophet_model.fit(prophet_df)
    future_dates = prophet_model.make_future_dataframe(periods=forecast_period)
    prophet_forecast = prophet_model.predict(future_dates)
    prophet_forecast = prophet_forecast.tail(forecast_period)['yhat']
    
    # Combine ARIMA and Prophet forecasts
    hybrid_forecast = (arima_forecast + prophet_forecast.values) / 2
    
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