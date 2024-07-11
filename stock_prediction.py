import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from datetime import datetime, timedelta
import japanize_matplotlib
import traceback

def is_valid_stock_code(code):
    return len(code) == 4 and code.isdigit()

def get_stock_data(code):
    end_date = datetime.now()
    start_date = end_date - timedelta(days=180)
    stock = yf.Ticker(f"{code}.T")
    df = stock.history(start=start_date, end=end_date)
    return df

def predict_stock_price(df):
    model = ARIMA(df['Close'], order=(1, 1, 1))
    results = model.fit()
    forecast = results.forecast(steps=10)
    return forecast

def create_stock_chart(df, forecast):
    plt.figure(figsize=(12, 6))
    sns.lineplot(data=df, x=df.index, y='Close', label='過去の株価')
    future_dates = pd.date_range(start=df.index[-1] + timedelta(days=1), periods=10)
    sns.lineplot(x=future_dates, y=forecast, label='予測株価', linestyle='--')
    plt.title('株価チャート（過去6ヶ月と今後10日間の予測）')
    plt.xlabel('日付')
    plt.ylabel('株価')
    plt.legend()
    return plt

def create_prediction_table(df, forecast):
    last_close = df['Close'].iloc[-1]
    today = pd.Timestamp.now().floor('D')
    dates = [today + timedelta(days=i) for i in range(11)]
    
    forecast_values = list(forecast.values)  # Convert forecast to a list
    
    table_data = {
        '日付': ['今日'] + [f'{i}日後' for i in range(1, 11)],
        '始値': [last_close] + forecast_values,
        '終値': forecast_values + [np.nan],
        '値差': [forecast_values[0] - last_close] + list(np.diff(forecast_values)) + [np.nan],
        '騰落率': [((forecast_values[0] - last_close) / last_close) * 100] + 
                  [((forecast_values[i] - last_close) / last_close) * 100 for i in range(1, 10)] + 
                  [np.nan]
    }
    prediction_df = pd.DataFrame(table_data)
    return prediction_df.set_index('日付').T  # Transpose the dataframe

def main():
    st.set_page_config(layout="wide")
    st.title('株価予測アプリ')

    stock_code = st.text_input('4桁の株式コードを入力してください:')

    if stock_code:
        if is_valid_stock_code(stock_code):
            try:
                df = get_stock_data(stock_code)
                forecast = predict_stock_price(df)

                st.subheader('株価チャート')
                chart = create_stock_chart(df, forecast)
                st.pyplot(chart)

                st.subheader('株価予測表')
                prediction_table = create_prediction_table(df, forecast)
                st.dataframe(prediction_table.style.format({
                    '始値': '{:.2f}',
                    '終値': '{:.2f}',
                    '値差': '{:.2f}',
                    '騰落率': '{:.2f}%'
                }), height=200)  # Adjust height to show full table

            except Exception as e:
                st.error(f'エラーが発生しました: {str(e)}')
                st.error(f'詳細なエラー情報:\n{traceback.format_exc()}')
        else:
            st.error('無効な株式コードです。4桁の数字を入力してください。')

if __name__ == '__main__':
    main()