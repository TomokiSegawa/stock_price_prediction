import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from prophet import Prophet
from datetime import datetime, timedelta
import japanize_matplotlib
import traceback

def is_valid_stock_code(code):
    return len(code) == 4 and code.isdigit()

def get_stock_data(code):
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365 * 2)  # 2年分のデータを取得
    stock = yf.Ticker(f"{code}.T")
    df = stock.history(start=start_date, end=end_date)
    return stock, df

def prepare_data_for_prophet(df):
    prophet_df = df.reset_index()[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']]
    prophet_df.columns = ['ds', 'open', 'high', 'low', 'y', 'volume']
    prophet_df['ds'] = prophet_df['ds'].dt.tz_localize(None)
    
    # 追加の特徴量
    prophet_df['price_range'] = prophet_df['high'] - prophet_df['low']
    prophet_df['prev_close'] = prophet_df['y'].shift(1)
    prophet_df['close_to_open'] = prophet_df['y'] / prophet_df['open']
    
    return prophet_df.dropna()

def predict_stock_price(df):
    prophet_df = prepare_data_for_prophet(df)
    
    model = Prophet(
        daily_seasonality=True,
        weekly_seasonality=True,
        yearly_seasonality=True,
        changepoint_prior_scale=0.05,
        seasonality_prior_scale=10
    )
    
    # 追加の回帰変数
    model.add_regressor('open')
    model.add_regressor('high')
    model.add_regressor('low')
    model.add_regressor('volume')
    model.add_regressor('price_range')
    model.add_regressor('prev_close')
    model.add_regressor('close_to_open')
    
    model.fit(prophet_df)
    
    future_dates = model.make_future_dataframe(periods=5)  # 5日間の予測に変更
    
    # 将来の回帰変数の値を設定（単純に直近の値を使用）
    for column in ['open', 'high', 'low', 'volume', 'price_range', 'prev_close', 'close_to_open']:
        future_dates[column] = prophet_df[column].iloc[-1]
    
    forecast = model.predict(future_dates)
    
    return forecast.tail(5)['yhat']  # 5日間の予測を返す

# ... (他の関数は変更なし) ...

def main():
    st.set_page_config(layout="wide")
    st.title('株価予測アプリ（改良版）')

    stock_code = st.text_input('4桁の株式コードを入力してください:')

    if stock_code:
        if is_valid_stock_code(stock_code):
            try:
                stock, df = get_stock_data(stock_code)
                company_name = stock.info['longName']
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