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
import requests

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

def get_company_name(code, stock):
    try:
        # まず、yfinanceから企業名を取得
        company_name = stock.info.get('longName') or stock.info.get('shortName')
        if company_name:
            return company_name

        # yfinanceで取得できない場合、JPXのウェブサイトから取得を試みる
        url = f"https://www.jpx.co.jp/markets/statistics-equities/misc/01.html"
        response = requests.get(url)
        tables = pd.read_html(response.text)
        for table in tables:
            if 'コード' in table.columns and '銘柄名' in table.columns:
                match = table[table['コード'] == int(code)]
                if not match.empty:
                    return match['銘柄名'].values[0]
        
        # どちらの方法でも取得できない場合はデフォルト名を返す
        return f"企業 {code}"
    except Exception as e:
        st.warning(f"企業名の取得に失敗しました: {str(e)}")
        return f"企業 {code}"

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

def create_stock_chart(df, forecast, company_name, code):
    plt.figure(figsize=(12, 6))
    sns.lineplot(data=df, x=df.index, y='Close', label='過去の株価')
    future_dates = pd.date_range(start=df.index[-1] + timedelta(days=1), periods=5)
    sns.lineplot(x=future_dates, y=forecast, label='予測株価', linestyle='--')
    plt.title(f'{company_name}（{code}）の株価チャート（過去2年間と今後5日間の予測）')
    plt.xlabel('日付')
    plt.ylabel('株価')
    plt.legend()
    return plt

def create_prediction_table(df, forecast):
    last_close = df['Close'].iloc[-1]
    today = pd.Timestamp.now().floor('D')
    dates = [today + timedelta(days=i) for i in range(6)]
    
    forecast_values = list(forecast.values)  # Convert forecast to a list
    
    table_data = {
        '日付': ['今日'] + [f'{i}日後' for i in range(1, 6)],
        '始値': [f'{round(last_close):,}'] + [f'{round(v):,}' for v in forecast_values],
        '終値': [f'{round(v):,}' for v in forecast_values] + [''],
        '値差': [f'{round(forecast_values[0] - last_close):,}'] + 
                [f'{round(forecast_values[i] - forecast_values[i-1]):,}' for i in range(1, 5)] + [''],
        '騰落率': [f'{((forecast_values[0] - last_close) / last_close * 100):.1f}%'] + 
                 [f'{((forecast_values[i] - forecast_values[i-1]) / forecast_values[i-1] * 100):.1f}%' for i in range(1, 5)] + ['']
    }
    prediction_df = pd.DataFrame(table_data)
    return prediction_df.set_index('日付').T  # Transpose the dataframe

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

                company_name = get_company_name(stock_code, stock)
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