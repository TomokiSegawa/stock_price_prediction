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
from pandas.tseries.offsets import BDay
from bs4 import BeautifulSoup

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
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
        response = requests.get(url, headers=headers)
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Google検索結果の最初のタイトルを取得
        search_result = soup.find('div', class_='yuRUbf')
        if search_result:
            title = search_result.find('h3')
            if title:
                return title.text.split('-')[0].strip()
        
        return f"企業 {code}"
    except Exception as e:
        st.warning(f"企業名の取得に失敗しました: {str(e)}")
        return f"企業 {code}"

def predict_stock_price(df):
    # Prophetのための日付フォーマット変更
    prophet_df = df.reset_index()[['Date', 'Close']]
    prophet_df.columns = ['ds', 'y']
    prophet_df['ds'] = prophet_df['ds'].dt.tz_localize(None)
    
    model = Prophet(daily_seasonality=True)
    model.fit(prophet_df)
    
    future_dates = model.make_future_dataframe(periods=10)
    forecast = model.predict(future_dates)
    
    return forecast.tail(10)[['ds', 'yhat']]

def create_stock_chart(df, forecast, company_name, code):
    plt.figure(figsize=(12, 6))
    sns.lineplot(data=df, x=df.index, y='Close', label='過去の株価')
    sns.lineplot(x=forecast['ds'], y=forecast['yhat'], label='予測株価', linestyle='--')
    plt.title(f'{company_name}（{code}）の株価チャート（過去2年間と今後10日間の予測）')
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