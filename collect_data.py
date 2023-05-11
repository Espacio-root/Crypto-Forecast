import requests
import os
import pandas as pd
from datetime import datetime
import time
import re


class Collect:

    def __init__(self, symbol: str or list, interval: str or list, start_time: str or list, end_time: str or list, unix: bool = True) -> None:
        self.symbol = symbol if type(symbol) == list else [symbol]
        self.interval = interval if type(interval) == list else [
            interval] * len(self.symbol)
        self.start_time = start_time if type(start_time) == list else [
            start_time] * len(self.symbol)
        self.end_time = end_time if type(end_time) == list else [
            end_time] * len(self.symbol)
        self.unix = unix
        self.count = 1
        self.iter = 1

        self.df = pd.DataFrame()
        
        

    def get_data(self, symbol, interval, start_time, end_time) -> pd.DataFrame:

        def datetime_to_unix_ms(date_string):
            if type(date_string) == int:
                return date_string

            return int(datetime.strptime(date_string, '%d-%m-%Y').timestamp()) * 1000
        

        def fetch(symbol, interval, start_time, end_time):
            start_time = datetime_to_unix_ms(start_time) if re.match(
                r'(\d{2})-(\d{2})-(\d{4})', str(start_time)) else start_time
            end_time = datetime_to_unix_ms(end_time) if re.match(
                r'(\d{2})-(\d{2})-(\d{4})', str(end_time)) else end_time

            response = requests.get(
                f'https://api.binance.com/api/v3/klines?symbol={symbol}&interval={interval}&limit=1000&startTime={start_time}&endTime={end_time}')

            if response.status_code == 200:
                data = response.json()

                if data and data[-1][0] < end_time:
                    time.sleep(1)
                    print(f'Fetched {self.count * 1000} datapoints')
                    self.count += 1
                    
                    tmp_data = fetch(symbol, interval, data[-1][0], end_time)
                    data.extend(tmp_data)

            else:
                data = []
                print(f'The program was only able to fetch {len(data)} datapoints from {start_time} to {end_time}')

            return data

        if f'{symbol}_{interval}_{start_time}_{end_time}.csv' in os.listdir('./cache'):
            print(f'[{self.iter}/{len(self.symbol)}] Cache for {symbol}_{interval}_{start_time}_{end_time}.csv found')
            return self.modify(pd.read_csv(f'./cache/{symbol}_{interval}_{start_time}_{end_time}.csv', index_col=0), symbol)

        data = fetch(symbol, interval, start_time, end_time)

        df = pd.DataFrame(data, columns=['_'.join(x.split(' ')) for x in ['Time', 'Open', 'High', 'Low', 'Close', 'Volume', 'Close time',
                          'Quote asset volume', 'Number of trades', 'Taker buy base asset volume', 'Taker buy quote asset volume', 'Ignore']])
        df.set_index('Time', inplace=True)

        self.store(symbol, interval, start_time, end_time, df)
        return self.modify(df, symbol)
    

    def modify(self, df, symbol) -> pd.DataFrame:
        if not self.unix:
            df.index = pd.to_datetime(df.index, unit='ms')
        return df.rename(columns={x: f'{symbol}_{x}' for x in df.columns})
    
    
    def store(self, symbol, interval, start_time, end_time, df) -> None:
        df.to_csv(f'./cache/{symbol}_{interval}_{start_time}_{end_time}.csv')
        print(f'Cache for {symbol}_{interval}_{start_time}_{end_time}.csv stored')
        
        
    def main(self):
        for i in range(len(self.symbol)):
            self.iter = i + 1
            df = self.get_data(
                self.symbol[i], self.interval[i], self.start_time[i], self.end_time[i])

            if len(self.df) == 0:
                self.df = df
            else:
                self.df = pd.concat([self.df, df], ignore_index=False, axis=1)
                
            self.count = 1

        return self.df


# Example
if __name__ == '__main__':
    collect = Collect(symbol=['BTCUSDT', 'ETHUSDT'], interval='1m',
                      start_time='01-01-2023', end_time='03-01-2023', unix=False)
    collect.main()
