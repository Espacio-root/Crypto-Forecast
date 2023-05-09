import requests
import os
import pandas as pd


class Collect:

    def __init__(self, symbol: str or list, interval: str or list, limit: int or list, unix: bool = True) -> None:
        self.symbol = symbol if type(symbol) == list else [symbol]
        self.interval = interval if type(interval) == list else [
            interval] * len(self.symbol)
        self.limit = limit if type(limit) == list else [
            limit] * len(self.symbol)
        self.unix = unix

        self.df = pd.DataFrame()

    def get_data(self, symbol, interval, limit) -> pd.DataFrame:

        if f'{symbol}_{interval}_{limit}.csv' in os.listdir('./cache'):
            print(f'Cache for {symbol}_{interval}_{limit}.csv found')
            return self.modify(pd.read_csv(f'./cache/{symbol}_{interval}_{limit}.csv', index_col=0), symbol)

        response = requests.get(
            f'https://api.binance.com/api/v3/klines?symbol={symbol}&interval={interval}&limit={limit}')

        if response.status_code == 200:
            data = response.json()

            df = pd.DataFrame(data, columns=['_'.join(x.split(' ')) for x in ['Time', 'Open', 'High', 'Low', 'Close', 'Volume', 'Close time',
                              'Quote asset volume', 'Number of trades', 'Taker buy base asset volume', 'Taker buy quote asset volume', 'Ignore']])
            df.set_index('Time', inplace=True)

            self.store(symbol, interval, limit, df)

            return self.modify(df, symbol)

        else:
            print(f'Request failed with status code {response.status_code}')

    def modify(self, df, symbol) -> pd.DataFrame:
        if not self.unix:
            df.index = pd.to_datetime(df.index, unit='ms')
        return df.rename(columns={x: f'{symbol}_{x}' for x in df.columns})

    def store(self, symbol, interval, limit, df) -> None:
        df.to_csv(f'./cache/{symbol}_{interval}_{limit}.csv')
        print(f'{symbol}_{interval}_{limit}.csv stored in ./cache/{symbol}_{interval}_{limit}.csv')

    def main(self):
        for i in range(len(self.symbol)):
            df = self.get_data(self.symbol[i], self.interval[i], self.limit[i])

            if len(self.df) == 0:
                self.df = df
            else:
                self.df = pd.concat([self.df, df], ignore_index=False, axis=1)

        return self.df


# Example
if __name__ == '__main__':
    collect = Collect(symbol=['BTCUSDT', 'ETHUSDT'], interval='1m', limit=10, unix=False)
    print(collect.main())
