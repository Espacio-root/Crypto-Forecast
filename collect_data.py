import requests
import pandas as pd

    
class Collect:
    
    def __init__(self, symbol: str or list, interval: str or list, limit: int or list) -> None:
        self.symbol = symbol if type(symbol) == list else [symbol]
        self.interval = interval if type(interval) == list else [interval]
        self.limit = limit if type(limit) == list else [limit]
        
        self.df = pd.DataFrame()
        
    def get_data(self, symbol, interval, limit) -> dict:
        
        if f'{symbol}_{interval}_{limit}.csv' in os.listdir('./cache'):
            return pd.read_csv(f'./cache/{symbol}_{interval}_{limit}.csv', index_col=0)
        
        response = requests.get(f'https://api.binance.com/api/v3/klines?symbol={symbol}&interval={interval}&limit={limit}')
        
        if response.status_code == 200:
            data = response.json()
            
            df = pd.DataFrame(data, columns=['Time', 'Open', 'High', 'Low', 'Close', 'Volume', 'Close time', 'Quote asset volume', 'Number of trades', 'Taker buy base asset volume', 'Taker buy quote asset volume', 'Ignore'])
            df.set_index('Time', inplace=True)
            
            self.store(symbol, interval, limit, df)
    
            return df
            
        else:
            print(f'Request failed with status code {response.status_code}')
            
    def store(self, symbol, interval, limit, df) -> None:
        df.to_csv(f'./cache/{symbol}_{interval}_{limit}.csv')
        
    def main(self):
        for i in len(self.symbol):
            self.df.append(self.get_data(self.symbol[i], self.interval[i], self.limit[i]))
            
        return self.df
    

if __name__ == '__main__':
    collect = Collect(symbol='BTCUSDT', interval='1m', limit=10)
    collect.get_data()