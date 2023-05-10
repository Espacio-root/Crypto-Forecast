import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from collect_data import Collect

collect = Collect(symbol=['BTCUSDT', 'ETHUSDT', 'LTCUSDT', 'BCHUSDT'], interval='1m', limit=100000, unix=False)

class LitLSTM(pl.LightningModule):
    
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout=0.2):
        super(LitLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        
        return out
    
    def training_step(self, batch, idx):
        x, y = batch
        y_hat = self(x)
        loss = F.mse_loss(y_hat, y)
        self.log('train_loss', loss)
        
        return loss

    def validation_step(self, batch, idx):
        x, y = batch
        y_hat = self(x)
        loss = F.mse_loss(y_hat, y)
        self.log('val_loss', loss)
        
        return loss

def prepare_data():
    df = collect.main()
    print(len(df))
    df.dropna(inplace=True)
    print(len(df))