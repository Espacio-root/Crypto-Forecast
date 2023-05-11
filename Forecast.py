import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import torch.nn.functional as F
import pytorch_lightning as pl
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from collect_data import Collect
import itertools
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor

SYMBOLS = ['BTCUSDT', 'ETHUSDT', 'LTCUSDT', 'BCHUSDT']
RATIO = 'BTCUSDT_Close'
EPOCHS = 100
BATCH_SIZE = 64
FUTURE_PREDICT = 3
PATH = './results'


class LitLSTM(pl.LightningModule):

    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout=0.2):
        super(LitLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # (batch, seq, feature) (64, 60, 7), (64, 60, 128)
        self.lstm = nn.LSTM(input_size, hidden_size,
                            num_layers, batch_first=True, dropout=dropout)
        # (64, 60, 128), (64, 60, 1)
        self.fc = nn.Linear(hidden_size, output_size)

        self.validation_predictions = []
        self.training_avg_loss = []
        self.validation_avg_loss = []

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(
            0), self.hidden_size)  # (2, 64, 128)
        c0 = torch.zeros(self.num_layers, x.size(
            0), self.hidden_size)  # (2, 64, 128)

        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])  # (64, 60, 1) -> (64, 1)

        return out

    def training_step(self, batch, idx):
        x, y = batch
        y = y.view(-1, 1)
        y_hat = self(x)
        loss = F.mse_loss(y_hat, y)
        self.training_avg_loss.append(loss)
        self.log('train_loss', loss, on_step=True,
                 on_epoch=True, prog_bar=True)

        return loss

    def validation_step(self, batch, idx):
        x, y = batch
        y_hat = self(x)
        loss = F.mse_loss(y_hat, y)
        self.validation_avg_loss.append(loss)
        self.validation_predictions.append(y_hat.view(-1).numpy())
        self.log('validation_loss', loss, on_step=True,
                 on_epoch=True, prog_bar=True)

        return loss

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=0.001)
        lr_scheduler = {'scheduler': optim.lr_scheduler.ExponentialLR(
            optimizer, gamma=0.95), 'name': 'expo_lr'}

        return [optimizer], [lr_scheduler]

    def on_training_epoch_end(self):
        avg_loss = torch.stack(self.training_avg_loss).mean()
        self.log('avg_train_loss', avg_loss, on_epoch=True, prog_bar=True)
        self.training_avg_loss.clear()

    def on_validation_epoch_end(self):
        avg_loss = torch.stack(self.validation_avg_loss).mean()
        self.log('avg_validation_loss', avg_loss, on_epoch=True, prog_bar=True)
        self.validation_avg_loss.clear()

        loader.visualize(self.validation_predictions,
                         'Predictions vs. Actuals')


class StockDataset(Dataset):
    def __init__(self, sequences, targets):
        self.sequences = sequences
        self.targets = targets

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.sequences[idx], self.targets[idx]


class StockDataLoader:
    def __init__(self, df, window_size, test_size, val_size, batch_size, shuffle=True):
        self.df = df.dropna()
        self.df = self.df[itertools.chain.from_iterable(
            [[f'{symbol}_Close', f'{symbol}_Volume'] for symbol in SYMBOLS])]  # 41k x 8
        self.window_size = window_size
        self.test_size = test_size
        self.val_size = val_size
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.scaler = MinMaxScaler()
        self.target_idx = df.columns.get_loc(RATIO)

    def preprocess_data(self):
        # Data Normalization
        normalized_data = self.df.values
        normalized_data = (normalized_data -
                           normalized_data[0]) / normalized_data[0]

        # Sliding Window
        input_sequences = []
        output_sequences = []

        for i in range(len(normalized_data) - self.window_size - FUTURE_PREDICT):
            input_sequences.append(
                np.delete(normalized_data[i:i + self.window_size], self.target_idx, axis=1))
            output_sequences.append(
                normalized_data[i + self.window_size + FUTURE_PREDICT - 1, self.target_idx])

        # Convert to NumPy arrays
        input_sequences = torch.from_numpy(np.array(input_sequences)).float()
        output_sequences = torch.from_numpy(np.array(output_sequences)).float()

        # Train-Test Split
        X_train, X_test, y_train, y_test = train_test_split(
            input_sequences,
            output_sequences,
            test_size=self.test_size,
            shuffle=False
        )  # 37k x 60 x 7  |   4k x 60 x 7  |   37k   |   4k

        # Create Dataset and DataLoader
        self.train_dataset = StockDataset(X_train, y_train)
        self.test_dataset = StockDataset(X_test, y_test)

        self.train_dataloader = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
        )

        self.test_dataloader = DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
        )

        return self.train_dataloader, self.test_dataloader

    def visualize(self, predicted, title):
        predicted = np.array(predicted)
        print('testa', self.test_dataset)
        predicted = (predicted + 1) * self.test_dataset.targets.numpy()  # Denormalize
        actual = [dp[0] for dp in self.test_dataset.sequences.numpy()]

        plt.figure(figsize=(15, 5))
        plt.title(title)
        plt.plot(predicted, label='Predicted')
        plt.plot(actual, label='Actual')
        plt.legend()
        plt.show()


collect = Collect(symbol=SYMBOLS, interval='1m',
                  start_time='01-01-2023', end_time='30-01-2023', unix=False)
df = collect.main()

loader = StockDataLoader(df, window_size=60, test_size=0.1,
                         val_size=0.1, batch_size=BATCH_SIZE)
train_data_loader, test_data_loader = loader.preprocess_data()

first_batch, label = next(iter(train_data_loader))

lr_logger = LearningRateMonitor(logging_interval='step')
checkpoint_callback = ModelCheckpoint(
    monitor='validation_loss', filename='mnist-{epoch:02d}-{val_loss:.2f}', dirpath=PATH)

model = LitLSTM(
    input_size=first_batch.shape[-1], hidden_size=128, num_layers=2, output_size=1, dropout=0.2)
trainer = pl.Trainer(max_epochs=EPOCHS, callbacks=[
                     lr_logger, checkpoint_callback], default_root_dir=PATH)

trainer.fit(model=model, train_dataloaders=train_data_loader,
            val_dataloaders=test_data_loader)
