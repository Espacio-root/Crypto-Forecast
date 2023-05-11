import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import torch.nn.functional as F
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import plotly.graph_objects as go
import os
import pandas as pd
import numpy as np
from collect_data import Collect
import itertools

SYMBOLS = ['BTCUSDT', 'ETHUSDT', 'LTCUSDT', 'BCHUSDT']
INTERVAL = '1m'
START_TIME = '01-01-2023'
END_TIME = '30-01-2023'
RATIO = 'BTCUSDT_Close'
EPOCHS = 100
BATCH_SIZE = 64
WINDOWS_SIZE = 60
FUTURE_PREDICT = 3
TEST_SIZE = 0.2
DEVICE = 'cpu'
DIR = 'results'
BEST_VAL_LOSS = float('inf')

os.makedirs(DIR, exist_ok=True)


class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout=0.2):
        super(LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.bn = nn.BatchNorm1d(hidden_size)
        self.fc = nn.Linear(hidden_size, 128)
        self.fc2 = nn.Linear(128, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(
            0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(
            0), self.hidden_size).to(x.device)

        out, _ = self.lstm(x, (h0, c0))
        out = self.bn(out[:, -1, :])
        out = F.relu(self.fc(out))
        out = self.fc2(out)

        return out


class StockDataset(Dataset):
    def __init__(self, sequences, targets):
        self.sequences = sequences
        self.targets = targets

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.sequences[idx], self.targets[idx]


class StockDataLoader:
    def __init__(self, df, window_size, val_size, batch_size, shuffle=True):
        self.df = df.dropna()
        self.df = self.df[itertools.chain.from_iterable(
            [[f'{symbol}_Close', f'{symbol}_Volume'] for symbol in SYMBOLS])]
        self.window_size = window_size
        self.val_size = val_size
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.scaler = MinMaxScaler()
        self.target_idx = df.columns.get_loc(RATIO)
        self.normalize_num = df.iloc[0, self.target_idx]
        self.val_unix_start = df.index[len(df) - int(len(df) * val_size)]

    def preprocess_data(self):
        
        normalized_data = self.df.values
        normalized_data = (normalized_data - normalized_data[0]) / normalized_data[0]

        input_sequences = []
        output_sequences = []

        for i in range(len(normalized_data) - self.window_size - FUTURE_PREDICT):
            input_sequences.append(
                np.delete(normalized_data[i:i + self.window_size], self.target_idx, axis=1))
            output_sequences.append(
                normalized_data[i + self.window_size + FUTURE_PREDICT - 1, self.target_idx])

        input_sequences = torch.from_numpy(np.array(input_sequences)).float()
        output_sequences = torch.from_numpy(np.array(output_sequences)).float()

        X_train, X_test, y_train, y_test = train_test_split(
            input_sequences,
            output_sequences,
            test_size=self.val_size,
            shuffle=False
        )

        self.train_dataset = StockDataset(X_train, y_train)
        self.test_dataset = StockDataset(X_test, y_test)

        self.train_dataloader = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=self.shuffle
        )

        self.test_dataloader = DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False
        )

        return self.train_dataloader, self.test_dataloader

    def visualize(self, predicted, actual, loss, title):
        predicted = np.array(predicted).T[0]
        actual = np.array(actual)
        predicted = (predicted * self.normalize_num) + self.normalize_num  # Denormalize
        actual = (actual * self.normalize_num) + self.normalize_num  # Denormalize
        predicted, actual = np.concatenate((np.full(self.window_size + FUTURE_PREDICT, np.nan), predicted)), np.concatenate((actual, np.full(self.window_size + FUTURE_PREDICT, np.nan)))
        
        num_points = len(predicted)
        time_index = pd.date_range(start=pd.to_datetime(self.val_unix_start, unit='s'), periods=num_points, freq=f'{Utils.convert_to_seconds(INTERVAL)}s')
        data = {'Predicted': predicted, 'Actual': actual}
        df = pd.DataFrame(data, index=time_index)
        df = Utils.reduce_dataframe_size(df)

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df.index, y=df['Predicted'], name='Predicted', line=dict(color='red')))
        fig.add_trace(go.Scatter(x=df.index, y=df['Actual'], name='Actual', line=dict(color='blue')))
        
        fig.update_layout(
            title=f'{RATIO.split("_")[0]}   Loss: {loss:.4f}    {title}',
            xaxis_title='Time',
            yaxis_title='Value',
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )

        fig.show()
        
class Utils:
    
    def best_model():
        files = []
        
        for filename in os.listdir(DIR):
            if filename.endswith('.pth'):
                files.append(filename)
                
        best_loss = sorted(files, key=lambda x: float(x.split('--loss-')[1].split('.pth')[0]))
        return best_loss[0] if len(best_loss) > 0 else float('inf')
    
    def model_checkpoint(epoch, loss):
        best_loss = Utils.best_model()
        best_loss = best_loss.split('--loss-')[1].split('.pth')[0] if type(best_loss) == str else best_loss

        if float(loss) > float(best_loss):
            return

        checkpoint_path = os.path.join(DIR, f'epoch-{epoch}--loss-{loss:.2f}.pth')
        torch.save(model.state_dict(), checkpoint_path)
        print("Saved the best model")
        
    def convert_to_seconds(duration):
        unit = duration[-1]  # Extract the unit from the end of the string
        value = int(duration[:-1])  # Extract the numeric value from the string

        if unit == 's':
            return value  # Convert seconds to seconds
        elif unit == 'm':
            return value * 60  # Convert minutes to seconds
        elif unit == 'h':
            return value * 3600  # Convert hours to seconds
        elif unit == 'd':
            return value * 86400  # Convert days to seconds
        elif unit == 'w':
            return value * 604800  # Convert weeks to seconds
        else:
            raise ValueError('Invalid duration format')
        
    def plot_losses(train_losses, validation_losses):
        epochs = list(range(1, len(train_losses) + 1))

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=epochs, y=train_losses, mode='lines', name='Train Loss'))
        fig.add_trace(go.Scatter(x=epochs, y=validation_losses, mode='lines', name='Validation Loss'))

        fig.update_layout(xaxis_title='Epoch', yaxis_title='Loss', title='Training and Validation Losses')
        fig.show()
        
    def reduce_dataframe_size(df):
        num_rows = len(df)
        if num_rows <= 1000:
            return df  # No need to reduce size if already within the desired limit
        
        indices_to_keep = np.linspace(0, num_rows - 1, num=1000, dtype=int)
        reduced_df = df.iloc[indices_to_keep]
        return reduced_df

collect = Collect(symbol=SYMBOLS, interval=INTERVAL, start_time=START_TIME, end_time=END_TIME, unix=False)
df = collect.main()

loader = StockDataLoader(df, window_size=WINDOWS_SIZE,
                         val_size=TEST_SIZE, batch_size=BATCH_SIZE)
train_data_loader, test_data_loader = loader.preprocess_data()

model = LSTM(input_size=train_data_loader.dataset.sequences.shape[-1],
             hidden_size=128, num_layers=2, output_size=1, dropout=0.2)
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.MSELoss()

train_losses = []
validation_losses = []

def train():

    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0.0

        predicted_train = []
        actual_train = []
        for batch_inputs, batch_targets in train_data_loader:
            optimizer.zero_grad()

            batch_inputs = batch_inputs.to(DEVICE)
            batch_targets = batch_targets.to(DEVICE)

            outputs = model(batch_inputs)
            loss = criterion(outputs, batch_targets.unsqueeze(1))
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            
            if (epoch + 1) % 10 == 0:
                predicted_train.extend(outputs.detach().numpy())
                actual_train.extend([input[0][loader.target_idx] for input in batch_inputs.detach().numpy()])
                

        train_loss /= len(train_data_loader)
        train_losses.append(train_loss)
        
        model.eval()
        validation_loss = 0.0
        predicted_validation = []
        actual_validation = []

        with torch.no_grad():
            for batch_inputs, batch_targets in test_data_loader:
                batch_inputs = batch_inputs.to(DEVICE)
                batch_targets = batch_targets.to(DEVICE)

                outputs = model(batch_inputs)
                loss = criterion(outputs, batch_targets.unsqueeze(1))

                validation_loss += loss.item()
                
                if (epoch + 1) % 10 == 0:
                    predicted_validation.extend(outputs.detach().numpy())
                    actual_validation.extend([input[0][loader.target_idx] for input in batch_inputs.detach().numpy()])

        validation_loss /= len(test_data_loader)
        validation_losses.append(validation_loss)

        print(f"Epoch [{epoch+1}/{EPOCHS}], Train Loss: {train_loss:.6f}, Validation Loss: {validation_loss:.6f}")
        
        if (epoch + 1) % 10 == 0:       
            loader.visualize(predicted_train, actual_train, train_loss, 'Train')
            loader.visualize(predicted_validation, actual_validation, validation_loss, 'Validation')

        Utils.model_checkpoint(epoch, validation_loss)

    # Plotting loss curves
    Utils.plot_losses(train_losses, validation_losses)

def test():
    # Generate predictions
    checkpoint_path = os.path.join(DIR, Utils.best_model())
    model.load_state_dict(torch.load(checkpoint_path))
    model.eval()
    validation_loss = 0.0
    predicted_validation = []
    actual_validation = []
    
    with torch.no_grad():
        for batch_inputs, batch_targets in test_data_loader:
            batch_inputs = batch_inputs.to(DEVICE)
            batch_targets = batch_targets.to(DEVICE)

            outputs = model(batch_inputs)
            predicted_validation.extend(outputs.cpu().numpy())
            actual_validation.extend([input[0][loader.target_idx] for input in batch_inputs.cpu().numpy()])
            loss = criterion(outputs, batch_targets.unsqueeze(1))

            validation_loss += loss.item()

    validation_loss /= len(test_data_loader)
    validation_losses.append(validation_loss)

    print(f"Test Loss: {validation_loss:.6f}")
    
    loader.visualize(predicted_validation, actual_validation, validation_loss)
    
if __name__ == '__main__':
    train()
    test()
