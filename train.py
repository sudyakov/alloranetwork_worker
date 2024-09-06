import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import json
import logging

# Проверка доступности CUDA
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Загрузка конфигурации
with open('config.json') as config_file:
    config = json.load(config_file)

class EnhancedBiLSTMModel(nn.Module):
    def __init__(self, input_size, hidden_layer_size=1024, output_size=1, num_layers=4, dropout=0.3):
        super(EnhancedBiLSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_layer_size, num_layers=num_layers, dropout=dropout, batch_first=True, bidirectional=True)
        self.attention = nn.MultiheadAttention(hidden_layer_size * 2, num_heads=8)
        self.fc1 = nn.Linear(hidden_layer_size * 2, hidden_layer_size)
        self.fc2 = nn.Linear(hidden_layer_size, hidden_layer_size // 2)
        self.fc3 = nn.Linear(hidden_layer_size // 2, output_size)
        self.relu = nn.ReLU()

    def forward(self, input_seq):
        lstm_out, _ = self.lstm(input_seq)
        attn_output, _ = self.attention(lstm_out, lstm_out, lstm_out)
        x = self.relu(self.fc1(attn_output[:, -1]))
        x = self.relu(self.fc2(x))
        return self.fc3(x)

def load_combined_dataset():
    return pd.read_csv('combined_dataset.csv')

def get_realtime_price(symbols):
    return pd.DataFrame({
        'symbol': symbols,
        'open': np.random.rand(len(symbols)),
        'high': np.random.rand(len(symbols)),
        'low': np.random.rand(len(symbols)),
        'close': np.random.rand(len(symbols)),
        'volume': np.random.rand(len(symbols))
    })

def prepare_data(df, seq_length):
    scaler = MinMaxScaler(feature_range=(-1, 1))
    scaled_data = scaler.fit_transform(df[['open', 'high', 'low', 'close', 'volume']].values)
    
    x, y = [], []
    for i in range(len(scaled_data) - seq_length):
        x.append(scaled_data[i:(i + seq_length)])
        y.append(scaled_data[i + seq_length, 3])  # Используем 'close' как целевую переменную
    
    return torch.FloatTensor(np.array(x)).to(device), torch.FloatTensor(np.array(y)).to(device), scaler

def train_model(model, criterion, optimizer, train_loader, val_loader, num_epochs, symbols, df, scaler, seq_length):
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y.unsqueeze(1))
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                outputs = model(batch_x)
                loss = criterion(outputs, batch_y.unsqueeze(1))
                val_loss += loss.item()

        print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss/len(train_loader):.4f}, Val Loss: {val_loss/len(val_loader):.4f}')

    return model, df, scaler, train_loader.dataset.tensors[0], train_loader.dataset.tensors[1]

def print_gpu_utilization():
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated()
        max_allocated = torch.cuda.max_memory_allocated()
        utilization = allocated / max_allocated * 100 if max_allocated > 0 else 0
        print(f"GPU utilization: {utilization:.2f}%")
        print(f"GPU memory allocated: {allocated / 1e9:.2f} GB")
        print(f"GPU memory cached: {torch.cuda.memory_reserved() / 1e9:.2f} GB")
    else:
        print("CUDA is not available. Using CPU.")

def main():
    logging.info("Начало обучения")
    print_gpu_utilization()

    df = load_combined_dataset()
    symbols = df['symbol'].unique().tolist()
    seq_length = 60
    
    x, y, scaler = prepare_data(df, seq_length)

    model = EnhancedBiLSTMModel(input_size=5, hidden_layer_size=512, output_size=1, num_layers=2, dropout=0.3).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Разделение данных на обучающую и валидационную выборки
    train_size = int(0.8 * len(x))
    train_x, val_x = x[:train_size], x[train_size:]
    train_y, val_y = y[:train_size], y[train_size:]

    train_dataset = torch.utils.data.TensorDataset(train_x, train_y)
    val_dataset = torch.utils.data.TensorDataset(val_x, val_y)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=32, shuffle=False)

    # Обучение модели
    num_epochs = 100
    model, df, scaler, x, y = train_model(model, criterion, optimizer, train_loader, val_loader, num_epochs, symbols, df, scaler, seq_length)

    # Сохранение модели
    torch.save(model.state_dict(), "enhanced_bilstm_model_all_symbols.pth")
    print("Модель обучена и сохранена для всех символов.")

if __name__ == "__main__":
    main()
