import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import requests
import json
import time
import datetime
import os
import torch.cuda as cuda

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


def get_binance_data(symbol="ETHUSDT", interval="1m", limit=1000):
    url = f"https://api.binance.com/api/v3/klines?symbol={symbol}&interval={interval}&limit={limit}"
    response = requests.get(url)
    data = response.json()
    df = pd.DataFrame(data, columns=["open_time", "open", "high", "low", "close", "volume",
                                     "close_time", "quote_asset_volume", "number_of_trades",
                                     "taker_buy_base_asset_volume", "taker_buy_quote_asset_volume", "ignore"])
    df["close_time"] = pd.to_datetime(df["close_time"], unit='ms')
    df = df[["close_time", "close"]]
    df.columns = ["date", "price"]
    df["price"] = df["price"].astype(float)
    return df


def prepare_data(df, seq_length):
    scaler = MinMaxScaler(feature_range=(-1, 1))
    scaled_data = scaler.fit_transform(df['price'].values.reshape(-1, 1))
    
    x = [scaled_data[i:(i + seq_length)] for i in range(len(scaled_data) - seq_length)]
    y = [scaled_data[i + seq_length] for i in range(len(scaled_data) - seq_length)]
    
    return torch.FloatTensor(np.array(x)).to(device), torch.FloatTensor(np.array(y)).to(device), scaler


def update_data(df, scaler):
    new_data = get_binance_data()
    df = pd.concat([df, new_data]).drop_duplicates().reset_index(drop=True).tail(1000)
    scaled_data = scaler.transform(df['price'].values.reshape(-1, 1))
    return df, scaled_data


def train_model(model, train_data, train_labels, df, scaler, epochs=100, lr=0.001, criterion=None, accumulation_steps=4):
    if criterion is None:
        criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    for epoch in range(epochs):
        for i in range(0, len(train_data), accumulation_steps):
            optimizer.zero_grad()
            for j in range(accumulation_steps):
                if i + j < len(train_data):
                    outputs = model(train_data[i+j:i+j+1])
                    loss = criterion(outputs, train_labels[i+j:i+j+1])
                    loss = loss / accumulation_steps
                    loss.backward()
            optimizer.step()
        
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')
    
    df, scaled_data = update_data(df, scaler)
    x, y, _ = prepare_data(df, len(train_data[0]))
    return model, df, scaler, x, y


def print_gpu_utilization():
    print(f"GPU utilization: {torch.cuda.memory_allocated() / torch.cuda.max_memory_allocated() * 100:.2f}%")
    print(f"GPU memory allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
    print(f"GPU memory cached: {torch.cuda.memory_reserved() / 1e9:.2f} GB")

def main():
    # Информация об устройстве
    print(f"Используемое устройство: {device}")
    if device.type == 'cuda':
        print(f"Модель GPU: {torch.cuda.get_device_name(0)}")
        print(f"Доступная память GPU: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} ГБ")
        print(f"Версия CUDA: {torch.version.cuda}")

    model = EnhancedBiLSTMModel(input_size=1, hidden_layer_size=512, output_size=1, num_layers=2, dropout=0.3).to(device)
    criterion = nn.MSELoss()
    df = get_binance_data()
    seq_length = 60
    x, y, scaler = prepare_data(df, seq_length)
    
    # Вывод параметров модели и обучения
    print(f"Архитектура модели: {model}")
    print(f"Количество параметров модели: {sum(p.numel() for p in model.parameters())}")
    print(f"Функция потерь: {criterion}")
    print(f"Длина последовательности: {seq_length}")
    print(f"Размер обучающего набора: {len(x)}")
    
    start_time = datetime.datetime.now()
    end_time = start_time + datetime.timedelta(hours=4)
    
    iteration_count = 0
    
    while datetime.datetime.now() < end_time:
        model, df, scaler, x, y = train_model(model, x, y, df, scaler, epochs=100, criterion=criterion)
        
        iteration_count += 1
        if iteration_count % 5 == 0:
            print_gpu_utilization()
            model_path = "enhanced_bilstm_model_latest.pth"
            torch.save(model.state_dict(), model_path)
            
            # Получаем размер файла модели
            model_size = os.path.getsize(model_path) / (1024 * 1024)  # в МБ
            
            # Получаем количество параметров модели
            total_params = sum(p.numel() for p in model.parameters())
            
            # Получаем последнее значение функции потерь
            last_loss = criterion(model(x), y).item()
            
            print(f"Модель сохранена после {iteration_count} итераций.")
            print(f"Размер файла модели: {model_size:.2f} МБ")
            print(f"Количество параметров модели: {total_params}")
            print(f"Последнее значение функции потерь: {last_loss:.4f}")
            print(f"Текущее время: {datetime.datetime.now()}")
            print(f"Осталось времени: {end_time - datetime.datetime.now()}")
            print("-" * 50)
        
        time.sleep(60)  # Подождем минуту перед следующей итерацией
    
    torch.save(model.state_dict(), "enhanced_bilstm_model_final.pth")
    print("Финальная модель успешно обучена и сохранена.")

    # Вывод первой и последней строки данных
    if len(df) > 1:
        print("Первая строка данных:")
        print(df.iloc[0])
        print("\nПоследняя строка данных:")
        print(df.iloc[-1])
    elif len(df) == 1:
        print("Единственная строка данных:")
        print(df.iloc[0])
    else:
        print("Получены пустые данные")


if __name__ == "__main__":
    main()

