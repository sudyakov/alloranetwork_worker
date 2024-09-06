from sklearn.preprocessing import MinMaxScaler
import torch
import numpy as np

def prepare_data(df, seq_length, symbols):
    if df.empty:
        raise ValueError("Input DataFrame is empty")
    
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(df[['open', 'high', 'low', 'close', 'volume']].values)
    
    close_idx = 3  # Индекс столбца 'close' в scaled_data
    x, y = [], []
    for i in range(len(scaled_data) - seq_length):
        x.append(scaled_data[i:(i + seq_length)])
        y.append(scaled_data[i + seq_length, close_idx])
    
    return torch.FloatTensor(np.array(x)), torch.FloatTensor(np.array(y)), scaler