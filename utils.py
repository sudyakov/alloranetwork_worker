from sklearn.preprocessing import MinMaxScaler
import torch

def prepare_data(df, seq_length):
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(df[['open', 'high', 'low', 'close', 'volume']].values)
    
    x, y = [], []
    for i in range(len(scaled_data) - seq_length):
        x.append(scaled_data[i:(i + seq_length)])
        y.append(scaled_data[i + seq_length, 3])
    
    return torch.FloatTensor(x), torch.FloatTensor(y), scaler
