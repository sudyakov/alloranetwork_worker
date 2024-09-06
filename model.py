import torch
import torch.nn as nn

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

class EarlyStopping:
    def __init__(self, patience=10, min_delta=0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif self.best_loss - val_loss > self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

def evaluate_model(model, test_loader, scaler):
    model.eval()
    predictions = []
    actual_values = []
    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            outputs = model(batch_x)
            predictions.extend(outputs.cpu().numpy())
            actual_values.extend(batch_y.cpu().numpy())
    
    predictions = scaler.inverse_transform(np.array(predictions).reshape(-1, 1))
    actual_values = scaler.inverse_transform(np.array(actual_values).reshape(-1, 1))
    
    mse = mean_squared_error(actual_values, predictions)
    mae = mean_absolute_error(actual_values, predictions)
    r2 = r2_score(actual_values, predictions)
    
    logging.info(f"Mean Squared Error: {mse}")
    logging.info(f"Mean Absolute Error: {mae}")
    logging.info(f"R2 Score: {r2}")

import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import logging

