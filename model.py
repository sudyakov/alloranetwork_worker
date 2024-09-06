import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import logging
from download_data import get_binance_data

def update_data(df, scaler, symbols):
    new_data = {symbol: get_binance_data(symbol) for symbol in symbols}

class GraphAttentionLayer(nn.Module):
    def __init__(self, in_features, out_features, dropout, alpha, concat=True):
        super(GraphAttentionLayer, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat

        self.W = nn.Parameter(torch.empty(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.empty(size=(2*out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, h, adj):
        Wh = torch.mm(h, self.W)
        e = self._prepare_attentional_mechanism_input(Wh)

        zero_vec = -9e15*torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        attention = F.softmax(attention, dim=1)
        attention = F.dropout(attention, self.dropout, training=self.training)
        h_prime = torch.matmul(attention, Wh)
        return F.elu(h_prime) if self.concat else h_prime

    def _prepare_attentional_mechanism_input(self, Wh):
        Wh1 = torch.matmul(Wh, self.a[:self.out_features, :])
        Wh2 = torch.matmul(Wh, self.a[self.out_features:, :])
        e = Wh1 + Wh2.T
        return self.leakyrelu(e)

class EnhancedBiLSTMModel(nn.Module):
    def __init__(self, input_size, hidden_layer_size=1024, output_size=1, num_layers=4, dropout=0.3, num_tokens=5):
        super(EnhancedBiLSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_layer_size, num_layers=num_layers, dropout=dropout, batch_first=True, bidirectional=True)
        self.attention = nn.MultiheadAttention(hidden_layer_size * 2, num_heads=8)
        self.gat1 = GraphAttentionLayer(hidden_layer_size * 2, hidden_layer_size, dropout=0.6, alpha=0.2)
        self.gat2 = GraphAttentionLayer(hidden_layer_size, hidden_layer_size // 2, dropout=0.6, alpha=0.2)
        self.fc1 = nn.Linear(hidden_layer_size // 2 * num_tokens, hidden_layer_size)
        self.fc2 = nn.Linear(hidden_layer_size, hidden_layer_size // 2)
        self.fc3 = nn.Linear(hidden_layer_size // 2, output_size)
        self.relu = nn.ReLU()

    def forward(self, input_seq, adj_matrix):
        batch_size, num_tokens, seq_len, features = input_seq.shape
        input_seq = input_seq.view(batch_size * num_tokens, seq_len, features)
        
        lstm_out, _ = self.lstm(input_seq)
        attn_output, _ = self.attention(lstm_out, lstm_out, lstm_out)
        
        gat_input = attn_output[:, -1].view(batch_size, num_tokens, -1)
        gat_out1 = self.gat1(gat_input, adj_matrix)
        gat_out2 = self.gat2(gat_out1, adj_matrix)
        
        x = gat_out2.view(batch_size, -1)
        x = self.relu(self.fc1(x))
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




