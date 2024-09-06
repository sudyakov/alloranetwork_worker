import torch
import torch.nn as nn
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import requests
from flask import Flask, Response, json
import logging
from datetime import datetime

app = Flask(__name__)

# Настройка базового логирования без JSON форматирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("main")

# Определение обновленной модели с правильной архитектурой
class EnhancedBiLSTMModel(nn.Module):
    def __init__(self, input_size, hidden_layer_size, output_size, num_layers, dropout):
        super(EnhancedBiLSTMModel, self).__init__()
        self.hidden_layer_size = hidden_layer_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_layer_size, num_layers=num_layers, dropout=dropout, batch_first=True, bidirectional=True)
        self.linear = nn.Linear(hidden_layer_size * 2, output_size * 2)  # *2 для двунаправленности и 2 временных рамок

    def forward(self, input_seq):
        h_0 = torch.zeros(self.num_layers * 2, input_seq.size(0), self.hidden_layer_size)
        c_0 = torch.zeros(self.num_layers * 2, input_seq.size(0), self.hidden_layer_size)
        
        lstm_out, _ = self.lstm(input_seq, (h_0, c_0))
        return self.linear(lstm_out[:, -1])

# Создаем словарь для хранения моделей
models = {}

# Инициализируем модели для каждой валюты
symbol_map = {
    'ETH': 'ETHUSDT',
    'BTC': 'BTCUSDT',
    'BNB': 'BNBUSDT',
    'SOL': 'SOLUSDT',
    'ARB': 'ARBUSDT'
}

for token in symbol_map:
    model = EnhancedBiLSTMModel(input_size=1, hidden_layer_size=115, output_size=1, num_layers=2, dropout=0.3)
    model.load_state_dict(torch.load(f"enhanced_bilstm_model_{token}.pth", weights_only=True))
    model.eval()
    models[token] = model

# Функция для получения исторических данных с Binance
def get_binance_url(symbol="ETHUSDT", interval="1m", limit=1000):
    return f"https://api.binance.com/api/v3/klines?symbol={symbol}&interval={interval}&limit={limit}"

@app.route("/inference/<string:token>")
def get_inference(token):
    token = token.upper()
    if token not in symbol_map:
        return Response(json.dumps({"error": "Неподдерживаемый токен"}), status=400, mimetype='application/json')

    symbol = symbol_map[token]
    model = models[token]

    url = get_binance_url(symbol=symbol)
    response = requests.get(url)
    if response.status_code != 200:
        return Response(json.dumps({"error": "Не удалось получить данные от API Binance", "details": response.text}),
                        status=response.status_code,
                        mimetype='application/json')

    data = response.json()
    df = pd.DataFrame(data, columns=[
        "open_time", "open", "high", "low", "close", "volume",
        "close_time", "quote_asset_volume", "number_of_trades",
        "taker_buy_base_asset_volume", "taker_buy_quote_asset_volume", "ignore"
    ])
    df["close_time"] = pd.to_datetime(df["close_time"], unit='ms')
    df = df[["close_time", "close"]]
    df.columns = ["date", "price"]
    df["price"] = df["price"].astype(float)

    df = df.tail(10 if symbol in ['BTCUSDT', 'SOLUSDT'] else 20)

    current_price = df.iloc[-1]["price"]
    current_time = df.iloc[-1]["date"]
    logger.info(f"Текущая цена: {current_price} на {current_time}")

    scaler = MinMaxScaler(feature_range=(-1, 1))
    scaled_data = scaler.fit_transform(df['price'].values.reshape(-1, 1))

    seq = torch.FloatTensor(scaled_data).view(1, -1, 1)

    with torch.no_grad():
        y_pred = model(seq)

    predicted_prices = scaler.inverse_transform(y_pred.numpy().reshape(-1, 1))

    predicted_price = round(float(predicted_prices[0 if symbol in ['BTCUSDT', 'SOLUSDT'] else 1][0]), 2)

    logger.info(f"Прогноз: {predicted_price}")

    return Response(json.dumps(predicted_price), status=200, mimetype='application/json')

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8000)