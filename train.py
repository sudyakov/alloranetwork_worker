import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda import amp
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import RandomizedSearchCV
from skorch import NeuralNetRegressor
from scipy.stats import uniform, randint
import matplotlib.pyplot as plt
import os
import logging
from datetime import datetime
from gpu_util import print_gpu_utilization, get_device_info
from downloads_data import load_data, update_data

from model import EnhancedBiLSTMModel, EarlyStopping, evaluate_model

# Настройка логирования
logging.basicConfig(filename=f'training_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log', level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Используется устройство: {device} - OK.")

def prepare_data(df, seq_length):
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(df[['open', 'high', 'low', 'close', 'volume']].values)
    
    x, y = [], []
    for i in range(len(scaled_data) - seq_length):
        x.append(scaled_data[i:(i + seq_length)])
        y.append(scaled_data[i + seq_length, 3])
    
    return torch.FloatTensor(x).to(device), torch.FloatTensor(y).to(device), scaler

def optimize_hyperparameters(X, y):
    net = NeuralNetRegressor(
        EnhancedBiLSTMModel,
        max_epochs=100,
        lr=0.01,
        device=device,
        optimizer=optim.Adam,
        criterion=nn.MSELoss,
    )

    param_dist = {
        'lr': uniform(0.0001, 0.1),
        'module__hidden_layer_size': randint(32, 256),
        'module__num_layers': randint(1, 4),
        'optimizer__weight_decay': uniform(0, 0.1),
    }

    search = RandomizedSearchCV(
        net,
        param_dist,
        n_iter=20,
        cv=3,
        scoring='neg_mean_squared_error',
        n_jobs=-1,
    )

    search.fit(X, y)
    return search.best_params_

def train_model(model, criterion, optimizer, train_loader, val_loader, num_epochs, symbols, df, scaler):
    scaler = amp.GradScaler()
    best_loss = float('inf')
    train_losses = []
    val_losses = []
    early_stopping = EarlyStopping(patience=10, min_delta=0.001)
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            
            with amp.autocast():
                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            total_loss += loss.item()
        
        avg_train_loss = total_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)
                val_loss += loss.item()
        
        avg_val_loss = val_loss / len(val_loader)
        val_losses.append(avg_val_loss)
        
        logging.info(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}')
        
        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            torch.save(model.state_dict(), "best_model.pth")
            logging.info(f"Лучшая модель сохранена с валидационной потерей: {best_loss:.4f}")
        
        early_stopping(avg_val_loss)
        if early_stopping.early_stop:
            print("Раннее остановка!")
            break
        
        if (epoch + 1) % 10 == 0:
            df, x_new, y_new = update_data(df, scaler, symbols)
            
            train_loader.dataset.tensors = (
                torch.cat([train_loader.dataset.tensors[0], x_new.repeat(5, 1, 1)]),
                torch.cat([train_loader.dataset.tensors[1], y_new.repeat(5)])
            )
            
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_val_loss,
            }, f"checkpoint_epoch_{epoch+1}.pth")
            
            print(f"Промежуточные результаты сохранены для эпохи {epoch+1}")
    
    # plot_training_results(train_losses, val_losses)

def main():
    logging.info("Начало обучения")
    print(get_device_info())
    
    df = load_data()
    symbols = df['symbol'].unique().tolist()
    seq_length = 60
    x, y, scaler = prepare_data(df, seq_length)
    
    best_params = optimize_hyperparameters(x.cpu().numpy(), y.cpu().numpy())
    print("Оптимальные гиперпараметры:", best_params)
    
    train_size = int(0.7 * len(x))
    val_size = int(0.15 * len(x))
    test_size = len(x) - train_size - val_size
    
    train_dataset = TensorDataset(x[:train_size], y[:train_size])
    val_dataset = TensorDataset(x[train_size:train_size+val_size], y[train_size:train_size+val_size])
    test_dataset = TensorDataset(x[train_size+val_size:], y[train_size+val_size:])
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    model = EnhancedBiLSTMModel(
        input_size=5, 
        hidden_layer_size=best_params['module__hidden_layer_size'], 
        output_size=1, 
        num_layers=best_params['module__num_layers']
    ).to(device)
    
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=best_params['lr'], weight_decay=best_params['optimizer__weight_decay'])
    
    train_model(model, criterion, optimizer, train_loader, val_loader, num_epochs=100, symbols=symbols, df=df, scaler=scaler)
    
    model.load_state_dict(torch.load("best_model.pth"))
    evaluate_model(model, test_loader, scaler)
    
    logging.info("Обучение завершено. Результаты сохранены.")
    print_gpu_utilization()

if __name__ == "__main__":
    main()
