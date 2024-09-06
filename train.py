import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda import amp
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import RandomizedSearchCV
from skorch import NeuralNetRegressor
from scipy.stats import uniform, randint
import logging
from datetime import datetime
from gpu_util import print_gpu_utilization, get_device_info
from download_data import load_data, update_data
from model import EnhancedBiLSTMModel, EarlyStopping, evaluate_model
from utils import prepare_data
from sklearn.preprocessing import MinMaxScaler

# Настройка логирования
logging.basicConfig(filename=f'training_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log', level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Используется устройство: {device} - OK.")

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

    search = RandomizedSearchCV(net, param_dist, n_iter=20, cv=3, scoring='neg_mean_squared_error')
    search.fit(X, y)

    return search.best_params_

def create_adjacency_matrix(symbols):
    num_tokens = len(symbols)
    adj_matrix = torch.ones(num_tokens, num_tokens) - torch.eye(num_tokens)
    return adj_matrix.to(device)

def train_model(model, criterion, optimizer, train_loader, val_loader, num_epochs, symbols, df, scaler):
    adj_matrix = create_adjacency_matrix(symbols)
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
                outputs = model(batch_x, adj_matrix)
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
                outputs = model(batch_x, adj_matrix)
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

def create_datasets(x, y, train_size, val_size):
    train_dataset = TensorDataset(x[:train_size], y[:train_size])
    val_dataset = TensorDataset(x[train_size:train_size+val_size], y[train_size:train_size+val_size])
    test_dataset = TensorDataset(x[train_size+val_size:], y[train_size+val_size:])
    return train_dataset, val_dataset, test_dataset

def create_dataloaders(train_dataset, val_dataset, test_dataset, batch_size=32):
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, test_loader

def create_model(best_params, symbols):
    return EnhancedBiLSTMModel(
        input_size=5, 
        hidden_layer_size=best_params['module__hidden_layer_size'], 
        output_size=1, 
        num_layers=best_params['module__num_layers'],
        num_tokens=len(symbols)
    ).to(device)

def create_criterion_and_optimizer(model, best_params):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=best_params['lr'], weight_decay=best_params['optimizer__weight_decay'])
    return criterion, optimizer

def main():
    seq_length = 60
    logging.info("Начало обучения")
    print(get_device_info())

    df = load_data()
    symbols = df['symbol'].unique().tolist()
    x, y, scaler = prepare_data(df, seq_length, symbols)

    best_params = optimize_hyperparameters(x.cpu().numpy(), y.cpu().numpy())
    print("Оптимальные гиперпараметры:", best_params)

    train_size = int(0.7 * len(x))
    val_size = int(0.15 * len(x))

    train_dataset, val_dataset, test_dataset = create_datasets(x, y, train_size, val_size)
    train_loader, val_loader, test_loader = create_dataloaders(train_dataset, val_dataset, test_dataset)

    model = create_model(best_params, symbols)
    criterion, optimizer = create_criterion_and_optimizer(model, best_params)

    train_model(model, criterion, optimizer, train_loader, val_loader, num_epochs=100, symbols=symbols, df=df, scaler=scaler)

    model.load_state_dict(torch.load("best_model.pth"))
    evaluate_model(model, test_loader, scaler)

    logging.info("Обучение завершено. Результаты сохранены.")
    print_gpu_utilization()

if __name__ == "__main__":
    main()