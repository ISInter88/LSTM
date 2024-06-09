import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd
import csv

# 自作データセットクラス
class TimeSeriesDataset(Dataset):
    def __init__(self, data, input_seq_length, output_seq_length):
        self.data = data
        self.input_seq_length = input_seq_length
        self.output_seq_length = output_seq_length

    def __len__(self):
        return len(self.data) - self.input_seq_length - self.output_seq_length + 1

    def __getitem__(self, idx):
        return (
            self.data[idx : idx + self.input_seq_length, :],
            self.data[idx + self.input_seq_length : idx + self.input_seq_length + self.output_seq_length, :]
        )

# バリデーションとテストの関数
def evaluate(loader, scaler):
    model.eval()
    all_targets = []
    all_outputs = []
    with torch.no_grad():
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            all_targets.append(targets.cpu().numpy())
            all_outputs.append(outputs.cpu().numpy())
    all_targets = np.concatenate(all_targets, axis=0)
    all_outputs = np.concatenate(all_outputs, axis=0)
    
    # スケーリング解除
    all_targets = inverse_transform(scaler, all_targets)
    all_outputs = inverse_transform(scaler, all_outputs)
    
    # 負の値を0に置き換え
    all_outputs = np.maximum(all_outputs, 0)
    
    # 形状を2次元に変換
    all_targets = all_targets.reshape(-1, all_targets.shape[-1])
    all_outputs = all_outputs.reshape(-1, all_outputs.shape[-1])
    
    mae = mean_absolute_error(all_targets, all_outputs)
    mse = mean_squared_error(all_targets, all_outputs)
    return mae, mse


# データの読み込みと前処理
def load_data(file_path, input_seq_length, output_seq_length):
    df = pd.read_csv(file_path)
    data = df.iloc[:, 1:].values.astype(np.float32)  # 日付列を除外
    scaler = StandardScaler()
    data = scaler.fit_transform(data)
    dataset = TimeSeriesDataset(data, input_seq_length, output_seq_length)
    return dataset, scaler


# RNNモデル（LSTMを使用）
class RNNModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, output_seq_length):
        super(RNNModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size * output_seq_length)
        self.output_seq_length = output_seq_length
        self.output_size = output_size

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out.view(-1, self.output_seq_length, self.output_size)


# データセットの分割
def split_dataset(dataset, train_ratio, valid_ratio):
    dataset_size = len(dataset)
    train_size = int(train_ratio * dataset_size)
    valid_size = int(valid_ratio * dataset_size)
    test_size = dataset_size - train_size - valid_size
    
    train_dataset = Subset(dataset, range(train_size))
    valid_dataset = Subset(dataset, range(train_size, train_size + valid_size))
    test_dataset = Subset(dataset, range(train_size + valid_size, dataset_size))
    
    return train_dataset, valid_dataset, test_dataset

# ハイパーパラメータの設定
input_size = 30  # 特徴量の数
num_layers = 2
hidden_size=64
output_size = 30  # 特徴量の数
input_seq_length = 24
output_seq_length = 12
num_epochs = 50
learning_rate = 0.001
batch_size = 32
best_loss = float('inf')
best_epoch = 0
counter=0
patience=5

P_data=False

file_path = 'datasets/Abilene_30.csv'
dataset, scaler = load_data(file_path, input_seq_length, output_seq_length)

def inverse_transform(scaler, data):
    data_reshaped = data.reshape(-1, data.shape[-1])
    data_inversed = scaler.inverse_transform(data_reshaped)
    return data_inversed.reshape(data.shape)


# データセットの分割
train_ratio = 0.6
valid_ratio = 0.2
train_dataset, valid_dataset, test_dataset = split_dataset(dataset, train_ratio, valid_ratio)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# モデル、損失関数、最適化手法の定義
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = RNNModel(input_size, hidden_size, num_layers, output_size, output_seq_length).to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# 学習ループ
for epoch in range(num_epochs):
    model.train()
    for inputs, targets in train_loader:
        inputs, targets = inputs.to(device), targets.to(device)

        # 順伝播
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        # 逆伝播と最適化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # バリデーションの評価
    valid_mae, valid_mse = evaluate(valid_loader, scaler)
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}, Validation MAE: {valid_mae:.4f}, Validation MSE: {valid_mse:.4f}')
    
    # Early Stoppingのチェック
    if valid_mse < best_loss:
        best_loss = valid_mse
        best_epoch = epoch
        counter = 0
        # モデルの保存
        torch.save(model.state_dict(), 'best_model.pth')
    else:
        counter += 1
        if counter >= patience:
            print(f'Early stopping at epoch {epoch+1}')
            break


# テストの評価
test_mae, test_mse = evaluate(test_loader, scaler)
print(f'Test MAE: {test_mae:.4f}, Test MSE: {test_mse:.4f}')

# モデルの保存
torch.save(model.state_dict(), 'rnn_model.pth')

print('Training complete!')

# 未来のデータに対する予測 24input 12output
future_dataset = DataLoader(test_dataset, batch_size=1, shuffle=False)
future_predictions = []
future_targets=[]
with torch.no_grad():
    for inputs, targets in future_dataset:
        inputs = inputs.to(device)
        outputs = model(inputs)
        future_predictions.append(outputs.cpu().numpy())
        future_targets.append(targets.cpu().numpy())

# Testに対する予測結果の取得,保存
future_list=[]
target_list=[]
for i in range(len(future_predictions)):
    if i %output_seq_length==0:
        if i!=0:
            target_list.append(future_targets[i])
        future_list.append(future_predictions[i])

future_list = np.array(future_list)
target_list = np.array(target_list)

future_list = np.concatenate(future_list, axis=0)
future_list = inverse_transform(scaler, future_list)
if(P_data==True):
    future_list = np.maximum(future_list, 0)  # 負の値を0に置き換え
future_list = future_list.reshape(-1, future_list.shape[-1])

target_list = np.concatenate(target_list, axis=0)
target_list = inverse_transform(scaler, target_list)
target_list = target_list.reshape(-1, target_list.shape[-1])

#future_predictions = np.concatenate(future_predictions, axis=0)
#future_predictions = inverse_transform(scaler, future_predictions)
#future_predictions = np.maximum(future_predictions, 0)  # 負の値を0に置き換え
#future_predictions = future_predictions.reshape(-1, future_predictions.shape[-1])

print(future_list.shape)
with open('prediction.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    for value in future_list:
        writer.writerow(value)

with open('target.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    for value in target_list:
        writer.writerow(value)
