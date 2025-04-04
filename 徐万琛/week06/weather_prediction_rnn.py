import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.tensorboard import SummaryWriter
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import os
import time

# 设置随机种子以确保结果可复现
torch.manual_seed(42)
np.random.seed(42)

# 设备配置
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {device}")

# 加载天气数据集
def load_weather_data(file_path):
    df = pd.read_csv(file_path, low_memory=False)
    # 选择需要的列：日期和最高温度
    df = df[['Date', 'MaxTemp']]
    # 将日期列转换为日期时间格式
    df['Date'] = pd.to_datetime(df['Date'])
    # 按日期排序
    df = df.sort_values('Date')
    # 删除缺失值
    df = df.dropna()
    return df

# 创建序列数据
def create_sequences(data, seq_length):
    xs, ys = [], []
    for i in range(len(data) - seq_length - 5):
        # 输入序列
        x = data[i:(i + seq_length)]
        # 输出：未来1天和5天的最高温度
        y1 = data[i + seq_length]  # 未来1天
        y5 = data[i + seq_length:i + seq_length + 5]  # 未来5天
        xs.append(x)
        ys.append((y1, y5))
    return np.array(xs), [(np.array(y1), np.array(y5).flatten()) for y1, y5 in ys]

# 定义RNN模型
class WeatherRNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, seq_length, rnn_type='lstm'):
        super(WeatherRNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size
        self.seq_length = seq_length
        self.rnn_type = rnn_type.lower()
        
        # 选择RNN类型
        if self.rnn_type == 'lstm':
            self.rnn = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        elif self.rnn_type == 'gru':
            self.rnn = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        elif self.rnn_type == 'rnn':
            self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        else:
            raise ValueError(f"不支持的RNN类型: {rnn_type}")
        
        # 输出层：预测未来1天
        self.fc1 = nn.Linear(hidden_size, 1)
        # 输出层：预测未来5天
        self.fc5 = nn.Linear(hidden_size, 5)
        
    def forward(self, x):
        batch_size = x.size(0)
        
        # 初始化隐藏状态
        if self.rnn_type == 'lstm':
            h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(device)
            c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(device)
            out, _ = self.rnn(x, (h0, c0))
        else:
            h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(device)
            out, _ = self.rnn(x, h0)
        
        # 获取最后一个时间步的输出
        out = out[:, -1, :]
        
        # 预测未来1天和5天的温度
        pred_1day = self.fc1(out)
        pred_5days = self.fc5(out)
        
        return pred_1day, pred_5days

# 训练函数
def train_model(model, model_name, train_loader, val_loader, criterion, optimizer, num_epochs=50):
    writer = SummaryWriter(f'runs/weather_prediction/{model_name}')
    
    # 记录训练开始时间
    start_time = time.time()
    
    # 训练历史记录
    train_losses = []
    val_losses = []
    
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        
        for i, (inputs, target_1day, target_5days) in enumerate(train_loader):
            inputs = inputs.to(device)
            target_1day = target_1day.to(device)
            target_5days = target_5days.to(device)
            
            # 前向传播
            pred_1day, pred_5days = model(inputs)
            
            # 计算损失
            loss_1day = criterion(pred_1day, target_1day)
            # 确保pred_5days和target_5days的维度匹配
            # pred_5days形状为[batch_size, 5]，target_5days形状可能为[batch_size, 5, 1]
            # 我们需要调整target_5days的形状以匹配pred_5days
            if target_5days.dim() > 2 and target_5days.size(-1) == 1:
                target_5days = target_5days.squeeze(-1)  # 移除最后一个维度，如果它是1
            loss_5days = criterion(pred_5days, target_5days)
            loss = loss_1day + loss_5days  # 总损失
            
            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
        
        # 计算平均训练损失
        epoch_loss = running_loss / len(train_loader)
        train_losses.append(epoch_loss)
        
        # 在验证集上评估模型
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, target_1day, target_5days in val_loader:
                inputs = inputs.to(device)
                target_1day = target_1day.to(device)
                target_5days = target_5days.to(device)
                
                # 前向传播
                pred_1day, pred_5days = model(inputs)
                
                # 计算损失
                loss_1day = criterion(pred_1day, target_1day)
                # 确保pred_5days和target_5days的维度匹配（验证阶段）
                if target_5days.dim() > 2 and target_5days.size(-1) == 1:
                    target_5days = target_5days.squeeze(-1)  # 移除最后一个维度，如果它是1
                loss_5days = criterion(pred_5days, target_5days)
                loss = loss_1day + loss_5days
                
                val_loss += loss.item()
        
        # 计算平均验证损失
        val_loss = val_loss / len(val_loader)
        val_losses.append(val_loss)
        
        # 记录到TensorBoard
        writer.add_scalar('Loss/train', epoch_loss, epoch)
        writer.add_scalar('Loss/validation', val_loss, epoch)
        
        print(f'Epoch [{epoch+1}/{num_epochs}], {model_name} - Train Loss: {epoch_loss:.4f}, Val Loss: {val_loss:.4f}')
    
    # 计算训练时间
    training_time = time.time() - start_time
    print(f'{model_name} 训练完成，用时: {training_time:.2f} 秒')
    
    # 保存模型
    torch.save(model.state_dict(), f'{model_name}_model.pth')
    
    writer.close()
    
    return train_losses, val_losses

# 评估函数
def evaluate_model(model, test_loader, scaler):
    model.eval()
    predictions_1day = []
    predictions_5days = []
    actual_1day = []
    actual_5days = []
    
    with torch.no_grad():
        for inputs, target_1day, target_5days in test_loader:
            inputs = inputs.to(device)
            target_1day = target_1day.numpy()
            target_5days = target_5days.numpy()
            
            # 前向传播
            pred_1day, pred_5days = model(inputs)
            
            # 转换为CPU并转换为NumPy数组
            pred_1day = pred_1day.cpu().numpy()
            pred_5days = pred_5days.cpu().numpy()
            
            # 确保target_5days的维度正确
            if target_5days.ndim > 2 and target_5days.shape[-1] == 1:
                target_5days = target_5days.squeeze(-1)  # 移除最后一个维度，如果它是1
            
            # 反归一化预测结果
            pred_1day = scaler.inverse_transform(pred_1day)
            pred_5days = np.array([scaler.inverse_transform(p.reshape(-1, 1)).flatten() for p in pred_5days])
            target_1day = scaler.inverse_transform(target_1day.reshape(-1, 1))
            target_5days = np.array([scaler.inverse_transform(t.reshape(-1, 1)).flatten() for t in target_5days])
            
            # 收集预测和实际值
            predictions_1day.extend(pred_1day)
            predictions_5days.extend(pred_5days)
            actual_1day.extend(target_1day)
            actual_5days.extend(target_5days)
    
    # 计算均方根误差
    rmse_1day = np.sqrt(np.mean((np.array(predictions_1day) - np.array(actual_1day)) ** 2))
    rmse_5days = np.sqrt(np.mean((np.array(predictions_5days) - np.array(actual_5days)) ** 2))
    
    print(f'1天预测RMSE: {rmse_1day:.4f}')
    print(f'5天预测RMSE: {rmse_5days:.4f}')
    
    return predictions_1day, predictions_5days, actual_1day, actual_5days

# 主函数
def main():
    # 数据路径
    data_path = 'archive/Summary of Weather.csv'
    
    # 加载数据
    df = load_weather_data(data_path)
    print(f"加载了 {len(df)} 条天气记录")
    
    # 提取最高温度数据
    temp_data = df['MaxTemp'].values.reshape(-1, 1)
    
    # 归一化数据
    scaler = MinMaxScaler(feature_range=(0, 1))
    temp_data_scaled = scaler.fit_transform(temp_data)
    
    # 创建序列数据
    seq_length = 10  # 使用过去10天的数据预测未来
    X, y = create_sequences(temp_data_scaled, seq_length)
    
    # 分割数据集
    X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.25, random_state=42)  # 0.25 * 0.8 = 0.2
    
    # 转换为PyTorch张量
    X_train = torch.FloatTensor(X_train)
    X_val = torch.FloatTensor(X_val)
    X_test = torch.FloatTensor(X_test)
    
    # 处理目标值 - 优化张量创建过程，避免低效的列表推导式
    # 首先将列表转换为numpy数组，然后一次性转换为张量
    y_train_1day_np = np.array([y[0] for y in y_train])
    y_train_5days_np = np.array([y[1] for y in y_train])
    y_val_1day_np = np.array([y[0] for y in y_val])
    y_val_5days_np = np.array([y[1] for y in y_val])
    y_test_1day_np = np.array([y[0] for y in y_test])
    y_test_5days_np = np.array([y[1] for y in y_test])
    
    y_train_1day = torch.FloatTensor(y_train_1day_np)
    y_train_5days = torch.FloatTensor(y_train_5days_np)
    y_val_1day = torch.FloatTensor(y_val_1day_np)
    y_val_5days = torch.FloatTensor(y_val_5days_np)
    y_test_1day = torch.FloatTensor(y_test_1day_np)
    y_test_5days = torch.FloatTensor(y_test_5days_np)
    
    # 创建数据加载器 - 修复TensorDataset参数问题
    train_dataset = TensorDataset(X_train, y_train_1day, y_train_5days)
    val_dataset = TensorDataset(X_val, y_val_1day, y_val_5days)
    test_dataset = TensorDataset(X_test, y_test_1day, y_test_5days)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    # 模型参数
    input_size = 1  # 输入特征维度
    hidden_size = 64  # 隐藏层大小
    num_layers = 2  # RNN层数
    output_size = 1  # 输出维度（温度）
    learning_rate = 0.001
    num_epochs = 50
    
    # 初始化不同类型的RNN模型
    lstm_model = WeatherRNN(input_size, hidden_size, num_layers, output_size, seq_length, 'lstm').to(device)
    gru_model = WeatherRNN(input_size, hidden_size, num_layers, output_size, seq_length, 'gru').to(device)
    rnn_model = WeatherRNN(input_size, hidden_size, num_layers, output_size, seq_length, 'rnn').to(device)
    
    # 损失函数和优化器
    criterion = nn.MSELoss()
    
    # 训练模型
    models = [
        (lstm_model, "LSTM"),
        (gru_model, "GRU"),
        (rnn_model, "RNN")
    ]
    
    results = {}
    
    for model, name in models:
        print(f"\n开始训练 {name} 模型...")
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        train_losses, val_losses = train_model(
            model, name, train_loader, val_loader, criterion, optimizer, num_epochs
        )
        results[name] = {
            'train_losses': train_losses,
            'val_losses': val_losses,
            'model': model
        }
    
    # 评估最佳模型
    best_model_name = min(results, key=lambda k: min(results[k]['val_losses']))
    best_model = results[best_model_name]['model']
    print(f"\n最佳模型: {best_model_name}")
    
    # 在测试集上评估最佳模型
    pred_1day, pred_5days, actual_1day, actual_5days = evaluate_model(best_model, test_loader, scaler)
    
    # 绘制损失对比图
    plt.figure(figsize=(12, 5))
    
    # 训练损失对比
    plt.subplot(1, 2, 1)
    for name, data in results.items():
        plt.plot(data['train_losses'], label=name)
    plt.title('训练损失对比')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    # 验证损失对比
    plt.subplot(1, 2, 2)
    for name, data in results.items():
        plt.plot(data['val_losses'], label=name)
    plt.title('验证损失对比')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('weather_prediction_loss_comparison.png')
    
    # 绘制预测结果
    plt.figure(figsize=(15, 10))
    
    # 1天预测结果
    plt.subplot(2, 1, 1)
    plt.plot(actual_1day[:100], label='实际值')
    plt.plot(pred_1day[:100], label='预测值')
    plt.title('未来1天最高温度预测')
    plt.xlabel('样本')
    plt.ylabel('温度 (°C)')
    plt.legend()
    
    # 5天预测结果（取前20个样本的第一天预测）
    plt.subplot(2, 1, 2)
    days = list(range(5))
    for i in range(5):
        plt.plot(days, actual_5days[i], 'b-', alpha=0.3)
        plt.plot(days, pred_5days[i], 'r-', alpha=0.3)
    
    # 添加平均线
    avg_actual = np.mean([actual_5days[i] for i in range(5)], axis=0)
    avg_pred = np.mean([pred_5days[i] for i in range(5)], axis=0)
    plt.plot(days, avg_actual, 'b-', linewidth=2, label='平均实际值')
    plt.plot(days, avg_pred, 'r-', linewidth=2, label='平均预测值')
    
    plt.title('未来5天最高温度预测（前5个样本）')
    plt.xlabel('天数')
    plt.ylabel('温度 (°C)')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('weather_prediction_results.png')
    plt.show()

if __name__ == "__main__":
    main()