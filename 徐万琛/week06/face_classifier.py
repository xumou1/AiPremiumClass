import torch
import torch.nn as nn
import numpy as np
from sklearn.datasets import fetch_olivetti_faces
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
import time

# 设置随机种子以确保结果可复现
torch.manual_seed(42)
np.random.seed(42)

# 设备配置
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {device}")

# 加载Olivetti人脸数据集
faces = fetch_olivetti_faces(shuffle=True)
X = faces.images
y = faces.target

# 数据预处理
X = X.reshape(-1, 64, 64)  # 重塑为序列形式 [样本数, 序列长度, 特征维度]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 转换为PyTorch张量
X_train = torch.FloatTensor(X_train).to(device)
X_test = torch.FloatTensor(X_test).to(device)
y_train = torch.LongTensor(y_train).to(device)
y_test = torch.LongTensor(y_test).to(device)

# 创建数据加载器
train_dataset = TensorDataset(X_train, y_train)
test_dataset = TensorDataset(X_test, y_test)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

# 定义不同的RNN模型
class RNN_Classifier(nn.Module):
    def __init__(self, input_size=64, hidden_size=128, num_layers=2, num_classes=40):
        super(RNN_Classifier, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.rnn = nn.RNN(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )
        self.fc = nn.Linear(hidden_size, num_classes)
        
    def forward(self, x):
        # 初始化隐藏状态
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        
        # 前向传播RNN
        out, _ = self.rnn(x, h0)
        
        # 解码最后一个时间步的隐藏状态
        out = self.fc(out[:, -1, :])
        return out

class LSTM_Classifier(nn.Module):
    def __init__(self, input_size=64, hidden_size=128, num_layers=2, num_classes=40):
        super(LSTM_Classifier, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )
        self.fc = nn.Linear(hidden_size, num_classes)
        
    def forward(self, x):
        # 初始化隐藏状态和细胞状态
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        
        # 前向传播LSTM
        out, _ = self.lstm(x, (h0, c0))
        
        # 解码最后一个时间步的隐藏状态
        out = self.fc(out[:, -1, :])
        return out

class GRU_Classifier(nn.Module):
    def __init__(self, input_size=64, hidden_size=128, num_layers=2, num_classes=40):
        super(GRU_Classifier, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )
        self.fc = nn.Linear(hidden_size, num_classes)
        
    def forward(self, x):
        # 初始化隐藏状态
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        
        # 前向传播GRU
        out, _ = self.gru(x, h0)
        
        # 解码最后一个时间步的隐藏状态
        out = self.fc(out[:, -1, :])
        return out

class BiRNN_Classifier(nn.Module):
    def __init__(self, input_size=64, hidden_size=128, num_layers=2, num_classes=40):
        super(BiRNN_Classifier, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.birnn = nn.RNN(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True
        )
        self.fc = nn.Linear(hidden_size * 2, num_classes)  # 双向RNN输出维度是hidden_size*2
        
    def forward(self, x):
        # 初始化隐藏状态 (num_layers * num_directions, batch, hidden_size)
        h0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(device)
        
        # 前向传播BiRNN
        out, _ = self.birnn(x, h0)
        
        # 解码最后一个时间步的隐藏状态
        out = self.fc(out[:, -1, :])
        return out

# 训练函数
def train_model(model, model_name, train_loader, test_loader, criterion, optimizer, num_epochs=20):
    writer = SummaryWriter(f'runs/face_classifier/{model_name}')
    
    # 记录训练开始时间
    start_time = time.time()
    
    # 训练历史记录
    train_losses = []
    test_accuracies = []
    
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for i, (images, labels) in enumerate(train_loader):
            # 前向传播
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
        # 计算平均训练损失
        epoch_loss = running_loss / len(train_loader)
        train_losses.append(epoch_loss)
        
        # 在测试集上评估模型
        model.eval()
        with torch.no_grad():
            correct = 0
            total = 0
            for images, labels in test_loader:
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
            
            accuracy = 100 * correct / total
            test_accuracies.append(accuracy)
        
        # 记录到TensorBoard
        writer.add_scalar('Loss/train', epoch_loss, epoch)
        writer.add_scalar('Accuracy/test', accuracy, epoch)
        
        print(f'Epoch [{epoch+1}/{num_epochs}], {model_name} - Loss: {epoch_loss:.4f}, Accuracy: {accuracy:.2f}%')
    
    # 计算训练时间
    training_time = time.time() - start_time
    print(f'{model_name} 训练完成，用时: {training_time:.2f} 秒')
    
    # 保存模型
    torch.save(model.state_dict(), f'{model_name}_model.pth')
    
    writer.close()
    
    return train_losses, test_accuracies

# 主函数
def main():
    # 模型参数
    input_size = 64
    hidden_size = 128
    num_layers = 2
    num_classes = 40
    learning_rate = 0.001
    num_epochs = 20
    
    # 初始化模型
    rnn_model = RNN_Classifier(input_size, hidden_size, num_layers, num_classes).to(device)
    lstm_model = LSTM_Classifier(input_size, hidden_size, num_layers, num_classes).to(device)
    gru_model = GRU_Classifier(input_size, hidden_size, num_layers, num_classes).to(device)
    birnn_model = BiRNN_Classifier(input_size, hidden_size, num_layers, num_classes).to(device)
    
    # 损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    
    # 训练模型
    models = [
        (rnn_model, "RNN"),
        (lstm_model, "LSTM"),
        (gru_model, "GRU"),
        (birnn_model, "BiRNN")
    ]
    
    results = {}
    
    for model, name in models:
        print(f"\n开始训练 {name} 模型...")
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        train_losses, test_accuracies = train_model(
            model, name, train_loader, test_loader, criterion, optimizer, num_epochs
        )
        results[name] = {
            'train_losses': train_losses,
            'test_accuracies': test_accuracies
        }
    
    # 绘制损失和准确率对比图
    plt.figure(figsize=(12, 5))
    
    # 损失对比
    plt.subplot(1, 2, 1)
    for name, data in results.items():
        plt.plot(data['train_losses'], label=name)
    plt.title('训练损失对比')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    # 准确率对比
    plt.subplot(1, 2, 2)
    for name, data in results.items():
        plt.plot(data['test_accuracies'], label=name)
    plt.title('测试准确率对比')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('face_classifier_comparison.png')
    plt.show()

if __name__ == "__main__":
    main()
