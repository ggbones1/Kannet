import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import wandb
from torch.utils.data import DataLoader, TensorDataset, random_split
import torch.optim as optim
from torchkan import KAN
from KACnet import KAC_Net
from KALnet import KAL_Net
import pandas as pd
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import r2_score, mean_squared_error
import matplotlib.pyplot as plt
import numpy as np
import torch
def train_and_validate_model(model, epochs, learning_rate, train_loader, val_loader, model_name):
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = nn.MSELoss()
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for x, y in train_loader:
            optimizer.zero_grad()
            predicted_y = model(x)
            loss = loss_fn(predicted_y, y.unsqueeze(1))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        scheduler.step()
        avg_loss = total_loss / len(train_loader)
        wandb.log({f"{model_name} Train Loss": avg_loss})

        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for x, y in val_loader:
                predicted_y = model(x)
                val_loss = loss_fn(predicted_y, y.unsqueeze(1))
                total_val_loss += val_loss.item()
        
        avg_val_loss = total_val_loss / len(val_loader)
        wandb.log({f"{model_name} Validation Loss": avg_val_loss})
        print(f"Epoch {epoch}, {model_name} Train Loss: {avg_loss}, Validation Loss: {avg_val_loss}")
# Evaluation function
def evaluate_model(model, eval_loader, model_name):
    model.eval()
    predictions, actuals = [], []
    with torch.no_grad():
        for x, y in eval_loader:
            predicted_y = model(x)
            predictions.extend(predicted_y.squeeze().cpu().numpy())
            actuals.extend(y.cpu().numpy())
    return predictions, actuals
class MLP(nn.Module):
    def __init__(self, layers):
        super(MLP, self).__init__()
        mlp_layers = []
        for i in range(len(layers) - 1):
            mlp_layers.append(nn.Linear(layers[i], layers[i+1]))
            if i < len(layers) - 2:
                mlp_layers.append(nn.ReLU())
        self.model = nn.Sequential(*mlp_layers)

    def forward(self, x):
        return self.model(x)
import pandas as pd
from sklearn.model_selection import train_test_split

# 读取xlsx文件
file_path = 'D:\OneDrive - Officials\OneDrive - Mraz Cindy\done\毕设资料\计算公式说明\数据库.xlsx'  # 替换为你的xlsx文件路径
df = pd.read_excel(file_path, engine='openpyxl')

# 打乱行顺序并按8:2的比例分成训练集和测试集
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
# 提取第2，3，4，5列的数据
input = train_df.iloc[:, 1:5]
output = train_df.iloc[:, 5:7]
test_input = test_df.iloc[:, 1:5]
test_label = test_df.iloc[:, 5:7]
# 将DataFrame转换为numpy数组并调整其维度为4
array1 = input.to_numpy()
array2 = output.to_numpy()
array3 = test_input.to_numpy()
array4 = test_label.to_numpy()
input=torch.tensor(array1, dtype=torch.float32)
ouput=torch.tensor(array2, dtype=torch.float32)
test_input=torch.tensor(array3, dtype=torch.float32)
test_label=torch.tensor(array4, dtype=torch.float32)
def normalize_columns(tensor):
    # 确保输入是2D张量
    assert tensor.dim() == 2, "Input tensor must be 2D"
    
    # 获取最小值和最大值
    col_min = tensor.min(dim=0, keepdim=True).values
    col_max = tensor.max(dim=0, keepdim=True).values
    
    # 防止除以零的情况
    denom = col_max - col_min
    denom[denom == 0] = 1  # 如果列中所有值相等，避免除以零
    
    # 进行归一化
    normalized_tensor = (tensor - col_min) / denom
    return normalized_tensor
# 对每一列进行归一化
input= normalize_columns(input)
ouput= normalize_columns(ouput)
test_input= normalize_columns(test_input)
test_label= normalize_columns(test_label)
dataset={'train_input':input,'test_input':test_input,'train_label':ouput,'test_label':test_label}

# 输出结果
print("训练集样本数:", len(train_df))
print("测试集样本数:", len(test_df))
wandb.init(project="kan")
dimension=4
# Define model layers
layers = [dimension, 9, 5, 2]
x_data=torch.cat((input,test_input),0)
y_data=torch.cat((ouput,test_label),0)

# 定义一个TensorDataset对象，将x_data和y_data传入
dataset = TensorDataset(x_data, y_data)
# 计算训练集和验证集的数量
train_size = int(0.7 * len(dataset))
val_size = len(dataset) - train_size
# 将数据集划分为训练集和验证集
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
# 创建训练集的DataLoader对象，批量大小为32，打乱数据
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
# 创建验证集的DataLoader对象，批量大小为32，不打乱数据
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    # Initialize and train the KAN model
kan_model = KAN(layers)
train_and_validate_model(kan_model, epochs=50, learning_rate=0.001, train_loader=train_loader, val_loader=val_loader, model_name=f"KAN")

    # Initialize and train the MLP model
mlp_model = MLP(layers)
train_and_validate_model(mlp_model, epochs=50, learning_rate=0.001, train_loader=train_loader, val_loader=val_loader, model_name=f"MLP")
    # Initialize and train the KAC_net model
kac_model = KAC_Net(layers)
train_and_validate_model(kac_model, epochs=50, learning_rate=0.001, train_loader=train_loader, val_loader=val_loader, model_name=f"KAC_Net")
    # Initialize and train the KAL_net model
kal_model = KAL_Net(layers)
train_and_validate_model(kal_model, epochs=50, learning_rate=0.001, train_loader=train_loader, val_loader=val_loader, model_name=f"KAL_Net")
    # Evaluate both models
kan_predictions, kan_actuals = evaluate_model(kan_model, val_loader, f"KAN")
mlp_predictions, mlp_actuals = evaluate_model(mlp_model, val_loader, f"MLP")
kac_predictions, kac_actuals = evaluate_model(kac_model, val_loader, f"KAC_Net")
kal_predictions, kal_actuals = evaluate_model(kal_model, val_loader, f"KAL_Net")
    # Log results to wandb
    # Log results to wandb
kan_data = [[pred, act] for pred, act in zip(kan_predictions, kan_actuals)]
mlp_data = [[pred, act] for pred, act in zip(mlp_predictions, mlp_actuals)]
kac_data = [[pred, act] for pred, act in zip(kac_predictions, kac_actuals)]
kal_data = [[pred, act] for pred, act in zip(kal_predictions, kal_actuals)]
wandb.log({
        f"KAN Predictions vs Actuals": wandb.Table(data=kan_data, columns=["KAN Predictions", "Actuals"]),
        f"MLP Predictions vs Actuals": wandb.Table(data=mlp_data, columns=["MLP Predictions", "Actuals"]),
        f"KAC_Net Predictions vs Actuals": wandb.Table(data=kac_data, columns=["KAC_Net Predictions", "Actuals"]),
        f"KAL_Net Predictions vs Actuals": wandb.Table(data=kal_data, columns=["KAL_Net Predictions", "Actuals"]),
    
    })

    # Save model states
# 保存kan_model的状态字典到文件"kan inverse.pth"
torch.save(kan_model.state_dict(), f"kan inverse.pth")
# 保存mlp_model的状态字典到文件"mlp inverse.pth"
torch.save(mlp_model.state_dict(), f"mlp inverse.pth")
# 保存kac_model的状态字典到文件"kac_net inverse.pth"
torch.save(kac_model.state_dict(), f"kac_net inverse.pth")
# 保存kal_model的状态字典到文件"kal_net inverse.pth"
torch.save(kal_model.state_dict(), f"kal_net inverse.pth")
# 保存"kan inverse.pth"文件到wandb
wandb.save(f"kan inverse.pth")
# 保存"mlp inverse.pth"文件到wandb
wandb.save(f"mlp inverse.pth")
# 保存"kac_net inverse.pth"文件到wandb
wandb.save(f"kac_net inverse.pth")
# 保存"kal_net inverse.pth"文件到wandb
wandb.save(f"kal_net inverse.pth")
