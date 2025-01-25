import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from pyro.nn import PyroModule, PyroSample
import pyro
import pyro.distributions as dist
from Load_data import train_loader, val_loader, test_loader
import matplotlib.pyplot as plt
import numpy as np
import random
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    auc,
    confusion_matrix,
)


# 设置随机种子以确保可重复性
seed = 40
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
pyro.set_rng_seed(seed)

# 在某些情况下，还需要添加以下代码以确保实验的可重复性
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

def plot_multiclass_roc_curve(true_labels, predicted_probabilities):
    num_classes = predicted_probabilities.shape[1]

    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    for i in range(num_classes):
        fpr[i], tpr[i], _ = roc_curve(true_labels == i, predicted_probabilities[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])  # 确保 auc 是从 sklearn 导入的函数

    # Plotting
    plt.figure()
    plt.rc('font', family='Times New Roman')
    for i in range(num_classes):
        plt.plot(fpr[i], tpr[i], label=f'Class {i} (AUC = {roc_auc[i]:.2f})')

    plt.plot([0, 1], [0, 1], 'k--')  # Diagonal line
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.tick_params(axis='both', which='major', labelsize=14)
    # 加粗刻度值
    for label in plt.gca().get_xticklabels() + plt.gca().get_yticklabels():
        label.set_fontweight('bold')
    plt.xlabel('False Positive Rate', fontsize=12, fontweight='bold')
    plt.ylabel('True Positive Rate', fontsize=12, fontweight='bold')

    plt.legend(loc='lower right')
    plt.tight_layout(rect=[0, 0.02, 1, 1])
    plt.figtext(0.5, 0.02,'ROC', ha='center', fontsize=14, fontweight='bold')

    # plt.savefig('pic/LLBNN_Phys_hard2_R.png', format='png', dpi=300)
    # plt.savefig('pic/LLBNN_Phys_hard2_R.pdf', format='pdf', dpi=300)
    # plt.savefig('pic/LLBNN_Phys_hard2_R.eps', format='eps', dpi=300)

    plt.show()


# 定义贝叶斯神经网络部分
class BayesianNN(PyroModule):
    def __init__(self, input_dim=128, hidden_dim=64, output_dim=3):
        super().__init__()
        # 输入是两个LSTM输出拼接后的特征维度，默认128维
        self.fc1 = PyroModule[nn.Linear](input_dim, hidden_dim).to(device)
        self.fc1.weight = PyroSample(dist.Normal(0., 1.).expand([hidden_dim, input_dim]).to_event(2))
        self.fc1.bias = PyroSample(dist.Normal(0., 1.).expand([hidden_dim]).to_event(1))

        self.fc2 = PyroModule[nn.Linear](hidden_dim, output_dim).to(device)
        self.fc2.weight = PyroSample(dist.Normal(0., 1.).expand([output_dim, hidden_dim]).to_event(2))
        self.fc2.bias = PyroSample(dist.Normal(0., 1.).expand([output_dim]).to_event(1))

        # 确保模型和参数移动到设备
        # self.to(device)

    def forward(self, x, y=None):
        # 将输入数据移动到设备
        x = x.view(x.size(0), -1).to(device)

        # 确保 fc1 和 fc2 的权重和偏置在同一设备上
        self.fc1.weight = self.fc1.weight.to(device)
        self.fc1.bias = self.fc1.bias.to(device)
        self.fc2.weight = self.fc2.weight.to(device)
        self.fc2.bias = self.fc2.bias.to(device)

        # 进行前向传播并确保结果在设备上
        x = F.relu(self.fc1(x))
        logits = F.linear(x, self.fc2.weight, self.fc2.bias)

        # 使用交叉熵损失函数
        with pyro.plate("data", x.shape[0]):
            obs = pyro.sample("obs", dist.Categorical(logits=logits), obs=y)
        # print(obs)
        return logits

class OrthogonalLayer(nn.Module):
    def __init__(self):
        super(OrthogonalLayer, self).__init__()

    def forward(self, x):
        batch_size = x.size(0)
        x = x.view(batch_size, 3, 3)
        u, _, v = torch.svd(x)
        orthogonal_x = torch.bmm(u, v.transpose(1, 2))
        return orthogonal_x.view(batch_size, -1)

class CombinedModel(nn.Module):
    def __init__(self, lstm_input_size1=3, lstm_hidden_size1=64, lstm_layers1=2,
                 lstm_input_size2=9, lstm_hidden_size2=9, lstm_layers2=3,
                 bayesian_model_class=BayesianNN):
        super(CombinedModel, self).__init__()

        # 定义 LSTM 层，lstm1 和 lstm2 使用不同的隐藏层大小和层数
        self.lstm1 = nn.LSTM(input_size=lstm_input_size1, hidden_size=lstm_hidden_size1,
                             num_layers=lstm_layers1, batch_first=True)
        self.lstm2 = nn.LSTM(input_size=lstm_input_size2, hidden_size=lstm_hidden_size2,
                             num_layers=lstm_layers2, batch_first=True)

        # 将 LSTM 输出从 32 维缩减到 9 维
        # self.linear = nn.Linear(lstm_hidden_size2, 9)

        self.ortho_layer = OrthogonalLayer()
        # 计算拼接后的输入维度，传递给 BayesianNN 的实例
        combined_input_dim = lstm_hidden_size1 + 9
        self.bayesian_nn = bayesian_model_class(input_dim=combined_input_dim)

    def forward(self, data):
        # 获取 batch size
        batch_size = data.size(0)

        # 初始化 LSTM1 和 LSTM2 的隐藏状态和记忆状态
        h0_1 = torch.zeros(self.lstm1.num_layers, batch_size, self.lstm1.hidden_size).to(data.device)
        c0_1 = torch.zeros(self.lstm1.num_layers, batch_size, self.lstm1.hidden_size).to(data.device)
        h0_2 = torch.zeros(self.lstm2.num_layers, batch_size, self.lstm2.hidden_size).to(data.device)
        c0_2 = torch.zeros(self.lstm2.num_layers, batch_size, self.lstm2.hidden_size).to(data.device)

        # 拆分数据：data[:, :, :3] 是坐标数据，data[:, :, 3:] 是旋转矩阵数据
        coords = data[:, :, :3]
        rotation_matrices = data[:, :, 3:]

        # LSTM1 处理坐标数据
        _, (h_n1, _) = self.lstm1(coords, (h0_1, c0_1))
        # LSTM2 处理旋转矩阵数据
        _, (h_n2, _) = self.lstm2(rotation_matrices, (h0_2, c0_2))


        # 使用线性层将 h_n2[-1] 转换为 9 维
        constrained_rotation = self.ortho_layer(h_n2[-1])

        # 将两个 LSTM 的最终隐藏状态拼接在一起
        combined_features = torch.cat((h_n1[-1], constrained_rotation), dim=1)

        # 将拼接后的特征输入贝叶斯神经网络
        logits = self.bayesian_nn(combined_features)
        return logits

# 直接在使用时配置不同的 LSTM 层数和隐藏层大小，无需调整 BayesianNN 的定义
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
net = CombinedModel(lstm_input_size1=3, lstm_hidden_size1=64, lstm_layers1=2,
                    lstm_input_size2=9, lstm_hidden_size2=9, lstm_layers2=2).to(device)
loss_func = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=0.001)
# 定义训练、验证和测试流程
losses_train = []
accuracies_train = []
losses_valid = []
accuracies_valid = []
best_acc = 0
epochs = 350
# 训练模型
for epoch in range(epochs):
    net.train()
    train_loss = 0.0
    train_accuracy = 0
    total = 0
    for batch_idx, (data, label) in enumerate(train_loader):
        x = data.to(device)
        optimizer.zero_grad()
        output = net(x)

        # 计算损失
        label = torch.argmax(label, dim=1).to(device)
        loss = loss_func(output, label)

        # 反向传播和优化
        loss.backward()
        optimizer.step()

        # 统计训练损失和准确率
        train_loss += loss.item() * x.size(0)
        _, predicted = torch.max(output, 1)
        train_accuracy += (predicted == label).sum().item()
        total += label.size(0)

    average_train_loss = train_loss / total
    average_train_accuracy = train_accuracy / total
    losses_train.append(average_train_loss)
    accuracies_train.append(average_train_accuracy)

    # 验证模型
    net.eval()
    valid_loss = 0.0
    valid_accuracy = 0
    total_valid = 0

    with torch.no_grad():
        for data, label in val_loader:
            x = data.to(device)
            output = net(x)

            label = torch.argmax(label, dim=1).to(device)
            loss = loss_func(output, label)

            valid_loss += loss.item() * x.size(0)
            _, predicted = torch.max(output, 1)
            valid_accuracy += (predicted == label).sum().item()
            total_valid += label.size(0)

    average_valid_loss = valid_loss / total_valid
    average_valid_accuracy = valid_accuracy / total_valid
    losses_valid.append(average_valid_loss)
    accuracies_valid.append(average_valid_accuracy)

    # 保存最优模型
    if average_valid_accuracy > best_acc:
        best_acc = average_valid_accuracy
        torch.save(net, 'pth/test.pt')

    # 打印每个epoch的训练和验证损失以及准确率
    print(f"Epoch {epoch + 1}/{epochs}:")
    print(f"  Train Loss: {average_train_loss:.6f}, Train Accuracy: {average_train_accuracy:.4f}")
    print(f"  Valid Loss: {average_valid_loss:.6f}, Valid Accuracy: {average_valid_accuracy:.4f}")

# 绘制损失曲线和准确率曲线
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(losses_train, label="Training Loss")
plt.plot(accuracies_train, label="Training Accuracy")
plt.xlabel("Epochs")
plt.ylabel("Loss / Accuracy")
plt.title("Training Loss and Accuracy over epochs")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(losses_valid, label="Validation Loss")
plt.plot(accuracies_valid, label="Validation Accuracy")
plt.xlabel("Epochs")
plt.ylabel("Loss / Accuracy")
plt.title("Validation Loss and Accuracy over epochs")
plt.legend()

plt.show()


# 测试最优模型
net = torch.load('pth/LLBNN_Phys_hard2_model.pt')
print(net)
# net = torch.load('pth/test.pt')
# record：lstm_hidden_size1, lstm_layers1; lstm_hidden_size2, lstm_layers2
print("Model loaded successfully, including all layers and parameters.")
net.eval()
predictions = []
true_labels = []
softmax_output = []

softmax = torch.nn.Softmax(dim=1)

with torch.no_grad():
    for data, label in test_loader:
        x = data.to(device)
        output = net(x)

        # 应用 softmax
        output = softmax(output)
        softmax_output.append(output)

        # 获取真实标签（转换为 class indices）
        label = torch.argmax(label, dim=1).to(device)

        # 获取预测结果
        _, predicted = torch.max(output, 1)
        predictions.extend(predicted.cpu().numpy())
        true_labels.extend(label.cpu().numpy())

accuracy = accuracy_score(true_labels, predictions)
precision = precision_score(true_labels, predictions, average="weighted")
recall = recall_score(true_labels, predictions, average="weighted")
f1 = f1_score(true_labels, predictions, average="weighted")

# 将 softmax_output 转换为形状 (num_samples, num_classes)
softmax_output = torch.cat(softmax_output, dim=0)

# 生成 one-hot 编码的真实标签
num_classes = softmax_output.size(1)
one_hot_labels = torch.eye(num_classes)[true_labels].to(device)

# 计算 AUC 得分（多分类）
auc_score = roc_auc_score(one_hot_labels.cpu().numpy(), softmax_output.cpu().numpy(), multi_class="ovo")

# Convert softmax output to probabilities
softmax_output_probabilities = torch.nn.functional.softmax(softmax_output, dim=1)
softmax_output_probabilities = softmax_output_probabilities.cpu().numpy()

plot_multiclass_roc_curve(np.array(true_labels), softmax_output_probabilities)

# 基于 softmax 概率预测类别
predictions = softmax_output_probabilities.argmax(axis=1)
# 计算并打印混淆矩阵
conf_matrix = confusion_matrix(true_labels, predictions)
print(f"模型在测试集上的准确率：{accuracy:.4f}")
print(f"模型在测试集上的精确率：{precision:.4f}")
print(f"模型在测试集上的召回率：{recall:.4f}")
print(f"模型在测试集上的F1分数：{f1:.4f}")
print(f"模型在测试集上的AUC：{auc_score:.4f}")

# print(conf_matrix)
# 计算每个类别的样本总数
class_counts = np.sum(conf_matrix, axis=1)
# 计算样本占比并替换混淆矩阵中的样本数量
conf_matrix_percentage = conf_matrix / class_counts[:, np.newaxis]

# 可视化混淆矩阵
plt.figure(figsize=(6, 6))
plt.rc('font', family='Times New Roman')
plt.imshow(conf_matrix_percentage, cmap=plt.cm.Blues)
# plt.title('EXP2—3 Average Confusion Matrix')
# plt.colorbar()

# 标注混淆矩阵上的预测概率
thresh = conf_matrix_percentage.max() / 2.
for i in range(conf_matrix_percentage.shape[0]):
    for j in range(conf_matrix_percentage.shape[1]):
        plt.text(j, i, format(conf_matrix_percentage[i, j], '.2f'),
                 horizontalalignment="center",
                 color="white" if conf_matrix_percentage[i, j] > thresh else "black", weight='bold', fontsize='20')
plt.ylabel('True label', fontsize=15, fontweight='bold')
plt.xlabel('Predicted label', fontsize=15, fontweight='bold')
plt.xticks([0, 1, 2], ['0', '1', '2'], fontsize=15, fontweight='bold')
plt.yticks([0, 1, 2], ['0', '1', '2'], fontsize=15, fontweight='bold')
plt.tight_layout(rect=[0, 0.02, 1, 1])
title = 'rr'
plt.figtext(0.5, 0.02, title, ha='center', fontsize=14, fontweight='bold')

# plt.savefig('pic/LLBNN_Phys_hard2_C.png', format='png', dpi=300)
# plt.savefig('pic/LLBNN_Phys_hard2_C.pdf', format='pdf', dpi=300)
# plt.savefig('pic/LLBNN_Phys_hard2_C.eps', format='eps', dpi=300)
plt.show()