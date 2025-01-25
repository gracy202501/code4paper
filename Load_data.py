import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split

# 加载NPY文件
data = np.load('NPY/V_data.npy', allow_pickle=True)[0:1100]
labels = np.load('NPY/V_labels.npy', allow_pickle=True)[0:1100]

# 打印数据的形状
print("原始数据形状：", data.shape)

# 打乱样本顺序
random_seed = 40 #LLBNN40
np.random.seed(random_seed)
indices = np.arange(len(data))
np.random.shuffle(indices)
data = data[indices]
labels = labels[indices]

# 划分数据集
# 定义训练集、验证集和测试集的比例
train_ratio = 0.8
val_ratio = 0.1
test_ratio = 0.1

# 计算划分后的样本数量
num_samples = len(data)
num_train = int(train_ratio * num_samples)
num_val = int(val_ratio * num_samples)
num_test = num_samples - num_train - num_val

data = data.astype(float)
labels = labels.astype(float)

# 划分数据集
train_data, val_data, test_data = np.split(data, [num_train, num_train + num_val])
train_labels, val_labels, test_labels = np.split(labels, [num_train, num_train + num_val])
# 将数据转换为 PyTorch 张量
train_data_tensor = torch.tensor(train_data, dtype=torch.float32)
train_label_tensor = torch.tensor(train_labels, dtype=torch.long)
val_data_tensor = torch.tensor(val_data, dtype=torch.float32)
val_label_tensor = torch.tensor(val_labels, dtype=torch.long)
test_data_tensor = torch.tensor(test_data, dtype=torch.float32)
test_label_tensor = torch.tensor(test_labels, dtype=torch.long)

# 使用 TensorDataset 和 DataLoader 加载数据
train_dataset = TensorDataset(train_data_tensor, train_label_tensor)
val_dataset = TensorDataset(val_data_tensor, val_label_tensor)
test_dataset = TensorDataset(test_data_tensor, test_label_tensor)

batch_size = 32  # 定义批量大小 tag

# train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
# test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=True)

# 打印每个数据集的大小
print("训练集大小：", len(train_dataset))
print("验证集大小：", len(val_dataset))
print("测试集大小：", len(test_dataset))

