import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import numpy as np


# 定义卷积神经网络模型
class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(64 * 4 * 4, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        conv1_output = self.conv1(x)
        x = self.relu(conv1_output)
        x = self.maxpool(x)
        conv2_output = self.conv2(x)
        x = self.relu(conv2_output)
        x = self.maxpool(x)
        conv3_output = self.conv3(x)
        x = self.relu(conv3_output)
        x = self.maxpool(x)
        x = x.view(x.size(0), -1)
#         x = self.fc1(x)
#         x = self.relu(x)
#         x = self.fc2(x)
        fc1_output = self.fc1(x)
        fc1_output = self.relu(fc1_output)
        fc2_output = self.fc2(fc1_output)
        return fc2_output, conv1_output, conv2_output, conv3_output, fc1_output, fc2_output
def get_shape(lst):
    shape = []
    while isinstance(lst, list):
        shape.append(len(lst))
        lst = lst[0] if lst else None
    return shape
def create_comparison_array(arr1, arr2):
    comparison_array = np.zeros(arr1.shape, dtype=np.int)

    def compare_elements(a, b, index):
        if isinstance(a, np.ndarray):
            for i in range(a.shape[0]):
                compare_elements(a[i], b[i], index + (i,))
        else:
            if np.sign(a) != np.sign(b):
                comparison_array[index] = 1

    compare_elements(arr1, arr2, tuple())

    return comparison_array

import torch.optim as optim
from torchvision import datasets, transforms

# 设置训练参数
device = torch.device("cpu")
num_epochs = 30
batch_size = 128
learning_rate = 0.001

# 加载CIFAR-10数据集并进行数据增强
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)

train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

# 初始化模型
model = ConvNet().to(device)

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# 训练模型
total_step = len(train_loader)
for epoch in range(num_epochs):
    model.train()
    for i, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)

        # 前向传播
        outputs, _, _, _, _, _ = model(images)
        loss = criterion(outputs, labels)

        # 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i + 1) % 100 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{total_step}], Loss: {loss.item():.4f}")

# 测试模型
net = ConvNet()
net = net.to(device)
model.eval()
with torch.no_grad():
    correct = 0
    total = 0
    conv1_outputs = []
    conv2_outputs = []
    conv3_outputs = []
    fc1_outputs = []
    fc2_outputs = []

    for images, labels in test_loader:
        images = images.to("cpu")  # 将图像数据移回CPU
        labels = labels.to("cpu")  # 将标签数据移回CPU
        # 保存卷积层输出
#         fc2_output, fc1_output, _ = model(images)
        outputs, conv1_output, conv2_output, conv3_output, fc1_output, fc2_output = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        conv1_outputs.append(conv1_output.cpu().numpy())
        conv2_outputs.append(conv2_output.cpu().numpy())
        conv3_outputs.append(conv3_output.cpu().numpy())
        fc1_outputs.append(fc1_output.cpu().numpy())
        fc2_outputs.append(fc2_output.cpu().numpy())

    print(f"Top-1 Accuracy: {100 * correct / total:.2f}%")
    
conv1_outputs = np.concatenate(conv1_outputs, axis=0)
conv2_outputs = np.concatenate(conv2_outputs, axis=0)
conv3_outputs = np.concatenate(conv3_outputs, axis=0)
fc1_outputs = np.concatenate(fc1_outputs, axis=0)
fc2_outputs = np.concatenate(fc2_outputs, axis=0)
# 量化模型
# quantized_model = torch.quantization.quantize_dynamic(
#     net, {nn.Linear}, dtype=torch.qint8
# )
quantized_model = torch.ao.quantization.quantize_dynamic(
    model,  # the original model
    {torch.nn.Linear},  # a set of layers to dynamically quantize
    dtype=torch.qint8)  # the target dtype for quantized weights


# 测试量化模型
quantized_model.eval()
quantized_model.to("cpu")  # 将量化模型转移到CPU上

with torch.no_grad():
    correct = 0
    total = 0
    quantized_conv1_outputs = []
    quantized_conv2_outputs = []
    quantized_conv3_outputs = []
    quantized_fc1_outputs = []
    quantized_fc2_outputs = []

    for images, labels in test_loader:
        images = images.to("cpu")  # 将图像数据移回CPU
        labels = labels.to("cpu")  # 将标签数据移回CPU
#         fc2_output, fc1_output, _ = quantized_model(images)
        # 保存卷积层输出
        outputs, conv1_output, conv2_output, conv3_output, fc1_output, fc2_output = quantized_model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        quantized_conv1_outputs.append(conv1_output.cpu().numpy())
        quantized_conv2_outputs.append(conv2_output.cpu().numpy())
        quantized_conv3_outputs.append(conv3_output.cpu().numpy())
        quantized_fc1_outputs.append(fc1_output.cpu().numpy())
        quantized_fc2_outputs.append(fc2_output.cpu().numpy())

    print(f"Quantized Model - Top-1 Accuracy: {100 * correct / total:.2f}%")
    
quantized_conv1_outputs = np.concatenate(quantized_conv1_outputs, axis=0)
quantized_conv2_outputs = np.concatenate(quantized_conv2_outputs, axis=0)
quantized_conv3_outputs = np.concatenate(quantized_conv3_outputs, axis=0)
quantized_fc1_outputs = np.concatenate(quantized_fc1_outputs, axis=0)
quantized_fc2_outputs = np.concatenate(quantized_fc2_outputs, axis=0)
# 保存模型
torch.save(model.state_dict(), 'model.pth')
torch.save(quantized_model.state_dict(), 'quantized_model.pth')
np.savetxt("quantized_fc1_outputs.txt", quantized_fc1_outputs, fmt='%d', delimiter=',')
# 保存卷积层输出结果为txt文件
# conv1_outputs = torch.cat(conv1_outputs, dim=0).cpu().numpy()
# quantized_conv1_outputs = torch.cat(quantized_conv1_outputs, dim=0).cpu().numpy()

# 比较符号并打印不同元素的索引
print(np.array(conv1_outputs).shape)
print(np.array(conv2_outputs).shape)
print(np.array(conv3_outputs).shape)
print(np.array(fc1_outputs).shape)
print(np.array(fc2_outputs).shape)
print(np.array(quantized_conv1_outputs).shape)
print(np.array(quantized_conv2_outputs).shape)
print(np.array(quantized_conv3_outputs).shape)
print(np.array(quantized_fc1_outputs).shape)
print(np.array(quantized_fc2_outputs).shape)
comparison_array_fc_1 = create_comparison_array(fc1_outputs, quantized_fc1_outputs)
print(comparison_array_fc_1)
comparison_array_fc_1_sum_array = np.sum(comparison_array_fc_1, axis=0)
print(comparison_array_fc_1_sum_array)
np.savetxt("comparison_array_fc_1_sum_array.txt", comparison_array_fc_1_sum_array.reshape((-1,)), fmt='%d', delimiter=',')
comparison_array_fc_2 = create_comparison_array(fc2_outputs, quantized_fc2_outputs)
print(comparison_array_fc_2)
comparison_array_fc_2_sum_array = np.sum(comparison_array_fc_2, axis=0)
print(comparison_array_fc_2_sum_array)
np.savetxt("comparison_array_fc_2_sum_array.txt", comparison_array_fc_2_sum_array.reshape((-1,)), fmt='%d', delimiter=',')
# np.save('quantized_fc1_weights.npy', quantized_model.fc1.module.weight.detach().cpu().numpy())
# np.save('quantized_fc2_weights.npy', quantized_model.fc2.module.weight.detach().cpu().numpy())
# np.save('quantized_conv1_weights.npy', quantized_model.conv1.module.weight.detach().cpu().numpy())
# np.save('quantized_conv2_weights.npy', quantized_model.conv2.module.weight.detach().cpu().numpy())
# np.save('quantized_conv3_weights.npy', quantized_model.conv3.module.weight.detach().cpu().numpy())

# diff_indices = np.where(np.sign(conv1_outputs) != np.sign(quantized_conv1_outputs))[0]
# print("Indices with different signs:")
# print(diff_indices.shape)

# 保存索引到文件
# np.savetxt('diff_indices.txt', diff_indices, fmt='%d', delimiter=',')
# quantized_conv1_outputs = torch.cat(quantized_conv1_outputs, dim=0).cpu().numpy()
# quantized_conv2_outputs = torch.cat(quantized_conv2_outputs, dim=0).cpu().numpy()
# quantized_conv3_outputs = torch.cat(quantized_conv3_outputs, dim=0).cpu().numpy()

# np.savetxt('conv1_outputs.txt', conv1_outputs.reshape(conv1_outputs.shape[0], -1), fmt='%.6f', delimiter=',')
# np.savetxt('conv2_outputs.txt', conv2_outputs.reshape(conv2_outputs.shape[0], -1), fmt='%.6f', delimiter=',')
np.savetxt('conv3_outputs.txt', conv3_outputs.reshape(conv3_outputs.shape[0], -1), fmt='%.6f', delimiter=',')
np.savetxt('fc1_outputs.txt', fc1_outputs.reshape(fc1_outputs.shape[0], -1), fmt='%d', delimiter=',')
np.savetxt('fc2_outputs.txt', fc2_outputs.reshape(fc2_outputs.shape[0], -1), fmt='%d', delimiter=',')
# np.savetxt('quantized_conv1_outputs.txt', quantized_conv1_outputs.reshape(quantized_conv1_outputs.shape[0], -1), fmt='%.6f', delimiter=',')
# np.savetxt('quantized_conv2_outputs.txt', quantized_conv2_outputs.reshape(quantized_conv2_outputs.shape[0], -1), fmt='%.6f', delimiter=',')

#conv1 2 are too huge to save as txt or npy, conv1~1.47 GB and conv2~737 MB
np.savetxt('quantized_conv3_outputs.txt', quantized_conv3_outputs.reshape(quantized_conv3_outputs.shape[0], -1), fmt='%.6f', delimiter=',')
np.savetxt('quantized_fc1_outputs.txt', quantized_fc1_outputs.reshape(quantized_fc1_outputs.shape[0], -1), fmt='%d', delimiter=',')
np.savetxt('quantized_fc2_outputs.txt', quantized_fc2_outputs.reshape(quantized_fc2_outputs.shape[0], -1), fmt='%d', delimiter=',')