import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
from torchvision.models import resnet50
from torch.quantization import QuantStub, DeQuantStub, quantize

# Step 1: Prepare the CIFAR-10 dataset
transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])
train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=128, shuffle=True)
test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=128, shuffle=False)

# Step 2: Define and initialize the model
model = resnet50(pretrained=False, num_classes=10)
model.train()

# Step 3: Define the loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4)
lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[150, 250], gamma=0.1)

# Step 4: Train the model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

num_epochs = 300
for epoch in range(num_epochs):
    model.train()
    train_loss = 0.0
    correct = 0
    total = 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    lr_scheduler.step()

    print(f"Epoch [{epoch+1}/{num_epochs}]: Train Loss: {train_loss / len(train_loader):.3f} | Train Acc: {100 * correct / total:.2f}%")

# Step 5: Evaluate the trained model
model.eval()

# Calculate top-1 and top-5 accuracy for the trained model
def calculate_accuracy(outputs, labels, top_k=1):
    _, predicted = outputs.topk(top_k, 1, True, True)
    predicted = predicted.t()
    correct = predicted.eq(labels.view(1, -1).expand_as(predicted))
    accuracy = correct[:top_k].reshape(-1).float().mean().item()
    return accuracy

top1_accuracy = 0.0
top5_accuracy = 0.0

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)

        top1_accuracy += calculate_accuracy(outputs, labels, top_k=1)
        top5_accuracy += calculate_accuracy(outputs, labels, top_k=5)

num_samples = len(test_dataset)
top1_accuracy /= num_samples
top5_accuracy /= num_samples

print(f"Top-1 Accuracy: {top1_accuracy * 100:.2f}%")
print(f"Top-5 Accuracy: {top5_accuracy * 100:.2f}%")

# Step 6: Quantize the model
quantized_model = quantize(model, test_loader)

# Step 7: Evaluate the quantized model
quantized_model.eval()

quantized_top1_accuracy = 0.0
quantized_top5_accuracy = 0.0

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = quantized_model(images)

        quantized_top1_accuracy += calculate_accuracy(outputs, labels, top_k=1)
        quantized_top5_accuracy += calculate_accuracy(outputs, labels, top_k=5)

quantized_top1_accuracy /= num_samples
quantized_top5_accuracy /= num_samples

print(f"Quantized Top-1 Accuracy: {quantized_top1_accuracy * 100:.2f}%")
print(f"Quantized Top-5 Accuracy: {quantized_top5_accuracy * 100:.2f}%")

# Step 8: Save the models
torch.save(model.state_dict(), "resnet50.pth")
torch.save(quantized_model.state_dict(), "resnet50_quantized.pth")