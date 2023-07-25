import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import argparse
DEFAULT_INPUT_TFLITE = "./resnet50_cifar10.pth"
parser = argparse.ArgumentParser()
parser.add_argument("-m", "--model", \
default= DEFAULT_INPUT_TFLITE, \
help="model to be tested")
model = torchvision.models.resnet50(pretrained=False, num_classes=10)
args = parser.parse_args()

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Normalize the data
])
test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=False)

# model.load_state_dict(torch.load(args.model), strict=False)
model = torch.load(args.model)
model.eval()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

def calculate_accuracy(outputs, labels, top_k=1):
    _, predicted = outputs.topk(top_k, 1, True, True)
    predicted = predicted.t()
    correct = predicted.eq(labels.reshape(1, -1).expand_as(predicted))
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

num_batches = len(test_loader)
top1_accuracy /= num_batches
top5_accuracy /= num_batches

print("Top-1 Accuracy: {:.2f}%".format(top1_accuracy * 100))
print("Top-5 Accuracy: {:.2f}%".format(top5_accuracy * 100))