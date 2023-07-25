import torch
import torch.nn as nn
import torchvision

import torchvision.transforms as transforms
import torchvision.datasets as datasets
import numpy as np
import argparse

class ResNet50(nn.Module):
    def __init__(self):
        super(ResNet50, self).__init__()
        self.model = torchvision.models.resnet50(pretrained=False, num_classes=10)

    def forward(self, x):
        x = self.model(x)
        return x

DEFAULT_INPUT_FLOAT = "./resnet50_cifar10.pth"
DEFAULT_INPUT_QUANTIZED = "./resnet50_cifar10_quantized.pth"

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Normalize the data
])
test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=False)

parser = argparse.ArgumentParser()
parser.add_argument("-f", "--floating_model", \
default= DEFAULT_INPUT_FLOAT, \
help="floating model to be tested")

parser.add_argument("-q", "--quantized_model", \
default= DEFAULT_INPUT_QUANTIZED, \
help="quantized model to be tested")
args = parser.parse_args()
model1 = ResNet50()
model2 = ResNet50()

model1.load_state_dict(torch.load(args.floating_model))
model1.eval()

device = torch.device("cpu")
model1.to(device)

quantization_config = torch.quantization.get_default_qconfig()
example_input = torch.rand(1, 3, 224, 224)
model2 = torch.quantization.quantize(model1, quantization_config, example_input)
torch.jit.save(model2, 'quantized_resnet_model.pt')


model2.to('cpu')

def calculate_accuracy(outputs, labels, top_k=1):
    _, predicted = outputs.topk(top_k, 1, True, True)
    predicted = predicted.t()
    correct = predicted.eq(labels.view(1, -1).expand_as(predicted))
    accuracy = correct[:top_k].reshape(-1).float().mean().item()
    return accuracy

def get_layer_output(model, input, layer_index):
    output = input
    for index, module in enumerate(model.modules()):
        if index == layer_index:
            output = module(output)
            break
    return output

top1_accuracy_1 = 0.0
top5_accuracy_1 = 0.0
outputs1_layer2 = []
outputs1_layer3 = []

top1_accuracy_2 = 0.0
top5_accuracy_2 = 0.0
outputs2_layer2 = []
outputs2_layer3 = []

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        
        
        outputs_1 = model1(images)
        top1_accuracy_1 += calculate_accuracy(outputs_1, labels, top_k=1)
        top5_accuracy_1 += calculate_accuracy(outputs_1, labels, top_k=5)
        outputs1_layer2.append(get_layer_output(model1, images, layer_index=-2).detach().cpu().numpy())
        outputs1_layer3.append(get_layer_output(model1, images, layer_index=-1).detach().cpu().numpy())

        
        outputs_2 = model2(images)
        top1_accuracy_2 += calculate_accuracy(outputs_2, labels, top_k=1)
        top5_accuracy_2 += calculate_accuracy(outputs_2, labels, top_k=5)
        outputs2_layer2.append(get_layer_output(model2, images, layer_index=-2).detach().cpu().numpy())
        outputs2_layer3.append(get_layer_output(model2, images, layer_index=-1).detach().cpu().numpy())

num_batches = len(test_loader)
top1_accuracy_1 /= num_batches
top5_accuracy_1 /= num_batches

top1_accuracy_2 /= num_batches
top5_accuracy_2 /= num_batches

print("Model 1 - Top-1 Accuracy: {:.2f}%".format(top1_accuracy_1 * 100))
print("Model 1 - Top-5 Accuracy: {:.2f}%".format(top5_accuracy_1 * 100))

print("Model 2 - Top-1 Accuracy: {:.2f}%".format(top1_accuracy_2 * 100))
print("Model 2 - Top-5 Accuracy: {:.2f}%".format(top5_accuracy_2 * 100))

np.save('model1_outputs_layer2.npy', np.concatenate(outputs1_layer2))
np.save('model1_outputs_layer1.npy', np.concatenate(outputs1_layer3))
np.save('model2_outputs_layer2.npy', np.concatenate(outputs2_layer2))
np.save('model2_outputs_layer1.npy', np.concatenate(outputs2_layer3))