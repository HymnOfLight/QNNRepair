import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
import torch.quantization
import numpy as np
import torchvision

# Load the pre-trained ResNet-18 model
model = models.resnet18(pretrained=True)

# Define a function to collect intermediate activations
def get_intermediate_activations(model, x):
    activations = []
    for module in model.modules():
        if isinstance(module, nn.ReLU):
            x = module(x)
            activations.append(x)
        elif isinstance(module, nn.MaxPool2d):
            x = module(x)
            activations.append(x)
        elif isinstance(module, nn.Linear):
            x = x.view(x.size(0), -1)
            x = module(x)
            activations.append(x)
        elif isinstance(module, nn.BatchNorm2d):
            x = module(x)
            activations.append(x)
        #elif isinstance(module, nn.Conv2d):
            #x = module(x)
            #activations.append(x)
    return activations

# Define a transform to normalize the CIFAR-10 images
normalize = transforms.Normalize(
    mean=[0.4914, 0.4822, 0.4465],
    std=[0.2023, 0.1994, 0.2010]
)

# Load the CIFAR-10 test set
testset = torchvision.datasets.CIFAR10(
    root='./data', train=False, download=True, transform=transforms.Compose([
        transforms.ToTensor(),
        normalize
    ])
)

# Get a single sample from the test set
input_data, target = testset[0]
input_data = input_data.unsqueeze(0)

# Collect intermediate activations for the floating-point model
model.eval()
with torch.no_grad():
    intermediate_activations_float = get_intermediate_activations(model, input_data)

# Convert the model to a quantized model
quantized_model = torch.quantization.quantize_dynamic(
    model, {nn.Conv2d}, dtype=torch.qint8
)

# Collect intermediate activations for the quantized model
quantized_model.eval()
with torch.no_grad():
    intermediate_activations_quantized = get_intermediate_activations
