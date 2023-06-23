# QNNRepair

Quantized Neural Network Repair

## Overview

We present QNNRepair, a method for repairing quantized neural networks (QNNs). It aims to improve the accuracy of a neural network model after quantization. QNNRepair accepts the full-precision and the weight-quantized neural networks, together with a repair dataset
of passing and failing tests. At first, QNNRepair applies the software fault localization method to identify these neurons for causing performance degradation during neural network quantization. Then, it formulates the repair problem into a linear programming problem of solving neuron weights parameters, which corrects the QNN’s performance on failing tests while not compromising its performance on passing tests. 


We evaluate QNNRepair with widely used neural network architectures such as MobileNetV2, ResNet, and VGGNet on popular datasets, including high-resolution images. We also compare QNNRepair with the state-of-the-art data-free quantization method SQuant. According to the experiment results, we conclude that QNNRepair is effective in improving the quantized model’s performance in most cases and its repaired models have higher accuracy than SQuant’s in the independent validation set.
