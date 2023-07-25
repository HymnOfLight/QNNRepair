import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import cifar10

(_, _), (test_images, test_labels) = cifar10.load_data()
test_images = test_images.astype('float32') / 255.0
test_labels = test_labels.flatten()

interpreter = tf.lite.Interpreter(model_path='./resnet50_cifar10.pth')
interpreter.allocate_tensors()

input_index = interpreter.get_input_details()[0]['index']
output_index = interpreter.get_output_details()[0]['index']

def calculate_accuracy(predictions, labels, top_k=1):
    if top_k == 1:
        correct_predictions = np.argmax(predictions, axis=1) == labels
    else:
        top_predictions = np.argsort(predictions, axis=1)[:, -top_k:]
        correct_predictions = np.any(top_predictions == np.expand_dims(labels, axis=1), axis=1)
    accuracy = np.mean(correct_predictions)
    return accuracy

num_examples = len(test_images)
batch_size = interpreter.get_input_details()[0]['shape'][0]
num_batches = int(np.ceil(num_examples / batch_size))

top1_accuracy = 0.0
top5_accuracy = 0.0

with open('resnet50_cifar10_quantized_output.txt', 'w') as f:
    for i in range(num_batches):
        start_index = i * batch_size
        end_index = min((i + 1) * batch_size, num_examples)
        batch_images = test_images[start_index:end_index]
        batch_labels = test_labels[start_index:end_index]

        interpreter.set_tensor(input_index, batch_images)

        interpreter.invoke()

        predictions = interpreter.get_tensor(output_index)

        top1_accuracy += calculate_accuracy(predictions, batch_labels, top_k=1)
        top5_accuracy += calculate_accuracy(predictions, batch_labels, top_k=5)

        predicted_classes = np.argmax(predictions, axis=1)
        for j in range(len(predicted_classes)):
            f.write(f"Input {i*batch_size + j + 1}: Class {predicted_classes[j]}\n")

top1_accuracy /= num_batches
top5_accuracy /= num_batches

print("Top-1 Accuracy: {:.2f}%".format(top1_accuracy * 100))
print("Top-5 Accuracy: {:.2f}%".format(top5_accuracy * 100))