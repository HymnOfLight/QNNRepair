import numpy as np
import tensorflow as tf
from tensorflow import keras

interpreter = tf.lite.Interpreter(model_path='./models/quantized_model_mnist.tflite')
interpreter.allocate_tensors()

(x_test, y_test), _ = keras.datasets.mnist.load_data()

x_test = x_test.astype('float32') / 255.0
y_test = keras.utils.to_categorical(y_test, num_classes=10)

input_index = interpreter.get_input_details()[0]['index']
output_index = interpreter.get_output_details()[0]['index']

predictions = []
layer_outputs = []

for i in range(len(x_test)):
    interpreter.set_tensor(input_index, x_test[i:i+1])
    interpreter.invoke()

    output = interpreter.get_tensor(output_index)
    predictions.append(output.flatten())

    for j, detail in enumerate(interpreter.get_tensor_details()):
        interpreter.allocate_tensors() 
        layer_output = interpreter.get_tensor(detail['index'])
#         print(layer_output)
        np.save(f"/local-data/e91868xs/mnist_layers/layer{j+1}_output.npy", layer_output)

predictions = np.array(predictions)
top1_accuracy = tf.keras.metrics.TopKCategoricalAccuracy(k=1)
top1_accuracy.update_state(y_test, predictions)
top1_acc = top1_accuracy.result().numpy()

top5_accuracy = tf.keras.metrics.TopKCategoricalAccuracy(k=5)
top5_accuracy.update_state(y_test, predictions)
top5_acc = top5_accuracy.result().numpy()

print("Top-1 Accuracy:", top1_acc)
print("Top-5 Accuracy:", top5_acc)

tf.io.gfile.makedirs("layer_outputs")

with open("mnist_predictions.txt", "w") as f:
    for i in range(len(x_test)):
        true_label = np.argmax(y_test[i])
        predicted_label = np.argmax(predictions[i])
        f.write(f"Input {i+1}: True Class {true_label}, Predicted Class {predicted_label}\n")