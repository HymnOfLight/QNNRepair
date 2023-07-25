import numpy as np
import tensorflow as tf
from tensorflow import keras


model = keras.models.load_model('./resnet50_cifar10.h5')

(x_test, y_test), _ = keras.datasets.cifar10.load_data()

x_test = x_test.astype('float32') / 255.0
y_test = keras.utils.to_categorical(y_test, num_classes=10)

predictions = model.predict(x_test)

top1_accuracy = tf.keras.metrics.TopKCategoricalAccuracy(k=1)
top1_accuracy.update_state(y_test, predictions)
top1_acc = top1_accuracy.result().numpy()

top5_accuracy = tf.keras.metrics.TopKCategoricalAccuracy(k=5)
top5_accuracy.update_state(y_test, predictions)
top5_acc = top5_accuracy.result().numpy()

print("Top-1 Accuracy:", top1_acc)
print("Top-5 Accuracy:", top5_acc)

with open("cifar10_resnet50_hdf5_predictions.txt", "w") as f:
    for i in range(len(x_test)):
        true_label = np.argmax(y_test[i])
        predicted_label = np.argmax(predictions[i])
        f.write(f"Input {i+1}: True Class {true_label}, Predicted Class {predicted_label}\n")

layer_outputs = []
for layer in model.layers:
    intermediate_model = keras.Model(inputs=model.input, outputs=layer.output)
    layer_output = intermediate_model.predict(x_test)
    layer_outputs.append(layer_output)

for i, layer_output in enumerate(layer_outputs):
    np.save(f"cifar10_resnet50_layer_outputs/float_layer{i+1}_output.npy", layer_output)