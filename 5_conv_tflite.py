import tensorflow as tf
from tensorflow import keras
import numpy as np
from tensorflow.keras import datasets, layers, models


(x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()

x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0
y_train = keras.utils.to_categorical(y_train, 10)
y_test = keras.utils.to_categorical(y_test, 10)

model = models.Sequential()


# Add the convolutional layers
model.add(layers.Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(32, 32, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu', padding='same'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation='relu', padding='same'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(256, (3, 3), activation='relu', padding='same'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(512, (3, 3), activation='relu', padding='same'))

# Flatten the output of the last convolutional layer
model.add(layers.Flatten())

# Add the fully connected layers
model.add(layers.Dense(512, activation='relu'))
model.add(layers.Dense(10, activation='softmax'))

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.fit(x_train, y_train, epochs=10, batch_size=32, validation_data=(x_test, y_test))

_, accuracy = model.evaluate(x_test, y_test)
print("Original Model Accuracy: %.2f%%" % (accuracy * 100))

model.save('original_model_conv5.h5')

def representative_dataset():
    for image in x_test:
        yield [np.expand_dims(image, axis=0)]

quantization_model = tf.keras.models.clone_model(model)

converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_dataset

quantized_model = converter.convert()

with open('quantized_model_conv5.tflite', 'wb') as f:
    f.write(quantized_model)

quantized_model = tf.lite.Interpreter(model_path='quantized_model_conv5.tflite')
quantized_model.allocate_tensors()

quantized_input_index = quantized_model.get_input_details()[0]['index']
quantized_output_index = quantized_model.get_output_details()[0]['index']

correct = 0
total = 0
quantized_fc1_outputs = []
for i in range(len(x_test)):
    input_data = np.expand_dims(x_test[i], axis=0).astype(np.float32)
    quantized_model.set_tensor(quantized_input_index, input_data)

    quantized_model.invoke()

    output_data = quantized_model.get_tensor(quantized_output_index)
    quantized_fc1_outputs.append(output_data)

    predicted_class = np.argmax(output_data)
    true_class = np.argmax(y_test[i])
    if predicted_class == true_class:
        correct += 1
    total += 1

quantized_accuracy = correct / total
print("Quantized Model Accuracy: %.2f%%" % (quantized_accuracy * 100))

quantized_fc1_outputs = np.array(quantized_fc1_outputs)
np.save('quantized_fc1_outputs.npy', quantized_fc1_outputs)
