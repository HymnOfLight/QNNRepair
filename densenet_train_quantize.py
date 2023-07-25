import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.datasets import cifar10
import numpy as np

# Load CIFAR-10 dataset
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# Normalize pixel values
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

# Convert labels to categorical one-hot encoding
num_classes = 10
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

# Define the DenseNet-like model
def create_densenet_model():
    growth_rate = 12
    num_layers = [6, 12, 24, 16]
    compression_factor = 0.5

    inputs = keras.Input(shape=(32, 32, 3))
    x = layers.Conv2D(64, kernel_size=3, padding='same')(inputs)

    # Dense blocks with transition layers
    num_blocks = len(num_layers)
    for block in range(num_blocks):
        x = dense_block(x, num_layers[block], growth_rate)
        if block < num_blocks - 1:
            x = transition_layer(x, compression_factor)

    # Global average pooling and classification layer
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dense(128, activation='relu')(x)

    outputs = layers.Dense(num_classes, activation='softmax')(x)

    model = keras.Model(inputs=inputs, outputs=outputs)
    return model

# Dense block
def dense_block(x, num_layers, growth_rate):
    for _ in range(num_layers):
        y = layers.BatchNormalization()(x)
        y = layers.Activation('relu')(y)
        y = layers.Conv2D(4 * growth_rate, kernel_size=1, padding='same')(y)
        y = layers.BatchNormalization()(y)
        y = layers.Activation('relu')(y)
        y = layers.Conv2D(growth_rate, kernel_size=3, padding='same')(y)
        x = layers.Concatenate()([x, y])
    return x

# Transition layer
def transition_layer(x, compression_factor):
    num_channels = int(x.shape[-1] * compression_factor)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Conv2D(num_channels, kernel_size=1, padding='same')(x)
    x = layers.AveragePooling2D(pool_size=2, strides=2)(x)
    return x

def representative_dataset():
    for image in x_test:
        yield [np.expand_dims(image, axis=0)]

# Create the DenseNet-like model
model = create_densenet_model()

# Compile the model
model.compile(
    loss='categorical_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)

# Train the model
model.fit(x_train, y_train, batch_size=64, epochs=1, validation_data=(x_test, y_test))

# Save the model
model.save('densenet_model.h5')

# Load the model
loaded_model = keras.models.load_model('densenet_model.h5')

# Evaluate the model accuracy
_, accuracy = model.evaluate(x_test, y_test)
_, loaded_accuracy = loaded_model.evaluate(x_test, y_test)

print('DenseNet Model Accuracy:', accuracy)
print('Loaded DenseNet Model Accuracy:', loaded_accuracy)

loaded_model = keras.models.load_model('densenet_model.h5')

# Quantize the model
quantized_model = tf.keras.models.clone_model(loaded_model)
quantized_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
quantized_model.summary()

# Quantize the weights
converter = tf.lite.TFLiteConverter.from_keras_model(quantized_model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_dataset
quantized_tflite_model = converter.convert()

# Save the quantized model to a file
with open('quantized_densenet_model.tflite', 'wb') as f:
    f.write(quantized_tflite_model)

# Load the quantized model
interpreter = tf.lite.Interpreter(model_content=quantized_tflite_model)
interpreter.allocate_tensors()

# Test the quantized model accuracy
input_index = interpreter.get_input_details()[0]["index"]
output_index = interpreter.get_output_details()[0]["index"]
num_correct = 0

for i in range(len(x_test)):
    input_data = x_test[i].reshape(1, 32, 32, 3)
    interpreter.set_tensor(input_index, input_data)
    interpreter.invoke()
    output = interpreter.get_tensor(output_index)
    predicted_label = output.argmax()
    true_label = y_test[i].argmax()
    if predicted_label == true_label:
        num_correct += 1

quantized_accuracy = num_correct / len(x_test)
print('Quantized DenseNet Model Accuracy:', quantized_accuracy)

