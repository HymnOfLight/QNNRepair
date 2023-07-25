import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import numpy as np

# 加载CIFAR-10数据集
(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()

# 数据预处理
train_images = train_images
test_images = test_images

# 定义全连接神经网络模型
model = models.Sequential()
model.add(layers.Flatten(input_shape=(32, 32, 3)))
model.add(layers.Dense(256, activation='relu'))
model.add(layers.Dense(128, activation='relu'))
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(32, activation='relu'))
model.add(layers.Dense(16, activation='relu'))
model.add(layers.Dense(8, activation='relu'))
model.add(layers.Dense(4, activation='relu'))
model.add(layers.Dense(2, activation='relu'))
model.add(layers.Dense(10, activation='softmax'))

# 编译和训练模型
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(train_images, train_labels, epochs=10, validation_data=(test_images, test_labels))

# 保存原始模型
model.save('FFNN_original_model.h5')

# 定义 representative_dataset 函数，用于量化过程
def representative_dataset():
    for image in test_images:
        yield [np.expand_dims(image, axis=0)]

# 使用TFLiteConverter将原始模型转换为TFLite模型并进行int8量化
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_dataset

quantized_model = converter.convert()

# 保存量化模型
with open('FFNN_quantized_model.tflite', 'wb') as f:
    f.write(quantized_model)

# 加载量化模型进行精确度测量
interpreter = tf.lite.Interpreter(model_content=quantized_model)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

correct_count = 0
total_count = 0

for i in range(len(test_images)):
    input_data = np.expand_dims(test_images[i], axis=0).astype(np.float32)
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])
    predicted_label = np.argmax(output_data)
    true_label = test_labels[i]
    
    if predicted_label == true_label:
        correct_count += 1
    total_count += 1

accuracy = correct_count / total_count
print('Quantized Model Accuracy:', accuracy)