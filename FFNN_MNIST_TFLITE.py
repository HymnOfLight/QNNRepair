import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense
from tensorflow.keras.optimizers import Adam

# 加载MNIST数据集
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

# 构建模型
model = Sequential([
    Flatten(input_shape=(28, 28)),
    Dense(256, activation='relu'),
    Dense(128, activation='relu'),
    Dense(64, activation='relu'),
    Dense(32, activation='relu'),
    Dense(10, activation='softmax')
])

# 编译模型
model.compile(optimizer=Adam(),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# 训练模型
model.fit(x_train, y_train, epochs=1, validation_data=(x_test, y_test))

# 保存原始模型
model.save('original_model_mnist.h5')

# 构建representative_dataset函数
def representative_dataset():
    for data in x_train[:100]:
        yield [data.reshape(1, 28, 28)]

# 量化模型
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_dataset
quantized_model = converter.convert()

# 保存量化模型
with open('quantized_model_mnist.tflite', 'wb') as f:
    f.write(quantized_model)

# 加载原始模型和量化模型
loaded_original_model = tf.keras.models.load_model('original_model_mnist.h5')
interpreter = tf.lite.Interpreter(model_path='quantized_model_mnist.tflite')
interpreter.allocate_tensors()

# 测量原始模型的精确度
_, original_model_accuracy = model.evaluate(x_test, y_test, verbose=0)
print("Original Model Accuracy:", original_model_accuracy)

# 测量量化模型的精确度
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
quantized_model_accuracy = 0
for i in range(len(x_test)):
    input_data = x_test[i].reshape(1, 28, 28)
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])
    predicted_label = tf.argmax(output_data, axis=1).numpy()[0]
    if predicted_label == y_test[i]:
        quantized_model_accuracy += 1
quantized_model_accuracy /= len(x_test)
print("Quantized Model Accuracy:", quantized_model_accuracy)