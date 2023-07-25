import tensorflow as tf
import glob
from PIL import Image
import os
from tensorflow.keras.datasets import cifar10


os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# 加载 TFLite 模型
interpreter = tf.lite.Interpreter(model_path='./flatc_output/vggnet_dense_corrected_top10all_dense_tarantula_with_all_images_maxn.tflite')
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# 加载 CIFAR-10 测试数据
(_, _), (x_test, y_test) = cifar10.load_data()
x_test = x_test / 255.0
import numpy as np

correct_count = 0
total_count = 0

for i in range(len(x_test)):
    input_data = x_test[i].reshape(input_details[0]['shape']).astype(input_details[0]['dtype'])
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])
    predicted_label = tf.argmax(output_data[0]).numpy()
    true_label = y_test[i][0]
    
    if predicted_label == true_label:
        correct_count += 1
    total_count += 1

accuracy = correct_count / total_count
print("Accuracy:", accuracy)