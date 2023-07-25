import subprocess
import tensorflow as tf
import glob
from PIL import Image
import os
from tensorflow.keras.datasets import cifar10

# 循环执行命令，i的序号从1到1000
for i in range(0, 999):
    command = f'flatc -b --strict-json --defaults-json -o flatc_output tflite.fbs /local-data/e91868xs/json/mobilenet_v2_1.0_224_quant_10images_corrected_suc_and_fail_#{i}_neuron.json'
    subprocess.run(command, shell=True)
with open('./mobilenet_single_neuron.txt', 'w') as tf_file:
    for correct_target in range(0, 999):
        if os.path.exists('./flatc_output/mobilenet_v2_1.0_224_quant_10images_corrected_suc_and_fail_#' + str(correct_target) + '_neuron.tflite'):
            interpreter = tf.lite.Interpreter(model_path='./flatc_output/mobilenet_v2_1.0_224_quant_10images_corrected_suc_and_fail_#' + str(correct_target) + '_neuron.tflite')
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
            tf_file.write(str(accuracy))