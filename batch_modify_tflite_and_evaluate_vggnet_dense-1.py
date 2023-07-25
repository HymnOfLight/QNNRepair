import subprocess
import tensorflow as tf
import glob
from PIL import Image
import os
from tensorflow.keras.datasets import cifar10

def find_tflite_files(folder_path):
    tflite_files = []
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if "vggnet_dense-1_corrected" in file and file.endswith(".tflite"):
                tflite_files.append(file)
    return tflite_files
directory = '/local-data/e91868xs/vggnet/json'  
command_template = 'flatc -b --strict-json --defaults-json -o flatc_output tflite.fbs /local-data/e91868xs/vggnet/json/'
for filename in os.listdir(directory):
    if filename.endswith('.json'):
        command = command_template+filename
        print(f"Executing command: {command}")
        subprocess.run(command, shell=True)
# for i in range(0, 10):
#     command = f'flatc -b --strict-json --defaults-json -o flatc_output tflite.fbs /local-data/e91868xs/json/vggnet/vggnet_dense1_corrected{i}.json'
#     subprocess.run(command, shell=True)
folder_path = "./flatc_output"
result = find_tflite_files(folder_path)
with open('./vggnet_dense-1_all_targets_neuron.txt', 'w') as tf_file:
    for file_name in result:        
        full_name = folder_path + '/' + file_name
        print(full_name)
        if os.path.exists(full_name):
            interpreter = tf.lite.Interpreter(full_name)
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
            print(file_name)
            print(str(accuracy)+'\n')
            tf_file.write(file_name)
            tf_file.write(' '+str(accuracy)+'\n')
        else:
            print('tflite doesnt exist')