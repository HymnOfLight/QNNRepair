import subprocess
import tensorflow as tf
import glob
from PIL import Image
import os
from tensorflow.keras.datasets import cifar10

# for i in range(0, 63):
#     command = f'flatc -b --strict-json --defaults-json -o flatc_output tflite.fbs /local-data/e91868xs/json/quantized_model_conv3_dense_corrected{i}.json'
#     subprocess.run(command, shell=True)
for correct_target in range(0, 63):
    if os.path.exists('./flatc_output/quantized_model_conv3_dense_corrected' + str(correct_target) + '.tflite'):
        interpreter = tf.lite.Interpreter(model_path='./flatc_output/quantized_model_conv3_dense_corrected' + str(correct_target) + '.tflite')
        interpreter.allocate_tensors()
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
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