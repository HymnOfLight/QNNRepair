import os
import numpy as np
import tensorflow as tf
assert tf.__version__.startswith('2')
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.utils import img_to_array


import matplotlib.pyplot as plt

interpreter = tf.lite.Interpreter(model_path="./model/resnet50_uint8_weightquantized.tflite")
interpreter.allocate_tensors()
# Get input and output tensors.
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
pathdir = "../imagemini-1000/imagenet-mini/val"
# Test the model on random input data.
input_shape = (224,224,3)
correct = 0
overall = 0
with open('Resnet50weightquantized.tflite_quant_result.txt', 'w') as f1:
    for d in os.listdir(pathdir):
            d = os.path.join(pathdir, d)
            if os.path.isdir(d):
                for f in sorted(os.listdir(d)):
                    f = os.path.join(d, f)
                    print(f)
                    if os.path.isfile(f):
                        if os.path.splitext(f)[-1][1:] == "JPEG":
                            overall = overall +1
                            img = load_img(f)
                            x = img_to_array(img)
                            size = (224,224)
                            x = tf.keras.preprocessing.image.smart_resize(x, size, interpolation='bilinear')                    
                            x = x.reshape((1,) + x.shape) # add one extra dimension to the front
                            #x /= 255. # rescale by 1/255.
                            input_data = x
                            interpreter.set_tensor(input_details[0]['index'], input_data)
                            interpreter.invoke()
                            # The function `get_tensor()` returns a copy of the tensor data.
                            # Use `tensor()` in order to get a pointer to the tensor.
                            output_data = interpreter.get_tensor(output_details[0]['index'])
                            im_class = tf.keras.applications.imagenet_utils.decode_predictions(output_data, top=5)
                            print(im_class)
                            classifi_result = np.argmax(output_data)
                            print(classifi_result)                            
                            f1.write(str(classifi_result) + '\n')
                            if im_class[0][0][0] not in d:
                                print("incorrect")
                            else:
                                print("correct")
                                correct = correct +1
    print("The floating-point classification accuracy:")
    print(correct/overall)
    f1.write("The floating-point classification accuracy:")
    f1.write(str(correct/overall) + '\n')
