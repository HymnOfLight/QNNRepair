import os
import numpy as np
import tensorflow as tf
assert tf.__version__.startswith('2')
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.utils import img_to_array
np.set_printoptions(threshold=np.inf)


import matplotlib.pyplot as plt

interpreter = tf.lite.Interpreter(model_path="./model/densenet.tflite")
tf.lite.experimental.Analyzer.analyze(model_path="./model/densenet.tflite")
interpreter.allocate_tensors()
# Get input and output tensors.
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
pathdir = "../imagenet_val"
# Test the model on random input data.
input_shape = (224,224,3)
correct = 0
overall = 0
label = 0
same = 0
filecount = -1
result_diff = np.zeros(3950)
# print("float_result")
# # float_classify = np.loadtxt('./MobileNetV2.tflite_weight_quant_result.txt', dtype=np.int32, delimiter='\n')
# # np.append(float_classify,0)
# # print(float_classify)
# val_label = np.loadtxt('./val.txt', dtype=np.int32, delimiter='\n')

f = open("./val.txt")
val_label = np.array([], dtype=np.int32)
lines = f.readlines()
for line in lines: 
    print(line)
    val_label = np.append(val_label, int(line.split(" ")[1]))
print(val_label)
with open('densenet.tflite_float_result.txt', 'w') as f1:
    path_dir=os.listdir(pathdir)
    path_dir.sort()
    for f in sorted(path_dir):
        f = os.path.join(pathdir, f)
        print(f)
        if os.path.isfile(f):
            if os.path.splitext(f)[-1][1:] == "JPEG":
                filecount = filecount + 1
                overall = overall +1                           
                img = load_img(f)
                x = img_to_array(img)
                size = (224,224)
                x = tf.keras.preprocessing.image.smart_resize(x, size, interpolation='bilinear')                    
                x = x.reshape((1,) + x.shape) # add one extra dimension to the front
                x /= 255. # rescale by 1/255.
                input_data = x
#                 input_data = input_data.astype('uint8')
                interpreter.set_tensor(input_details[0]['index'], input_data)
                interpreter.invoke()
                # The function `get_tensor()` returns a copy of the tensor data.
                # Use `tensor()` in order to get a pointer to the tensor.
                output_data = interpreter.get_tensor(output_details[0]['index'])
                #im_class = tf.keras.applications.imagenet_utils.decode_predictions(output_data, top=5)
                #print(im_class)
#                 print(output_data)
                classifi_result = np.argmax(output_data) -1
                print(classifi_result)
                print(val_label[filecount])
                if classifi_result == val_label[filecount]:
                    correct = correct + 1
                    print("correct")
                else:
                    print("incorrect")                                   
#                             if classifi_result == float_classify[filecount]:
#                                 same = same + 1
#                                 result_diff[filecount] = 1
#                                 print("same as float model")

#                             else:
#                                 print("not the same")
                f1.write(str(classifi_result) + '\n')
#                             if im_class[0][0][0] not in d:
#                                 print("incorrect")
#                             else:
#                                 print("correct")
#                                 correct = correct +1
    print("The full integer val set classification accuracy:")
    print(correct/overall)
#     print("The fidelity accuracy:")
#     print(same/overall)
#     print(result_diff)
#     np.savetxt('result_diff_corrected_topn.txt',result_diff)
    f1.write("The full integer val set classification accuracy:")
    f1.write(str(correct/overall) + '\n')
