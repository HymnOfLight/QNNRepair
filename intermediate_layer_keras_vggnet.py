from keras import backend as K 
from keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.utils import img_to_array


import tensorflow as tf
from keract import get_activations
import numpy as np
import keras
from PIL import Image
import os
from keract import get_activations, persist_to_json_file, load_activations_from_json_file

model_path = "./original_vggnet_model.h5"
model = load_model(model_path)
categories = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
layer_outputs = [layer.output for layer in model.layers]
activation_model = tf.keras.models.Model(inputs=model.input, outputs=layer_outputs)
pathdir = "/local-data/e91868xs/cifar-10-images/cifar10/cifar10/test/"
correct = 0
overall = 0

with open('vggnet_cifar10_float_result.txt', 'w') as f1:
    for d in os.listdir(pathdir):
            d = os.path.join(pathdir, d)
            if os.path.isdir(d):
                for f in sorted(os.listdir(d)):
                    f = os.path.join(d, f)                   
                    overall = overall + 1
                    if os.path.isfile(f):
                            print(f)
                        #if os.path.splitext(f)[-1][1:] == "jpg":
                            parts = f.split("/")
                            ora_class = parts[-2]
                            print(ora_class)
                            index = categories.index(ora_class)
                            img = load_img(f)                       
                            x = img_to_array(img)
                            size = (32,32)
                            x = tf.keras.preprocessing.image.smart_resize(x, size, interpolation='bilinear')                    
                            x = x.reshape((1,) + x.shape) # add one extra dimension to the front
                            x /= 255. # rescale by 1/255.
                            prediction = model.predict(x)
                            print(np.argmax(prediction))
                            classifi_result = np.argmax(prediction)
                            f1.write(str(classifi_result) + '\n')
                            if classifi_result != index:
                                print("incorrect")
                            else:
                                print("correct")
                                correct = correct +1
                            #print(get_activations(model, x, layer_names=layer.name))
#                                     acts = get_activations(model, x, layer_names=layer.name)[layer.name]
                            #print(np.sign(acts))
                            save_path = '/local-data/e91868xs/quantized_vggnet_model/'+ ora_class + "/" + os.path.split(f)[1] + '/'
                            if not os.path.exists(save_path):
                                os.makedirs(save_path)
#                                     np.savetxt(save_path+ os.path.split(f)[1] + "_resnet18_" +str(relu_layer)+"_"+layer.name+'_tensor.txt',np.array(acts).reshape(-1),delimiter=',')
                            layer_outputs = activation_model.predict(x)

                            # Save the outputs of each layer to separate TXT files
                            for i, output in enumerate(layer_outputs):
#                                 output_filename = f'image_{image_file}_layer_{i}_output.txt'
                                np.savetxt(save_path+ os.path.split(f)[1] + '_vggnet_' +str(i)+'_tensor.txt', output.flatten())
    print("The floating-point classification accuracy:")
    print(correct/overall)
    f1.write("The floating-point classification accuracy:")
    f1.write(str(correct/overall) + '\n')





