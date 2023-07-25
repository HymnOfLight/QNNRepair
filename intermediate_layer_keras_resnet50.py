from keras import backend as K 
from keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img
from keras_preprocessing.image import img_to_array

import tensorflow as tf
from keract import get_activations
import numpy as np
import keras
from PIL import Image
import os
from keract import get_activations, persist_to_json_file, load_activations_from_json_file
model2 = tf.keras.applications.mobilenet_v2.MobileNetV2(
    input_shape=None,
    alpha=1.0,
    include_top=True,
    weights='imagenet',
    input_tensor=None,
    pooling=None,
    classes=1000,
    classifier_activation='softmax'
)

model = tf.keras.applications.resnet50.ResNet50(
    include_top=True,
    weights='imagenet',
    input_tensor=None,
    input_shape=None,
    pooling=None,
    classes=1000,
)
#model.load_weights('.\\models\\mobilenet_1_0_224_tf.h5')
#model = load_model('./models/tensorflow/mobilenetv2/mobilenetv2.h5')
layer_outputs = [layer.output for layer in model.layers[1:]]
visual_model = tf.keras.models.Model(inputs = model.input, outputs = layer_outputs)
pathdir = "../imagemini-1000/imagenet-mini/val"
correct = 0
overall = 0
model.summary()
model.save("./modelresnet50fullprecision.h5")
converter = tf.lite.TFLiteConverter.from_keras_model(model)

converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.inference_input_type =  tf.float32
converter.inference_output_type =  tf.float32
tflite_quant_model = converter.convert()
with open('./model/resnet50_uint8_weightquantized.tflite', 'wb') as f:
    f.write(tflite_quant_model)

# def representative_dataset_gen():
#   for _ in range(num_calibration_steps):
#     # Get sample input data as a numpy array in a method of your choosing.
#     yield [input]

# converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
# converter.optimizations = [tf.lite.Optimize.DEFAULT]
# converter.representative_dataset = representative_dataset_gen
# tflite_quant_model = converter.convert()
with open('resnet50_float_result.txt', 'w') as f1:
    for layer in model.layers:
        #print(layer.name)
        layer_k = model.get_layer(layer.name)
        weights = layer_k.get_weights()
        weights_path = ".\\output\\ResNet50\\"
        if not os.path.exists(weights_path):
            os.makedirs(weights_path)
        #print(weights)
        np.savetxt(weights_path + layer.name + "_"+"resnet50.txt", np.array(weights).reshape(-1),delimiter=',', fmt='%s')
    for d in os.listdir(pathdir):
            d = os.path.join(pathdir, d)
            if os.path.isdir(d):
                for f in sorted(os.listdir(d)):
                    f = os.path.join(d, f)
                    ora_class = os.path.split(d)[1]
                    print(ora_class)
                    overall = overall + 1
                    if os.path.isfile(f):
                        #if os.path.splitext(f)[-1][1:] == "jpg":
                            img = load_img(f)                       
                            x = img_to_array(img)
                            size = (224,224)
                            x = tf.keras.preprocessing.image.smart_resize(x, size, interpolation='bilinear')                    
                            x = x.reshape((1,) + x.shape) # add one extra dimension to the front
                            #x /= 255. # rescale by 1/255.
                            prediction = model.predict(x)
                            im_class = tf.keras.applications.imagenet_utils.decode_predictions(prediction, top=5)
                            print(np.argmax(prediction))
                            classifi_result = np.argmax(prediction)
                            f1.write(str(classifi_result) + '\n')
                            print(d)
                            print(im_class)
                            if im_class[0][0][0] not in d:
                                print("incorrect")
                            else:
                                print("correct")
                                correct = correct +1
                            relu_layer = 0
                            for layer in model.layers:                          
                                if 'global_average_pooling2d' in layer.name:
                                    relu_layer+=1                               
                                    #print(get_activations(model, x, layer_names=layer.name))
                                    acts = get_activations(model, x, layer_names=layer.name)[layer.name]
                                    #print(np.sign(acts))
                                    save_path = ".\\output\\layer_outputs_" + ora_class +"_intermediate_float\\"
                                    if not os.path.exists(save_path):
                                        os.makedirs(save_path)
                                    np.savetxt(save_path+ os.path.split(f)[1] + "_" +str(relu_layer)+"_"+layer.name+'_tensor.txt',np.array(acts).reshape(-1),delimiter=',')
                                    np.savetxt(save_path+os.path.split(f)[1] + "_" +str(relu_layer)+"_"+layer.name+'_activation_status_tensor.txt',np.sign(np.array(acts)).reshape(-1),delimiter=',')
    print("The floating-point classification accuracy:")
    print(correct/overall)
    f1.write("The floating-point classification accuracy:")
    f1.write(str(correct/overall) + '\n')





