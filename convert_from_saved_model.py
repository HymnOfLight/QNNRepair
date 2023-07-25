import os
import numpy as np
import tensorflow as tf
import keras
from tqdm import tqdm
import cv2
import matplotlib.pyplot as plt
from tensorflow.keras.applications.resnet50 import ResNet50

model = ResNet50(weights='imagenet')



pathdir = "/local-data/e91868xs/val"
rep_ds = np.array([])
path_dir=os.listdir(pathdir)
path_dir.sort()
imagefolder_count = 0
# for d in path_dir:
#     imagefolder_count += 1
#     d = os.path.join(pathdir, d)
#     if os.path.isdir(d):
#         for f in sorted(os.listdir(d)):
#             f = os.path.join(d, f)
#             print("read:" + str(f))
#             # Read the image from the current path, change the datatype, resize the image,
#             # add batch dimension, normalize the pixel values
#             image_pixels = plt.imread(f).astype(np.float32)
#             image_pixels = cv2.resize(image_pixels, (224, 224))
#             image_pixels = np.expand_dims(image_pixels, 0)
#             image_pixels = image_pixels / 255.
# #             print(image_pixels)
#             # Append to the list
#             rep_ds = np.append(image_pixels, rep_ds)
#     if imagefolder_count >100:
#         break
             
# rep_ds = np.array(rep_ds)
test_dir = "/local-data/e91868xs/val"
def representative_dataset():
    dataset_list = tf.data.Dataset.list_files(test_dir + '/*')
    for i in range(99):
        image = next(iter(dataset_list))
        image = tf.io.read_file(image)       
        image = tf.io.decode_jpeg(image, channels=3)
        image = tf.image.resize(image, [224,224])
        image = tf.cast(image / 255., tf.float32)
        image = tf.expand_dims(image, 0)       
    # Model has only one input so each data point has one element
        yield [image]
correct = 0
overall = 0
model.summary()
# model.save("./model/ResNet50_fullprecision.h5")

converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.experimental_new_converter = True

tflite_model = converter.convert()
with open('resnet50_full_precision.tflite', 'wb') as f:
    f.write(tflite_model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.representative_dataset = representative_dataset
# print(np.array(converter.representative_dataset).shape())
converter.target_spec.supported_types = [tf.int8]
converter.inference_input_type =  tf.uint8
converter.inference_output_type =  tf.uint8

tflite_quant_model = converter.convert()
with open('./model/ResNet50_input_uint8_full_quantized.tflite', 'wb') as f:
    f.write(tflite_quant_model)