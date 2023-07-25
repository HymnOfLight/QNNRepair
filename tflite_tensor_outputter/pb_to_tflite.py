from keras import backend as K 
from keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
import tensorflow as tf
from keract import get_activations
import numpy as np
from keract import get_activations, persist_to_json_file, load_activations_from_json_file


model = load_model('../models/tensorflow/mobilenetv2/mobilenetv2.h5')
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()
tflite_name = "mobilenetv2"
open(tflite_name, "wb").write(tflite_model)