import json
import os
import numpy as np
import math
import random

topn = 5
dstar_file = "./logs/quantized_model_conv5.json"
dstar_result = np.loadtxt(dstar_file, delimiter='\n')
minn = dstar_result.argsort()
maxn = minn[::-1]

random_numbers = []

for i in range(100):
    random_numbers.append(random.randint(1, 1000))

#correct_target = 108
buffer_index = 15
corrected_neuron = 0
with open('./json/mobilenet_v2_1.0_224_quant.json', 'r') as tf_file:
    tflite_model_dict = json.load(tf_file)
    print(type(tflite_model_dict))
    print(tflite_model_dict.keys())
    for correct_target in maxn:        
        if os.path.exists('./correction/MobileNetV2_#' + str(correct_target) + 'neuron_repair_result_with_all_pictures.txt'):
            with open('./correction/MobileNetV2_#' + str(correct_target) + 'neuron_repair_result_with_all_pictures.txt', 'r') as solution:
                print("correcting #" + str(correct_target))
                solution_dic = eval(solution.read())
                #print(solution_dic[0])
                #print("original weight")
                for i in range(0, 999): #139520
                    weight = tflite_model_dict['buffers'][buffer_index]['data'][correct_target*1280 + i]
                    #print(weight)
                    delta = math.ceil(solution_dic[i])
                    #print(delta)
                    if delta <127:
                        if weight + delta >255:
                            tflite_model_dict['buffers'][buffer_index]['data'][correct_target*1280 + i] = weight+delta-256
                        else:
                            tflite_model_dict['buffers'][buffer_index]['data'][correct_target*1280 + i] = weight+delta
                corrected_neuron = corrected_neuron +1
        if corrected_neuron > topn:
            break
        #         print("modified weight")
        #         for i in range(0, 1000):
        #             weight = tflite_model_dict['buffers'][2]['data'][correct_target*1280 + i]
        #             print(weight)
    with open('mobilenet_v2_1.0_224_quant_corrected_top10_random10_with_all_images_maxn.json','w') as r:
        json.dump(tflite_model_dict,r)
    r.close()
        
            
            
            
    
