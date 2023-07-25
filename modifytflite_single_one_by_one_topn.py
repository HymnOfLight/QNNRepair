import json
import os
import numpy as np
import math
import random

topn = 63
dstar_file = "./logs/Mobilenetv2_euclid.txt"
dstar_result = np.loadtxt(dstar_file, delimiter='\n')
minn = dstar_result.argsort()
maxn = minn[::-1]

for j in range(0, 10):
    random_numbers = []
    for i in range(64):
        random_numbers.append(random.randint(0, 63))
    buffer_index = 7
    corrected_neuron = 0
    with open('quantized_model_conv3.json', 'r') as tf_file:
        tflite_model_dict = json.load(tf_file)
        print(type(tflite_model_dict))
        print(tflite_model_dict.keys())
        for correct_target in random_numbers:
            if os.path.exists('/local-data/e91868xs/conv3_cifar/correction/dense_conv3_#' + str(correct_target) + 'neuron_repair_result_with_10_pictures.txt'):
                with open('/local-data/e91868xs/conv3_cifar/correction/dense_conv3_#' + str(correct_target) + 'neuron_repair_result_with_10_pictures.txt', 'r') as solution:
                    solution_dic = eval(solution.read())
                    #print(solution_dic[0])
                    print("original weight")
                    for i in range(0, 63): #139520
                        weight = tflite_model_dict['buffers'][buffer_index]['data'][correct_target*64 + i]
                        delta = math.ceil(solution_dic[i])                   
                        if delta <127:
                            if weight + delta >255:
                                tflite_model_dict['buffers'][buffer_index]['data'][correct_target*64 + i] = weight+delta-256
                            else:
                                tflite_model_dict['buffers'][buffer_index]['data'][correct_target*64 + i] = weight+delta
            #         print("modified weight")
            #         for i in range(0, 1000):
            #             weight = tflite_model_dict['buffers'][2]['data'][correct_target*1280 + i]
            #             print(weight)
            corrected_neuron = corrected_neuron +1
            if corrected_neuron > topn:
                break
        with open('/local-data/e91868xs/json/quantized_model_conv3_dense_randomtop63_corrected'+str(j)+'.json','w') as r:
#         with open('/local-data/e91868xs/json/quantized_model_conv3_dense_all_corrected.json','w') as r:
            json.dump(tflite_model_dict,r)
        r.close()
        
            
            
            
    
