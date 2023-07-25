import json
import os
import numpy as np
import math

#correct_target = 108
buffer_index = 131
with open('./json/mobilenet_v2_1.0_224_quant.json', 'r') as tf_file:
    tflite_model_dict = json.load(tf_file)
    print(type(tflite_model_dict))
    print(tflite_model_dict.keys())
    for correct_target in range(0,1000):
        if os.path.exists('./correction/MobileNetV2_#' + str(correct_target) + 'neuron_repair_result_with_1000_pictures_suc_and_fail.txt'):
            with open('./correction/MobileNetV2_#' + str(correct_target) + 'neuron_repair_result_with_1000_pictures_suc_and_fail.txt', 'r') as solution:
                solution_dic = eval(solution.read())
                #print(solution_dic[0])
                print("original weight")
                for i in range(0, 999): #139520
                    weight = tflite_model_dict['buffers'][buffer_index]['data'][correct_target*1280 + i]
                    print(weight)
                    delta = math.ceil(solution_dic[i])
                    print(delta)
                    if delta <127:
                        if weight + delta >255:
                            tflite_model_dict['buffers'][buffer_index]['data'][correct_target*1280 + i] = weight+delta-256
                        else:
                            tflite_model_dict['buffers'][buffer_index]['data'][correct_target*1280 + i] = weight+delta
        #         print("modified weight")
        #         for i in range(0, 1000):
        #             weight = tflite_model_dict['buffers'][2]['data'][correct_target*1280 + i]
        #             print(weight)
    with open('./json/mobilenet_v2_1.0_224_quant_1000images_corrected_suc_and_fail.json','w') as r:
        json.dump(tflite_model_dict,r)
    r.close()
        
            
            
            
    
