import json
import os
import numpy as np
import math


buffer_index = 27
#buffer_index = 15 for conv5 dense_1
#buffer_index = 13 for conv5 dense
#buffer_index = 43 for resnet18 dense
#buffer_index = 27 for vggnet dense_1
#buffer_index = 25 for vggnet dense
#buffer_index = 23 for vggnet dense-1
#buffer_index = 115 for lenet dense


for correct_target in range(0, 10):
    with open('quantized_vggnet_model.json', 'r') as tf_file:
        tflite_model_dict = json.load(tf_file)
        print(type(tflite_model_dict))
        print(tflite_model_dict.keys())
        if os.path.exists('/local-data/e91868xs/resnet18_correction/vggnet_dense1_#' + str(correct_target) + 'neuron_repair_result_with_all_pictures.txt'):
            with open('/local-data/e91868xs/resnet18_correction/vggnet_dense1_#' + str(correct_target) + 'neuron_repair_result_with_all_pictures.txt', 'r') as solution:
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
        with open('/local-data/e91868xs/json/vggnet/vggnet_dense1_corrected'+str(correct_target)+'.json','w') as r:
            json.dump(tflite_model_dict,r)
        r.close()
        
            
            
            
    
