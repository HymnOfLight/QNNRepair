import json
import os
import numpy as np
import math
import random

def get_indices_of_largest_elements(arr, n):
    # Create a list of tuples with element-value pairs
    elements_with_values = [(index, element) for index, element in enumerate(arr)]

    # Sort the list in descending order based on element values
    elements_with_values.sort(key=lambda x: x[1], reverse=True)

    # Get the indices of the n largest elements
    indices = [pair[0] for pair in elements_with_values[:n]]

    return indices

topn_values = [1, 5, 10, 100, 512]
dstar_files = ["./vggnet_dense-1_dstar.txt", "./vggnet_dense-1_tarantula.txt", "./vggnet_dense-1_ochiai.txt", "./vggnet_dense-1_euclid.txt", "./vggnet_dense-1_ample.txt", "./vggnet_dense-1_wong3.txt", "./vggnet_dense-1_jaccard.txt"]
n = 512
buffer_index = 23



for topn in topn_values:
    for dstar_file in dstar_files:
        print(f"topn: {topn}")
        dstar_result = np.loadtxt(dstar_file, delimiter='\n')
        maxn = get_indices_of_largest_elements(dstar_result, n)
        #correct_target = 108

        corrected_neuron = 0
        print(f"dstar_file: {dstar_file}")
        with open('./quantized_vggnet_model.json', 'r') as tf_file:
            tflite_model_dict = json.load(tf_file)
            print(type(tflite_model_dict))
            print(tflite_model_dict.keys())
            for correct_target in maxn: 
                if os.path.exists('/local-data/e91868xs/vggnet/correction/dense-1_vggnet_#' + str(correct_target) + 'neuron_repair_result_with_all_pictures.txt'):
                    with open('/local-data/e91868xs/vggnet/correction/dense-1_vggnet_#' + str(correct_target) + 'neuron_repair_result_with_all_pictures.txt', 'r') as solution:
                        print("correcting #" + str(correct_target))
                        solution_dic = eval(solution.read())
                        #print(solution_dic[0])
                        #print("original weight")
                        for i in range(0, 2048): #139520
                            weight = tflite_model_dict['buffers'][buffer_index]['data'][correct_target*2048 + i]
                            #print(weight)
                            delta = math.ceil(solution_dic[i])
                            #print(delta)
                            if delta <127:
                                if weight + delta >255:
                                    tflite_model_dict['buffers'][buffer_index]['data'][correct_target*2048 + i] = weight+delta-256
                                else:
                                    tflite_model_dict['buffers'][buffer_index]['data'][correct_target*2048 + i] = weight+delta
                        corrected_neuron = corrected_neuron +1
                if corrected_neuron == topn:
                    break
                #         print("modified weight")
                #         for i in range(0, 1000):
                #             weight = tflite_model_dict['buffers'][2]['data'][correct_target*1280 + i]
                #             print(weight)
            start_index = dstar_file.find('_') + 1  
            end_index = dstar_file.find('.txt')
            if start_index != -1 and end_index != -1:
                dstar_substring = dstar_file[start_index:end_index]
            with open('/local-data/e91868xs/vggnet/json/vggnet_dense-1_corrected_top'+ str(topn) + 'all_' + dstar_substring + '_with_all_images_maxn.json','w') as r:
                json.dump(tflite_model_dict,r)
            r.close()        

    with open('./quantized_vggnet_model.json', 'r') as tf_file:
        tflite_model_dict = json.load(tf_file)
        print(type(tflite_model_dict))
        print(tflite_model_dict.keys())
        for random_repeat in range(0,20):
            random_numbers = []
            for i in range(512):
                random_numbers.append(random.randint(1, 512))
            print(f"random_time: {random_repeat}")
            for correct_target in random_numbers:                
                print(f"correcting target: {correct_target}")
                if os.path.exists('/local-data/e91868xs/vggnet/correction/dense-1_vggnet_#' + str(correct_target) + 'neuron_repair_result_with_all_pictures.txt'):
                    with open('/local-data/e91868xs/vggnet/correction/dense-1_vggnet_#' + str(correct_target) + 'neuron_repair_result_with_all_pictures.txt', 'r') as solution:
                        print("correcting #" + str(correct_target))
                        solution_dic = eval(solution.read())
                        #print(solution_dic[0])
                        #print("original weight")
                        for i in range(0, 512): #139520
                            weight = tflite_model_dict['buffers'][buffer_index]['data'][correct_target*2048 + i]
                            #print(weight)
                            delta = math.ceil(solution_dic[i])
                            #print(delta)
                            if delta <127:
                                if weight + delta >255:
                                    tflite_model_dict['buffers'][buffer_index]['data'][correct_target*2048 + i] = weight+delta-256
                                else:
                                    tflite_model_dict['buffers'][buffer_index]['data'][correct_target*2048 + i] = weight+delta
                        corrected_neuron = corrected_neuron +1
                if corrected_neuron == topn:
                    break
                #         print("modified weight")
                #         for i in range(0, 1000):
                #             weight = tflite_model_dict['buffers'][2]['data'][correct_target*1280 + i]
                #             print(weight)
            start_index = dstar_file.find('_') + 1  
            end_index = dstar_file.find('.txt')
            if start_index != -1 and end_index != -1:
                dstar_substring = dstar_file[start_index:end_index]
            with open('/local-data/e91868xs/vggnet/json/vggnet_dense-1_corrected_top'+ str(topn) + 'random_#' + str(random_repeat) + 'all_with_all_images_maxn.json','w') as r:
                json.dump(tflite_model_dict,r)
            r.close()

            
            
            
    
