import numpy as np
import os
from math import sqrt
import math

# np.set_printoptions(threshold=np.inf)


#not_equal = np.loadtxt('unequal_indexes.txt', dtype=np.intp, delimiter='\n')

pathdir = "/local-data/e91868xs/quantized_model_conv5"
dir = 0 
categories = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]

activate_diff_all_cnas = np.zeros(10)
activate_diff_all_cnaf = np.zeros(10)
activate_status_quant = np.zeros(10)
activate_status_float = np.zeros(10)
dstar = np.zeros(10)
tarantula = np.zeros(10)
jaccard = np.zeros(10)
ochiai = np.zeros(10)
ample = np.zeros(10)
euclid = np.zeros(10)
wong3 = np.zeros(10)
h = np.zeros(10)
activate_diff = np.zeros(10)
filecount = -1
stack = [pathdir]
while stack:
    current_dir = stack.pop()
    for filename in os.listdir(current_dir):
        filepath = os.path.join(current_dir, filename)
        if os.path.isfile(filepath) and filepath.endswith(".txt"):               
            parts = filepath.split("/")
            ora_class = parts[-3]
            folder_name = parts[-2]
            index = categories.index(ora_class)
            if "47_weight_quant_tensor.txt" in filepath:
                last_output = np.loadtxt(filepath, dtype=np.float32, delimiter=',')
                classifi_result = np.argmax(last_output)                    
                if classifi_result != index:
                    array = np.random.rand(10)
                    last_input_q = os.path.join(os.path.dirname(filepath), "28_weight_quant_tensor.txt")
                    last_input_f = os.path.join(os.path.dirname(filepath), folder_name + "_resnet18_10_tensor.txt")
                    last_output_q = os.path.join(os.path.dirname(filepath), "29_weight_quant_tensor.txt")
                    last_output_f = os.path.join(os.path.dirname(filepath), folder_name + "_resnet18_11_tensor.txt")
                    f_results = np.loadtxt(os.path.join(os.path.dirname(filepath), folder_name + "_resnet18_11_tensor.txt"), dtype=np.float32, delimiter='\n')
                    q_results = np.loadtxt(os.path.join(os.path.dirname(filepath), "29_weight_quant_tensor.txt"),dtype=np.float32, delimiter=',') 
                    f_classifi_result = np.argmax(f_results)
                    q_classifi_result = np.argmax(q_results)
                    acti_quant = np.loadtxt(last_output_q, dtype=np.float32, delimiter=',')
                    acti_quant = [0 if x < 0 else x for x in acti_quant]
                    activate_status_quant = (np.rint(np.sign(acti_quant))).astype(int)
                    acti_float = np.loadtxt(last_output_f, dtype=np.float32, delimiter='\n')
                    acti_float = [0 if x < 0 else x for x in acti_float]
                    activate_status_float = (np.rint(np.sign(acti_float))).astype(int)
                    activate_diff = np.add(activate_status_float, activate_status_quant)
#                         activate_status_float = np.append(activate_status_float, 1)
                    if f_classifi_result == q_classifi_result:
                        activate_diff_all_cnas = np.add(activate_diff, activate_diff_all_cnas)
                    else:
                        activate_diff_all_cnaf = np.add(activate_diff, activate_diff_all_cnaf)
                    
        elif os.path.isdir(filepath):
            stack.append(filepath)  


for i in range(0, len(activate_diff_all_cnaf)):  
    dstar[i] = (activate_diff_all_cnaf[i] * activate_diff_all_cnaf[i])/(activate_diff_all_cnas[i]+(10000-activate_diff_all_cnaf[i]))
for i in range(0, len(activate_diff_all_cnaf)):  
    jaccard[i] = activate_diff_all_cnaf[i]/(activate_diff_all_cnas[i]+(10000-activate_diff_all_cnaf[i])+ activate_diff_all_cnaf[i])
for i in range(0, len(activate_diff_all_cnaf)):  
    tarantula[i] = ((activate_diff_all_cnaf[i]) / 10000) / (((activate_diff_all_cnaf[i]) / 10000) + (activate_diff_all_cnas[i] / 10000))
    if activate_diff_all_cnaf[i] == 0:
        tarantula[i] = 0
for i in range(0, len(activate_diff_all_cnaf)):  
    ochiai[i] = activate_diff_all_cnaf[i]/sqrt((activate_diff_all_cnas[i]+(10000-activate_diff_all_cnaf[i]))*(activate_diff_all_cnaf[i]+ activate_diff_all_cnas[i])) 
    if activate_diff_all_cnaf[i] == 0:
        ochiai[i] = 0

for i in range(0, len(activate_diff_all_cnaf)):
    euclid[i] = sqrt(activate_diff_all_cnaf[i]+ 10000-activate_diff_all_cnas[i])

for i in range(0, len(activate_diff_all_cnaf)):
    ample[i] = abs(activate_diff_all_cnaf[i]/10000-activate_diff_all_cnas[i]/10000)


for i in range(0, len(activate_diff_all_cnaf)):
    h[i] = 0
    if activate_diff_all_cnas[i]<=2:
        h[i] = activate_diff_all_cnas[i]
    elif activate_diff_all_cnas[i]<=10 and activate_diff_all_cnas[i]>2:
        h[i] = 2 + 0.1*(activate_diff_all_cnas[i] - 2)
    elif activate_diff_all_cnas[i]>10:
        h[i] = 2.8 + 0.01*(activate_diff_all_cnas[i] - 10)
    
    wong3[i] = activate_diff_all_cnaf[i] - h[i]
                                            
with open('conv5_dense_1_importance_result.txt', 'w') as f1:
    print("dstar")
#     print(dstar)
#     print(np.argmax(dstar))
    f1.write("Dstar Importance:")
    f1.write(str(np.argmax(dstar)) + '\n')
    f1.write("Dstar Importance sequence:\n")
    f1.write(str(dstar.argsort()))
    f1.write("\n")
    np.savetxt('conv5_dense_1_dstar.txt',dstar)
    print("jaccard")
#     print(jaccard)
#     print(np.argmax(jaccard))
    f1.write("jaccard Importance:")
    f1.write(str(np.argmax(jaccard)) + '\n')
    f1.write("jaccard Importance sequence:\n")
    f1.write(str(jaccard.argsort()))
    f1.write("\n")
    np.savetxt('conv5_dense_1_jaccard.txt',jaccard)
    print("tarantula")
#     print(tarantula)
#     print(np.argmax(tarantula))
    f1.write("tarantula Importance:")
    f1.write(str(np.argmax(tarantula)) + '\n')
    f1.write("tarantula Importance sequence:\n")
    f1.write(str(tarantula.argsort()))
    f1.write("\n")
    np.savetxt('conv5_dense_1_tarantula.txt',tarantula)
    print("ochiai")
#     print(ochiai)
#     print(np.argmax(ochiai))
    f1.write("ochiai Importance:")
    f1.write(str(np.argmax(ochiai)) + '\n')
    f1.write("ochiai Importance sequence:\n")
    f1.write(str(ochiai.argsort()))
    f1.write("\n")
    np.savetxt('conv5_dense_1_ochiai.txt',ochiai)
    print("euclid")
#     print(euclid)
#     print(np.argmax(euclid))
    f1.write("euclid Importance:")
    f1.write(str(np.argmax(euclid)) + '\n')
    f1.write("euclid Importance sequence:\n")
    f1.write(str(euclid.argsort()))
    f1.write("\n")
    np.savetxt('conv5_dense_1_euclid.txt',euclid)
    print("ample")
#     print(ample)
#     print(np.argmax(ample))
    f1.write("ample Importance:")
    f1.write(str(np.argmax(ample)) + '\n')
    f1.write("ample Importance sequence:\n")
    f1.write(str(ample.argsort()))
    f1.write("\n")
    np.savetxt('conv5_dense_1_ample.txt',ample)

    print("wong3")
#     print(wong3)
#     print(np.argmax(wong3))
    f1.write("wong3 Importance:")
    f1.write(str(np.argmax(wong3)) + '\n')
    f1.write("wong3 Importance sequence:\n")
    f1.write(str(wong3.argsort()))
    f1.write("\n")
    np.savetxt('conv5_dense_1_wong3.txt',wong3)


                
                
