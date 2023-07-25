import numpy as np
import os
import gurobipy as gp
from gurobipy import GRB
np.set_printoptions(threshold=np.inf)

n = 512
categories = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
dir = -1
pathdir = "/local-data/e91868xs/quantized_model_conv5/"
weight_file = "./models/conv5_cifar10/sequentialdense_1MatMul.npy"
dense_weight = np.load(weight_file)
print(dense_weight.shape)
bias_file = "./models/conv5_cifar10/sequentialdense_1BiasAddReadVariableOp.npy"
bias = np.load(bias_file)
#neuron_index = 108
print(dense_weight.shape)
for neuron_index in range (0,10):
    image_count = 0
    m = gp.Model('neuron')
    vars = m.addVars(n, vtype=GRB.INTEGER, name="W")
    M = m.addVar(name = "M")
    for v in range(n):
        m.addConstr(vars[v] -M <= 0)
        m.addConstr(0 <= M + vars[v])
    m.setObjective(M, GRB.MINIMIZE)
    dir = 0
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
                if "29_weight_quant_tensor.txt" in filepath:
                    last_output = np.loadtxt(filepath, dtype=np.float32, delimiter=',')
                    classifi_result = np.argmax(last_output)                    
                    if classifi_result != index:
                        array = np.random.rand(10)
                        last_input_q = os.path.join(os.path.dirname(filepath), "28_weight_quant_tensor.txt")
                        last_input_f = os.path.join(os.path.dirname(filepath), folder_name + "resnet18_10_tensor.txt")
                        last_output_q = os.path.join(os.path.dirname(filepath), "29_weight_quant_tensor.txt")
                        last_output_f = os.path.join(os.path.dirname(filepath), folder_name + "resnet18_11_tensor.txt")
                        f_results = os.path.join(os.path.dirname(filepath), folder_name + "resnet18_11_tensor.txt")                     
                        f_classifi_result = np.argmax(f_results)
                        largest_indices = np.argsort(f_results)[-3:]
                        result_sign_array = np.ones_like(array)                    
                        result_sign_array[largest_indices] = -1
                        if f_classifi_result == index:
                            image_count = image_count + 1
                            last_input = np.loadtxt(last_input_q, dtype=np.float32, delimiter=',')
                            m.addConstr((gp.quicksum((dense_weight[neuron_index][i]+vars[i]) * last_input[i] for i in range(0, 511))+ bias[neuron_index])*result_sign_array[neuron_index]<=0)
#                         print(index)
#                         print(classifi_result)
#                         print(dense_weight[neuron_index][0])
#                         m.addConstr((gp.quicksum((dense_weight[index][i]+vars[i]) * last_input[i] for i in range(0, 512))+ bias[index])-gp.quicksum((dense_weight[classifi_result][i]+vars[i]) * last_input[i] for i in range(0, 512)+ bias[classifi_result])>=0)
            elif os.path.isdir(filepath):
                stack.append(filepath)  # 将子目录添加到堆栈中
#             if image_count > 100: #max94
#                 break
#     m.write('repair_dense_conv3_#' + str(neuron_index) + 'neuron.lp')
    m.optimize()
    if m.Status == GRB.OPTIMAL:
        solution = m.getAttr('X', vars)
        with open('/local-data/e91868xs/resnet18_correction/conv5_dense1_#' + str(neuron_index) + 'neuron_repair_result_with_all_pictures.txt', 'w') as f1:
            f1.write(str(solution))
    else:
        continue

                                                                
                        
                        
                
                