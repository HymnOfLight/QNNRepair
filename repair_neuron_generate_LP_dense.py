import numpy as np
import os
import gurobipy as gp
from gurobipy import GRB
np.set_printoptions(threshold=np.inf)

n = 16384
categories = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
dir = -1
pathdir = "/local-data/e91868xs/output_conv3/"
weight_file = "./models/conv3_cifar10/sequential-dense-MatMul-weight.npy"
dense_weight = np.load(weight_file)
print(dense_weight.shape)
bias_file = "./models/conv3_cifar10/sequential-dense-MatMul-bias.npy"
bias = np.load(bias_file)
#neuron_index = 108
print(dense_weight.shape)
for neuron_index in range (46,63):
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
                if "16_weight_quant_tensor" in filepath:
                    last_output = np.loadtxt(filepath, dtype=np.float32, delimiter=',')
                    classifi_result = np.argmax(last_output)
                    if classifi_result != index:
                        last_input_q = os.path.join(os.path.dirname(filepath), "15_weight_quant_tensor.txt")
                        last_input_f = os.path.join(os.path.dirname(filepath), folder_name + "_conv3_4_flatten_tensor")
                        last_output_q = os.path.join(os.path.dirname(filepath), "16_weight_quant_tensor.txt")
                        last_output_f = os.path.join(os.path.dirname(filepath), folder_name + "_conv3_5_flatten_tensor")
                        f_results = os.path.join(os.path.dirname(filepath), folder_name + "_conv3_6_dense_1_tensor")                     
                        f_classifi_result = np.argmax(f_results)
                        if f_classifi_result == index:
                            image_count = image_count + 1
                            last_input = np.loadtxt(last_input_q, dtype=np.float32, delimiter=',')
                            m.addConstr((gp.quicksum((dense_weight[neuron_index][i]+vars[i]) * last_input[i] for i in range(0, 16384))+ bias[neuron_index])*last_output[neuron_index]>=0)
            elif os.path.isdir(filepath):
                stack.append(filepath)  # 将子目录添加到堆栈中
            if image_count > 100: #max94
                break
#     m.write('repair_dense_conv3_#' + str(neuron_index) + 'neuron.lp')
    m.optimize()
    if m.Status == GRB.OPTIMAL:
        solution = m.getAttr('X', vars)
        with open('/local-data/e91868xs/conv3_cifar/correction/dense_conv3_#' + str(neuron_index) + 'neuron_repair_result_with_10_pictures.txt', 'w') as f1:
            f1.write(str(solution))
    else:
        continue

                                                                
                        
                        
                
                