import numpy as np
import os
import gurobipy as gp
from gurobipy import GRB
# m = gp.Model('neuron')
np.set_printoptions(threshold=np.inf)

result_diff = np.loadtxt('./result_diff.txt', dtype=np.int32, delimiter='\n')
np.append(result_diff,0)

n = 1280
# vars = m.addVars(n, vtype=GRB.CONTINUOUS, name="W")
# M = m.addVar(name = "M")
# for v in range(n):
#     m.addConstr(vars[v] -M <= 0)
#     m.addConstr(0 <= M + vars[v])
# m.setObjective(M, GRB.MINIMIZE)

dir = -1
pathdir = "./output"
weight_file = "./MobileNetv2_last_tensor.npy"
conv_weight = np.load(weight_file)
print(conv_weight.shape)
bias_file = "./tensor_bias.npy"
bias = np.load(bias_file)
#neuron_index = 108

for neuron_index in range (1,1000):
    image_count = 0
    m = gp.Model('neuron')
    vars = m.addVars(n, vtype=GRB.INTEGER, name="W")
    M = m.addVar(name = "M")
    for v in range(n):
        m.addConstr(vars[v] -M <= 0)
        m.addConstr(0 <= M + vars[v])
    m.setObjective(M, GRB.MINIMIZE)
    dir = 0
    for d in os.listdir(pathdir):
        d = os.path.join(pathdir, d)
        dir = dir + 1
        if os.path.isdir(d):
    #         for f in os.listdir(d):
            for f in os.listdir(d):           
                f = os.path.join(d, f)
                #print(f)        
                if os.path.isfile(f):
                    if "58_quant_tensor" in f:                   
                        if result_diff[dir] == 0:
                            image_count = image_count + 1
                            last_input = np.loadtxt(f, dtype=np.float32, delimiter=',')
                            #m.addConstr(gp.quicksum((conv_weight[neuron_index][0][0][i]+vars[i]) * last_input[i] + bias[neuron_index] for i in range(1, 1280))*gp.quicksum((conv_weight[neuron_index][0][0][i]) * last_input[i] + bias[neuron_index] for i in range(0, 1280))<=0)
                            m.addConstr((gp.quicksum((conv_weight[neuron_index][0][0][i]+vars[i]) * last_input[i] for i in range(0, 1279))+ bias[neuron_index])*(gp.quicksum((conv_weight[neuron_index][0][0][i]) * last_input[i] for i in range(0, 1279))+ bias[neuron_index])<=0)

                                #m.addConstr(0<=(gp.quicksum([(conv_weight[neuron_index][0][0][i]+vars[i]) * last_input[i] + bias[i]])) * (gp.quicksum([conv_weight[neuron_index][0][0][i] * last_input[i] + bias[i]])))
        if image_count > 10: #max94
            break
    #m.write('repair_#' + str(neuron_index) + 'neuron.lp')
    m.optimize()

    if m.Status == GRB.OPTIMAL:
        solution = m.getAttr('X', vars)
        print(solution)
        with open('./correction/MobileNetV2_#' + str(neuron_index) + 'neuron_repair_result_with_10_pictures.txt', 'w') as f1:
            f1.write(str(solution))
    else:
        continue

                                                                
                        
                        
                
                