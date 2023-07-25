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
quant_result_file = "MobileNetV2.tflite_quant_result.txt"
quant_result = np.loadtxt(quant_result_file, delimiter='\n')
exact_result_file = "./exact_results.txt"
exact_result = np.loadtxt(exact_result_file, delimiter='\n')


m = gp.Model('neuron')
vars = m.addVars(n, vtype=GRB.INTEGER, name="W")
M = m.addVar(name = "M")
for v in range(n):
    m.addConstr(vars[v] -M <= 0)
    m.addConstr(1 <=vars[v])
m.addConstr(M<=127)
m.addConstr(M>=-127)
m.setObjective(M, GRB.MINIMIZE)
dir = -1
for d in os.listdir(pathdir):
    d = os.path.join(pathdir, d)
    dir = dir + 1
    if os.path.isdir(d):
#         for f in os.listdir(d):
        for f in sorted(os.listdir(d)):           
            f = os.path.join(d, f)
            #print(f)        
            if os.path.isfile(f):
                if "58_quant_tensor" in f:                   
                    last_input = np.loadtxt(f, dtype=np.float32, delimiter=',')
                    #print(conv_weight[neuron_index][0][0])
                    if quant_result[dir] != exact_result[dir]:
                        m = gp.Model('neuron')
                        vars = m.addVars(n, vtype=GRB.INTEGER, name="W")
                        M = m.addVar(name = "M")
                        for v in range(n):
                            m.addConstr(vars[v] -M <= 0)
                            m.addConstr(1 <=vars[v])
                        m.addConstr(M<=127)
                        m.addConstr(M>=-127)
                        m.setObjective(M, GRB.MINIMIZE)
                        print("The #" +str(exact_result[dir]) + " need to be corrected")
                        q_result = int(quant_result[dir])
                        print("wrong result:" + str(q_result))
                        e_result = int(exact_result[dir])
                        print("correct result:" + str(e_result))
#                                     print(gp.quicksum((conv_weight[neuron_index][0][0][i]+vars[i]) * last_input[i]  for i in range(1, 1280))-gp.quicksum((conv_weight[e_result][0][0][i]) * last_input[i] for i in range(0, 1280))>=- (bias[neuron_index] + bias[q_result]))
                        for j in range (0,1000):
                            m.addConstr(gp.quicksum((conv_weight[e_result][0][0][i]+vars[i]) * last_input[i]  for i in range(1, 1280))-gp.quicksum((conv_weight[j][0][0][i]+vars[i]) * last_input[i] for i in range(0, 1280))>=- (bias[e_result] - bias[j]))
                        m.write('repair_#' + str(exact_result[dir]) + 'neuron.lp')
                        m.optimize()
                        if m.Status == GRB.OPTIMAL:
                            solution = m.getAttr('X', vars)
                            print(solution)
                            with open('MobileNetV2_#' + str(exact_result[dir]) + 'neuron_repair_result_with_all_pictures.txt', 'w') as f1:
                                f1.write(str(solution))
                        else:
                            continue
#                         elif neuron_index == quant_result[dir]:
#                             if quant_result[dir] != exact_result[dir]:
#                                 print("The #" +str(exact_result[dir]) + " was classified as " + str(quant_result[dir]))
#                                 q_result = int(quant_result[dir])
#                                 print("wrong result:" + str(q_result))
#                                 e_result = int(exact_result[dir])
#                                 print("correct result:" + str(e_result))
#                                 m.addConstr(gp.quicksum((conv_weight[neuron_index][0][0][i]+vars[i]) * last_input[i] for i in range(1, 1280))-gp.quicksum((conv_weight[e_result][0][0][i]) * last_input[i] for i in range(0, 1280))<=- (bias[neuron_index] - bias[q_result]))
                        #m.addConstr(0<=(gp.quicksum([(conv_weight[neuron_index][0][0][i]+vars[i]) * last_input[i] + bias[i]])) * (gp.quicksum([conv_weight[neuron_index][0][0][i] * last_input[i] + bias[i]])))
#         if dir > 1000-dirlimit: #max94
#             break


                                                                
                        
                        
                
                