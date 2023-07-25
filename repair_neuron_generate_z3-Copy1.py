import numpy as np
import os
import z3
#from z3 import *
import json
z3.set_param('parallel.enable', True)
z3.set_option("parallel.threads.max", 16)

from json import JSONEncoder
class MyEncoder(JSONEncoder):
    def default(self, o):
        return o.__dict__  

o = z3.Optimize() 
np.set_printoptions(threshold=np.inf)

result_diff = np.loadtxt('./result_diff.txt', dtype=np.int32, delimiter='\n')
np.append(result_diff,0)


n = 1001
vars = z3.RealVector('x', 1001) 
M = z3.Real("M")
print(vars)
for v in range(n):
    o.add(vars[v] <= M)
    o.add(-M < vars[v])
o.minimize(M)

dir = -1
pathdir = "./output"
weight_file = "./MobileNetv2_last_tensor.npy"
conv_weight = np.load(weight_file)
print(conv_weight.shape)
bias_file = "./tensor_bias.npy"
bias = np.load(bias_file)
neuron_index = 108
for d in os.listdir(pathdir):
    d = os.path.join(pathdir, d)
    dir = dir + 1
    if os.path.isdir(d):
        for f in sorted(os.listdir(d)):           
            f = os.path.join(d, f)
            print(f)        
            if os.path.isfile(f):
                if "58_quant_tensor" in f:                   
                    if result_diff[dir] == 0:
                        last_input = np.loadtxt(f, dtype=np.float32, delimiter=',')
                        for i in range(n):                       
                            print(conv_weight[neuron_index][0][0][i])
                            print(last_input[i])
                            print(bias[i])
                            print(z3.RealVal(conv_weight[neuron_index][0][0][i]))
                            
                            #o.add(RealVal(conv_weight[neuron_index][0][0][i]) + vars[i] * RealVal(last_input[i]) + RealVal(bias[i]) >0)
                            
                            #print(gp.quicksum([(conv_weight[neuron_index][0][0][i]+vars[i]) * last_input[i] + bias[i]]))
                            #m.addConstr(gp.quicksum([(conv_weight[neuron_index][0][0][i]+vars[i]) * last_input[i] + bias[i]])>=0)
                            o.add(z3.Sum((z3.RealVal(conv_weight[neuron_index][0][0][i]) + vars[i]) * z3.RealVal(last_input[i]) + z3.RealVal(bias[i])) * z3.Sum((z3.RealVal(conv_weight[neuron_index][0][0][i])+vars[i]) * z3.RealVal(last_input[i]) + z3.RealVal(bias[i])) >=0)
                            print(z3.Sum((z3.RealVal(conv_weight[neuron_index][0][0][i]) + vars[i]) * z3.RealVal(last_input[i]) + z3.RealVal(bias[i])) * z3.Sum((z3.RealVal(conv_weight[neuron_index][0][0][i])) * z3.RealVal(last_input[i]) + z3.RealVal(bias[i])) >=0)
                            #o.add(Sum((RealVal(conv_weight[neuron_index][0][0][i]) + vars[i]) * RealVal(last_input[i]) + RealVal(bias[i]) )>=0)
    if dir > 10:
        break
result = o.check()
print(result)
mm = o.model()
print(o.model())

for d in mm:
    print("%s -> %s" % (d, mm[d]))