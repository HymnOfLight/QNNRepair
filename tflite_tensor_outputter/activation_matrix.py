import numpy as np
import tensorflow as tf

import os 

g = os.walk(r"./output/layer_outputs_n01440764_tench_quant")
for path,dir_list,file_list in g:  
    for file_name in file_list:  
        if "Relu" in file_name:
            tensor_num=file_name.split("_")[0]
            print(tensor_num)
            relu_tensor = np.loadtxt('./output/layer_outputs_n01440764_tench_quant/'+tensor_num+'_tensor.txt', dtype=np.float32, delimiter=',')
            relu_activate_status = (np.rint(np.sign(relu_tensor))).astype(int)
            np.savetxt('./output/layer_outputs_n01440764_tench_quant/'+tensor_num+'_activate_status_tensor.txt',relu_activate_status)
            print(relu_activate_status)
            print(np.average(relu_activate_status, axis =None, weights=None))