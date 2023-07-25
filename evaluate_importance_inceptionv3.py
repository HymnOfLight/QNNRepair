import numpy as np
import os
from math import sqrt
import math

np.set_printoptions(threshold=np.inf)


#not_equal = np.loadtxt('unequal_indexes.txt', dtype=np.intp, delimiter='\n')

pathdir = "/local-data/e91868xs/output_inceptionv3"
dir = 0 
result_diff = np.loadtxt('./result_diff.txt', dtype=np.int32, delimiter='\n')
np.append(result_diff,0)
activate_diff_all_cnas = np.zeros(1001)
activate_diff_all_cnaf = np.zeros(1001)
activate_status_quant = np.zeros(1001)
activate_status_float = np.zeros(1001)
dstar = np.zeros(1001)
tarantula = np.zeros(1001)
jaccard = np.zeros(1001)
ochiai = np.zeros(1001)
ample = np.zeros(1001)
euclid = np.zeros(1001)
wong3 = np.zeros(1001)
h = np.zeros(1001)
activate_diff = np.zeros(1001)
filecount = -1
for d in os.listdir(pathdir):
    d = os.path.join(pathdir, d)
    filecount = filecount + 1
    quant_file = 0           
    float_file = 0
    if os.path.isdir(d):
        for f in sorted(os.listdir(d)):           
            f = os.path.join(d, f)
            print(f)        
            if os.path.isfile(f):                
                if "quant" in f:
                    if "317_weight_quant_tensor" in f:
                        acti_quant = np.loadtxt(f, dtype=np.float32, delimiter=',')
                        acti_quant = [0 if x < 0 else x for x in acti_quant]
                        activate_status_quant = (np.rint(np.sign(acti_quant))).astype(int)                 
                        #print(len(activate_status_quant))
                        quant_file = 1
                if "val" in f:
                    if "inceptionv3_2_predictions_tensor" in f:
                        print(f)
                        acti_float = np.loadtxt(f, dtype=np.float32, delimiter='\n')
                        acti_float = [0 if x < 0 else x for x in acti_float]
                        activate_status_float = (np.rint(np.sign(acti_float))).astype(int)
                        activate_status_float = np.append(activate_status_float, 1)
                        #print(len(activate_status_float))
                        float_file = 1
    #print(quant_file)
    #print(float_file)
    if quant_file == float_file:
        activate_status_quant = np.append(activate_status_quant, 0)
        activate_diff = np.add(activate_status_float, activate_status_quant)
    else:
        continue
        #print(activate_diff)
    print(d)
    if result_diff[filecount] == 1:
        activate_diff_all_cnas = np.add(activate_diff, activate_diff_all_cnas)
    elif result_diff[filecount] == 0:
        activate_diff_all_cnaf = np.add(activate_diff, activate_diff_all_cnaf)
    #print(activate_diff_all_cnas)
    #print(activate_diff_all_cnaf)
print(filecount)


for i in range(0, len(activate_diff_all_cnaf)):  
    dstar[i] = (activate_diff_all_cnaf[i] * activate_diff_all_cnaf[i])/(activate_diff_all_cnas[i]+(3924-activate_diff_all_cnaf[i]))
for i in range(0, len(activate_diff_all_cnaf)):  
    jaccard[i] = activate_diff_all_cnaf[i]/(activate_diff_all_cnas[i]+(3924-activate_diff_all_cnaf[i])+ activate_diff_all_cnaf[i])
for i in range(0, len(activate_diff_all_cnaf)):  
    tarantula[i] = ((activate_diff_all_cnaf[i]) / 3924) / (((activate_diff_all_cnaf[i]) / 3924) + (activate_diff_all_cnas[i] / 3924))
    if activate_diff_all_cnaf[i] == 0:
        tarantula[i] = 0
for i in range(0, len(activate_diff_all_cnaf)):  
    ochiai[i] = activate_diff_all_cnaf[i]/sqrt((activate_diff_all_cnas[i]+(3924-activate_diff_all_cnaf[i]))*(activate_diff_all_cnaf[i]+ activate_diff_all_cnas[i])) 
    if activate_diff_all_cnaf[i] == 0:
        ochiai[i] = 0

for i in range(0, len(activate_diff_all_cnaf)):
    euclid[i] = sqrt(activate_diff_all_cnaf[i]+ 3924-activate_diff_all_cnas[i])

for i in range(0, len(activate_diff_all_cnaf)):
    ample[i] = abs(activate_diff_all_cnaf[i]/3924-activate_diff_all_cnas[i]/3924)


for i in range(0, len(activate_diff_all_cnaf)):
    h[i] = 0
    if activate_diff_all_cnas[i]<=2:
        h[i] = activate_diff_all_cnas[i]
    elif activate_diff_all_cnas[i]<=10 and activate_diff_all_cnas[i]>2:
        h[i] = 2 + 0.1*(activate_diff_all_cnas[i] - 2)
    elif activate_diff_all_cnas[i]>10:
        h[i] = 2.8 + 0.01*(activate_diff_all_cnas[i] - 10)
    
    wong3[i] = activate_diff_all_cnaf[i] - h[i]
                                            
with open('Inceptionv3_importance_result.txt', 'w') as f1:
    print("dstar")
    print(dstar)
    print(np.argmax(dstar))
    f1.write("Dstar Importance:")
    f1.write(str(np.argmax(dstar)) + '\n')
    f1.write("Dstar Importance sequence:\n")
    f1.write(str(dstar.argsort()))
    f1.write("\n")
    np.savetxt('Inceptionv3_importance_dstar.txt',dstar)
    print("jaccard")
    print(jaccard)
    print(np.argmax(jaccard))
    f1.write("jaccard Importance:")
    f1.write(str(np.argmax(jaccard)) + '\n')
    f1.write("jaccard Importance sequence:\n")
    f1.write(str(jaccard.argsort()))
    f1.write("\n")
    np.savetxt('Inceptionv3_importance_jaccard.txt',jaccard)
    print("tarantula")
    print(tarantula)
    print(np.argmax(tarantula))
    f1.write("tarantula Importance:")
    f1.write(str(np.argmax(tarantula)) + '\n')
    f1.write("tarantula Importance sequence:\n")
    f1.write(str(tarantula.argsort()))
    f1.write("\n")
    np.savetxt('Inceptionv3_importance_tarantula.txt',tarantula)
    print("ochiai")
    print(ochiai)
    print(np.argmax(ochiai))
    f1.write("ochiai Importance:")
    f1.write(str(np.argmax(ochiai)) + '\n')
    f1.write("ochiai Importance sequence:\n")
    f1.write(str(ochiai.argsort()))
    f1.write("\n")
    np.savetxt('Inceptionv3_importance_ochiai.txt',ochiai)
    print("euclid")
    print(euclid)
    print(np.argmax(euclid))
    f1.write("euclid Importance:")
    f1.write(str(np.argmax(euclid)) + '\n')
    f1.write("euclid Importance sequence:\n")
    f1.write(str(euclid.argsort()))
    f1.write("\n")
    np.savetxt('Inceptionv3_importance_euclid.txt',euclid)
    print("ample")
    print(ample)
    print(np.argmax(ample))
    f1.write("ample Importance:")
    f1.write(str(np.argmax(ample)) + '\n')
    f1.write("ample Importance sequence:\n")
    f1.write(str(ample.argsort()))
    f1.write("\n")
    np.savetxt('Inceptionv3_importance_ample.txt',ample)

    print("wong3")
    print(wong3)
    print(np.argmax(wong3))
    f1.write("wong3 Importance:")
    f1.write(str(np.argmax(wong3)) + '\n')
    f1.write("wong3 Importance sequence:\n")
    f1.write(str(wong3.argsort()))
    f1.write("\n")
    np.savetxt('Inceptionv3_importance_wong3.txt',wong3)


                
                