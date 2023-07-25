import numpy as np
import tensorflow as tf
import os
import numpy as np
import scipy.sparse as sp
from gurobipy import *

def tarantula_analysis(trainable_layers, scores, num_cf, num_uf, num_cs, num_us, suspicious_num):
    """
    More information on Tarantula fault localization technique can be found in
    Abreu, Rui, Peter Zoeteweij, and Arjan JC Van Gemund. "An evaluation of similarity coefficients for software fault localization." 2006 12th Pacific Rim International Symposium on Dependable Computing (PRDC'06). IEEE, 2006.
    """
    def tarantula(i, j):
        return float(float(num_cf[i][j]) / (num_cf[i][j] + num_uf[i][j])) / \
            (float(num_cf[i][j]) / (num_cf[i][j] + num_uf[i][j]) + float(num_cs[i][j]) / (num_cs[i][j] + num_us[i][j]))

    return scores_with_foo(trainable_layers, scores, num_cf, num_uf, num_cs, num_us, suspicious_num, tarantula)

def dstar_analysis(trainable_layers, scores, num_cf, num_uf, num_cs, num_us, suspicious_num, star):
    """
    More information on DStar fault localization technique can be found in
    Wong, W. Eric, et al. "The DStar method for effective software fault localization." IEEE Transactions on Reliability 63.1 (2013): 290-308.
    """

    def dstar(i, j):
        return float(num_cf[i][j]**star) / (num_cs[i][j] + num_uf[i][j])

    return scores_with_foo(trainable_layers, scores, num_cf, num_uf, num_cs, num_us, suspicious_num, dstar)


def jaccard_analysis(trainable_layers, scores, num_cf, num_uf, num_cs, num_us, suspicious_num, star):
    """
    More information on DStar fault localization technique can be found in
    Abreu, Rui, Peter Zoeteweij, and Arjan JC Van Gemund. "An evaluation of similarity coefficients for software fault localization." 2006 12th Pacific Rim International Symposium on Dependable Computing (PRDC'06). IEEE, 2006.
    """

    def dstar(i, j):
        return float(num_cf[i][j]) / (num_cs[i][j] + num_uf[i][j] + num_cf[i][j])

    return scores_with_foo(trainable_layers, scores, num_cf, num_uf, num_cs, num_us, suspicious_num, dstar)

def read_activation_diff_from_txt(suc_full_precision_filename, suc_int8_filename, fail_full_precision_filename, fail_suc_int8_filename, num_input, num_output, num_cases, regression=False):
    #return the diff for a single input image, success test case or failed test case
    suc_full_precision_matrix = []
    count_suc_full_precision_neurons = 0
    with open(suc_full_precision_filename, 'r') as f:
        lines = f.readlines()
        for line in lines:
            curline=line.strip().split(" ")
            full_precision_matrix.append(curline[:])
            count_suc_full_precision_filename +=1
        
    suc_int8_matrix = []
    count_suc_int8_neurons = 0
    
    with open(suc_int8_filename, 'r') as f:
        lines = f.readlines()
        for line in lines:
            curline=line.strip().split(" ")
            suc_in8_matrix.append(curline[:])
            count_suc_int8_neurons +=1
       
    fail_full_precision_matrix = []
    count_fail_full_precision_neurons = 0
        
    with open(fail_full_precision_filename, 'r') as f:
        lines = f.readlines()
        for line in lines:
            curline=line.strip().split(" ")
            fail_full_precision_matrix.append(curline[:])
            count_fail_full_precision_neurons+=1
        
    fail_int8_matrix = []
    count_fail_int8_neurons = 0
        
    with open(fail_int8_filename, 'r') as f:
        lines = f.readlines()
        for line in lines:
            curline=line.strip().split(" ")
            fail_in8_matrix.append(curline[:])
            count_fail_int8_neurons+=1
    
    suc_diff = [suc_full_precision_matrix[i] - suc_int8_filename[i] for i in range(len(suc_full_precision_matrix))]
    fail_diff = [fail_full_precision_matrix[i] - fail_int8_filename[i] for i in range(len(suc_full_precision_matrix))]
    return (suc_diff, fail_diff, count_suc_full_precision_neurons)
    
                           
def retract_relu_status(model, num_input, num_output, num_cases, regression=False):
    for layer in model.layers:
        print(layer.name)
        if 'relu' in layer.name:
            relu_layer+=1
            print(get_activations(model, x, layer_names=layer.name))
            acts = get_activations(model, x, layer_names=layer.name)[layer.name]
            print(np.sign(acts))
            #os.mkdir('./output/layer_outputs_n01440764_tench_quant/')
    np.savetxt('./output/layer_outputs_n01440764_tench_quant/'+str(relu_layer)+"_"+layer.name+'_tensor.txt',np.array(acts).reshape(-1),delimiter=',')
    np.savetxt('./output/layer_outputs_n01440764_tench_quant/'+layer.name+'_activation_status_tensor.txt',np.sign(np.array(acts)).reshape(-1),delimiter=',')

def calculate_cnaf_cnas_layer(layer):
    #for each input
    g = os.walk(r"./test_image_weights") #path to the weights of a single layer
    suc_diff_for_layer = []
    fail_diff_for_layer = []
    for path,dir_list,file_list in g:  
        for file_name in file_list:    
            suc_int8_filename = os.path.join(path, file_name)
            suc_diff, fail_diff, neuron_count = read_activation_diff_from_txt(suc_full_precision_filename, suc_int8_filename, fail_full_precision_filename, fail_suc_int8_filename, num_input, num_output, num_cases, regression=False)
            suc_diff_for_layer = suc_diff_for_layer + suc_diff
            fail_diff_for_layer = fail_diff_for_layer + fail_diff
    cnaf_for_layer = fail_diff_for_layer.copy()
    cnas_for_layer = suc_diff_for_layer.copy()
    return (cnaf_for_layer, cnas_for_layer)

def calculate_cnnf_cnns_layer(layer, count):
    cnaf_for_layer_temp, cnas_for_layer_temp = calculate_cnaf_cnas_layer(layer)
    cnnf_for_layer = []
    cnns_for_layer = []
    cnnf_for_layer = [count-i for i in cnaf_for_layer_temp]
    cnns_for_layer = [count-i for i in cnas_for_layer_temp]
    return (cnnf_for_layer, cnns_for_layer)

def neuron_T(layer):
    

    
        
def count_num_cf(suc_full_precision_filename, suc_int8_filename, fail_full_precision_filename, fail_suc_int8_filename, num_input, num_output, num_cases, regression=False):

def get_correction(layer, index, weight):
    equations = network.equList
    numOfVar = network.numVars
    networkEpsilons = network.epsilons
    epsilonsShape = networkEpsilons.shape 
    model = Model("my model")
    modelVars = model.addVars(numOfVar, lb=-GRB.INFINITY, ub=GRB.INFINITY)
    epsilon = model.addVar(name="epsilon")
    model.setObjective(epsilon, GRB.MINIMIZE)
    for i in range(epsilonsShape[0]):
        for j in range(epsilonsShape[1]):
            model.addConstr(modelVars[networkEpsilons[i][j]], GRB.LESS_EQUAL, epsilon)
            model.addConstr(modelVars[networkEpsilons[i][j]], GRB.GREATER_EQUAL, -1*epsilon)

    for eq in equations:
        addends = map(lambda addend: modelVars[addend[1]] * addend[0], eq.addendList)
        eq_left = reduce(lambda x,y: x+y, addends)
        if eq.EquationType == MarabouCore.Equation.EQ:
            model.addConstr(eq_left, GRB.EQUAL, eq.scalar)
        if eq.EquationType == MarabouCore.Equation.LE:
            model.addConstr(eq_left, GRB.LESS_EQUAL, eq.scalar)
        if eq.EquationType == MarabouCore.Equation.GE:
            model.addConstr(eq_left, GRB.GREATER_EQUAL, eq.scalar)

    model.optimize()
    # epsilons_vals = np.array([[modelVars[networkEpsilons[i][j]].x for j in range(epsilonsShape[1])] for i in range(epsilonsShape[0])])
    all_vals = np.array([modelVars[i].x for i in range(numOfVar)])
    return epsilon.x, epsilon.x, all_vals 
    

def apply_correction(layer, index, weight):
                           
               




def read_weight_from_c(cfile, num_input, num_output, regression=False):
    
    with open(cfile, 'r') as f:
        lines = f.readlines()
    lines = [line.strip() for line in lines]
    assert len(lines) > 0
    layers = 0
    weights = []
    current_statement = ''
    for line in lines:
        comment_index = line.find(';')

        if comment_index != -1:
            line = line[:comment_index].rstrip()

        if not line:
            continue
               
def calculate_importance(layers, a=1, b=1, c=1):
    T = tarantula_analysis(layers, 

                           
def get_grb_solution(grb_model, reference, bound_type, eps=1e-5):
    try:

        # Create a new model
        m = gp.Model("matrix1")

        # Create variables
        x = m.addMVar(shape=3, vtype=GRB.BINARY, name="x")

        # Set objective
        obj = np.array([1.0, 1.0, 2.0])
        m.setObjective(obj @ x, GRB.MAXIMIZE)

        # Build (sparse) constraint matrix
        val = np.array([1.0, 2.0, 3.0, -1.0, -1.0])
        row = np.array([0, 0, 0, 1, 1])
        col = np.array([0, 1, 2, 0, 1])

        A = sp.csr_matrix((val, (row, col)), shape=(2, 3))

        # Build rhs vector
        rhs = np.array([4.0, -1.0])

        # Add constraints
        m.addConstr(A @ x <= rhs, name="c")

        # Optimize model
        m.optimize()

        print(x.X)
        print('Obj: %g' % m.ObjVal)

    except gp.GurobiError as e:
        print('Error code ' + str(e.errno) + ": " + str(e))

    except AttributeError:
        print('Encountered an attribute error')
    # Set objective

def modify_weight(model, grb_solution):
