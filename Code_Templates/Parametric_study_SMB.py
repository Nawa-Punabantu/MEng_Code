# IMPORTING LIBRARIES
###########################################
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
# Loading the Plotting Libraries
from matplotlib.pyplot import subplots
import matplotlib.pyplot as plt
import matplotlib.animation as animation
# from PIL import Image
from scipy import integrate
import plotly.graph_objects as go
###########################################
# IMPORTING MY OWN FUNCTIONS
###########################################
from SMB_func_general import SMB

def calculate_mse(matrix1, matrix2):
    """
    Calculate the Mean Squared Error (MSE) between two matrices.
    
    Parameters:
        matrix1 (np.array): The first matrix of concentrations.
        matrix2 (np.array): The second matrix of concentrations.
    
    Returns:
        float: The Mean Squared Error.
    """
    mse = np.mean((matrix1 - matrix2) ** 2)
    return mse

def calculate_log_error(matrix1, matrix2, epsilon=1e-10):
    """
    Calculate the Logarithmic Error between two matrices.
    
    Parameters:
        matrix1 (np.array): The first matrix of concentrations.
        matrix2 (np.array): The second matrix of concentrations.
        epsilon (float): A small value added to avoid log of zero (default is 1e-10).
    
    Returns:
        float: The Logarithmic Error.
    """
    log_error1 = np.mean(np.abs(np.log(matrix1 + epsilon) - np.log(matrix2 + epsilon)))
    log_error2 = np.mean(np.abs(np.log((matrix1 + epsilon) / (matrix2 + epsilon))))
    return log_error1, log_error2

def find_indices(t_ode_times, t_schedule):
    """
    t_schedule -> vector of times when (events) port switches happen e.g. at [0,5,10] seconds
    t_ode_times -> vector of times from ODE

    We want to know where in t_ode_times, t_schedule occures
    These iwll be stored as indecies in t_idx
    Returns:np.ndarray: An array of indices in t_ode_times corresponding to each value in t_schedule.
    """
    t_idx = np.searchsorted(t_ode_times, t_schedule)
    t_idx = np.append(t_idx, len(t_ode_times))

    return t_idx

def point_value_parametric_study(quantity_name, lower_bound, upper_bound, resolution):
    """
    Perform a parametric study on a given SMB input parameter.

    Args:
        quantity_name (str): The name of the SMB input to vary (must match exactly with SMB_inputs_names).
        lower_bound (float): The starting value of the pacrameter.
        upper_bound (float): The ending value of the parameter.
        resolution (float): The step size for the parameter variation.

    Returns:
        dict: A dictionary containing the varied parameter values and corresponding SMB results.
    """
    
    # Ensure the input name matches one of the SMB input names
    if quantity_name not in SMB_inputs_names:
        raise ValueError(f"{quantity_name} is not a valid SMB input name. Please choose from {SMB_inputs_names}.")

    output = {'parameter_values': [], 'results': []}
    variable = np.arange(lower_bound, upper_bound + resolution, resolution)

    # Get the index of the parameter to vary
    name_idx = SMB_inputs_names.index(quantity_name)

    for i, var_value in enumerate(variable):  # Iterate over each value in the range
        print("\n\n\n\n\n\n\n-----------------------------------------------------------")
        print("-----------------------------------------------------------")
        print(f'Iteration {i+1}: Setting {quantity_name} to {var_value}')
        print("-----------------------------------------------------------")
        print("-----------------------------------------------------------\n\n")
        
        # Update the current SMB input parameter
        SMB_inputs[name_idx] = var_value

        # Run the SMB simulation
        y_matrices, nx, t, t_sets, t_schedule, C_feed, m_in, m_out, raff_cprofile, ext_cprofile, raff_intgral_purity, raff_recov, ext_intgral_purity, ext_recov, raff_vflow, ext_vflow, Model_Acc, Expected_Acc, Error_percent = SMB(SMB_inputs)

        # Save the parameter value and corresponding results
        output['parameter_values'].append(var_value)
        output['results'].append({
            'y_matrices': y_matrices,
            'nx': nx,
            't': t,
            't_sets': t_sets,
            't_schedule': t_schedule,
            'C_feed': C_feed,
            'm_in': m_in,
            'm_out': m_out,
            'raff_cprofile': raff_cprofile,
            'ext_cprofile': ext_cprofile,
            'raff_intgral_purity': raff_intgral_purity,
            'raff_recov': raff_recov,
            'ext_intgral_purity': ext_intgral_purity,
            'ext_recov': ext_recov,
            'raff_vflow': raff_vflow,
            'ext_vflow': ext_vflow,
            'Model_Acc': Model_Acc,
            'Expected_Acc': Expected_Acc,
            'Error_percent': Error_percent
        })
    return output, variable, quantity_name

def vector_value_parametric_study(quantity_name, vecotr_set):
    """
    Func to varry vecoter values like zone_config or flowrate
    
    vecotr_set => set of different config vectors to try
    """
    counter = []

    # Ensure the input name matches one of the SMB input names
    if quantity_name not in SMB_inputs_names:
        raise ValueError(f"{quantity_name} is not a valid SMB input name. Please choose from {SMB_inputs_names}.")

    output = {'parameter_values': [], 'results': []}
    
    
    # Get the index of the parameter to vary
    name_idx = SMB_inputs_names.index(quantity_name)
    print('name_idx:', name_idx)
    print(f'Changing:{SMB_inputs_names[name_idx]}')

    for i in range(len(vecotr_set)):  # Iterate over each value in the range
        var_value = vecotr_set[i]
        print("\n\n\n\n\n-----------------------------------------------------------")
        print("-----------------------------------------------------------\n\n")
        print(f'Iteration {i+1}: Setting {quantity_name} to {var_value}')
        print("-----------------------------------------------------------")
        print("-----------------------------------------------------------\n\n")
        counter.append(i+1) # count num of iterations


        # Update the current SMB input parameter
        SMB_inputs[name_idx] = var_value

        # Run the SMB simulation
        (y_matrices, nx, t, t_sets, t_schedule, C_feed, m_in, m_out, raff_cprofile, ext_cprofile, raff_intgral_purity, raff_recov, ext_intgral_purity, ext_recov, raff_vflow, ext_vflow, Model_Acc, Expected_Acc, Error_percent) = SMB(SMB_inputs)

        # Save the parameter value and corresponding results
        output['parameter_values'].append(var_value)
        output['results'].append({
            'y_matrices': y_matrices,
            'nx': nx,
            't': t,
            't_sets': t_sets,
            't_schedule': t_schedule,
            'C_feed': C_feed,
            'm_in': m_in,
            'm_out': m_out,
            'raff_cprofile': raff_cprofile,
            'ext_cprofile': ext_cprofile,
            'raff_intgral_purity': raff_intgral_purity,
            'raff_recov': raff_recov,
            'ext_intgral_purity': ext_intgral_purity,
            'ext_recov': ext_recov,
            'raff_vflow': raff_vflow,
            'ext_vflow': ext_vflow,
            'Model_Acc': Model_Acc,
            'Expected_Acc': Expected_Acc,
            'Error_percent': Error_percent
        })

    return output, counter, quantity_name   

def plot_parametric_results(output, x_values, y_variable_name, x_variable_name, color):
    """
    Plot the parametric study results for a given output variable.
    
    Args:
        output (dict): The results dictionary returned from `point_value_parametric_study`.
        y_variable_name (str): The name of the output variable to plot (e.g., 'raff_recov', 'ext_recov').
        x_variable_name (str): The name of the independent variable (default is 'parameter_values').
    
    Raises:
        ValueError: If the y_variable_name is not found in the results.
    """
    # Extract the x-axis values (independent variable)
    
    if x_values is None:
        raise ValueError(f"{x_variable_name} is not a valid independent variable.")
    
    # Extract the y-axis values (dependent variable)
    y_values = []
    for result in output['results']:
        if y_variable_name in result:
            y_values.append(result[y_variable_name])
        else:
            raise ValueError(f"{y_variable_name} is not a valid output variable. Available keys: {list(output['results'][0].keys())}")
    # print('y_values:\n',y_values)
    # print('y_values[0]:\n',y_values[0])
    # Plot the results
    plt.figure(figsize=(10, 6))
    if y_variable_name == 'raff_intgral_purity' or y_variable_name =="ext_intgral_purity":
        y_values = np.concatenate(y_values)
        y_values_A = y_values[0::2]
        y_values_B = y_values[1::2]
        
        plt.plot(x_values, y_values_A, marker='o', linestyle='-', color = color[0], label = Names[0])
        plt.plot(x_values, y_values_B, marker='o', linestyle='-', color = color[1], label = Names[1])
    else:
        plt.plot(x_values, y_values, marker='o', linestyle='-', color = 'purple')
    plt.title(f"Parametric Study: {y_variable_name} vs {x_variable_name}")
    plt.xlabel(x_variable_name)
    plt.ylabel(y_variable_name)
    plt.legend()
    plt.grid(True)
    plt.show()


def mesh_ind_study(mesh_sizes, SMB_inputs, n):
    """
    Func that returns the error related to using different mesh (x-descritizsation) sizes
    mesh_sizes => np array of mesh sizes we are interested in.
    Note that the inpute to mesh_sizes are the the x-descritizsations just one col, nx_per_col
    n => num of error points in time (delult 100)

    """
    # Initialized where we will store the output matrices
    # Organise into accending order
    mesh_sizes = np.sort(mesh_sizes)
    num_mesh = len(mesh_sizes) # numner of meshes to be looked at
    # Solve the SMB solver at the different mesh sizes:
    y_mesh = []
    t_idx_all = []
    t_odes_all = []
    
    error = []
    for mesh in mesh_sizes:
        nx_per_col = mesh
        SMB_inputs = [iso_type, Names, color, num_comp, nx_per_col, e, Pe, Bm, zone_config, L, d_col, d_in, t_index_min, n_num_cycles, Q_internal, parameter_sets]
        y_matrices, nx, t, t_sets, t_schedule, C_feed, m_in, m_out, raff_cprofile, ext_cprofile, raff_intgral_purity, raff_recov, ext_intgral_purity, ext_recov, raff_vflow, ext_vflow, Model_Acc, Expected_Acc, Error_percent = SMB(SMB_inputs)
        y_mesh.append(y_matrices[0])
        t_odes_all.append(t_sets[0])

    print('size of y_mesh:', y_mesh[0].shape)
    # Times at which the concentrations will be compared
    tend = t_odes_all[0][-1]
    t_sch = np.linspace(0, tend, n)
    print('len(t_sch):\n', len(t_sch))

    for i in range(num_mesh):
        t_idx = find_indices(t_odes_all[i], t_sch)
        t_idx_all.append(t_idx)
    
    # Initialize
    Ncol_num = sum(zone_config)
    y_mesh_new = []

    for k in range(len(mesh_sizes)): # go through each mesh
        y_mesh_new_add = np.zeros((Ncol_num, n))
        for i in range(Ncol_num):
            j = i*(mesh_sizes[k] - 1)
            y_add = []
            for w in  range(len(t_sch)): # go throght the col of y_mesh_new
                y_add.append(y_mesh[k][j,t_idx_all[k][w]])
            y_mesh_new_add[i,:] = y_add

        y_mesh_new.append(y_mesh_new_add)

    
    print('np.shape(y_mesh_new[0]):\n', np.shape(y_mesh_new[0]))

    # # Find the errors:
    error_vector = []
    mse_all = []
    for i in range(num_mesh-1): # for each matrix in y_mesh_new
        error_matrix = (y_mesh_new[i]-y_mesh_new[i+1])
        for j in range(Ncol_num): # for the rows in error_matrix
            error_vector.append(sum(error_matrix[j]))
        error_value = sum(error_vector)
        mse = calculate_mse(y_mesh_new[i], y_mesh_new[i+1])
        log_error1, log_error2 = calculate_log_error(y_mesh_new[i], y_mesh_new[i+1])
        error.append(error_value)
        mse_all.append(mse)

    # print(f'Error b/n nx_per_col = {mesh_sizes[0]} vs {mesh_sizes[1]}: {error} ')
    print(f'MSE Error b/n nx_per_col = {mesh_sizes[0]} vs {mesh_sizes[1]}: {mse_all} ')
    print(f'Log Error b/n nx_per_col = {mesh_sizes[0]} vs {mesh_sizes[1]}: log_error1: {log_error1}, log_error2: {log_error2} ')
    return y_mesh_new, error, mse



###################### PRIMARY INPUTS #########################
# Define the names, colors, and parameter sets for 6 components
Names = ["Glucose", "Fructose"]#, "C"]#, "D", "E", "F"]
color = ["g", "orange"]#, "b"]#, "r", "purple", "brown"]
num_comp = len(Names) # Number of components
e = 0.40         # bed voidage
Pe = 500
Bm = 300

########################## Column Dimensions #########################

# How many columns in each Zone?
Z1, Z2, Z3, Z4 = 1, 1, 1, 1
zone_config = np.array([Z1, Z2, Z3, Z4])
L = 30 # cm # Length of one column
d_col = 2.6 # cm # column diameter
# Dimensions of the tubing and from each column:
# Assuming the pipe diameter is 20% of the column diameter:
d_in = 0.2 * d_col # cm
nx_per_col = 10
# Time Specs
t_index_min = 19/60 # min # Index time # How long the pulse holds before swtiching
n_num_cycles = 5    # Number of Cycles you want the SMB to run for

###############  FLOWRATES   #########################

# Jochen et al:
Q_P, Q_Q, Q_R, Q_S = 5.21, 4, 5.67, 4.65 # x10-7 m^3/s
conv_fac = 0.1 # x10-7 m^3/s => cm^3/s
Q_P, Q_Q, Q_R, Q_S  = Q_P*conv_fac, Q_Q*conv_fac, Q_R*conv_fac, Q_S*conv_fac

Q_I, Q_II, Q_III, Q_IV = Q_R,  Q_S, Q_P, Q_Q
Q_internal = np.array([Q_I, Q_II, Q_III, Q_IV])


parameter_sets = [
                    {"kh": 3.15/100, "H": 0.27, "C_feed": 1},  # Component A
                    {"kh": 2.217/100, "H": 0.53, "C_feed": 1}] #, # Component B

# ISOTHERM PARAMETERS
###########################################################################################
iso_type = "UNC" 
theta_lin = [parameter_sets[i]['H'] for i in range(num_comp)] # [HA, HB]
print('theta_lin:', theta_lin)
# theta_lang = [1, 2, 3, 4 ,5, 6] # [HA, HB]
theta_cup_lang = [5.29, 3.24, 2.02, 0.03] # [HA, HB, KA, KB]
# theta_fre = [1.2, 0.5]
# theta_blang = [[2.69, 0.0336, 0.0466, 0.1, 1, 3],\
#                 [3.73, 0.0336, 0.0466, 0.3, 1, 3]] # [HA, HB]

# SMB setup (modify these accordingly)
SMB_inputs = [iso_type, Names, color, num_comp, nx_per_col, e, Pe, Bm, zone_config, L, d_col, d_in, t_index_min, n_num_cycles, Q_internal, parameter_sets]
SMB_inputs_names = ['iso_type', 'Names', 'color', 'num_comp', 'nx_per_col', 'e', 'Pe', 'Bm', 'zone_config', 'L', 'd_col', 'd_in', 't_index_min', 'n_num_cycles', 'Q_internal', 'parameter_sets']


################ VECTORS TO VARRY ####################################
zone_config_varry = [np.array([2,2,3,1])]
# print('zone_config_varry[0]:\n', zone_config_varry[0])
###########################################################################################

################ EXCUTING THE FUNCTIONS ####################################
# print('Solving Mesh Independence Study . . . ')
# mesh_sizes = [60, 60]
# y, error, mse = mesh_ind_study(mesh_sizes, SMB_inputs, 100)
# print("""
# Interprating the Log Error:
# Log Error = 0: The matrices are identical.
# Log Error < 0.5: The differences are relatively small.
# Log Error ≈ 0.5: The values differ by about 65% on average.
# Log Error > 0.5: The matrices have substantial differences, with values differing by more than 65%.\n\n\n""")

print('\n\n\n\nSolving Parametric Study #1. . . . . . ')
quantity_name = 'e'
min_val = 0.1
max_val = 1
dist_bn_points = 0.1

Output, x_variable, x_variable_name = point_value_parametric_study(quantity_name, min_val, max_val , dist_bn_points) # (Name of quantitiy, lower_bound, upper_bound + resolution, resolution)
plot_parametric_results(output= Output, x_values = x_variable, y_variable_name = 'Model_Acc', x_variable_name = x_variable_name, color=color)
# plot_parametric_results(output= Output, x_values = x_variable, y_variable_name = 'Expected_Acc', x_variable_name = x_variable_name, color=color)
# plot_parametric_results(output= Output, x_values = x_variable, y_variable_name = 'm_in', x_variable_name = x_variable_name, color=color)
# plot_parametric_results(output= Output, x_values = x_variable, y_variable_name = 'm_out', x_variable_name = x_variable_name, color=color)
# plot_parametric_results(output= Output, x_values = x_variable, y_variable_name = 'Error_percent', x_variable_name = x_variable_name, color=color)

# print('Solving Parametric Study #2. . . . . . ')
# Output, x_variable, x_variable_name = point_value_parametric_study('nx_per_col', 5, 100 , 10) # (Name of quantitiy, lower_bound, upper_bound + resolution, resolution)
# # plot_parametric_results(output= Output, x_values = x_variable, y_variable_name = 'Model_Acc', x_variable_name = x_variable_name, color=color)
# # plot_parametric_results(output= Output, x_values = x_variable, y_variable_name = 'Expected_Acc', x_variable_name = x_variable_name, color=color)
# # plot_parametric_results(output= Output, x_values = x_variable, y_variable_name = 'm_in', x_variable_name = x_variable_name, color=color)
# # plot_parametric_results(output= Output, x_values = x_variable, y_variable_name = 'm_out', x_variable_name = x_variable_name, color=color)
# plot_parametric_results(output= Output, x_values = x_variable, y_variable_name = 'Error_percent', x_variable_name = x_variable_name, color=color)

# print('Solving Parametric Study #3. . . . . . ')
# Output, x_variable, x_variable_name = point_value_parametric_study('t_index_min', 100/60, 600/60 , 100/60) # (Name of quantitiy, lower_bound, upper_bound + resolution, resolution)
# plot_parametric_results(output= Output, x_values = x_variable, y_variable_name = 'Model_Acc', x_variable_name = x_variable_name, color=color)
# plot_parametric_results(output= Output, x_values = x_variable, y_variable_name = 'Expected_Acc', x_variable_name = x_variable_name, color=color)
# plot_parametric_results(output= Output, x_values = x_variable, y_variable_name = 'm_in', x_variable_name = x_variable_name, color=color)
# plot_parametric_results(output= Output, x_values = x_variable, y_variable_name = 'm_out', x_variable_name = x_variable_name, color=color)
# plot_parametric_results(output= Output, x_values = x_variable, y_variable_name = 'Error_percent', x_variable_name = x_variable_name, color=color)

# Output, counter, x_variable_name = vector_value_parametric_study('zone_config', zone_config_varry)
# print('counter:\n', counter)
# plot_parametric_results(output= Output, x_values = counter, y_variable_name = 'Error_percent', x_variable_name = x_variable_name, color=color)
# plot_parametric_results(output= Output, x_values = counter, y_variable_name = 'ext_intgral_purity', x_variable_name = x_variable_name, color=color)
# plot_parametric_results(output= Output, x_values = counter, y_variable_name = 'raff_intgral_purity', x_variable_name = x_variable_name, color=color)

