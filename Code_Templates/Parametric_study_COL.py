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
from Col_test_model_func import column_func


# Function to solve the concentration using the column model given a Pe, and tend
def solve_concentration(param_name, param_val, column_func_inputs, Hkfp):
    
    """
    param => string | name of parameter to be changed as listed in column_func_inputs_names
    Hkfp = ['H' or 'kfp'] | string if we want to varry the kinetics :)
    """
    column_func_inputs_names = ["iso_type", "Names", "color", "parameter_sets", "Pe", "Bm", "e", "Q_S", "Q_inj", "t_index", "tend_min", "nx", "L", "d_col"]
        # Ensure the input name matches one of the SMB input names
    if param_name not in column_func_inputs_names:
        raise ValueError(f"\n\n{param_name} is not a valid SMB input name. Please choose from {column_func_inputs_names}.")

    
    # Get the index of the parameter to vary
    name_idx = column_func_inputs_names.index(param_name)
    
    # print('name_idx:', name_idx)
    # print('column_func_inputs_names[name_idx]:', column_func_inputs_names[name_idx])
    # print('column_func_inputs[name_idx]:', column_func_inputs[name_idx])
    # print('column_func_inputs[name_idx][0][Hkfp]:', column_func_inputs[name_idx][0][Hkfp])

    if Hkfp != None and column_func_inputs_names[name_idx] == 'parameter_sets' :
        column_func_inputs[name_idx][0][Hkfp] =  param_val # Insert the Peclet number in the correct position
    else:
        column_func_inputs[name_idx] =  param_val # Insert the Peclet number in the correct position
    
    col_elution, y_matrices, nx, t, t_sets, t_schedule, C_feed, m_in, m_out, Model_Acc, Expected_Acc, Error_percent = column_func(column_func_inputs)
    return col_elution, y_matrices, nx, t, t_sets, t_schedule, C_feed, m_in, m_out, Model_Acc, Expected_Acc, Error_percent


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




def point_value_parametric_study(param_name, lower_bound, upper_bound, num_points, Hkfp):
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
    if param_name not in column_func_inputs_names:
        raise ValueError(f"\n\n{param_name} is not a valid SMB input name. Please choose from {column_func_inputs_names}.")

    output = {'parameter_values': [], 'results': []}
    
    # array of the values of the variable to varry:
    variable = np.arange(lower_bound, upper_bound + num_points, num_points)

    # Get the index of the parameter to vary:
    # name_idx = column_func_inputs_names.index(param_name)
    # if column_func_inputs_names[name_idx] == 'parameter_sets':

    for i, var_value in enumerate(variable):  # Iterate over each value in the range
        
        print("\n\n\n\n\n\n\n-----------------------------------------------------------")
        print("-----------------------------------------------------------")
        print(f'Iteration {i+1}: Setting {param_name} to {var_value}')
        print("-----------------------------------------------------------")
        print("-----------------------------------------------------------\n\n")
        
        # Update the current SMB input parameter
        # column_func_inputs[name_idx] = var_value

        param_val = var_value

        # Run the SMB simulation
        col_elution, y_matrices, nx, t, t_sets, t_schedule, C_feed, m_in, m_out, Model_Acc, Expected_Acc, Error_percent = solve_concentration(param_name, param_val, column_func_inputs, Hkfp)
        t_end = t_sets[0][-1]
        
        # Save the parameter value and corresponding results
        output['parameter_values'].append(var_value)
        output['results'].append({
            'col_elution': col_elution,
            'y_matrices': y_matrices,
            'nx': nx,
            't': t,
            't_sets': t_sets,
            't_schedule': t_schedule,
            'C_feed': C_feed,
            'm_in': m_in,
            'm_out': m_out,
            'Model_Acc': Model_Acc,
            'Expected_Acc': Expected_Acc,
            'Error_percent': Error_percent
        })
    
    return output, variable, param_name, t_end

 

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

import matplotlib.pyplot as plt
import numpy as np

def plot_elution_curves(output, tend, lower_bound, upper_bound, dist_bn_points, var_name):
    """
    Plots multiple elution curves on the same graph.

    Parameters:
    - output: dict, contains the results of each simulation.
        Must be in the format:
        {
            'parameter_values': [param1, param2, ...],
            'results': [
                {
                    'col_elution': col_elution_vector,
                    't': time_vector,
                    ...
                },
                ...
            ]
        }
    - tend: float, the final time point to be used as the maximum x-axis limit for all curves.
    """
    plt.figure(figsize=(10, 6))
    # array of the values of the variable to varry:
    variable = np.arange(lower_bound, upper_bound + dist_bn_points, dist_bn_points)

    for i, result in enumerate(output['results']):
        # Extract the elution curve and time vector
        col_elution = result['col_elution'][0]
        time_vector = result['t_sets'][0]

        # Normalize time vector to start at 0 and end at tend
        # normalized_time = np.linspace(0, tend, len(time_vector))

        # Plot the elution curve with a label for parameter value
        # param_value = output['parameter_values'][i]
        if var_name == 'parameter_sets':
            plt.plot(time_vector, col_elution, label=f'{Hkfp}: {variable[i]}')
        else:
            plt.plot(time_vector, col_elution, label=f'{var_name}: {variable[i]}')

    plt.xlabel("Time")
    plt.ylabel("Concentration")
    plt.title("Elution Curves")
    plt.legend()
    plt.grid(True)
    plt.xlim(0, tend)
    plt.show()


###################### PRIMARY INPUTS #########################

# What tpye of isoherm is required?
# Coupled: "CUP"
# Uncoupled: "UNC"
iso_type = "UNC" 
Names = ["A"]#, "B"]#, "C"]#, "D", "E", "F"]
color = ["g"]#, "orange"]#, "b"]#, "r", "purple", "brown"]
num_comp = len(Names)

# Parameter sets for different components
# Units:
# - Concentrations: g/cm^3
# - kfp: 1/s
parameter_sets = [{"kfp": 3.15/100, "H": 1, "C_feed": 5} ]#,    # Component A
     #{"kfp": 2.217/100, "H": 0.23, "C_feed": 1}#] #, # Component B
    # {"kfp": 0.05, "H": 2.5, "C_feed": 1.8},  # Component C
#     {"kfp": 0.02, "H": 1.2, "C_feed": 3.0},  # Component D
#     {"kfp": 0.03, "H": 4.0, "C_feed": 2.5},  # Component E
#     {"kfp": 0.07, "H": 2.0, "C_feed": 1.5}   # Component  
# ]
# print("size:\n", np.shape(parameter_sets))  #]#

Pe = 500 # Da = (u * L)/Pe 
Bm = 300
e = 0.4    # (0, 1]     # voidage
Q_S = 1 # cm^3/s | The volumetric flowrate of the feed to the left of the feed port (pure solvent)
Q_inj = 0.01 # cm^3/s | The volumetric flowrate of the injected concentration slug
t_index = 10 # s # Index time # How long the SINGLE pulse holds for
tend_min = 80/60 # min
nx = 70
###################### COLUMN DIMENTIONS ########################
L = 20 # cm
d_col = 2.6 # cm


column_func_inputs = [iso_type,  Names, color, parameter_sets, Pe, Bm, e, Q_S, Q_inj, t_index, tend_min, nx, L, d_col]


################ EXCUTING THE FUNCTIONS ####################################
column_func_inputs_names = ["iso_type", "Names", "color", "parameter_sets", "Pe", "Bm", "e", "Q_S", "Q_inj", "t_index", "tend_min", "nx", "L", "d_col"]

print('\n\n\n\nSolving Parametric Study #1 . . . . . . ')

lower_bound =  0.1
upper_bound = 0.9
dist_bn_points = 0.1
var_name = 'e'
# var_name = 'parameter_sets'
Hkfp = None # 'H', 'kfp', None

Output, x_variable, x_variable_name, tend = point_value_parametric_study(var_name, lower_bound, upper_bound, dist_bn_points, Hkfp) # (Name of quantitiy, lower_bound, upper_bound, resolution(=space between points))
# Output, x_variable, x_variable_name = point_value_parametric_study('parameter_sets', 2, 10, 1, Hkfp='H') # 
# Where resolution => space between points
plot_parametric_results(output= Output, x_values = x_variable, y_variable_name = 'Error_percent', x_variable_name = x_variable_name, color=color)
# plot_parametric_results(output= Output, x_values = x_variable, y_variable_name = 'Model_Acc', x_variable_name = x_variable_name, color=color)
# plot_parametric_results(output= Output, x_values = x_variable, y_variable_name = 'Expected_Acc', x_variable_name = x_variable_name, color=color)
plot_parametric_results(output= Output, x_values = x_variable, y_variable_name = 'm_in', x_variable_name = x_variable_name, color=color)
plot_parametric_results(output= Output, x_values = x_variable, y_variable_name = 'm_out', x_variable_name = x_variable_name, color=color)
plot_elution_curves(Output, tend, lower_bound, upper_bound, dist_bn_points, var_name)
