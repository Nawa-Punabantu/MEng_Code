#%%
# 
# # IMPORTING LIBRARIES
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
import time
###########################################
# IMPORTING MY OWN FUNCTIONS
###########################################
from Col_test_model_func import column_func


# Function to solve the concentration using the column model given a Pe, and tend
def solve_concentration(param_name, param_val, column_func_inputs, Hkfp, column_func_inputs_names):
    
    """
    param => string | name of parameter to be changed as listed in column_func_inputs_names
    Hkfp = ['H' or 'kfp'] | string if we want to varry the kinetics :)
    """
        # Ensure the input name matches one of the SMB input names
    
    if param_name != "C_feed":
        print(f'param_name: {param_name}')
        if param_name not in column_func_inputs_names:
            raise ValueError(f"\n\n{param_name} is not a valid SMB input name. Please choose from {column_func_inputs_names}.")

        # Get the index of the parameter to vary
        name_idx = column_func_inputs_names.index(param_name)
        column_func_inputs[name_idx] =  param_val # Insert the Peclet number in the correct position
    
    elif param_name == "C_feed":
        column_func_inputs[3][0][param_name] =  param_val

    col_elution, y_matrices, nx, t, t_sets, t_schedule, C_feed, m_in, m_out, Model_Acc, Expected_Acc, Error_percent = column_func(column_func_inputs)
    return col_elution, y_matrices, nx, t, t_sets, t_schedule, C_feed, m_in, m_out, Model_Acc, Expected_Acc, Error_percent




def point_value_parametric_study(param_name, lower_bound, upper_bound, dist_bn_points, Hkfp, column_func_inputs_names):
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
    if param_name != 'C_feed':
        if param_name not in column_func_inputs_names:
            raise ValueError(f"\n\n{param_name} is not a valid SMB input name. Please choose from {column_func_inputs_names}.")

    output = {'parameter_values': [], 'results': []}
    
    # array of the values of the variable to varry:
    variable = np.arange(lower_bound, upper_bound + dist_bn_points, dist_bn_points)

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

        # Run the simulation
        start_time = time.time()

        col_elution, y_matrices, nx, t, t_sets, t_schedule, C_feed, m_in, m_out, Model_Acc, Expected_Acc, Error_percent = solve_concentration(param_name, param_val, column_func_inputs, Hkfp, column_func_inputs_names)
        end_time = time.time()
        sim_time = (end_time - start_time)/60 # min
        
        if iso_type == "UNC":
            t_end = t_sets[0][-1]   
        else:
            t_end = t[-1]
        
        # Save the parameter value and corresponding results
        output['parameter_values'].append(var_value)
        output['results'].append({
            # Not interested in matrices:
            # 'y_matrices': y_matrices,
            # 't_schedule': t_schedule,


            # Appended scalars:
            'nx': nx,
            'C_feed': C_feed,
            'm_in': m_in,
            'm_out': m_out,
            'Model_Acc': Model_Acc,
            'Expected_Acc': Expected_Acc,
            'Error_percent': Error_percent,
            'Simulation_time':sim_time,

            # Appended vectors (np arrays)
            'col_elution': col_elution,
            't_sets': t_sets,
            't': t,


            })
    
    return output, variable, param_name, t_end

# FUNCTION TO SAVE RESULTS:
import json
import os

import json
import numpy as np

import json
import numpy as np

import json
import numpy as np
import os

import json
import os
import numpy as np
import os
import json
import numpy as np

def convert_ndarrays(obj):
    """
    Recursively converts numpy arrays in an object to lists for JSON serialization.
    """
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, list):
        return [convert_ndarrays(item) for item in obj]
    elif isinstance(obj, dict):
        return {key: convert_ndarrays(value) for key, value in obj.items()}
    else:
        return obj
import os
import json
import numpy as np

def convert_ndarrays(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, list):
        return [convert_ndarrays(item) for item in obj]
    elif isinstance(obj, dict):
        return {key: convert_ndarrays(value) for key, value in obj.items()}
    else:
        return obj

def infer_param_range(variable):
    variable = np.asarray(variable)
    sorted_vals = np.sort(variable)
    unique_steps = np.unique(np.round(np.diff(sorted_vals), 10))  # round to avoid float noise
    spacing = float(unique_steps[0]) if len(unique_steps) > 0 else None
    return float(sorted_vals[0]), float(sorted_vals[-1]), spacing

# def save_output_to_json(
#     output,
#     variable,
#     param_name,
#     description_text,
#     save_path=None
# ):
#     if save_path is None:
#         filename = f'study_{param_name}_variation.json'
#         save_path = os.path.join(os.getcwd(), filename)

#     lower_bound, upper_bound, dist_bn_points = infer_param_range(variable)

#     results_dict = {
#         "description_text": description_text,
#         f"variable,{param_name}": convert_ndarrays(variable),
#         "param_name": param_name,
#         "lower_bound": lower_bound,
#         "upper_bound": upper_bound,
#         "dist_bn_points": dist_bn_points,
#         "Error_percent": [],
#         "Simulation_time": [],
#         "col_elution": [],
#         "t_sets": [],
#         "t": [],
#         "m_in": [],
#         "m_out": [],
#     }

#     keys = [
#         'Error_percent', 'Simulation_time', 'col_elution',
#         't_sets', 't', 'm_in', 'm_out'
#     ]

#     for result in output.get('results', []):
#         for key in keys:
#             if key in result:
#                 results_dict[key].append(convert_ndarrays(result[key]))

#     with open(save_path, 'w') as f:
#         json.dump(results_dict, f, indent=4)

#     print(f"✅ Results saved to: {save_path}")
#     print(f"   ➤ lower_bound = {lower_bound}")
#     print(f"   ➤ upper_bound = {upper_bound}")
#     print(f"   ➤ dist_bn_points = {dist_bn_points}")


def save_output_to_json(
    output,
    variable,
    param_name,
    description_text,
    save_path=None
):
    """
    Saves a parametric study output dictionary to a JSON file with clear structure.

    Parameters:
    - output: dict containing 'results', each of which is a dict with simulation outputs
    - variable: np.ndarray or list of parameter values (varied across runs)
    - param_name: str, name of the varied parameter
    - description_text: str, explanation of what the study represents
    - save_path: optional full file path; if None, defaults to current directory
    """

    # Set default save path
    if save_path is None:
        filename = f'study_{param_name}_variation.json'
        save_path = os.path.join(os.getcwd(), filename)

    # Initialize the output dictionary
    results_dict = {
        "description_text": description_text,
        f"variable,{param_name}": convert_ndarrays(variable),
        "param_name": param_name,
        "Error_percent": [],
        "Simulation_time": [],
        "col_elution": [],
        "t_sets": [],
        "t": [],
        "m_in": [],
        "m_out": [],
    }

    keys = [
        'Error_percent', 'Simulation_time', 'col_elution', 
        't_sets', 't', 'm_in', 'm_out'
    ]

    for result in output.get('results', []):
        for key in keys:
            if key in result:
                results_dict[key].append(convert_ndarrays(result[key]))

    # Save to JSON
    with open(save_path, 'w') as f:
        json.dump(results_dict, f, indent=4)

    print(f"✅ Results saved to: {save_path}")






def plot_parametric_results(output, x_values, y_variable_name, x_variable_name, color):
    """
    Plot the parametric study results for a given output variable.

    Args:
        output (dict): The results dictionary returned from `point_value_parametric_study`.
        x_values (array-like): The values of the independent variable.
        y_variable_name (str): The name of the output variable to plot.
        x_variable_name (str): The name of the independent variable.
        color (list or str): Color(s) used for plotting.

    Raises:
        ValueError: If the y_variable_name is not found in the results.
    """
    if x_values is None:
        raise ValueError(f"{x_variable_name} is not a valid independent variable.")
    
    y_values = []

    for result in output['results']:
        if y_variable_name == "diff":
            if 'Model_Acc' in result and 'Expected_Acc' in result:
                diff = np.array(result['Model_Acc']) - np.array(result['Expected_Acc'])
                # If it's a scalar, wrap into a list
                if np.isscalar(diff):
                    y_values.append(diff)
                else:
                    y_values.append(np.mean(diff))  # or np.max(np.abs(diff)) depending on what you want
            else:
                raise ValueError("Both 'Model_Acc' and 'Expected_Acc' must be in result when y_variable_name is 'diff'.")
        else:
            if y_variable_name in result:
                y_values.append(result[y_variable_name])
            else:
                raise ValueError(f"{y_variable_name} not found. Available keys: {list(result.keys())}")

    plt.figure(figsize=(10, 6))
    
    if y_variable_name == 'raff_intgral_purity' or y_variable_name == "ext_intgral_purity":
        y_values = np.concatenate(y_values)
        y_values_A = y_values[0::2]
        y_values_B = y_values[1::2]
        
        plt.plot(x_values, y_values_A, marker='o', linestyle='-', color=color[0], label=Names[0])
        plt.plot(x_values, y_values_B, marker='o', linestyle='-', color=color[1], label=Names[1])

        
        
    elif y_variable_name == "diff":
        plt.plot(x_values, y_values, marker='o', linestyle='-', color='purple')
        plt.axhline(0, color='black', linestyle='--', linewidth=1.2, label="Zero Error")
        plt.title(f"Parametric Study: {y_variable_name} vs {x_variable_name}")
        plt.xlabel(x_variable_name)
        plt.ylabel(f'{y_variable_name}, grams')
        plt.legend()
        plt.grid(True)
        plt.show()

    elif y_variable_name == "Error_percent":
        plt.plot(x_values, y_values, marker='o', linestyle='-', color='red')
        plt.axhline(0, color='black', linestyle='--', linewidth=1.2, label="Zero Error")
        plt.title(f"Effect of {x_variable_name} on Mass Balance Error Percent")
        plt.xlabel(x_variable_name)
        plt.ylabel(f'{y_variable_name}, (%)')


        
        y_limit = max(np.max(y_values), abs(np.min(y_values)))
        y_padding =  y_limit * 0.05
        if y_limit < 10:
            y_plot = 10 + y_padding
        else:
            y_plot = y_limit + y_padding
        
        plt.ylim(-y_plot , y_plot)
        

        plt.legend()
        plt.grid(True)
        plt.show()

    elif y_variable_name == "Simulation_time":
        plt.plot(x_values, y_values, marker='o', linestyle='-', color='blue')
        plt.title(f"Effect of {x_variable_name} on Computation Time (min)")
        plt.xlabel(x_variable_name)
        plt.ylabel(f'{y_variable_name}, (min)')
        plt.ylim(0,)
        plt.legend()
        plt.grid(True)
        plt.show()
    else:
        plt.plot(x_values, y_values, marker='o', linestyle='-', color='purple')
        plt.title(f"Parametric Study: {y_variable_name} vs {x_variable_name}")
        plt.xlabel(x_variable_name)
        plt.ylabel(y_variable_name)
        plt.legend()
        plt.grid(True)
        plt.show()
        



# def plot_parametric_results(output, x_values, y_variable_name, x_variable_name, color):
#     """
#     Plot the parametric study results for a given output variable.
    
#     Args:
#         output (dict): The results dictionary returned from `point_value_parametric_study`.
#         y_variable_name (str): The name of the output variable to plot (e.g., 'raff_recov', 'ext_recov').
#         x_variable_name (str): The name of the independent variable (default is 'parameter_values').
    
#     Raises:
#         ValueError: If the y_variable_name is not found in the results.
#     """
#     # Extract the x-axis values (independent variable)
    
#     if x_values is None:
#         raise ValueError(f"{x_variable_name} is not a valid independent variable.")
    
#     # Extract the y-axis values (dependent variable)
#     y_values = []
#     for result in output['results']:
#         if y_variable_name in result:
#             y_values.append(result[y_variable_name])
#         else:
#             raise ValueError(f"{y_variable_name} is not a valid output variable. Available keys: {list(output['results'][0].keys())}")
#     # print('y_values:\n',y_values)
#     # print('y_values[0]:\n',y_values[0])
    
#     # Plot the results
#     plt.figure(figsize=(10, 6))
#     if y_variable_name == 'raff_intgral_purity' or y_variable_name =="ext_intgral_purity":
#         y_values = np.concatenate(y_values)
#         y_values_A = y_values[0::2]
#         y_values_B = y_values[1::2]
        
#         plt.plot(x_values, y_values_A, marker='o', linestyle='-', color = color[0], label = Names[0])
#         plt.plot(x_values, y_values_B, marker='o', linestyle='-', color = color[1], label = Names[1])
#     else:
#         plt.plot(x_values, y_values, marker='o', linestyle='-', color = 'purple')
#     plt.title(f"Parametric Study: {y_variable_name} vs {x_variable_name}")
#     plt.xlabel(x_variable_name)
#     plt.ylabel(y_variable_name)
#     plt.legend()
#     plt.grid(True)
#     plt.show()

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
        if iso_type == "UNC":
            time_vector = result['t_sets'][0]
        else:
            time_vector = result['t']

        # Normalize time vector to start at 0 and end at tend
        # normalized_time = np.linspace(0, tend, len(time_vector))

        # Plot the elution curve with a label for parameter value
        # param_value = output['parameter_values'][i]
        if var_name == 'parameter_sets':
            plt.plot(time_vector, col_elution, label=f'{Hkfp}: {variable[i]}')
        else:
            plt.plot(time_vector, col_elution, label=f'{var_name}: {variable[i]}')

    plt.xlabel("Time (min)")
    plt.ylabel("Concentration (g/mL)")
    plt.title("Elution Curves")
    plt.legend()
    plt.grid(True)
    plt.xlim(0, tend)
    plt.show()

import matplotlib.pyplot as plt
import numpy as np

def plot_all_parametric_results(output, x_values, x_variable_name, lower_bound, upper_bound, dist_bn_points, tend, var_name, Hkfp=None):
    """
    Plots three subplots:
    (1) Mass Balance Error vs Varied Variable
    (2) Simulation Time vs Varied Variable
    (3) Elution Curves
    """

    # Extract y-values
    mb_errors = []
    sim_times = []

    for result in output['results']:
        mb_errors.append(result.get("Error_percent", 0))
        sim_times.append(result.get("Simulation_time", 0))

    # Elution data
    variable = np.arange(lower_bound, upper_bound + dist_bn_points, dist_bn_points)

    # Start figure
    fig, axs = plt.subplots(1, 3, figsize=(18, 5))

    # Plot 1: MB Error
    axs[0].plot(x_values, mb_errors, marker='o', linestyle='-', color='red')
    axs[0].axhline(0, color='black', linestyle='--', linewidth=1.2, label="Zero Error")
    axs[0].set_title(f"Mass Balance Error (%) vs {x_variable_name}")
    axs[0].set_xlabel(x_variable_name)
    axs[0].set_ylabel("Error (%)")

    y_limit = max(np.max(mb_errors), abs(np.min(mb_errors)))
    y_padding = y_limit * 0.05
    if y_limit < 10:
        y_plot = 10 + y_padding
    else:
        y_plot = y_limit + y_padding
    axs[0].set_ylim(-y_plot, y_plot)
    axs[0].legend()
    axs[0].grid(True)

    # Plot 2: Simulation Time
    axs[1].plot(x_values, sim_times, marker='o', linestyle='-', color='blue')
    axs[1].set_title(f"Computation Time (min) vs {x_variable_name}")
    axs[1].set_xlabel(x_variable_name)
    # axs[1].set_ylabel("Simulation Time (min)")
    axs[1].set_ylim(0,)
    axs[1].grid(True)

    # Plot 3: Elution Curves
    for i, result in enumerate(output['results']):
        col_elution = result['col_elution'][0]
        if iso_type == "UNC":
            time_vector = result['t_sets'][0]
        else:
            time_vector = result['t']

        label_val = f'{Hkfp}: {variable[i]}' if var_name == 'parameter_sets' else f'{var_name}: {variable[i]}'
        axs[2].plot(time_vector, col_elution, label=label_val)

    axs[2].set_title("Elution Curves (g/mL) vs Time (min)")
    axs[2].set_xlabel("Time (min)")
    # axs[2].set_ylabel("Concentration (g/mL)")
    axs[2].set_xlim(0, tend)
    axs[2].grid(True)
    axs[2].legend()

    plt.tight_layout()
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
parameter_sets = [ {"C_feed": 0.1}]
kav_params_all = [[0.05]] # [[A], [B]]
cusotom_isotherm_params_all = np.array([[1]])
Da_all = np.array([1e-6]) 

# ]
# print("size:\n", np.shape(parameter_sets))  #]#

 
Bm = 0
e = 0.4    # (0, 1]     # voidage
Q_S = 1 # cm^3/s | The volumetric flowrate of the feed to the left of the feed port (pure solvent)
Q_inj = 0.5 # cm^3/s | The volumetric flowrate of the injected concentration slug
t_index = 2 # s    # Index time # How long the SINGLE pulse holds for
tend_min = 1 # min
nx = 100
###################### COLUMN DIMENTIONS ########################
L = 30 # cm
d_col = 2 # cm


column_func_inputs = [iso_type,  Names, color, parameter_sets, Da_all, Bm, e, Q_S, Q_inj, t_index, tend_min, nx, L, d_col, cusotom_isotherm_params_all, kav_params_all]


################ EXCUTING THE FUNCTIONS ####################################
 


print('\n\n\n\nSolving Parametric Study #1 . . . . . . ')
# Units Note: 
# - All lengths are in cm
# - All concentrations are in g/cm^3 (g/mL)
# 
lower_bound = 30         # cm or g/cm^3
upper_bound = 100         # cm or g/cm^3
dist_bn_points = 10    # cm or g/cm^3
var_name = 'L'     # C_feed

Hkfp = None # 'H', 'kfp', None

column_func_inputs_names = ["iso_type", "Names", "color", "parameter_sets", "Da_all", "Bm", "e", "Q_S", "Q_inj", "t_index", "tend_min", "nx", "L", "d_col", "cusotom_isotherm_params_all", "kav_params_all" ]
Output, x_variable, x_variable_name, tend = point_value_parametric_study(var_name, lower_bound, upper_bound, dist_bn_points, Hkfp, column_func_inputs_names) # (Name of quantitiy, lower_bound, upper_bound, resolution(=space between points))
# print(F'Output: {Output}')
# Output, x_variable, x_variable_name = point_value_parametric_study('parameter_sets', 2, 10, 1, Hkfp='H') # 
# # Where resolution => space between points
# plot_parametric_results(output= Output, x_values = x_variable, y_variable_name = 'Error_percent', x_variable_name = x_variable_name, color=color)
# plot_parametric_results(output= Output, x_values = x_variable, y_variable_name = 'Simulation_time', x_variable_name = x_variable_name, color=color)
# # # plot_parametric_results(output= Output, x_values = x_variable, y_variable_name = 'Model_Acc', x_variable_name = x_variable_name, color=color)
# # # plot_parametric_results(output= Output, x_values = x_variable, y_variable_name = 'Expected_Acc', x_variable_name = x_variable_name, color=color)
# plot_parametric_results(output= Output, x_values = x_variable, y_variable_name = 'm_in', x_variable_name = x_variable_name, color=color)
# # plot_parametric_results(output= Output, x_values = x_variable, y_variable_name = 'm_out', x_variable_name = x_variable_name, color=color)
# plot_parametric_results(output= Output, x_values = x_variable, y_variable_name = 'diff', x_variable_name = x_variable_name, color=color)
plot_elution_curves(Output, tend, lower_bound, upper_bound, dist_bn_points, var_name)

plot_all_parametric_results(
    output=Output,
    x_values=x_variable,
    x_variable_name=x_variable_name,
    lower_bound=lower_bound,
    upper_bound=upper_bound,
    dist_bn_points=dist_bn_points,
    tend=tend,
    var_name=var_name,
    Hkfp=Hkfp
)


#%%
# Save Files
par_std_save_location = r"C:\Users\28820169\Downloads\BO_Papers\MEng_Code\Results_and_Discussion\Parametric_Study"
description_text = (
    f"Parametric study varying {x_variable_name} from {lower_bound} to {upper_bound} "
    f"with a step of {dist_bn_points}. Results include col_elution and mass metrics. "
    f"All lengths are in cm, time in s, concentrations in g/cm³, volumes in cm³."
)

save_output_to_json(
    output=Output,
    variable=x_variable,
    param_name=x_variable_name,
    description_text=description_text,
    save_path= None

)

# %%
