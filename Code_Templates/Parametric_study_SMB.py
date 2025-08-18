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
from SMB_func_general import SMB


# parameter_sets = [
#                     {"C_feed": 0.22},
#                     {"C_feed": 0.22}
                    
#                     ] 

# Da_all = np.array([6.218e-6, 6.38e-6]) 
# kav_params_all = np.array([[0.027], [0.053]]) 
# cusotom_isotherm_params_all = np.array([[0.27], [0.53]]) # H_glu, H_fru 


# Function to solve the concentration using the column model given a Pe, and tend
def solve_concentration(param_name, param_val, SMB_inputs, SMB_inputs_names):
    
    """
    param => string | name of parameter to be changed as listed in column_func_inputs_names

    SMB_inputs_names = ['iso_type', 'Names', 'color', 'num_comp', 'nx_per_col', 'e', 'D_all', 'Bm', 'zone_config', 'L', 'd_col', 'd_in', 't_index_min', 'n_num_cycles', 'Q_internal', 'parameter_sets', 'cusotom_isotherm_params_all', 'kav_params_all']

    """
    # Ensure the input name matches one of the SMB input names
    
    if param_name == "parameter_sets":
        name_idx = SMB_inputs_names.index(param_name)
        SMB_inputs[name_idx][0][param_name] =  [{"C_feed": param_val}, {"C_feed": 0.22}] 
    
    elif param_name == "cusotom_isotherm_params_all":
        name_idx = SMB_inputs_names.index(param_name)
        SMB_inputs[name_idx] =  np.array([[param_val], [0.53]])

    elif param_name == "kav_params_all":
        name_idx = SMB_inputs_names.index(param_name)
        SMB_inputs[name_idx][0] =  np.array([[param_val], [0.053]])

    elif param_name == "D_all":
        name_idx = SMB_inputs_names.index(param_name)
        SMB_inputs[name_idx][0] =  np.array([[param_val], [6.38e-6]])
    
    else:
        name_idx = SMB_inputs_names.index(param_name)
        SMB_inputs[name_idx] =  param_val # Insert the Peclet number in the correct position

    results = SMB(SMB_inputs)
    
    y_matrices, nx, t, t_sets, t_schedule, C_feed, m_in, m_out, raff_cprofile, ext_cprofile, raff_intgral_purity, raff_recov, ext_intgral_purity, ext_recov, raff_vflow, ext_vflow, Model_Acc, Expected_Acc, Error_percent, raff_inst_purity, ext_inst_purity, raff_inst_output_recovery, ext_inst_output_recovery, raff_output_recov, ext_output_recov = results[0:]

    return results



def generate_exponential_array(start, stop, step):
    exponents = np.arange(start, stop - 1, -step)
    return 10.0 ** exponents

def scalar_value_parametric_study(param_name, lower_bound, upper_bound, dist_bn_points, SMB_inputs, SMB_inputs_names):
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
    # if param_name != 'C_feed':
    #     if param_name not in SMB_inputs_names:
    #         raise ValueError(f"\n\n{param_name} is not a valid SMB input name. Please choose from {SMB_inputs_names}.")

    output = {'parameter_values': [], 'results': []}
    
    # array of the values of the variable to varry:
    if param_name == 'D_all':
        variable = generate_exponential_array(lower_bound, upper_bound + dist_bn_points, dist_bn_points)
        print(f'variable: {variable}')
    else:    
        variable = np.arange(lower_bound, upper_bound + dist_bn_points, dist_bn_points)

    # Get the index of the parameter to vary:
    # name_idx = SMB_inputs_names.index(param_name)
    # if SMB_inputs_names[name_idx] == 'parameter_sets':

    for i, var_value in enumerate(variable):  # Iterate over each value in the range
        
        print("\n\n\n\n\n\n\n-----------------------------------------------------------")
        print("-----------------------------------------------------------")
        print(f'Iteration {i+1}: Setting {param_name} to {var_value}')
        print("-----------------------------------------------------------")
        print("-----------------------------------------------------------\n\n")
        
        # Update the current SMB input parameter
        # column_func_inputs[name_idx] = var_value

        param_val = var_value

        # if param_name == 'cusotom_isotherm_params_all' or param_name == 'Da_all' or param_name == 'kav_params_all':
        #     param_val = np.array([[param_val]])
        # Run the simulation
        start_time = time.time()

        y_matrices, nx, t, t_sets, t_schedule, C_feed, m_in, m_out, raff_cprofile, ext_cprofile, raff_intgral_purity, raff_recov, ext_intgral_purity, ext_recov, raff_vflow, ext_vflow, Model_Acc, Expected_Acc, Error_percent, raff_inst_purity, ext_inst_purity, raff_inst_output_recovery, ext_inst_output_recovery, raff_output_recov, ext_output_recov = solve_concentration(param_name, param_val, SMB_inputs, SMB_inputs_names)
        
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
            'Error_percent': Error_percent, 
            'Simulation_time':sim_time

            })
    
    return output, variable, param_name, t_end

# FUNCTION TO SAVE RESULTS:
import json
import os
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
        filename = f'SMB_parametric_study_on_{param_name}_variation.json'
        save_path = os.path.join(os.getcwd(), filename)

    # Initialize the output dictionary
    results_dict = {
        "description_text": description_text,
        f"variable,{param_name}": convert_ndarrays(variable),
        "param_name": param_name,
        "Error_percent": [],
        "Simulation_time": [],

        'raff_intgral_purity': [],
        'raff_recov': [], 
        'ext_intgral_purity': [], 
        'ext_recov': [],
        
        "t_sets": [],
        "t": [],
        "m_in": [],
        "m_out": [],
    }

    keys = [
        'Error_percent', 'Simulation_time', 'raff_intgral_purity', 'raff_recov', 'ext_intgral_purity', 'ext_recov', 
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
        output (dict): The results dictionary returned from `scalar_value_parametric_study`.
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
        

def plot_all_parametric_results(output, x_values, x_variable_name, lower_bound, upper_bound, dist_bn_points, tend, var_name):
    """
    Plots three subplots:
    (1) Mass Balance Error vs Varied Variable
    (2) Simulation Time vs Varied Variable
    (3) Elution Curves
    """

    # Extract y-values
    mb_errors = []
    sim_times = []

    ext_intgral_purityA = [] 
    ext_intgral_purityB = []
    ext_recovA = []
    ext_recovB = []


    raff_intgral_purityA = [] 
    raff_intgral_purityB = []
    raff_recovA= []
    raff_recovB = []


# ----------- Components:
    if var_name == 'C_feed':
        var_name = 'Feed Concentration, c_feed (g/ml)'
        var_name2 = 'c_feed'
    
    elif var_name == 'cusotom_isotherm_params_all':
        var_name = 'Henry Constant, H'
        var_name2 = 'H'

    elif var_name == 'kav_params_all':
        var_name = 'Mass Transfer Coefficient, kfp (1/s)'
        var_name2 = 'kfp'

    elif var_name == 'D_all':
        var_name = 'Dispersion Coefficient, Da (cm^3/s)'
        var_name2 = 'Da'


# ----------- Flowrates:

    elif var_name == 'Q1':
        var_name = 'Section 1 Flowrate, Q (mL/s)'
        var_name2 = 'Q1'

    elif var_name == 'Q2':
        var_name = 'Section 2 Flowrate, Q (mL/s)'
        var_name2 = 'Q2'

    elif var_name == 'Q3':
        var_name = 'Section 3 Flowrate, Q (mL/s)'
        var_name2 = 'Q3'

    elif var_name == 'Q4':
        var_name = 'Section 4 Flowrate, Q (mL/s)'
        var_name2 = 'Q4'

    elif var_name == 't_index':
        var_name = 'Index Time, t_index (seconds)'
        var_name2 = 't_index'



# ----------- Columns:
    elif var_name == 'zone_config':
        var_name = 'Column Configuration'
        var_name2 = 'config'

    elif var_name == 'L':
        var_name = 'Column Length, L (cm)'
        var_name2 = 'L'

    elif var_name == 'd_col':
        var_name = 'Column Diameter, d_col (cm)'
        var_name2 = 'd_col'

    elif var_name == 'tend_min':
        var_name = 'End Time, (min)'
        var_name2 = 't_end'

    elif var_name == 'e':
        var_name = 'void fraction, e'
        var_name2 = 'e'

    elif var_name == 'nx_per_col':
        var_name = '# Spacial Nodes per column, nx'
        var_name2 = 'nx'



    for result in output['results']:
        mb_errors.append(result.get("Error_percent", 0))
        sim_times.append(result.get("Simulation_time", 0))

        ext_intgral_purityA.append(result.get("ext_intgral_purity", 0)[0])
        ext_intgral_purityB.append(result.get("ext_intgral_purity", 1)[1])
        ext_recovA.append(result.get("ext_recov", 0)[0])
        ext_recovB.append(result.get("ext_recov", 1)[1])

        raff_intgral_purityA.append(result.get("raff_intgral_purity", 0)[0])
        raff_intgral_purityB.append(result.get("raff_intgral_purity", 1)[1])
        raff_recovA.append(result.get("raff_recov", 0)[0])
        raff_recovB.append(result.get("raff_recov", 1)[1])

    # Elution data
    if x_variable_name == 'D_all':
        variable = generate_exponential_array(lower_bound, upper_bound + dist_bn_points, dist_bn_points)
        
        # print(f'variable: {variable}')
    else:
        variable = np.arange(lower_bound, upper_bound + dist_bn_points, dist_bn_points)

    # Start figure
    fig, axs = plt.subplots(1, 3, figsize=(18, 5))

    # Plot 1: MB Error
    axs[0].plot(x_values, mb_errors, marker='o', linestyle='-', color='red')
    axs[0].axhline(0, color='black', linestyle='--', linewidth=1.2, label="Zero Error")
    axs[0].set_title(f"Mass Balance Error (%)") # vs {x_variable_name}")
    axs[0].set_xlabel(var_name)
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



    # Extract
    axs[1].plot(x_values, np.array(ext_intgral_purityA)*100, marker='o', linestyle='-', color='red', label = 'Pur_A')
    axs[1].plot(x_values, np.array(ext_intgral_purityB)*100, marker='s', linestyle='-', color='green', label = 'Pur_B')
    axs[1].plot(x_values, np.array(ext_recovA)*100,         marker='^', linestyle='-', color='k', label = 'Rec_A')
    axs[1].plot(x_values, np.array(ext_recovB)*100,        marker='d', linestyle='-', color='orange', label = 'Rec_B')

    axs[1].set_title("Extract Purity and Recoveries")
    axs[1].set_xlabel(var_name)
    # axs[2].set_ylabel("Concentration (g/mL)")
    # axs[2].set_xlim(0, tend)
    axs[1].grid(True)
    axs[1].set_ylim(0, 100)
    axs[1].legend()

    
    # Raffinate
    axs[2].plot(x_values, np.array(raff_intgral_purityA)*100, marker='o', linestyle='-', color='red', label = 'Pur_A')
    axs[2].plot(x_values, np.array(raff_intgral_purityB)*100, marker='s', linestyle='-', color='green', label = 'Pur_B')
    axs[2].plot(x_values, np.array(raff_recovA)*100,         marker='^', linestyle='-', color='k', label = 'Rec_A')
    axs[2].plot(x_values, np.array(raff_recovB)*100,        marker='d', linestyle='-', color='orange', label = 'Rec_B')

    axs[2].set_title("Raffinate Purity and Recoveries")
    axs[2].set_xlabel(var_name)
    # axs[2].set_ylabel("Concentration (g/mL)")
    # axs[2].set_xlim(0, tend)
    axs[2].grid(True)
    axs[2].set_ylim(0, 100)
    axs[2].legend()

    plt.tight_layout()
    plt.show()

    # Start new figure for simulation time:

    fig, bxs = plt.subplots(1, 1, figsize=(18, 5))
        # Plot 2: Simulation Time
    bxs.plot(x_values, sim_times, marker='o', linestyle='-', color='blue')
    bxs.set_title(f"Computation Time (min)") # vs {x_variable_name}")
    bxs.set_xlabel(var_name)
    # axs[1].set_ylabel("Simulation Time (min)")
    # axs[1].set_ylim(0,)
    bxs.grid(True)



def mj_to_Qj(mj, t_index_min):
    '''
    Converts flowrate ratios to internal flowrates - flowrates within columns
    '''
    Qj = (mj*V_col*(1-e) + V_col*e)/(t_index_min*60) # cm^3/s
    return Qj

##################### PRIMARY INPUTS #########################

# --------------- FUNCTION EVALUATION SECTION

# SMB VARIABLES
# ######################################################
# What tpye of isoherm is required?
# Coupled: "CUP"
# Uncoupled: "UNC"
iso_type = "CUP"

###################### PRIMARY INPUTS #########################
# Define the names, colors, and parameter sets for 6 components
Names = ["Glucose", "Fructose"]#, 'C', 'D']#, "C"]#, "D", "E", "F"]
colors = ["green", "orange"]    #, "purple", "brown"]#, "b"]#, "r", "purple", "brown"]
num_comp = len(Names) # Number of components
e = 0.56 # 0.56         # bed voidage
Bm = 300

# Column Dimensions

# How many columns in each Zone?

Z1, Z2, Z3, Z4 = 1, 1, 1, 1 # *3 for smb config
zone_config = np.array([Z1, Z2, Z3, Z4])

# sub_zone information - EASIER TO FILL IN IF YOU DRAW THE SYSTEM
# -----------------------------------------------------
# sub_zone_j = [[feed_bays], [reciveinig_bays]]
# -----------------------------------------------------
# feed_bay = the bay(s) that feed the set of reciveing bays in "reciveinig_bays" e.g. [2] or [2,3,4] 
# reciveinig_bays = the set of bayes that recieve material from the feed bay

"""
sub-zones are counted from the feed onwards i.e. sub_zone_1 is the first subzone "seen" by the feed stream. 
Bays are counted in the same way, starting from 1 rather than 0
"""
# Borate-HCL
sub_zone_1 = [[22, 23, 24], [1]] # ---> in subzone 1, there are 2 columns stationed at bay 3 and 4. Bay 3 and 4 recieve feed from bay 1"""

sub_zone_2 = [[3], [4,5,6]] 
sub_zone_3 = [[4,5,6], [7]] 

sub_zone_4 = [[9], [10, 11, 12]] 
sub_zone_5 = [[10,11,12], [13]]

sub_zone_6 = [[15],[16, 17, 18]]
sub_zone_7 = [[16, 17, 18],[19]]

sub_zone_8 = [[21],[22, 23, 24]]


subzone_set = [sub_zone_1,sub_zone_2,sub_zone_3,sub_zone_4,sub_zone_5,sub_zone_6, sub_zone_7, sub_zone_8]
subzone_set = []
# # PACK:
# subzone_set = [sub_zone_1,sub_zone_2,sub_zone_3,sub_zone_4,sub_zone_5,sub_zone_6,sub_zone_7,sub_zone_8]

# Glucose Fructose
# sub_zone_1 = [[3], [4,5,6]] # ---> in subzone 1, there are 2 columns stationed at bay 3 and 4. Bay 3 and 4 recieve feed from bay 1"""
# sub_zone_2 = [[4,5,6], [7,8,9]] 

# sub_zone_3 = [[7,8,9], [10,11,12]] 
# sub_zone_4 = [[10,11,12], [13]]

# sub_zone_5 = [[15],[16, 17, 18]]
# sub_zone_6 = [[16, 17, 18],[19, 20, 21]]
# sub_zone_7 = [[19, 20, 21], [22, 23, 24]]
# sub_zone_8 = [[22, 23, 24], [1]]

# PACK:
# subzone_set = [sub_zone_1,sub_zone_2,sub_zone_3,sub_zone_4,sub_zone_5,sub_zone_6,sub_zone_7,sub_zone_8]
# subzone_set = [] # no subzoning

# PLEASE ASSIGN THE BAYS THAT ARE TO THE IMMEDIATE LEFT OF THE RAFFIANTE AND EXTRACT
# product_bays = [2, 5] # [raff, extract]



L = 70 # cm # Length of one column
d_col = 4.5 # cm # column internal diameter

# Calculate the radius
r_col = d_col / 2
# Calculate the area of the base
A_col = np.pi * (r_col ** 2) # cm^2
V_col = A_col*L # cm^3
# Dimensions of the tubing and from each column:
# Assuming the pipe diameter is 20% of the column diameter:
d_in = 0.2 * d_col # cm
nx_per_col = 15


################ Time Specs #################################################################################
t_index_min = 10 # min # Index time # How long the pulse holds before swtiching
n_num_cycles = 6   # Number of Cycles you want the SMB to run for
t_simulation_end = None # HRS
###############  FLOWRATES  #################################################################################

# # Jochen et al:
# Q_P, Q_Q, Q_R, Q_S = 5.21, 4, 5.67, 4.65 # x10-7 m^3/s
# conv_fac = 0.1 # x10-7 m^3/s => cm^3/s
# Q_P, Q_Q, Q_R, Q_S  = Q_P*conv_fac, Q_Q*conv_fac, Q_R*conv_fac, Q_S*conv_fac

# Q_I, Q_II, Q_III, Q_IV = Q_R,  Q_S, Q_P, Q_Q

# # Q_I, Q_II, Q_III, Q_IV = 2,1,2,1
# Q_I, Q_II, Q_III, Q_IV = 11/3.6, 8.96/3.6, 9.96/3.6, 7.96/3.6 # L/h





# # Parameter Sets for different components
################################################################

# Units:
# - Concentrations: g/cm^3
# - kh: 1/s
# - Da: cm^2/s

# A must have a less affinity to resin that B - FOUND IN EXtract purity
# Parameter sets for different components
# Units:
# - Concentrations: g/cm^3
# - kfp: 1/s
# parameter_sets = [
#                     {"C_feed": 0.09222},    # Glucose SMB Launch
#                     {"C_feed": 0.061222}]   # Fructose

# kav_params_all = np.array([[0.1467], [0.1462]])
# cusotom_isotherm_params_all = np.array([[2.71],[2.94]])
# Da_all = np.array([3.218e-6, 8.38e-6 ]) 

parameter_sets = [ {"C_feed": 0.003190078*1.4}, {"C_feed": 0.012222*0.8}] 
Da_all = np.array([5.77e-7, 2.3812e-7]) 
kav_params_all = np.array([[0.170], [0.154]])
cusotom_isotherm_params_all = np.array([[2.13], [2.4]]) # [ [H_borate], [H_hcl] ]

m1, m2, m3, m4 = 3.5, 2.13, 2.4, 1.5
Q_I, Q_II, Q_III, Q_IV = mj_to_Qj(m1, t_index_min), mj_to_Qj(m2, t_index_min), mj_to_Qj(m3, t_index_min), mj_to_Qj(m4, t_index_min)

Q_internal = np.array([Q_I, Q_II, Q_III, Q_IV])


# Linear, H
parameter_sets = [{"C_feed": 0.22}, {"C_feed": 0.22}] 
D_all = np.array([6.218e-6, 6.38e-6]) 
kav_params_all = np.array([[0.027], [0.053]]) 
cusotom_isotherm_params_all = np.array([[0.27], [0.53]]) # H_glu, H_fru 
# Sub et al = np.array([[0.27], [0.53]])


# STORE/INITALIZE SMB VAIRABLES
SMB_inputs = [iso_type, Names, colors, num_comp, nx_per_col, e, Da_all, Bm, zone_config, L, d_col, d_in, t_index_min, n_num_cycles, Q_internal, parameter_sets, cusotom_isotherm_params_all, kav_params_all, subzone_set, t_simulation_end]
SMB_inputs_names = ['iso_type', 'Names', 'color', 'num_comp', 'nx_per_col', 'e', 'D_all', 'Bm', 'zone_config', 'L', 'd_col', 'd_in', 't_index_min', 'n_num_cycles', 'Q_internal', 'parameter_sets', 'cusotom_isotherm_params_all', 'kav_params_all',  'subzone_set', 't_simulation_end']
#%% ---------- SAMPLE RUN IF NECESSARY
# start_test = time.time()
# y_matrices, nx, t, t_sets, t_schedule, C_feed, m_in, m_out, raff_cprofile, ext_cprofile, raff_intgral_purity, raff_recov, ext_intgral_purity, ext_recov, raff_vflow, ext_vflow, Model_Acc, Expected_Acc, Error_percent = SMB(SMB_inputs)
# end_test = time.time()

# duration = end_test - start_test
# print(f'Simulation Took: {duration/60} min')




################ EXCUTING THE FUNCTIONS ####################################
 


print('\n\n\n\nSolving Parametric Study #1 . . . . . . ')
# Units Note: 
# - All lengths are in cm
# - All concentrations are in g/cm^3 (g/mL)
# 
lower_bound = 10       # cm or g/cm^3
upper_bound = 30   # cm or g/cm^3
dist_bn_points = 10  # cm or g/cm^3
var_name = 'nx_per_col'     # C_feed

Output, x_variable, x_variable_name, tend = scalar_value_parametric_study(var_name, lower_bound, upper_bound, dist_bn_points, SMB_inputs, SMB_inputs_names) # (Name of quantitiy, lower_bound, upper_bound, resolution(=space between points))
# print(F'Output: {Output}')
# Output, x_variable, x_variable_name = scalar_value_parametric_study('parameter_sets', 2, 10, 1, Hkfp='H') # 
# # Where resolution => space between points
# plot_parametric_results(output= Output, x_values = x_variable, y_variable_name = 'Error_percent', x_variable_name = x_variable_name, color=color)
# plot_parametric_results(output= Output, x_values = x_variable, y_variable_name = 'Simulation_time', x_variable_name = x_variable_name, color=color)
# plot_parametric_results(output= Output, x_values = x_variable, y_variable_name = 'Model_Acc', x_variable_name = x_variable_name, color=color)
# plot_parametric_results(output= Output, x_values = x_variable, y_variable_name = 'Expected_Acc', x_variable_name = x_variable_name, color=color)
# plot_parametric_results(output= Output, x_values = x_variable, y_variable_name = 'm_in', x_variable_name = x_variable_name, color=color)
# # plot_parametric_results(output= Output, x_values = x_variable, y_variable_name = 'm_out', x_variable_name = x_variable_name, color=color)
# plot_parametric_results(output= Output, x_values = x_variable, y_variable_name = 'diff', x_variable_name = x_variable_name, color=color)
# plot_elution_curves(Output, tend, lower_bound, upper_bound, dist_bn_points, var_name)

#%%
plot_all_parametric_results(
    output=Output,
    x_values=x_variable,
    x_variable_name=x_variable_name,
    lower_bound=lower_bound,
    upper_bound=upper_bound,
    dist_bn_points=dist_bn_points,
    tend=tend,
    var_name=var_name,
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
