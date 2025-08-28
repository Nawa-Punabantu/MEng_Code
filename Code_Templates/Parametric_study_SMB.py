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


"""
All studies are for binary components only.

For kineatic stuff, only changes to the more adsorbing component, "A" in [A, B] are implimented i.e. the other is kept fixed

"""
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

    SMB_inputs_names = ['iso_type', 'Names', 'color', 'num_comp', 'nx_per_col', 'e', 'D_all', 'Bm', 'zone_config', 'L', 'd_col', 'd_in', 't_index_min', 'n_num_cycles', 'Q_internal', 'parameter_sets', 'cusotom_isotherm_params_all', 'kav_params_all', 'subzone_set', 't_simulation_end']

    """
    # Ensure the input name matches one of the SMB input names
    
    if param_name == "C_feed":
        name_idx = SMB_inputs_names.index('parameter_sets')
        SMB_inputs[name_idx][0][param_name] =  param_val 
    
    elif param_name == "cusotom_isotherm_params_all":
        name_idx = SMB_inputs_names.index(param_name)
        SMB_inputs[name_idx][0] =  param_val

    elif param_name == "kav_params_all":
        name_idx = SMB_inputs_names.index(param_name)
        SMB_inputs[name_idx][0] =  [param_val]

    elif param_name == "D_all":
        
        print(f'param_name: {param_name}')
        name_idx = SMB_inputs_names.index(param_name)
        print(f'name_idx: {name_idx}')
        SMB_inputs[name_idx][0] =  param_val

    elif param_name == "zone_config":
        print(f'param_name: {param_name}')
        name_idx = SMB_inputs_names.index(param_name)
        print(f'name_idx: {name_idx}')
        SMB_inputs[name_idx] =  param_val
    
    else:
        name_idx = SMB_inputs_names.index(param_name)
        SMB_inputs[name_idx] =  param_val # Insert the Peclet number in the correct position

    results = SMB(SMB_inputs)
    
    
    y_matrices, nx, t, t_sets, t_schedule, C_feed, m_in, m_out, raff_cprofile, ext_cprofile, raff_intgral_purity, raff_feed_recov, ext_intgral_purity, ext_feed_recov, raff_vflow, ext_vflow, Model_Acc, Expected_Acc, Error_percent, raff_inst_purity, ext_inst_purity, raff_inst_feed_recovery, ext_inst_feed_recovery, raff_inst_output_recovery, ext_inst_output_recovery, raff_avg_cprofile, ext_avg_cprofile, raff_avg_mprofile, ext_avg_mprofile, t_schedule, raff_output_recov, ext_output_recov = results[0:]
                 
    

    return results



def generate_exponential_array(start, stop, step):
    exponents = np.arange(start, stop - 1, -step)
    return 10.0 ** exponents

def scalar_value_parametric_study(param_name, lower_bound, upper_bound, dist_bn_points, SMB_inputs, SMB_inputs_names, ZONE_CONFIGS = None):
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

    if param_name == 'zone_config' and ZONE_CONFIGS != None:
        variable = ZONE_CONFIGS

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

        y_matrices, nx, t, t_sets, t_schedule, C_feed, m_in, m_out, raff_cprofile, ext_cprofile, raff_intgral_purity, raff_feed_recov, ext_intgral_purity, ext_feed_recov, raff_vflow, ext_vflow, Model_Acc, Expected_Acc, Error_percent, raff_inst_purity, ext_inst_purity, raff_inst_feed_recovery, ext_inst_feed_recovery, raff_inst_output_recovery, ext_inst_output_recovery, raff_avg_cprofile, ext_avg_cprofile, raff_avg_mprofile, ext_avg_mprofile, t_schedule, raff_output_recov, ext_output_recov = solve_concentration(param_name, param_val, SMB_inputs, SMB_inputs_names)
        
        end_time = time.time()
        sim_time = (end_time - start_time)/60 # min
        

        
        # Save the parameter value and corresponding results
        output['parameter_values'].append(var_value)
        output['results'].append({      # Appended scalars:
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
                                        'raff_feed_recov': raff_feed_recov,
                                        'ext_intgral_purity': ext_intgral_purity,
                                        'ext_feed_recov': ext_feed_recov,

                                        'raff_vflow': raff_vflow,
                                        'ext_vflow': ext_vflow,

                                        'Model_Acc': Model_Acc,
                                        'Expected_Acc': Expected_Acc,
                                        'Error_percent': Error_percent,

                                        'raff_inst_purity': raff_inst_purity,
                                        'ext_inst_purity': ext_inst_purity,
                                        'raff_inst_feed_recovery': raff_inst_feed_recovery,
                                        'ext_inst_feed_recovery': ext_inst_feed_recovery,
                                        'raff_inst_output_recovery': raff_inst_output_recovery,
                                        'ext_inst_output_recovery': ext_inst_output_recovery,

                                        'raff_avg_cprofile': raff_avg_cprofile,
                                        'ext_avg_cprofile': ext_avg_cprofile,
                                        'raff_avg_mprofile': raff_avg_mprofile,
                                        'ext_avg_mprofile': ext_avg_mprofile,

                                        'raff_output_recov': raff_output_recov,
                                        'ext_output_recov': ext_output_recov,

                                        'Simulation_time': sim_time
                                        })

    if iso_type == "UNC":
        t_end = t_sets[0][-1]   
    elif iso_type == 'CUP':
            t_end = t[-1]                        
    
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


def plot_all_parametric_results(output, x_values, var_name):
    """
    Plots:
    (1) Mass Balance Error vs Varied Variable
    (2) Raffinate instantaneous purity & recovery (comp 0)
    (3) Extract instantaneous purity & recovery (comp 0)
    """

    # Extract MB error + sim time
    mb_errors = [r.get("Error_percent", 0) for r in output["results"]]
    sim_times = [r.get("Simulation_time", 0) for r in output["results"]]

    n_cases = len(output["results"])

    # Build colormap scaled from min→max x_values
    norm = plt.Normalize(vmin=min(x_values), vmax=max(x_values))
    cmap = plt.cm.get_cmap("viridis")  # could swap with other
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)

    fig, axs = plt.subplots(1, 3, figsize=(18, 5))

    # --- Plot 1: MB Error vs parameter ---
    axs[0].plot(x_values, mb_errors, marker="o", color="red")
    axs[0].axhline(0, color="black", linestyle="--", linewidth=1)
    axs[0].set_title("Mass Balance Error (%)")
    axs[0].set_xlabel(var_name)
    axs[0].set_ylabel("Error (%)")
    axs[0].grid(True)

    # --- Plot 2: Raffinate inst. purity & recovery (comp 0) ---
    for idx, result in enumerate(output["results"]):
        raff_purity = np.array(result["raff_inst_purity"][0]) * 100
        raff_recovery = np.array(result["raff_inst_output_recovery"][0]) * 100
        t_schedule_1 = result["t_schedule"].copy()
        t_indexing  = t_schedule_1[1] - t_schedule_1[0] # s

        for i, t in enumerate(t_schedule_1):
            t_schedule_1[i] = t + t_indexing
        
        where_to_insert = 0
        t_schedule_1.insert(where_to_insert, 0)

        color = sm.to_rgba(x_values[idx])

        axs[1].plot(np.array(t_schedule_1)/3600, raff_purity, color=color, linestyle="-", label=f"Purity {x_values[idx]}")
        axs[1].plot(np.array(t_schedule_1)/3600, raff_recovery, color=color, linestyle="--", label=f"Recovery {x_values[idx]}")

    axs[1].set_title(f"Raffinate {Names[0]} Inst. Purity & Recovery")
    axs[1].set_xlabel("Time (s)")
    axs[1].set_ylabel("Value (%)")
    axs[1].set_ylim(0, 105)
    axs[1].grid(True)

    # --- Plot 3: Extract inst. purity & recovery (comp 0) ---
    for idx, result in enumerate(output["results"]):
        ext_purity = np.array(result["ext_inst_purity"][1]) * 100
        ext_recovery = np.array(result["ext_inst_output_recovery"][1]) * 100
        t_schedule = result["t_schedule"].copy()
        t_indexing  = t_schedule[1] - t_schedule[0] # s

        for i, t in enumerate(t_schedule):
            t_schedule[i] = t + t_indexing
        
        where_to_insert = 0
        t_schedule.insert(where_to_insert, 0)
        color = sm.to_rgba(x_values[idx])

        axs[2].plot(np.array(t_schedule)/3600, ext_purity, color=color, linestyle="-", label=f"Purity {x_values[idx]}")
        axs[2].plot(np.array(t_schedule)/3600, ext_recovery, color=color, linestyle="--", label=f"Recovery {x_values[idx]}")

    axs[2].set_title(f"Extract {Names[1]} Inst. Purity & Recovery")
    axs[2].set_xlabel("Time (s)")
    axs[2].set_ylabel("Value (%)")
    axs[2].set_ylim(0, 105)
    axs[2].grid(True)

    plt.tight_layout()
    plt.show()

    # --- Separate legend figure ---
    fig_leg, ax_leg = plt.subplots(figsize=(8, 1))
    fig_leg.subplots_adjust(bottom=0.5)
    cb = plt.colorbar(sm, cax=ax_leg, orientation="horizontal")
    cb.set_label(var_name)
    plt.show()

    # --- Separate sim time figure ---
    fig, bx = plt.subplots(figsize=(8, 5))
    bx.plot(x_values, sim_times, marker="o", color="blue")
    bx.set_title("Simulation Time (min)")
    bx.set_xlabel(var_name)
    bx.set_ylabel("Time (min)")
    bx.grid(True)
    plt.show()

def plot_parametric_inst_purity_recovery(output, x_values, var_name, component_index=0):
    """
    Plots instantaneous purity and recovery for raffinate and extract
    vs t_schedule for a parametric study.
    
    Parameters:
    - output: dict, output from simulation
    - x_values: list or array, values of the varied parameter
    - var_name: str, name of the varied parameter for labeling
    - component_index: int, component to plot (default 0)
    """

    # Color gradient for lighter to darker for low to high x_values
    from matplotlib.cm import get_cmap
    cmap = get_cmap("viridis")
    colors = [cmap(i/(len(x_values)-1)) for i in range(len(x_values))]

    fig, axs = plt.subplots(1, 2, figsize=(18, 5))

    for i, result in enumerate(output['results']):
        # t_schedule vector
        t_schedule = np.array(result['t_schedule'])

        # Raffinate: purity and recovery of first component
        raff_purity = np.array(result['raff_inst_purity'])[component_index] * 100
        raff_recovery = np.array(result['raff_inst_output_recovery'])[component_index] * 100

        # Extract: purity and recovery of first component
        ext_purity = np.array(result['ext_inst_purity'])[component_index] * 100
        ext_recovery = np.array(result['ext_inst_output_recovery'])[component_index] * 100

        # Label
        if isinstance(x_values[i], float):
            label_val = f"{var_name}: {x_values[i]:.2f}"
        else:
            label_val = f"{var_name}: {x_values[i]}"

        # Raffinate plot
        axs[0].plot(t_schedule, raff_purity, color=colors[i], linestyle='-', label=f'Purity {label_val}')
        axs[0].plot(t_schedule, raff_recovery, color=colors[i], linestyle='--', label=f'Recovery {label_val}')

        # Extract plot
        axs[1].plot(t_schedule, ext_purity, color=colors[i], linestyle='-', label=f'Purity {label_val}')
        axs[1].plot(t_schedule, ext_recovery, color=colors[i], linestyle='--', label=f'Recovery {label_val}')

    # Titles and labels
    axs[0].set_title("Raffinate Instantaneous Purity and Recovery (%)")
    axs[0].set_xlabel("Time (s)")
    axs[0].set_ylabel("Percent (%)")
    axs[0].grid(True)
    axs[0].legend(fontsize=8)

    axs[1].set_title("Extract Instantaneous Purity and Recovery (%)")
    axs[1].set_xlabel("Time (s)")
    axs[1].set_ylabel("Percent (%)")
    axs[1].grid(True)
    axs[1].legend(fontsize=8)

    plt.tight_layout()
    plt.show()

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

Z1, Z2, Z3, Z4 = 2, 2, 2, 2 # *3 for smb config
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
d_col = 5 # cm # column internal diameter

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
n_num_cycles = 10   # Number of Cycles you want the SMB to run for
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

# parameter_sets = [ {"C_feed": 0.003190078*1.4}, {"C_feed": 0.012222*0.8}] 
parameter_sets = [ {"C_feed": 0.01222}, {"C_feed": 0.012222}] 
D_all = np.array([5.77e-7, 2.3812e-7]) 
kav_params_all = np.array([[0.170], [0.154]])
cusotom_isotherm_params_all = np.array([[2.13], [2.4]]) # [ [H_borate], [H_hcl] ]

# Linear, H
# parameter_sets = [{"C_feed": 0.22}, {"C_feed": 0.22}] 
# D_all = np.array([6.218e-6, 6.38e-6]) 
# kav_params_all = np.array([[0.027], [0.053]]) 
# cusotom_isotherm_params_all = np.array([[0.27], [0.53]]) # H_glu, H_fru 
# Sub et al = np.array([[0.27], [0.53]])


m1, m2, m3, m4 = 3.5, 2.13, 2.4, 1.5
Q_I, Q_II, Q_III, Q_IV = mj_to_Qj(m1, t_index_min), mj_to_Qj(m2, t_index_min), mj_to_Qj(m3, t_index_min), mj_to_Qj(m4, t_index_min)

Q_internal = np.array([Q_I, Q_II, Q_III, Q_IV])




# STORE/INITALIZE SMB VAIRABLES
SMB_inputs = [iso_type, Names, colors, num_comp, nx_per_col, e, D_all, Bm, zone_config, L, d_col, d_in, t_index_min, n_num_cycles, Q_internal, parameter_sets, cusotom_isotherm_params_all, kav_params_all, subzone_set, t_simulation_end]
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
lower_bound = 1      # cm or g/cm^3 -10
upper_bound = 3   # cm or g/cm^3       -5
dist_bn_points = 1  # cm or g/cm^3     -1
# ZONE_CONFIGS = None

# //// if varying hte zone zone config:
# config1 = np.array([1,1,1,1])
# config2 = np.array([2,2,2,2])
# config3 = np.array([2,1,2,1])
# config4 = np.array([1,2,1,2])

# ZONE_CONFIGS = [config1, config2, config3, config4]
var_name = 'cusotom_isotherm_params_all'     # C_feed, 

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

plot_all_parametric_results(output = Output, x_values = x_variable, var_name = x_variable_name)
plot_parametric_inst_purity_recovery()
# plot_all_parametric_results(
#     output=Output,
#     x_values=x_variable,
#     x_variable_name=x_variable_name,
#     lower_bound=lower_bound,
#     upper_bound=upper_bound,
#     dist_bn_points=dist_bn_points,
#     tend=tend,
#     var_name=var_name,
# )


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
