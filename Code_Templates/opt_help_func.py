# -*- coding: utf-8 -*-
# # %%


#%%
import numpy as np
# -*- coding: utf-8 -*-
# # %%


#%%
import numpy as np
from scipy.optimize import minimize
from scipy.integrate import solve_ivp
from scipy import integrate
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C, Matern
from scipy.optimize import differential_evolution
from scipy.optimize import minimize, NonlinearConstraint
import json
from scipy.stats import norm
from scipy.integrate import solve_ivp
from scipy import integrate
import warnings
import time



# DEFINE INPUTS:

# BATCH 1: GLUCOSE FRUCTOSE

# SMB VARIABLES
#######################################################


###################### PRIMARY INPUTS #########################
# Define the names, colors, and parameter sets for 6 components
Names = ["Borate", "HCl"]#, 'C', 'D']#, "C"]#, "D", "E", "F"]
color = ["red", "green"]#, "purple", "brown"]#, "b"]#, "r", "purple", "brown"]
num_comp = len(Names) # Number of components
e = 0.56        # bed voidage
Bm = 300

# Column Dimensions

# How many columns in each Zone?

Z1, Z2, Z3, Z4 = 1,1,1,1 # *3 for smb config
zone_config = np.array([Z1, Z2, Z3, Z4])
nnn = Z1 + Z2 + Z3 + Z4
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

# # PACK:
subzone_set = [sub_zone_1,sub_zone_2,sub_zone_3,sub_zone_4,sub_zone_5,sub_zone_6, sub_zone_7, sub_zone_8]
subzone_set = []





L = 60 # cm # Length of one column
d_col = 5 # cm # column internal diameter

# Calculate the radius
r_col = d_col / 2
# Calculate the area of the base
A_col = np.pi * (r_col ** 2) # cm^2
V_col = A_col*L # cm^3
# Dimensions of the tubing and from each column:
# Assuming the pipe diameter is 20% of the column diameter:
d_in = 1 # cm
nx_per_col = 15


################ Time Specs #################################################################################
t_index_min = 10      # min    # Index time # How long the pulse holds before swtiching
n_num_cycles = None   # Number of Cycles you want the SMB to run for
t_simulation_end = 12 # hrs
###############  FLOWRATES   #################################################################################

# Jochen et al:
# Q_P, Q_Q, Q_R, Q_S = 5.21, 4, 5.67, 4.65 # x10-7 m^3/s
conv_fac = 0.1 # x10-7 m^3/s => cm^3/s
# # Q_P, Q_Q, Q_R, Q_S  = Q_P*conv_fac, Q_Q*conv_fac, Q_R*conv_fac, Q_S*conv_fac
# Q_I, Q_II, Q_III, Q_IV  = Q_P, Q_Q, Q_R, Q_S 
# Q_internal = np.array([Q_I, Q_II, Q_III, Q_IV])

# Other flowrates:
Q_I, Q_II, Q_III, Q_IV = 5.8, 4.2, 5.26, 4.11 # L/h
Q_internal = np.array([Q_I, Q_II, Q_III, Q_IV])/3.6 # L/h => cm^3/s




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



# ISOTHERM PARAMETERS
###########################################################################################
# What tpye of isoherm is required?
# Coupled: "CUP"
# Uncoupled: "UNC"
iso_type = "CUP"


# Do you want to maximize or minimize the objective function?
job_max_or_min = 'maximize'

# - - - - -
# Primary Varaibles
# - - - - -
t_reff = 10 # min
# - - - - -
Q_max = 12 # L/h
Q_min = 1 # L/h
# - - - - -
m_max = 1.5
m_min = 0.27
# - - - - -
sampling_budget = 1 #
optimization_budget = 1 #100
constraint_threshold = [0.995, 0.995] # [Glu, Fru]
# - - - - -
PF_weight = 10 # Weight applied to the probability of feasibility
# - - - - -
bounds = [  
(0.6, m_max), # m1
(0.1, 0.7), # m2
(0.4, m_max), # m3
(0.05, m_max*0.5), # m4
(0.2, 1) # t_index/t_reff (normalized)
]




# ---------------------------- BATCHES (DIFFERENT IN TERMS OF KINETICS)


# BATCH 1: GLUCOSE FRUCTOSE
parameter_sets = [
                    {"C_feed": 0.09222},    # Glucose SMB Launch
                    {"C_feed": 0.061222}]   # Fructose

test_kfp = 0.001
kav_params_all = np.array([[0.1467], [0.1462]])
kav_params_all = np.array([[test_kfp], [test_kfp]])

cusotom_isotherm_params_all = np.array([[2.71],[2.94]])
Da_all = np.array([3.218e-6, 8.38e-6 ]) 
# When saving the json:
Description = [f"Description example: Optimizting the Glu-Fru system for the sythetic solution on PCR.Ca. {optimization_budget} iterations, with no isotherom data. We placed a upper flowrate constraint of {Q_min}<Q<{Q_max} L/h. This is with the sub-zoning for bor-hcl"]
save_name_inputs = f"Glu_Fru-type1_{optimization_budget+1}iter_norm_config_all_inputs.json" # (1) "ILLOVO_PCR_borhcl-type1_40iter_norm_config_all_inputs.json", (2) "ILLOVO_PCR-borhcl-type1_40iter_subzone_config_all_inputs.json"
save_name_outputs = f"Glu_Fru-type1_{optimization_budget+1}iter_norm_config_all_outputs.json" # (1) "ILLOVO_PCR-borhcl-type1_40iter_norm_config_all_outputs.json", (2) "ILLOVO_PCR-borhcl-type1_40iter_subzone_config_all_outputs.json"
    
# PACK:
opt_inputs_b1 = [Description, save_name_inputs, save_name_outputs, job_max_or_min, t_reff, Q_max, Q_min, m_max, m_min, sampling_budget, optimization_budget, constraint_threshold, PF_weight, bounds]
SMB_inputs_b1 = [iso_type, Names, color, num_comp, nx_per_col, e, Da_all, Bm, zone_config, L, d_col, d_in, t_index_min, n_num_cycles, Q_internal, parameter_sets, cusotom_isotherm_params_all, kav_params_all, subzone_set, t_simulation_end]
# -------------------------------
batch_1 = [opt_inputs_b1, SMB_inputs_b1]


# --------------------------------------------------------------------------
# BATCH 2: BORATE-HCL - SYNTHETIC SOLUTION - UBK
# # --------------------------------------------------------------------------

# Glucose Fructose
sub_zone_1 = [[3], [4,5,6]] # ---> in subzone 1, there are 2 columns stationed at bay 3 and 4. Bay 3 and 4 recieve feed from bay 1"""
sub_zone_2 = [[4,5,6], [7,8,9]] 

sub_zone_3 = [[7,8,9], [10,11,12]] 
sub_zone_4 = [[10,11,12], [13]]

sub_zone_5 = [[15],[16, 17, 18]]
sub_zone_6 = [[16, 17, 18],[19, 20, 21]]
sub_zone_7 = [[19, 20, 21], [22, 23, 24]]
sub_zone_8 = [[22, 23, 24], [1]]
# PACK:
subzone_set = [sub_zone_1,sub_zone_2,sub_zone_3,sub_zone_4,sub_zone_5,sub_zone_6,sub_zone_7,sub_zone_8]
subzone_set = []

parameter_sets = [ {"C_feed": 0.003190078*1.4}, {"C_feed": 0.012222*0.8}] 
Da_all = np.array([3.218e-6, 8.38e-6 ]) 
kav_params_all = np.array([[0.395], [0.151]])
kav_params_all = np.array([[test_kfp], [test_kfp]])

cusotom_isotherm_params_all = np.array([[3.63], [2.40]]) # [ [H_borate], [H_hcl] ]
# When saving the json:
Description = [f"Description example: Optimizting the Borate HCl system for the sythetic solution on PCR.Ca. {optimization_budget} iterations, with no isotherom data. We placed a upper flowrate constraint of {Q_max} L/h. This is with the sub-zoning for bor-hcl"]
save_name_inputs = "SYNTH_UBK-borhcl-type1_40iter_norm_config_all_inputs.json" # (1) "ILLOVO_PCR_borhcl-type1_40iter_norm_config_all_inputs.json", (2) "ILLOVO_PCR-borhcl-type1_40iter_subzone_config_all_inputs.json"
save_name_outputs = "SYNTH_UBK-borhcl-type1_40iter_norm_config_all_outputs.json" # (1) "ILLOVO_PCR-borhcl-type1_40iter_norm_config_all_outputs.json", (2) "ILLOVO_PCR-borhcl-type1_40iter_subzone_config_all_outputs.json"
# PACK:
opt_inputs_b2 = [Description, save_name_inputs, save_name_outputs, job_max_or_min, t_reff, Q_max, Q_min, m_max, m_min, sampling_budget, optimization_budget, constraint_threshold, PF_weight, bounds]
SMB_inputs_b2 = [iso_type, Names, color, num_comp, nx_per_col, e, Da_all, Bm, zone_config, L, d_col, d_in, t_index_min, n_num_cycles, Q_internal, parameter_sets, cusotom_isotherm_params_all, kav_params_all, subzone_set, t_simulation_end]
# -------------------------------
batch_2 = [opt_inputs_b2, SMB_inputs_b2]



# --------------------------------------------------------------------------
# BATCH 3: BORATE-HCL - SYNTHETIC SOLUTION - PCR
# --------------------------------------------------------------------------
parameter_sets = [ {"C_feed": 0.003190078*1.4}, {"C_feed": 0.012222*0.8}] 
Da_all = np.array([3.218e-6, 8.38e-6 ]) 
kav_params_all = np.array([[0.215], [0.132]])
kav_params_all = np.array([[test_kfp], [test_kfp]])

cusotom_isotherm_params_all = np.array([[3.93], [3.23]]) # [ [H_borate], [H_hcl] ]
# When saving the json:
Description = [f"Description example: Optimizting the Borate HCl system for the sythetic solution on PCR.Ca. {optimization_budget} iterations, with no isotherom data. We placed a upper flowrate constraint of {Q_max} L/h. This is with the sub-zoning for bor-hcl"]
save_name_inputs = "SYNTH_PCR-borhcl-type1_40iter_norm_config_all_inputs.json" # (1) "ILLOVO_PCR_borhcl-type1_40iter_norm_config_all_inputs.json", (2) "ILLOVO_PCR-borhcl-type1_40iter_subzone_config_all_inputs.json"
save_name_outputs = "SYNTH_PCR-borhcl-type1_40iter_norm_config_all_outputs.json" # (1) "ILLOVO_PCR-borhcl-type1_40iter_norm_config_all_outputs.json", (2) "ILLOVO_PCR-borhcl-type1_40iter_subzone_config_all_outputs.json"
   # PACK:
opt_inputs_b3 = [Description, save_name_inputs, save_name_outputs, job_max_or_min, t_reff, Q_max, Q_min, m_max, m_min, sampling_budget, optimization_budget, constraint_threshold, PF_weight, bounds]
SMB_inputs_b3 = [iso_type, Names, color, num_comp, nx_per_col, e, Da_all, Bm, zone_config, L, d_col, d_in, t_index_min, n_num_cycles, Q_internal, parameter_sets, cusotom_isotherm_params_all, kav_params_all, subzone_set, t_simulation_end]
# -------------------------------
batch_3 = [opt_inputs_b3, SMB_inputs_b3] 


# --------------------------------------------------------------------------
# BATCH 4: BORATE-HCL - ILLOVO SOLUTION - UBK
# --------------------------------------------------------------------------
parameter_sets = [ {"C_feed": 0.003190078*1.4}, {"C_feed": 0.012222*0.8}] 
Da_all = np.array([1.83e-5, 5.6-5]) 
kav_params_all = np.array([[0.173], [0.151]])
kav_params_all = np.array([[test_kfp], [test_kfp]])

cusotom_isotherm_params_all = np.array([[4.13], [2.3]]) # [ [H_borate], [H_hcl] ]
# When saving the json:
Description = [f"Description example: Optimizting the Borate HCl system for the sythetic solution on PCR.Ca. {optimization_budget} iterations, with no isotherom data. We placed a upper flowrate constraint of {Q_max} L/h. This is with the sub-zoning for bor-hcl"]
save_name_inputs = "ILLOVO_UBK-borhcl-type1_40iter_norm_config_all_inputs.json" # (1) "ILLOVO_PCR_borhcl-type1_40iter_norm_config_all_inputs.json", (2) "ILLOVO_PCR-borhcl-type1_40iter_subzone_config_all_inputs.json"
save_name_outputs = "ILLOVO_UBK-borhcl-type1_40iter_norm_config_all_outputs.json" # (1) "ILLOVO_PCR-borhcl-type1_40iter_norm_config_all_outputs.json", (2) "ILLOVO_PCR-borhcl-type1_40iter_subzone_config_all_outputs.json"
   # PACK:
opt_inputs_b4 = [Description, save_name_inputs, save_name_outputs, job_max_or_min, t_reff, Q_max, Q_min, m_max, m_min, sampling_budget, optimization_budget, constraint_threshold, PF_weight, bounds]
SMB_inputs_b4 = [iso_type, Names, color, num_comp, nx_per_col, e, Da_all, Bm, zone_config, L, d_col, d_in, t_index_min, n_num_cycles, Q_internal, parameter_sets, cusotom_isotherm_params_all, kav_params_all, subzone_set, t_simulation_end]
# -------------------------------
batch_4 = [opt_inputs_b4, SMB_inputs_b4] 



# --------------------------------------------------------------------------
# BATCH 5: BORATE-HCL - ILLOVO SOLUTION - PCR
# --------------------------------------------------------------------------
parameter_sets = [ {"C_feed": 0.003190078*1.4}, {"C_feed": 0.012222*0.8}] 
Da_all = np.array([5.77e-7, 2.3812e-6]) 
kav_params_all = np.array([[0.170], [0.154]])
kav_params_all = np.array([[test_kfp], [test_kfp]])

cusotom_isotherm_params_all = np.array([[2.13], [2.3]]) # [ [H_borate], [H_hcl] ]
# When saving the json:
Description = [f"Description example: Optimizting the Borate HCl system for the sythetic solution on PCR.Ca. {optimization_budget} iterations, with no isotherom data. We placed a upper flowrate constraint of {Q_max} L/h. This is with the sub-zoning for bor-hcl"]
save_name_inputs = "ILLOVO_PCR-borhcl-type1_40iter_norm_config_all_inputs.json" # (1) "ILLOVO_PCR_borhcl-type1_40iter_norm_config_all_inputs.json", (2) "ILLOVO_PCR-borhcl-type1_40iter_subzone_config_all_inputs.json"
save_name_outputs = "ILLOVO_PCR-borhcl-type1_40iter_norm_config_all_outputs.json" # (1) "ILLOVO_PCR-borhcl-type1_40iter_norm_config_all_outputs.json", (2) "ILLOVO_PCR-borhcl-type1_40iter_subzone_config_all_outputs.json"
   # PACK:
opt_inputs_b5 = [Description, save_name_inputs, save_name_outputs, job_max_or_min, t_reff, Q_max, Q_min, m_max, m_min, sampling_budget, optimization_budget, constraint_threshold, PF_weight, bounds]
SMB_inputs_b5 = [iso_type, Names, color, num_comp, nx_per_col, e, Da_all, Bm, zone_config, L, d_col, d_in, t_index_min, n_num_cycles, Q_internal, parameter_sets, cusotom_isotherm_params_all, kav_params_all, subzone_set, t_simulation_end]
# -------------------------------
batch_5 = [opt_inputs_b5, SMB_inputs_b5] 


opt_batches = [batch_1, batch_2, batch_3, batch_4, batch_5]
