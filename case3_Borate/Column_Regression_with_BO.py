#%%
import numpy as np
from scipy.optimize import minimize
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import pandas as pd
from scipy import integrate
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C, Matern
from scipy.optimize import differential_evolution
from scipy.optimize import minimize, NonlinearConstraint
import json

import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.integrate import solve_ivp
from scipy import integrate
import warnings
import time
# import numba

# Import the column model
from Col_test_model_func import column_func





# column_func_inputs = [
#     iso_type, Names, color, parameter_sets, Pe, Bm, e, 
#     Q_S, Q_inj, Ncol_num, t_index, tend_min, nx, L, d_col
# ]




# Function to solve the concentration using the column model given a Pe, and tend
def solve_concentration(Da_all, kfp_all, cusotom_isotherm_params_all, column_func_inputs):
    """
    cusotom_isotherm_params_all = 1 < len(.) < n
    Da =>  len(.) = 1
    kfp =>  len(.) = 1
    """
    column_func_inputs[4] =  np.array(Da_all) # Insert the Dispersion in the correct position
    column_func_inputs[3][0]["kfp"] = kfp_all[0] # Update kfp in the parameter set
    column_func_inputs[3][1]["kfp"] = kfp_all[1]  # Update kfp in the parameter set
    column_func_inputs[-1] = cusotom_isotherm_params_all  # Update H in the parameter set

    solution = column_func(column_func_inputs)

    col_elution_borate = solution[0][0]  # For 1st component
    col_elution_hcl = solution[0][1]  # For 2nd component

    t_values = solution[3]  # Corresponding time values
    error = solution[-1]

    return t_values, col_elution_borate, col_elution_hcl

# Generate synthetic noisy data for testing
def generate_synthetic_data(Da, kfp, cusotom_isotherm_params_all, resolution):
    np.random.seed(0)

    # Column model parameters
    iso_type = "CUP"
    Names = ["A"]
    color = ["g"]
    parameter_sets = [{"kfp": 3.15/100, "C_feed": 1}] # [1/s, __ , g/cm^3]
    Q_S = 0.1 # cm^3/s
    Q_inj = 0.015 # cm^3/s
    t_index = 70 # s
    tend_min = 20 # min
    nx = 60
    e = 0.4
    L = 40 # cm
    d_col = 2 # cm
    Bm = 500

    column_func_inputs = [iso_type,  Names, color, parameter_sets, Da, Bm, e, Q_S, Q_inj, t_index, tend_min, nx, L, d_col, cusotom_isotherm_params_all]

    t_values, col_elution_borate, col_elution_hcl = solve_concentration(Da, kfp, cusotom_isotherm_params_all, column_func_inputs)
    # noise = 0.1 * np.random.normal(size=len(t_values))
    noise = 0
    col_elution_borate_noisy = col_elution_borate + noise
    col_elution_hcl_noisy = col_elution_hcl + noise

    plt.scatter(t_values, col_elution_borate_noisy)
    plt.scatter(t_values, col_elution_hcl_noisy)
    plt.xlabel('Time (s)')
    plt.ylabel('Concentration (g/cm^3)')
    plt.title('Full Synthetic Data')
    plt.show()

    while len(col_elution_hcl_noisy) > resolution:
        col_elution_hcl_noisy = col_elution_hcl_noisy[::2]
        col_elution_borate_noisy = col_elution_borate_noisy[::2]
        t_values = t_values[::2]
    
    plt.scatter(t_values, )
    plt.scatter(t_values, col_elution_hcl_noisy)
    plt.xlabel('Time (s)')
    plt.ylabel('Concentration (g/cm^3)')
    plt.title('Reduced Synthetic Data')
    plt.show()
    return t_values, col_elution_borate_noisy, col_elution_hcl_noisy, column_func_inputs

# Objective function for the regression routine
def objective_function(params, t_data, conc_data, column_func_inputs, max_of_each_input):
    """
    Func that: 
    (1) Interpolates to find the models conc values at the time points that correspond to the experimental data
    (2) Checks the "error" between the model concs and the experimental concs 

    t_data => experimental data time points
    conc_data => experimental data => [[borate elution curve vector], [hcl elution curve vector]]
    params  = [Da_bor,
               Da_hcl, 
               kfp_bor, 
               kfp_hcl, 
               K_bor, 
               K_hcl] Ki=isotherm parameters -> normalized according to max and minimum values
    
    max_of_each_input = [Da_max, kfp_max, Kmax]
    
    """
    params = np.array(params)
    Da_max = max_of_each_input[0]
    kfp_max = max_of_each_input[2]
    K_max = max_of_each_input[4]

    # Apply:
    Da_bor = params[0]*Da_max
    Da_hcl = params[1]*Da_max

    kfp_bor = params[2]*kfp_max
    kfp_hcl = params[3]*kfp_max

    K_bor = params[4]*K_max
    K_hcl = params[5]*K_max

    # PACK:
    kfp_all = [kfp_bor, kfp_hcl]
    Da_all = [Da_bor, Da_hcl]
    cusotom_isotherm_params_all = np.array([[K_bor], [K_hcl]]) #, [params[3]*K2_max]]) # params[4]*K3_max]
    

    t_predicted, predicted_conc_borate,  predicted_conc_hcl = solve_concentration(Da_all, kfp_all, cusotom_isotherm_params_all, column_func_inputs)
    # Interpolate the predicted concentrations to match the time points in t_data
    # predicted_conc_interpolated = np.interp(t_data, np.linspace(0, tend_min, len(predicted_conc)), predicted_conc)
    t_data = t_data.to_numpy()
    print(f't_data {t_data}, t_predicted: {np.shape(t_predicted)}, predicted_conc_borate: {np.shape(predicted_conc_borate)}')
    print(f't_data {t_data}, t_predicted: {np.shape(t_predicted)}, predicted_conc_borate: {np.shape(predicted_conc_hcl)}')
    predicted_conc_borate_interpolated = np.interp(t_data, t_predicted, predicted_conc_borate) # concentratio values that match the experimental data
    predicted_conc_hcl_interpolated = np.interp(t_data, t_predicted, predicted_conc_hcl) # concentratio values that match the experimental data
    
    # Calculate the sum of squared errors between the actual data and the predicted concentrations
    # Change objective function as necessary:
    # SSE:
    borate_error = np.sum((conc_data[0] - predicted_conc_borate_interpolated) ** 2)
    hcl_error = np.sum((conc_data[1] - predicted_conc_hcl_interpolated) ** 2)

    total_error = borate_error + hcl_error

    return total_error, predicted_conc_borate_interpolated, predicted_conc_hcl_interpolated

def get_feed_concs(slug_vol):
    # Specify the path to your CSV file
    # Note that this path includes the file name
    file_path_masses = "D:\Education\MSc\School Work\code\Inlet Concentrations.xlsx"

    # Read the CSV file
    df = pd.read_excel(file_path_masses)

    # Display the DataFrame
    print(df.columns)


    # Selecting a specific column

    Gluc = df.iloc[1:80]['Glucose']
    Fruc = df.iloc[1:80]['Fructose']
    Suc = df.iloc[1:80]['Sucrose']
    Kest = df.iloc[1:80]['Kestose']
    Nyst = df.iloc[1:80]['Nystose']
    GF4 = df.iloc[1:80]['GF4']
    Time = df.iloc[1:80]['Time, min']

    Components = [Gluc, Fruc, Suc, Kest, Nyst, GF4]

    masses = np.zeros(len(Components))
    for i in range(len(Components)):
        masses[i] = integrate.simpson(Components[i], Time)

    slug_vol = 2.5 # cm^3

    feed_concs = masses/slug_vol

    return feed_concs

def get_data_from_excel(file_path, resolution):

    # Read the CSV file
    df = pd.read_excel(file_path)
    # Display the DataFrame
    print(df)
    print(df.columns.str.strip())

    # Unpack the input parameters
    iso_type = df.iloc[0]['isotherm type']

    Names = ['Borate', 'HCl']


    color = df.iloc[0]['color']
    
    slug_vol = df.iloc[0]['slug volume (cm^3)']
    
    feed_conc_borate = df.iloc[0]['borate conc in slug (g/cm^3)']
    
    feed_conc_hcl = df.iloc[0]['hcl conc in slug (g/cm^3)']
    
    parameter_sets = [{"kfp": df.iloc[0]['kfp'], "C_feed": feed_conc_borate}, {"kfp": df.iloc[0]['kfp'], "C_feed": feed_conc_hcl}] # [1/s, __ , g/cm^3]
    
    e = df.iloc[0]['voidage']

    Q_S = df.iloc[0]['Flowrate (cm^3/s)'] # cm^3/s

    Q_inj = df.iloc[0]['pulse vol flowrate (cm^3/s)'] # cm^3/s

    nx = int(df.iloc[0]['number of x nodes']) 

    L = df.iloc[0]['column length (cm)']   # cm

    d_col = df.iloc[0]['column diameter (cm)'] # cm

    t_index = df.iloc[0]['time of slug pulse (s)'] # s

    t_start = 0
    t_end = 25 # Fraction number - limit to n/25 fractions

    tend_min = df.iloc[t_end]['Time, min'] # min

    cusotom_isotherm_params_all = [[1], [1]]

    t_data = df.iloc[t_start:t_end]['Time, s']
    
    col_elution_data_borate = df.iloc[t_start:t_end]['borate g/cm^3']
    
    col_elution_data_hcl = df.iloc[t_start:t_end]['HCl g/cm^3']


    
    # Place Olders:
    Da = 0 
    Bm = 0 

    column_func_inputs = [iso_type,  Names, color, parameter_sets, Da, Bm, e, Q_S, Q_inj, t_index, tend_min, nx, L, d_col, cusotom_isotherm_params_all]
    

    if resolution != None:
            while len(col_elution_data_borate) > resolution:
                col_elution_data_borate = col_elution_data_borate[::2]
                col_elution_data_hcl = col_elution_data_hcl[::2]
                t_data = t_data[::2]
    num_points = len(t_data)

    plt.scatter(t_data, col_elution_data_borate)
    plt.scatter(t_data, col_elution_data_hcl)
    plt.axvline(x=t_index, linestyle = '--', label = 'End of Pulse')
    plt.xlabel('Time (s)')
    plt.ylabel('Concentration (g/cm^3)')
    plt.title(f'Experimental Data\n {column_func_inputs[1]}\n Number of Points: {num_points}')
    plt.show()

    return t_data, [col_elution_data_borate, col_elution_data_hcl], column_func_inputs


#%%
# ---------------- Functions for BO

# 1.
# --- Surrogate model creation ---
def surrogate_model(X_train, y_train):
    # X_train = np.atleast_2d(X_train)
    

    # if y_train.ndim == 2 and y_train.shape[1] == 1:
    #     y_train = y_train.ravel()

    # kernel = C(1.0, (1e-4, 10.0)) * RBF(1.0, (1e-4, 10.0))
    kernel = Matern(length_scale=1.0, nu=1.5)

    gp = GaussianProcessRegressor(kernel=kernel, alpha=1e-5, normalize_y=True, n_restarts_optimizer=5)

    gp.fit(X_train, y_train)

    return gp


# Aquisition Funciton

#  Expected Improvement ---
def expected_improvement(x, surrogate_gp, y_best, xi=0.005):
    """
    Computes the Expected Improvement at a point x where x=[Da, kfp, H].
    Scalarizes the surrogate predictions using Tchebycheff, then computes EI.

    Note that the surrogate GP already has the weights applied to it

    the greater the value of xi, the more we encourage exploration
    """
    x = np.array(x).reshape(1, -1)

    mu, sigma = surrogate_gp.predict(x, return_std=True)

    # print(f'mu: {mu}')
    # print(f'y_best: {y_best}')
    # Compute EI

    with np.errstate(divide='warn'):
        Z = (y_best - mu - xi) / sigma
        ei = (y_best - mu - xi) * norm.cdf(Z) + sigma * norm.pdf(Z)
        ei[sigma == 0.0] = 0.0

    return -ei[0]  # Negative for minimization - DOES NOT AFFECT STRUCTURE

# 2. Probability of Imporovement:
def probability_of_improvement(x, surrogate_gp, y_best, xi=0.005):
    """
    Computes the Probability of Improvement (PI) acquisition function.

    Parameters:
    - mu: np.array of predicted means (shape: [n_samples])
    - sigma: np.array of predicted std deviations (shape: [n_samples])
    - best_f: scalar, best objective value observed so far
    - xi: float, small value to balance exploration/exploitation (default: 0.01)

    Returns:
    - PI: np.array of probability of improvement values
    """
    x = np.array(x).reshape(1, -1)

    mu, sigma = surrogate_gp.predict(x, return_std=True)

    # Avoid division by zero
    if sigma == 0:
      sigma = 1e-8

    z = (y_best - mu - xi) / sigma

    pi = norm.cdf(z)

    return -pi


def constrained_BO(optimization_budget, bounds, all_initial_inputs, all_initial_ouputs, t_data, conc_data, column_func_inputs, max_of_each_input, xi, sampling_budget=1):
    """
    max_of_each_input = [Da_max, kfp_max, K1_max, K2_max, ... Kn_max]

    """
    # xi = exploration parameter (the larger it is, the more we explore)

    # Initial values

    # Unpack from: all_initial_ouputs: [SSE]

    # iObjectives
    f_vals = np.array([all_initial_ouputs])
    # print(f'f_vals: {f_vals}')

    num_inputs = len(max_of_each_input)
    all_inputs = all_initial_inputs# columns: [Da, kfp, K1, K2, ..., Kn] Ki=isotherm parameters

    # print(f'np.shape(all_inputs):{np.shape(all_inputs)}')
    # print(f'np.shape(all_initial_inputs):{np.shape(all_initial_inputs)}')

    # all_inputs = np.vstack((all_inputs, all_initial_inputs))
    population = all_initial_inputs.reshape(1, -1)

    # Unpack from: all_initial_inputs


    # Initialize where we will store solutions
    population_all = []

    for gen in range(optimization_budget):
        # generation = iteration
        print(f"\n\nStarting gen {gen+1}")



        # Note that we generate new weights in each iteration/generation
        # i.e. each time we update the training set

        # Fit GP to scalarized_surrogate_objective
        # print(f'population { population}, \nscalarized_f_vals {scalarized_f_vals} ')
        surrogate_gp = surrogate_model(population, f_vals)
        # Pull mean at relevant poputlation points
        # Mean & Varriance
        surrogate_gp_mean, surrogate_gp_std = surrogate_gp.predict(population, return_std=True)
        # The best value so far:
        y_best = np.min(surrogate_gp_mean)
        # y_best = 0.60


        # Define the constraint function for the ei optimizer
        # Constraint function with correct shape
        # --- Run the optimization ---
        print(f'Maxing ECI')
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            result = differential_evolution(
                func=probability_of_improvement, # (i) probability_of_improvement(x, surrogate_gp, y_best, xi=0.005) , OR (ii) expected_improvement(x, surrogate_gp, y_best)
                args=(surrogate_gp, y_best),
                bounds=bounds,
                strategy='best1bin',
                maxiter=200,
                popsize=15,
                disp=False,
                )

                # Perform the optimization using L-BFGS-B method
        # result = minimize(
        #     expected_improvement,
        #     initial_guess,
        #     args=(scalarized_surrogate_gp, y_best),
        #     method='L-BFGS-B',
        #     bounds=bounds,
        #     options={'maxiter': 100, 'disp': True})

        x_new = result.x # [Da, kfp, e, K1, K2, ..., Kn] Ki=isotherm parameters
        # print(f"x_new: { x_new}") 

        f_new, __, __ = objective_function(x_new, t_data, conc_data, column_func_inputs, max_of_each_input)

        # Add the new row to all_inputs
        all_inputs = np.vstack((all_inputs, x_new))

        # Add to population
        population_all.append(population)
        population = np.vstack((population, x_new))

        f_vals = np.vstack([f_vals.reshape(-1,1), f_new])


        print(f"Gen {gen+1} Status:\n | Sampled Inputs:{x_new} [Da, kfp, e, H]|\n Outputs (SSE): f1: {f_new}")

    return f_vals, all_inputs

# %%
# Main routine

if __name__ == "__main__":
    # PLease note that most inputs are obtained from the Excel Sheets and in the 'GENERATE SYNTHETIC DATA' func
    # Because the system is coupled, is solves for all parameters at the same time
    ##
    max_of_each_input = np.array([1e-4, 1e-4, # Da_max,
                                  0.8, 0.8,   # kfp_max
                                  5, 5])      # K_max]
    ##

    # Da guesses:
    Da_bor_guess = 1e-6 # cm^2/s
    Da_hcl_guess = 1e-6 # cm^2/s
    # kfp guesses:
    kfp_bor_guess = 0.096
    kfp_hcl_guess = 0.0542241278
    # Isotherm Guesses
    K_bor_guess = 4.7
    K_hcl_guess = 3.5
    
    # Load Initial Guesses in vector:
    # When doing Fru:
    x_initial_guess = np.array([Da_bor_guess,
                                Da_hcl_guess,
                                kfp_bor_guess,
                                kfp_hcl_guess,
                                K_bor_guess,
                                K_hcl_guess,
                                ])
        
    print(f'x_initial_guess: {x_initial_guess}')
    # Normalize Initial Guess
    x_initial_guess = x_initial_guess/max_of_each_input
    

    optimization_budget = 6
    bounds = [  (0.01, 1), # Da_bor
                (0.01, 1), # Da_hcl

                (0.0001, 1), # kfp_bor
                (0.0001, 1), # kfp_hcl

                (0.0001, 1), # K_bor
                (0.0001, 1), # K_hcl
            ]

    # ---- PART 1: GET DATA  -------#

    # ---- OPTION 1: GENERATE SYNTHETIC DATA  -------#
    # Set the TRUE values of parameters
    # Da_true = 1e-5
    # Bm_true = 300
    # kfp_true =  3.15 / 100
    # e_true = 0.45
    # cusotom_isotherm_params_all_true = np.array([1])
    # num_points = 20

    # t_data, col_elution_data, column_func_inputs = generate_synthetic_data(Da_true, kfp_true, e_true, cusotom_isotherm_params_all_true, num_points)

    # ---- OPTION 2: GET DATA FROM EXCEL  -------#


    # ------ Where in your PC is the Excel File?:
    # Get data from Excel file

    # Get files in folders for respective Porjects:

    # --------- 3. BORATE & HCL DATA
    # UBK:
    file_path_bor_hcl_UBK = r"_bor_hcl_UBK_530.xlsx"
    file_path_BORATE_UBK = r"_borate_UBK_530.xlsx"
    file_path_HCL_UBK = r"_HCL_UBK_530.xlsx"
    # PCR
    file_path_BORATE_PCR = r"_borate_PCR642Ca.xlsx"
    file_path_HCL_PCR = r"_HCL_PCR642Ca.xlsx"




    # --------------------------------
    t_data, col_elution_data, column_func_inputs = get_data_from_excel(file_path_bor_hcl_UBK, resolution=None)
    num_points = len(t_data)

    print(f't_data: {t_data}')
    tend_min = t_data.iloc[-1]/60 # min
#%%
    # --------------------------------

    # ---- PART 2: GET DATA  -------#

    # ---- Set Iniital Guess
    # Adjust INITIAL GUESS Based On Isotherm Choice:

    # Evaluate Initial guess:
    f_initial, bor_elution_curve_initial_guess, hcl_elution_curve_initial_guess = objective_function(x_initial_guess, t_data, col_elution_data, column_func_inputs,max_of_each_input)
    
#%%
    print(f'{f_initial} for {x_initial_guess}')

    # Perform the Bayesian Optimization
    start = time.time()

    f_vals, norm_all_inputs = constrained_BO(optimization_budget, bounds, x_initial_guess, f_initial, t_data, col_elution_data, column_func_inputs, max_of_each_input, xi=0.005, sampling_budget=1)
    # print(f'all_inputs !!!!!!!!: \n{ all_inputs}')
    all_inputs  = norm_all_inputs*max_of_each_input
    end = time.time()
    



    # ---------------- SAVE THE OUTPUTS TO JSONS
    data_dict = {
    "f_vals": f_vals.tolist(),
    "all_inputs": all_inputs.tolist(),
    }

    # SAVE to JSON:
    with open("BO_COL_REG.json", "w") as f:
        json.dump(data_dict, f, indent=4)


    # ---------------- Visualise THE OUTPUTS

    #%%
    # 1. Function Progression:
    idx_min = np.argmin(f_vals)
    norm_best_inputs = norm_all_inputs[idx_min,:]
    best_inputs = all_inputs[idx_min,:]

    print(f'Best Fitted Points: {best_inputs}')
    __, bor_elution_curve_best, hcl_elution_curve_best= objective_function(norm_best_inputs, t_data, col_elution_data, column_func_inputs,max_of_each_input)
    
#%%
    fig, ax = plt.subplots(1, 1, figsize=(15, 5))
    x_plot = range(len(f_vals))
    ax.scatter(x_plot, f_vals, label='SSE', color='blue', alpha=0.6)
    ax.plot(x_plot, f_vals, label='SSE', color='blue', alpha=0.6)

    ax.scatter(x_plot[idx_min], f_vals[idx_min], marker='*', color='red', s=200, label='Minimum SSE')
    ax.axvline(x=x_plot[0], linestyle='--', label='Initial Guess', color='green')

    ax.set_xlabel('Function Calls')
    ax.set_ylabel('SSE')
    ax.set_title(f'BO of {column_func_inputs[1]}\nElution Curves\n{optimization_budget+1} Interations')
    ax.legend()
        # Enforce whole numbers on the x-axis
    ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    plt.show()

    # How did the inputs change?
    # fig, axs = plt.subplots(2, 2, figsize=(15, 10))

    # # Plot Da on the top-left plot
    # axs[0, 0].scatter(x_plot, norm_all_inputs[:, 0], label='Da', color='blue', alpha=0.6)
    # axs[0, 0].axvline(x=x_plot[idx_min], linestyle='--', label='Best Point', color='k')
    # axs[0, 0].plot(x_plot, norm_all_inputs[:, 0], color='blue', alpha=0.6)
    # axs[0, 0].set_xlabel('Function Calls')
    # axs[0, 0].set_ylabel('Input Parameters')
    # axs[0, 0].set_title('Normalized Da')
    # axs[0, 0].legend()

    # # Plot Q_max on the top-right plot
    # axs[0, 1].scatter(x_plot, norm_all_inputs[:, 2], label='Q_max', color='orange', alpha=0.6)
    # axs[0, 1].axvline(x=x_plot[idx_min], linestyle='--', label='Best Point', color='k')
    # axs[0, 1].plot(x_plot, norm_all_inputs[:, 2], color='orange', alpha=0.6)
    # axs[0, 1].set_xlabel('Function Calls')
    # axs[0, 1].set_ylabel('Input Parameters')
    # axs[0, 1].set_title('Normalized Q_max')
    # axs[0, 1].legend()

    # # Plot kfp on the bottom-left plot
    # axs[1, 0].scatter(x_plot, norm_all_inputs[:, 1], label='kfp', color='green', alpha=0.6)
    # axs[1, 0].axvline(x=x_plot[idx_min], linestyle='--', label='Best Point', color='k')
    # axs[1, 0].plot(x_plot, norm_all_inputs[:, 1], color='green', alpha=0.6)
    # axs[1, 0].set_xlabel('Function Calls')
    # axs[1, 0].set_ylabel('Input Parameters')
    # axs[1, 0].set_title('Normalized kfp')
    # axs[1, 0].legend()

    # # Plot b on the bottom-right plot
    # axs[1, 1].scatter(x_plot, norm_all_inputs[:, 3], label='b', color='blue', alpha=0.6)
    # axs[1, 1].axvline(x=x_plot[idx_min], linestyle='--', label='Best Point', color='k')
    # axs[1, 1].plot(x_plot, norm_all_inputs[:, 3], color='blue', alpha=0.6)
    # axs[1, 1].set_xlabel('Function Calls')
    # axs[1, 1].set_ylabel('Input Parameters')
    # axs[1, 1].set_title('Normalized b')
    # axs[1, 1].legend()
    # fig.suptitle('Optimization Trajectories')
    
    # # Enforce whole numbers on the x-axis
    # for ax in axs.flat:
    #     ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))

    # # axs.set_title(f'Optimization Trajectories')
    # plt.tight_layout()
    # plt.show()

#%%
    # 2. Elution Curves
    fig, bx = plt.subplots(1, 1, figsize=(15, 5))
    # Experimental Data:
    bx.scatter(t_data/60, col_elution_data[0], label='Borate Experimental Data', color='red', alpha=0.6)
    bx.scatter(t_data/60, col_elution_data[1], label='HCl Experimental Data', color='green', alpha=0.6)
    # Model:
    # Initial Guess bor_elution_curve_initial_guess, hcl_elution_curve_initial_guess
    bx.plot(t_data/60, bor_elution_curve_initial_guess, linestyle='--', label='Initial Guess', color='grey', alpha=0.6)
    bx.plot(t_data/60, hcl_elution_curve_initial_guess, linestyle='--', color='grey', alpha=0.6)
    # Fitted Model , 
    bx.plot(t_data/60, bor_elution_curve_best, label='Borate Model', color='red', alpha=0.6)
    bx.scatter(t_data/60, bor_elution_curve_best, marker='s', color='red', alpha=0.6)

    bx.plot(t_data/60, hcl_elution_curve_best, label='HCl Model', color='green', alpha=0.6)
    bx.scatter(t_data/60, hcl_elution_curve_best, marker='s', color='green', alpha=0.6)

    # Lables
    bx.set_xlabel('time, min')
    bx.set_ylabel('g/mL')
    bx.set_title(f'{column_func_inputs[1]} Elution Curves\n Borate:[Da, kfp, K] {best_inputs[::2]}\n HCl:[Da, kfp, K] {best_inputs[1::2]} \n{optimization_budget+1} Interations')
    bx.legend()
    plt.show()



# %%
