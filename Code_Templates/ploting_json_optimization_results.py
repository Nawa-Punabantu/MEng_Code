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
from matplotlib.ticker import MaxNLocator, MultipleLocator

import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.integrate import solve_ivp
from scipy import integrate
import warnings
import time


#%% Run nd Define the Funcitons
def load_inputs_outputs(inputs_path, outputs_path):
    """
    Loads all_inputs and output values (f1, f2, c1, c2) from saved JSON files and reconstructs them as numpy arrays.
    
    Args:
        inputs_path (str): Path to 'all_inputs.json'.
        outputs_path (str): Path to 'all_outputs.json'.

    Returns:
        all_inputs (np.ndarray): Loaded inputs array.
        f1_vals (np.ndarray): Glucose recovery values.
        f2_vals (np.ndarray): Fructose recovery values.
        c1_vals (np.ndarray): Glucose purity values.
        c2_vals (np.ndarray): Fructose purity values.
    """
    # Load inputs
    with open(inputs_path, "r") as f:
        all_inputs_list = json.load(f)
    all_inputs = np.array(all_inputs_list)

    # Load outputs
    with open(outputs_path, "r") as f:
        data_dict = json.load(f)
    f1_vals = np.array(data_dict["f1_vals"])
    f2_vals = np.array(data_dict["f2_vals"])
    c1_vals = np.array(data_dict["c1_vals"])
    c2_vals = np.array(data_dict["c2_vals"])

    return all_inputs, f1_vals, f2_vals, c1_vals, c2_vals


#%%

#------------------------------------------------------- 1. Table

def create_output_optimization_table(f1_vals, f2_vals, c1_vals, c2_vals, sampling_budget):
    # Create a data table with recoveries first
    data = np.column_stack((f1_vals*100, f2_vals*100, c1_vals*100, c2_vals*100))
    columns = ['Recovery F1 (%)', 'Recovery F2 (%)', 'Purity C1 (%)', 'Purity C2 (%)']
    rows = [f'Iter {i+1}' for i in range(len(c1_vals))]

    # Identify "star" entries (where f1_vals, f2_vals > 70 and c1_vals, c2_vals > 90)
    # star_indices = np.where((f1_vals*100 > 50) & (f2_vals*100 > 50) & (c1_vals*100 > 80) & (c2_vals*100 > 95))[0]
    star_indices = np.where((c2_vals*100 > 99.5))[0]
    # Create figure
    fig, ax = plt.subplots(figsize=(8, len(c1_vals) * 0.2))
    # ax.set_title("Optimization Iterations: Recovery & Purity Table", fontsize=12, fontweight='bold', pad=5)  # Reduced padding
    ax.axis('tight')
    ax.axis('off')

    # Create the table
    table = ax.table(cellText=data.round(2),
                     colLabels=columns,
                     rowLabels=rows,
                     cellLoc='center',
                     loc='center')

    # Adjust font size
    table.auto_set_font_size(False)
    table.set_fontsize(5)
    table.auto_set_column_width(col=list(range(len(columns))))

    # Apply colors
    for i in range(len(c1_vals)):
        for j in range(len(columns)):
            cell = table[(i+1, j)]  # (row, column) -> +1 because row labels shift index
            if i < sampling_budget:
                cell.set_facecolor('lightgray')  # Grey out first 20 rows
            if i in star_indices:
                cell.set_facecolor('yellow')  # Highlight star entries in yellow

    # Save the figure as an image
    image_filename = "output_optimization_table.png"
    fig.savefig(image_filename, dpi=300, bbox_inches='tight')
    plt.show()

    return image_filename



def calculate_flowrates(input_array, V_col, e):
    # Initialize the external flowrate array with the same shape as input_array
    internal_flowrate = np.zeros_like(input_array[:,:-1])
    external_flowrate = np.zeros_like(input_array)
    
    # Reshape the last column to be a 2D array for broadcasting
    input_last_col = input_array[:, -1]
    
    for i, t_index in enumerate(input_last_col):
        # Calculate the flow rates using the provided formula
        # Fill each row in external_flowrate:
        print(f't_index: {t_index}')
        internal_flowrate[i, :] = (input_array[i, :-1] * V_col * (1 - e) + V_col * e) / (t_index * 60)  # cm^3/s
    

    internal_flowrate = internal_flowrate*3.6 # cm^3/s => L/h
    print(f'internal_flowrate: {internal_flowrate}')
    # Calculate Internal FLowtates:
    Qfeed = internal_flowrate[:,2] - internal_flowrate[:,1] # Q_III - Q_II 
    Qraffinate = internal_flowrate[:,2] - internal_flowrate[:,3] # Q_III - Q_IV 
    Qdesorbent = internal_flowrate[:,0] - internal_flowrate[:,3] # Q_I - Q_IV 
    Qextract = internal_flowrate[:,0] - internal_flowrate[:,1] # Q_I - Q_II

    external_flowrate[:,0] = Qfeed
    external_flowrate[:,1] = Qraffinate
    external_flowrate[:,2] = Qdesorbent
    external_flowrate[:,3] = Qextract
    external_flowrate[:,4] = input_last_col

    return internal_flowrate, external_flowrate

def create_input_optimization_table(input_array, V_col, e, sampling_budget, f1_vals, f2_vals, c1_vals, c2_vals):
    # Calculate flow rates
    internal_flowrate, external_flowrate = calculate_flowrates(input_array, V_col, e)
    flowrates = external_flowrate
    # Create a data table with flow rates
    data = external_flowrate
    columns = ['Feed (L/h)', 'Raffinate (L/h)', 'Desorbent (L/h)', 'Extract(L/h)', 'Index Time (min)']
    rows = [f'Iter {i+1}' for i in range(len(input_array))]

    # star_indices = np.where((f1_vals*100 > 50) & (f2_vals*100 > 50) & (c1_vals*100 > 80) & (c2_vals*100 > 95))[0]
    star_indices = np.where((c2_vals*100 > 95))[0]
    # Create figure
    fig, ax = plt.subplots(figsize=(8, len(input_array) * 0.2))
    # ax.set_title("Optimization Iterations: Flowrate Table", fontsize=12, fontweight='bold', pad=1)  # Reduced padding
    ax.axis('tight')
    ax.axis('off')

    # Create the table
    table = ax.table(cellText=data.round(3),
                     colLabels=columns,
                     rowLabels=rows,
                     cellLoc='center',
                     loc='center')

    # Adjust font size
    table.auto_set_font_size(False)
    table.set_fontsize(5)
    table.auto_set_column_width(col=list(range(len(columns))))

    # Apply colors
    for i in range(len(input_array)):
        for j in range(len(columns)):
            cell = table[(i+1, j)]  # (row, column) -> +1 because row labels shift index
            if i < sampling_budget:
                cell.set_facecolor('lightgray')  # Grey out first sampling_budget rows
            if i in star_indices:
                cell.set_facecolor('yellow')  # Highlight star entries in yellow

    # Save the figure as an image
    image_filename = "input_optimization_table.png"
    fig.savefig(image_filename, dpi=300, bbox_inches='tight')
    plt.show()

    return image_filename





#------------------------------------------------------- 2. Recovery Pareto

def create_recovery_pareto_plot(f1_vals, f2_vals, zone_config, sampling_budget, optimization_budget):
    # Convert to percentages
    f1_vals_plot = f1_vals * 100
    f2_vals_plot = f2_vals * 100

    # Function to find Pareto front
    def find_pareto_front(f1, f2):
        pareto_mask = np.ones(len(f1), dtype=bool)  # Start with all points assumed Pareto-optimal

        for i in range(len(f1)):
            if pareto_mask[i]:  # Check only if not already removed
                pareto_mask[i] = not np.any((f1 >= f1[i]) & (f2 >= f2[i]) & ((f1 > f1[i]) | (f2 > f2[i])))

        return pareto_mask

    # Identify Pareto-optimal points
    pareto_mask = find_pareto_front(f1_vals_plot, f2_vals_plot)

    plt.figure(figsize=(10, 6))

    # Plot non-Pareto points in blue
    plt.scatter(f1_vals_plot[~pareto_mask], f2_vals_plot[~pareto_mask], c='blue', marker='o', label='Optimization Iterations')
    # Plot Pareto-optimal points in red
    plt.scatter(f1_vals_plot[pareto_mask], f2_vals_plot[pareto_mask], c='red', marker='o', label='Pareto Frontier')

    # Plot initial samples in grey
    # plt.scatter(f1_initial, f2_initial, c='grey', marker='o', label='Initial Samples')

    # Labels and formatting
    plt.title(f'Pareto Curve of Recoveries\nGlucose in Raffinate vs Fructose in Extract\nInitial Samples: {sampling_budget}, Opt Iterations: {optimization_budget}')
    plt.xlabel('Glucose Recovery in Raffinate (%)', fontsize=12)
    plt.ylabel('Fructose Recovery in Extract (%)', fontsize=12)
    plt.xlim(0, 100+1)
    plt.ylim(0, 100+1)

    # Set x-axis limits with buffer to avoid clipping edge markers
    plt.grid(True)
    plt.legend()

    # Save the figure as an image
    image_filename = "recovery_pareto.png"
    plt.savefig(image_filename, dpi=300, bbox_inches='tight')
    plt.show()

    return image_filename


# ------ comparing multiple pareot
import numpy as np

def compare_pareto_similarity(f1_vals_100, f2_vals_100, f1_vals_20, f2_vals_20, tolerance_percent=5):
    """
    Compares Pareto fronts from 100-iteration and 20-iteration runs.
    Returns number and fraction of 20-iteration Pareto points within X% of any 100-iteration point.
    """

    # Convert to percent scale
    f1_100 = f1_vals_100 * 100
    f2_100 = f2_vals_100 * 100
    f1_20 = f1_vals_20 * 100
    f2_20 = f2_vals_20 * 100

    # Find Pareto masks
    pareto_mask_100 = find_pareto_front(f1_100, f2_100)
    pareto_mask_20 = find_pareto_front(f1_20, f2_20)

    pareto_100 = np.column_stack((f1_100[pareto_mask_100], f2_100[pareto_mask_100]))
    pareto_20 = np.column_stack((f1_20[pareto_mask_20], f2_20[pareto_mask_20]))

    count_within = 0

    for point in pareto_20:
        f1_p, f2_p = point
        for f1_ref, f2_ref in pareto_100:
            f1_close = abs(f1_p - f1_ref) <= (tolerance_percent / 100) * f1_ref
            f2_close = abs(f2_p - f2_ref) <= (tolerance_percent / 100) * f2_ref
            if f1_close and f2_close:
                count_within += 1
                break  # Move to next point in 20-front

    total_points = len(pareto_20)
    fraction_within = count_within / total_points if total_points > 0 else 0

    print(f"{count_within} out of {total_points} points in the 20-iteration Pareto front "
          f"are within {tolerance_percent}% of a point in the 100-iteration front "
          f"({fraction_within:.2%})")

    return count_within, total_points, fraction_within

def compare_recovery_pareto_plot(f1_vals_20, f2_vals_20, f1_vals_100, f2_vals_100, c1_vals_20, c2_vals_20, c1_vals_100, c2_vals_100):
    # Convert to percentages
    f1_vals_plot_100 = f1_vals_100 * 100
    f2_vals_plot_100 = f2_vals_100 * 100
    c1_vals_plot_100 = c1_vals_100 * 100
    c2_vals_plot_100 = c2_vals_100 * 100
    # -----
    f1_vals_plot_20 = f1_vals_20 * 100
    f2_vals_plot_20 = f2_vals_20 * 100
    c1_vals_plot_20 = c1_vals_20 * 100
    c2_vals_plot_20 = c2_vals_20 * 100



    # Function to find Pareto front
    def find_pareto_front(f1, f2):
        pareto_mask = np.ones(len(f1), dtype=bool)  # Start with all points assumed Pareto-optimal

        for i in range(len(f1)):
            if pareto_mask[i]:  # Check only if not already removed
                pareto_mask[i] = not np.any((f1 >= f1[i]) & (f2 >= f2[i]) & ((f1 > f1[i]) | (f2 > f2[i])))

        return pareto_mask

    # Identify Pareto-optimal points
    pareto_mask_100 = find_pareto_front(f1_vals_plot_100, f2_vals_plot_100)
    pareto_mask_20 = find_pareto_front(f1_vals_plot_20, f2_vals_plot_20)

    plt.figure(figsize=(10, 6))

    # WE ARE ONLY INTERESTED ON COMPARING PARETO FRONTS
    # Compare Recovery
    plt.scatter(f1_vals_plot_100[pareto_mask_100], f2_vals_plot_100[pareto_mask_100], c='red', marker='o', label='100 iterations Recovery Pareto Frontier')
    plt.scatter(f1_vals_plot_20[pareto_mask_20], f2_vals_plot_20[pareto_mask_20], c='purple', marker='s', label='20 iterations Recovery Pareto Frontier')
    # Compare Purity
    plt.scatter(c1_vals_plot_100[pareto_mask_100], c2_vals_plot_100[pareto_mask_100], c='grey', marker='o', label='100 iterations Purity at Recovery Frontier')
    plt.scatter(c1_vals_plot_20[pareto_mask_20], c2_vals_plot_20[pareto_mask_20], c='grey', marker='s', label='20 iterations Purity at Recovery Frontier')
    # Labels and formatting
    plt.title(f'Comparison of Sampling Efficeiny\n{len(f2_vals_20)-1} vs  {len(f2_vals_100)-1} iterations')
    plt.xlabel('Glucose (%)', fontsize=12)
    plt.ylabel('Fructose (%)', fontsize=12)

    plt.xlim(0, 100+1)
    plt.ylim(0, 100+1)

    # Set x-axis limits with buffer to avoid clipping edge markers
    plt.grid(True)
    plt.legend()

    # Save the figure as an image
    image_filename = "20_vs_100_recovery_pareto.png"
    plt.savefig(image_filename, dpi=300, bbox_inches='tight')
    plt.show()

    return image_filename


#------------------------------------------------------- 2. Purity Pareto

def create_purity_pareto_plot(c1_vals, c2_vals, zone_config, sampling_budget, optimization_budget):
    # Convert to percentages
    c1_vals_plot = c1_vals * 100
    c2_vals_plot = c2_vals * 100

    # Function to find Pareto front
    def find_pareto_front(c1, c2):
        pareto_mask = np.ones(len(c1), dtype=bool)  # Start with all points assumed Pareto-optimal

        for i in range(len(c1)):
            if pareto_mask[i]:  # Check only if not already removed
                pareto_mask[i] = not np.any((c1 >= c1[i]) & (c2 >= c2[i]) & ((c1 > c1[i]) | (c2 > c2[i])))

        return pareto_mask

    # Identify Pareto-optimal points
    pareto_mask = find_pareto_front(c1_vals_plot, c2_vals_plot)

    plt.figure(figsize=(10, 6))

    # Plot non-Pareto points in blue
    plt.scatter(c1_vals_plot[~pareto_mask], c2_vals_plot[~pareto_mask], c='blue', marker='o', label='Optimization Iterations')
    # Plot Pareto-optimal points in red
    plt.scatter(c1_vals_plot[pareto_mask], c2_vals_plot[pareto_mask], c='red', marker='o', label='Pareto Frontier')

    # Plot initial samples in grey
    # plt.scatter(c1_initial, c2_initial, c='grey', marker='o', label='Initial Samples')

    # Labels and formatting
    plt.title(f'Pareto Curve of Purities\nGlucose in Raffinate vs Fructose in Extract\nInitial Samples: {sampling_budget}, Opt Iterations: {optimization_budget}')
    plt.xlabel('Glucose Purity in Raffinate (%)', fontsize=12)
    plt.ylabel('Fructose Purity in Extract (%)', fontsize=12)
    plt.xlim(0, 100+1)
    plt.ylim(0, 100+1)
    plt.grid(True)
    # plt.legend()

    # Save the figure as an image
    image_filename = "purity_pareto.png"
    plt.savefig(image_filename, dpi=300, bbox_inches='tight')
    plt.show()

    return image_filename


def create_purity_vs_rec_pareto_plot(f1_vals, f2_vals, c1_vals, c2_vals, zone_config, sampling_budget, optimization_budget, comp):
    # Convert to percentages
    c1_vals_plot = c1_vals * 100
    c2_vals_plot = c2_vals * 100
    f1_vals_plot = f1_vals * 100
    f2_vals_plot = f2_vals * 100

    # Function to find Pareto front
    def find_pareto_front(c1, c2):
        pareto_mask = np.ones(len(c1), dtype=bool)  # Start with all points assumed Pareto-optimal

        for i in range(len(c1)):
            if pareto_mask[i]:  # Check only if not already removed
                pareto_mask[i] = not np.any((c1 >= c1[i]) & (c2 >= c2[i]) & ((c1 > c1[i]) | (c2 > c2[i])))

        return pareto_mask

    # Identify Pareto-optimal points
    # Glucose
    pareto_mask_glu = find_pareto_front(c1_vals_plot, f1_vals_plot)
    #Fructose
    pareto_mask_fru = find_pareto_front(c2_vals_plot, f2_vals_plot)

    plt.figure(figsize=(10, 6))

    # Plot non-Pareto points in blue

    plt.scatter(c1_vals_plot[~pareto_mask_glu], f1_vals_plot[~pareto_mask_glu], c='grey', marker='o', label='Glucose')
    plt.scatter(c2_vals_plot[~pareto_mask_fru], f2_vals_plot[~pareto_mask_fru], c='grey', marker='^', label='Fructose')

    plt.scatter(c1_vals_plot[pareto_mask_glu], f1_vals_plot[pareto_mask_glu], c='red', marker='o',label='Glucose Pareto Front')

    plt.scatter(c2_vals_plot[pareto_mask_fru], f2_vals_plot[pareto_mask_fru], c='orange', marker='^',label='Fructose Pareto Front')

    # Labels and formatting
    plt.title(f'Pareto Curves of {comp} Recovery vs Purity \nInitial Samples: {sampling_budget}, Opt Iterations: {optimization_budget}')
    plt.ylabel(f'Recovery (%)', fontsize=12)
    plt.xlabel(f'Purity (%)', fontsize=12)
    plt.xlim(0, 100+1)
    plt.ylim(0, 100+1)
    plt.grid(True)
    plt.legend()

    # Save the figure as an image
    image_filename = f"{comp}_recovery_vs_purity_pareto.png"
    plt.savefig(image_filename, dpi=300, bbox_inches='tight')
    plt.show()

    return image_filename



#------------------------------------------------------- 4. Pareto Outputs Trace
def find_pareto_front(f1, f2):
    pareto_mask = np.ones(len(f1), dtype=bool)  # Start with all points assumed Pareto-optimal

    for i in range(len(f1)):
        if pareto_mask[i]:  # Check only if not already removed
            pareto_mask[i] = not np.any((f1 >= f1[i]) & (f2 >= f2[i]) & ((f1 > f1[i]) | (f2 > f2[i])))

    return pareto_mask

def plot_inputs_vs_iterations(input_array, f1_vals, f2_vals):
    input_names = ['Feed (L/h)', 'Raffinate (L/h)', 'Desorbent (L/h)', 'Extract (L/h)', 'Index Time (min)']
    # Convert to percentages
    f1_vals_plot = f1_vals * 100
    f2_vals_plot = f2_vals * 100

    # Identify Pareto-optimal points
    pareto_mask = find_pareto_front(f1_vals_plot, f2_vals_plot)

    # Filter input_array for Pareto-optimal points
    pareto_inputs = input_array[pareto_mask]
    internal_flowrate, external_flowrate = calculate_flowrates(pareto_inputs, V_col, e)

    # Plot inputs vs iterations for Pareto-optimal points
    iterations = np.arange(1, len(external_flowrate) + 1)

    fig, ax1 = plt.subplots(figsize=(12, 8))

    # Plot all inputs except the last one
    for i in range(external_flowrate.shape[1] - 1):
        ax1.plot(iterations, external_flowrate[:, i], marker='o', label=f'{input_names[i]}')

    ax1.set_xlabel('Position on Pareto Front (Left-to-Right)', fontsize=12)
    ax1.set_ylabel('Flowrates (L/h)', fontsize=12)
    ax1.grid(True)
    ax1.legend(loc='upper left', bbox_to_anchor=(1.05, 1.0), borderaxespad=0.)

    # Create a second y-axis for the indexing time
    ax2 = ax1.twinx()
    ax2.plot(iterations, external_flowrate[:, -1], marker='o', color='grey', linestyle = "--", label=f'Input {input_names[-1]}')
    ax2.set_ylabel('Index Time (min)')
    ax2.legend(loc='upper left', bbox_to_anchor=(2.05, 1.0), borderaxespad=0.)
    # Ensure integer ticks only (no half-values)
    ax2.xaxis.set_major_locator(MultipleLocator(1))

    plt.title('Operating Conditions at Pareto-Optimal Operating Points')
    plt.tight_layout()  # Adjust layout so nothing gets cut off
    plt.show()


def plot_outputs_vs_iterations(f1_vals, f2_vals, c1_vals, c2_vals):
    input_names = ['Feed (L/h)', 'Raffinate (L/h)', 'Desorbent (L/h)', 'Extract (L/h)', 'Index Time (min)']
    # Convert to percentages
    c1_vals_plot = c1_vals * 100
    c2_vals_plot = c2_vals * 100
    f1_vals_plot = f1_vals * 100
    f2_vals_plot = f2_vals * 100

    # Plot inputs vs iterations for Pareto-optimal points
    iterations = np.arange(1, len(f1_vals_plot) + 1)

    fig, ax1 = plt.subplots(figsize=(12, 8))

    # Plot all inputs except the last one
    ax1.plot(iterations[50:], f1_vals_plot[50:], marker='o', label=f'Glucose Recovery')
    ax1.plot(iterations[50:], f2_vals_plot[50:], marker='o', label=f'Fructose Recovery')
    # ax1.plot(iterations, c1_vals_plot, marker='o', label=f'Glucose Purity')
    # ax1.plot(iterations, c1_vals_plot, marker='o', label=f'Fructose Purity')

    ax1.set_xlabel('Function Calls', fontsize=12)
    ax1.set_ylabel('Recovery Objective Functions (%)', fontsize=12)
    # ax1.grid(True)
    # ax1.legend(loc='upper left', bbox_to_anchor=(1.05, 1.0), borderaxespad=0.)

    # Ensure integer ticks only (no half-values)
    ax1.xaxis.set_major_locator(MultipleLocator(1))

    plt.title('Plot of Recovery Objectives and Purity Constraints vs Number of Iterations')
    plt.tight_layout()  # Adjust layout so nothing gets cut off
    plt.show()





# -------------------------- Constraint Porgression Over-Time
def constraints_vs_iterations(c1_vals, c2_vals):
    # Convert to percentages
    c1_vals_plot = c1_vals * 100
    c2_vals_plot = c2_vals * 100

    iterations = np.arange(0, len(c1_vals))

    fig, ax1 = plt.subplots()

    # Plot data
    ax1.scatter(iterations, c1_vals_plot, marker='o', label='Glucose Purity')
    ax1.scatter(iterations, c2_vals_plot, marker='o', label='Fructose Purity')
    ax1.axhline(y=99.5, linestyle="--", color="red", label='Constraint Threshold')

    # Axis labels and grid
    ax1.set_xlabel('Iteration', fontsize=12)
    ax1.set_ylabel('(%)', fontsize=12)
    ax1.grid(True)

    # Set x-axis limits with buffer to avoid clipping edge markers
    x_end = max(iterations)
    ax1.set_xlim(-0.5, x_end + 0.5)

    # Ensure integer ticks only (no half-values)
    ax1.xaxis.set_major_locator(MultipleLocator(1))

    # Place legend outside the plot (upper right)
    ax1.legend(loc='upper left', bbox_to_anchor=(1.05, 1.0), borderaxespad=0.)

    plt.tight_layout()  # Adjust layout so nothing gets cut off
    plt.show()





# %%
#%% Define in the Inputs
# Input the location if the saved jsons
# Make sure the path include the folder AND the file names!
inputs_path = r"C:\Users\28820169\ILLOVO_PCR-borhcl-type1_40iter_norm_config_all_inputs.json"
outputs_path = r"C:\Users\28820169\ILLOVO_PCR-borhcl-type1_40iter_norm_config_all_outputs.json"

# for comparing
inputs_path_20 = r"C:\Users\28820169\Downloads\BO Papers\Regression_Analysis\SIMS_20_iter_all_inputs.json"
outputs_path_20 = r"C:\Users\28820169\Downloads\BO Papers\Regression_Analysis\SIMS_20_iter_all_outputs.json"

# inputs_path_100 = r"C:\Users\28820169\Downloads\BO Papers\Regression_Analysis\SIMS_100_iter_all_inputs.json"
# outputs_path_100 = r"C:\Users\28820169\Downloads\BO Papers\Regression_Analysis\SIMS_100_iter_all_outputs.json"

# inputs_path_50 = r"C:\Users\28820169\Downloads\BO Papers\Regression_Analysis\SIMS_50_3333_iter_all_inputs.json"
# outputs_path_50 = r"C:\Users\28820169\Downloads\BO Papers\Regression_Analysis\SIMS_50_3333_iter_all_outputs.json"

# all_inputs_20, f1_vals_20, f2_vals_20, c1_vals_20, c2_vals_20 = load_inputs_outputs(inputs_path_20, outputs_path_20)
# all_inputs_100, f1_vals_100, f2_vals_100, c1_vals_100, c2_vals_100 = load_inputs_outputs(inputs_path_100, outputs_path_100)
# all_inputs_50, f1_vals_50, f2_vals_50, c1_vals_50, c2_vals_50 = load_inputs_outputs(inputs_path_50, outputs_path_50)


# # Load the file and the data
all_inputs, f1_vals, f2_vals, c1_vals, c2_vals = load_inputs_outputs(inputs_path, outputs_path)
# From some varianles:
sampling_budget = 1 # always
optimization_budget=f1_vals.shape[0]-sampling_budget
zone_config = np.array([3,3,3,3])
d_col = 2.6  # cm
L = 30 # cm
A_col = np.pi * d_col *0.25 # cm^2
V_col = A_col * L # cm^3
e = 0.4  # porosity

# Run the Functions and Visualise
# --- Paretos
rec_pareto_image_filename = create_recovery_pareto_plot(f1_vals, f2_vals, zone_config, sampling_budget, optimization_budget)
pur_pareto_image_filename = create_purity_pareto_plot(c1_vals, c2_vals, zone_config, sampling_budget, optimization_budget)
comp_1_name = "Fructose"
comp_2_name = "Glucose"
create_purity_vs_rec_pareto_plot(f1_vals, f2_vals, c1_vals, c2_vals, zone_config, sampling_budget, optimization_budget, comp_1_name)
# create_purity_vs_rec_pareto_plot(f1_vals, f2_vals, c1_vals, c2_vals, zone_config, sampling_budget, optimization_budget, comp_2_name)
plot_outputs_vs_iterations(f1_vals, f2_vals, c1_vals, c2_vals)
# compare_recovery_pareto_plot(f1_vals_50, f2_vals_50, f1_vals_100, f2_vals_100, c1_vals_50, c2_vals_50, c1_vals_100, c2_vals_100) # "rec" "pur"
# # --- Constraints
# constraints_vs_iterations(c1_vals, c2_vals)
# plot_inputs_vs_iterations(all_inputs, f1_vals, f2_vals)

# # # # ---- Tables
# opt_table_for_outputs_image_filename = create_output_optimization_table(f1_vals, f2_vals, c1_vals, c2_vals, sampling_budget)
# opt_table_for_inputs_image_filename = create_input_optimization_table(all_inputs, V_col, e, sampling_budget, f1_vals, f2_vals, c1_vals, c2_vals)
# # compare_pareto_similarity(f1_vals_100, f2_vals_100, f1_vals_50, f2_vals_50, tolerance_percent=15)


# %%
