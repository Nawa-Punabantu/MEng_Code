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



import numpy as np
import matplotlib.pyplot as plt

def find_pareto_front(x, y):
    """Return boolean mask of Pareto-optimal points."""
    pareto_mask = np.ones(len(x), dtype=bool)
    for i in range(len(x)):
        if pareto_mask[i]:
            pareto_mask[i] = not np.any((x >= x[i]) & (y >= y[i]) & ((x > x[i]) | (y > y[i])))
    return pareto_mask

import numpy as np
import matplotlib.pyplot as plt

def find_pareto_front(x, y):
    """Return boolean mask of Pareto-optimal points."""
    pareto_mask = np.ones(len(x), dtype=bool)
    for i in range(len(x)):
        if pareto_mask[i]:
            pareto_mask[i] = not np.any(
                (x >= x[i]) & (y >= y[i]) &
                ((x > x[i]) | (y > y[i]))
            )
    return pareto_mask

def plot_raff_ext_pareto(c1_vals, c2_vals, r1_vals, r2_vals):
    """
    Plots Pareto frontiers for raffinate (top row) and extract (bottom row).
    c1_vals, c2_vals, r1_vals, r2_vals are lists of length 2:
      [ [raffinate data], [extract data] ]
    """
    fig, axs = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle("Pareto Optimality — Raffinate (Top) vs Extract (Bottom)", fontsize=16)

    for row_idx, stream_name in enumerate(["Raffinate", "Extract"]):
        # Convert to % arrays
        c1 = np.array(c1_vals[row_idx]) * 100
        c2 = np.array(c2_vals[row_idx]) * 100
        r1 = np.array(r1_vals[row_idx]) * 100
        r2 = np.array(r2_vals[row_idx]) * 100

        # Pareto masks
        pur_mask = find_pareto_front(c1, c2)
        rec_mask = find_pareto_front(r1, r2)
        both_mask = pur_mask & rec_mask

        # --- Purity plot ---
        ax_pur = axs[row_idx, 0]
        ax_pur.scatter(c1[~pur_mask], c2[~pur_mask], color='lightgray', label="Non-Pareto")
        ax_pur.scatter(c1[pur_mask], c2[pur_mask], color='red', label="Purity Frontier")
        ax_pur.scatter(c1[rec_mask], c2[rec_mask], facecolors='none', edgecolors='blue', label="Recovery Optimal")
        ax_pur.scatter(c1[both_mask], c2[both_mask], marker='*', color='gold', s=205, label="Both Optimal")
        ax_pur.set_title(f"{stream_name} — Purity Frontier (recovery-optimal highlighted)")
        ax_pur.set_xlabel("Comp1 Purity (%)")
        ax_pur.set_ylabel("Comp2 Purity (%)")
        ax_pur.grid(True)
        ax_pur.set_xlim(0, 101)
        ax_pur.set_ylim(0, 101)
        ax_pur.legend()

        # --- Recovery plot ---
        ax_rec = axs[row_idx, 1]
        ax_rec.scatter(r1[~rec_mask], r2[~rec_mask], color='lightgray', label="Non-Pareto")
        ax_rec.scatter(r1[rec_mask], r2[rec_mask], color='blue', label="Recovery Frontier")
        ax_rec.scatter(r1[pur_mask], r2[pur_mask], facecolors='none', edgecolors='red', label="Purity Optimal")
        ax_rec.scatter(r1[both_mask], r2[both_mask], marker='*', color='gold', s=205, label="Both Optimal")
        ax_rec.set_title(f"{stream_name} — Recovery Frontier (purity-optimal highlighted)")
        ax_rec.set_xlabel("Comp1 Recovery (%)")
        ax_rec.set_ylabel("Comp2 Recovery (%)")
        ax_rec.grid(True)
        ax_rec.set_xlim(0, 101)
        ax_rec.set_ylim(0, 101)
        ax_rec.legend()

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()


import numpy as np

def generate_dummy_inputs(n, V_col, L, A_col, d_col, e, zone_config, Description="Dummy Optimization Run", bounds = [
                                                                                                        (2.5, 5),   # m1
                                                                                                        (1, 2),     # m2
                                                                                                        (2.5, 5),   # m3
                                                                                                        (1, 2),     # m4
                                                                                                        (2, 10)    # t_index/t_ref
                                                                                                    ]):
    """
    Generate dummy optimization data consistent with all_inputs_dict format.

    Args:
        n (int): Number of iterations (rows of all_inputs).
        bounds (list of tuple): Bounds for each variable [(low, high), ...].
        V_col (float): Column volume (mL).
        L (float): Column length (cm).
        A_col (float): Column area (cm²).
        d_col (float): Column diameter (cm).
        e (float): Voidage.
        zone_config (list): Zone configuration.
        Description (str): Description of the dataset.

    Returns:
        dict: all_inputs_dict with dummy data.
    """
    m = len(bounds)  # number of decision variables
    all_inputs = np.zeros((n, m))

    # Sample random values within the provided bounds
    for j, (low, high) in enumerate(bounds):
        all_inputs[:, j] = np.random.uniform(low, high, n)

    # Extract t_index_min (last column)
    t_index_min = all_inputs[:, 4]

    # Build dictionary
    all_inputs_dict = {
        "Description": Description,
        "m1": all_inputs[:, 0].tolist(),
        "m2": all_inputs[:, 1].tolist(),
        "m3": all_inputs[:, 2].tolist(),
        "m4": all_inputs[:, 3].tolist(),
        "t_index_min": t_index_min.tolist(),

        "Q1_(L/h)": ((3.6 * all_inputs[:, 0] * V_col * (1 - e) + V_col * e) / (t_index_min * 60)).tolist(),
        "Q2_(L/h)": ((3.6 * all_inputs[:, 1] * V_col * (1 - e) + V_col * e) / (t_index_min * 60)).tolist(),
        "Q3_(L/h)": ((3.6 * all_inputs[:, 2] * V_col * (1 - e) + V_col * e) / (t_index_min * 60)).tolist(),
        "Q4_(L/h)": ((3.6 * all_inputs[:, 3] * V_col * (1 - e) + V_col * e) / (t_index_min * 60)).tolist(),

        "V_col_(mL)": [V_col],
        "L_col_(cm)": [L],
        "A_col_(cm)": [A_col],
        "d_col_(cm)": [d_col],
        "config": zone_config,
        "e": [e]
    }

    return all_inputs_dict
import numpy as np

def parse_inputs_dict(all_inputs_dict, include_t_index_as="col"):
    """
    Parse optimization results dictionary into structured matrices and scalars.

    Args:
        all_inputs_dict (dict): Input dictionary from optimization run.
        include_t_index_as (str): "row" or "col" - include t_index_min as extra row or column.

    Returns:
        dict with:
            - "M_matrix": numpy array of m ratios (+ t_index_min)
            - "Q_matrix": numpy array of Q values (+ t_index_min)
            - "vectors": dict of vectors (lists)
            - "scalars": dict of scalars (single values)
    """
    # Extract mj ratios
    m1 = np.array(all_inputs_dict["m1"])
    m2 = np.array(all_inputs_dict["m2"])
    m3 = np.array(all_inputs_dict["m3"])
    m4 = np.array(all_inputs_dict["m4"])
    t_index_min = np.array(all_inputs_dict["t_index_min"])

    # Extract Q values
    Q1 = np.array(all_inputs_dict["Q1_(L/h)"])
    Q2 = np.array(all_inputs_dict["Q2_(L/h)"])
    Q3 = np.array(all_inputs_dict["Q3_(L/h)"])
    Q4 = np.array(all_inputs_dict["Q4_(L/h)"])

    # Stack into matrices
    M_matrix = np.vstack([m1, m2, m3, m4]).T
    Q_matrix = np.vstack([Q1, Q2, Q3, Q4]).T

    if include_t_index_as == "col":
        M_matrix = np.column_stack([M_matrix, t_index_min])
        Q_matrix = np.column_stack([Q_matrix, t_index_min])
    elif include_t_index_as == "row":
        M_matrix = np.vstack([M_matrix, t_index_min])
        Q_matrix = np.vstack([Q_matrix, t_index_min])
    else:
        raise ValueError("include_t_index_as must be 'row' or 'col'")

    # Collect vectors and scalars
    vectors = {
        "config": np.array(all_inputs_dict["config"])
    }

    scalars = {
        "V_col_(mL)": all_inputs_dict["V_col_(mL)"][0],
        "L_col_(cm)": all_inputs_dict["L_col_(cm)"][0],
        "A_col_(cm)": all_inputs_dict["A_col_(cm)"][0],
        "d_col_(cm)": all_inputs_dict["d_col_(cm)"][0],
        "e": all_inputs_dict["e"][0]
    }

    return {
        "M_matrix": M_matrix,
        "Q_matrix": Q_matrix,
        "vectors": vectors,
        "scalars": scalars
    }


# %%
#%% Define in the Inputs
# Input the location if the saved jsons
# Make sure the path include the folder AND the file names!
inputs_path = r"C:\Users\28820169\SIMS_20_iter_all_inputs.json"
outputs_path = r"C:\Users\28820169\SIMS_20_iter_all_outputs.json"

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


import numpy as np

np.random.seed(42)  # for reproducibility


# GET dummy Data
# Number of points
n_points = 50

# Raffinate purity (comp1, comp2) — random but correlated
c1_raff = np.clip(np.random.normal(0.7, 0.15, n_points), 0, 1)
c2_raff = np.clip(1 - c1_raff + np.random.normal(0, 0.1, n_points), 0, 1)

# Extract purity (comp1, comp2) — different distribution
c1_ext = np.clip(np.random.normal(0.6, 0.2, n_points), 0, 1)
c2_ext = np.clip(1 - c1_ext + np.random.normal(0, 0.12, n_points), 0, 1)

# Raffinate recovery (comp1, comp2) — another distribution
r1_raff = np.clip(np.random.normal(0.75, 0.1, n_points), 0, 1)
r2_raff = np.clip(1 - r1_raff + np.random.normal(0, 0.08, n_points), 0, 1)

# Extract recovery (comp1, comp2)
r1_ext = np.clip(np.random.normal(0.65, 0.12, n_points), 0, 1)
r2_ext = np.clip(1 - r1_ext + np.random.normal(0, 0.1, n_points), 0, 1)

# Pack into nested lists as expected by the plotting function
c1_vals = [c1_raff, c1_ext]
c2_vals = [c2_raff, c2_ext]
f1_vals = [r1_raff, r1_ext]
f2_vals = [r2_raff, r2_ext]

import numpy as np

def generate_dummy_data(n_points=50, description="Dummy Optimization Results"):
    """
    Generate dummy purity and recovery data for raffinate and extract streams.
    Returns a data_dict in the required format.
    """
    # --- Raffinate Purity ---
    c1_raff = np.clip(np.random.normal(0.7, 0.15, n_points), 0, 1)
    c2_raff = np.clip(1 - c1_raff + np.random.normal(0, 0.1, n_points), 0, 1)

    # --- Extract Purity ---
    c1_ext = np.clip(np.random.normal(0.6, 0.2, n_points), 0, 1)
    c2_ext = np.clip(1 - c1_ext + np.random.normal(0, 0.12, n_points), 0, 1)

    # --- Raffinate Recovery ---
    r1_raff = np.clip(np.random.normal(0.75, 0.1, n_points), 0, 1)
    r2_raff = np.clip(1 - r1_raff + np.random.normal(0, 0.08, n_points), 0, 1)

    # --- Extract Recovery ---
    r1_ext = np.clip(np.random.normal(0.65, 0.12, n_points), 0, 1)
    r2_ext = np.clip(1 - r1_ext + np.random.normal(0, 0.1, n_points), 0, 1)

    # --- Pack into dict ---
    data_dict = {
        "Description": description,
        "c1_vals": [c1_raff.tolist(), c1_ext.tolist()],   # Purity Comp1
        "c2_vals": [c2_raff.tolist(), c2_ext.tolist()],   # Purity Comp2
        "f1_vals": [r1_raff.tolist(), r1_ext.tolist()],   # Recovery Comp1
        "f2_vals": [r2_raff.tolist(), r2_ext.tolist()],   # Recovery Comp2
    }

    return data_dict


# Example usage
dummy_outputs = generate_dummy_data()
print(dummy_outputs.keys())

# V_col, L, A_col, d_col, e, zone_config = 3, 1, 1, 1, 0.1, [1, 1, 1, 1]


# From some varianles:
sampling_budget = 1 # always
optimization_budget=np.shape(f1_vals)[0]-sampling_budget
zone_config = np.array([3,3,3,3])
d_col = 5  # cm
L = 70 # cm
A_col = np.pi * d_col *0.25 # cm^2
V_col = A_col * L # cm^3
e = 0.4  # porosity
import matplotlib.pyplot as plt
import numpy as np

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FormatStrFormatter

def plot_flows_and_indexing(all_inputs_dict):
    """
    Plot Q1-Q4 flowrates (L/h) and indexing time (min) across optimization iterations.
    Only markers are plotted (no lines), with grey vertical lines for readability.
    Legend is drawn in a separate figure.
    """

    # Extract values
    Q1 = np.array(all_inputs_dict["Q1_(L/h)"])
    Q2 = np.array(all_inputs_dict["Q2_(L/h)"])
    Q3 = np.array(all_inputs_dict["Q3_(L/h)"])
    Q4 = np.array(all_inputs_dict["Q4_(L/h)"])
    t_index = np.array(all_inputs_dict["t_index_min"])  # minutes

    iterations = np.arange(len(Q1))  # x-axis

    # Create figure
    fig, ax1 = plt.subplots(figsize=(10, 6))

    # Grey vertical lines at each iteration
    for i in iterations:
        ax1.axvline(x=i, color="grey", linestyle="--", alpha=0.3)

    # Plot only markers for flowrates
    m1 = ax1.plot(iterations, Q1, "o", label="Q1 (L/h)")
    m2 = ax1.plot(iterations, Q2, "s", label="Q2 (L/h)")
    m3 = ax1.plot(iterations, Q3, "^", label="Q3 (L/h)")
    m4 = ax1.plot(iterations, Q4, "d", label="Q4 (L/h)")
    ax1.set_xlabel("Iteration")
    ax1.set_ylabel("Flow rate (L/h)")
    ax1.yaxis.set_major_formatter(FormatStrFormatter("%.2g"))  # 2 sig figs

    # Right-hand axis for indexing time
    ax2 = ax1.twinx()
    m5 = ax2.plot(iterations, t_index, "x", color="k", label="Indexing Time (min)")
    ax2.set_ylabel("Indexing Time (min)", color="k")
    ax2.tick_params(axis="y", labelcolor="k")
    ax2.yaxis.set_major_formatter(FormatStrFormatter("%.2g"))  # 2 sig figs

    plt.title("Flow Rates and Indexing Time per Iteration")
    plt.tight_layout()
    plt.show()

    # --- Legend in separate figure ---
    fig_leg = plt.figure(figsize=(8, 1))
    ax_leg = fig_leg.add_subplot(111)
    ax_leg.axis("off")
    handles = m1 + m2 + m3 + m4 + m5
    labels = [h.get_label() for h in handles]
    ax_leg.legend(handles, labels, loc="center", ncol=5, frameon=False)
    plt.show()

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FormatStrFormatter

def plot_flows_indexing_grid(all_inputs_dict):
    """
    Plot Q1-Q4 flowrates (L/h) and indexing time (min) across optimization iterations
    on a (2,2) grid. Each subplot highlights one Q in color while others remain grey.
    Indexing time is always shown in black.
    """

    # Extract values
    Qs = [
        np.array(all_inputs_dict["Q1_(L/h)"]),
        np.array(all_inputs_dict["Q2_(L/h)"]),
        np.array(all_inputs_dict["Q3_(L/h)"]),
        np.array(all_inputs_dict["Q4_(L/h)"]),
    ]
    Q_labels = ["Q1 (L/h)", "Q2 (L/h)", "Q3 (L/h)", "Q4 (L/h)"]
    Q_markers = ["o", "s", "^", "d"]  # different symbols for each
    colors = ["tab:blue", "tab:orange", "tab:green", "tab:red"]

    t_index = np.array(all_inputs_dict["t_index_min"])  # minutes
    iterations = np.arange(len(Qs[0]))

    # Setup subplots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharex=True)
    axes = axes.flatten()

    for k in range(4):  # loop over Q1-Q4
        ax1 = axes[k]

        # Grey vertical lines at each iteration
        for i in iterations:
            ax1.axvline(x=i, color="grey", linestyle="--", alpha=0.2)

        # # Plot all Q’s in grey
        # for j in range(4):
        #     ax1.plot(iterations, Qs[j], Q_markers[j], 
        #              color="grey", alpha=0.5, label=f"{Q_labels[j]} (other)" if j != k else None)

        # Highlight only the target Q in color
        ax1.plot(iterations, Qs[k], Q_markers[k], color=colors[k], label=Q_labels[k])

        # Left y-axis (flow rates)
        ax1.set_ylabel("Flow rate (L/h)")
        ax1.yaxis.set_major_formatter(FormatStrFormatter("%.2g"))

        # Right-hand axis for indexing time
        ax2 = ax1.twinx()
        ax2.plot(iterations, t_index, "x", color="k", label="Indexing Time (min)")
        ax2.set_ylabel("Indexing Time (min)", color="k")
        ax2.tick_params(axis="y", labelcolor="k")
        ax2.yaxis.set_major_formatter(FormatStrFormatter("%.2g"))

        ax1.set_title(f"Highlight: {Q_labels[k]}")
        ax1.set_xlabel("Iteration")

    plt.tight_layout()
    plt.show()

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FormatStrFormatter

def find_pareto_front(f1, f2):
    """
    Finds Pareto front for 2D objective space (f1, f2).
    Accepts inputs as lists, 1D arrays, or 2D arrays (2, n_points).
    Returns boolean mask of Pareto-optimal points.
    """
    f1 = np.asarray(f1)
    f2 = np.asarray(f2)

    # --- Flatten if shape is (2, n_points) or (n_points, 2)
    if f1.ndim > 1:
        if f1.shape[0] == 2:   # (2, n_points)
            f1 = f1[0, :]
            f2 = f1[1, :] if f2.ndim > 1 else f2
        elif f1.shape[1] == 2: # (n_points, 2)
            f1, f2 = f1[:, 0], f1[:, 1]
        else:
            raise ValueError(f"Unexpected shape for f1/f2: {f1.shape}")

    if f2.ndim > 1:
        if f2.shape[0] == 2:
            f2 = f2[1, :]
        elif f2.shape[1] == 2:
            f2 = f2[:, 1]
        else:
            raise ValueError(f"Unexpected shape for f2: {f2.shape}")

    n_points = len(f1)
    is_optimal = np.ones(n_points, dtype=bool)

    for i in range(n_points):
        if is_optimal[i]:
            # Point i is dominated if another point is >= in both and > in at least one
            is_optimal[is_optimal] = ~(
                (f1[i] <= f1[is_optimal]) &
                (f2[i] <= f2[is_optimal]) &
                ((f1[i] < f1[is_optimal]) | (f2[i] < f2[is_optimal]))
            )

    return is_optimal


def plot_flows_indexing_grid_with_dict(input_dict, output_dict):
    """
    Plots Q1-Q4 flows and indexing times across optimization iterations.
    Vertical dashed lines at each iteration.
    Red lines = purity-optimal, Blue lines = recovery-optimal.
    Grid: 2x2, each plot highlights one Q flow in color.
    """
    import matplotlib.pyplot as plt
    import numpy as np

    # --- Extract matrices from input_dict ---
    m_mat = np.vstack([input_dict["m1"], input_dict["m2"],
                       input_dict["m3"], input_dict["m4"]]).T
    Q_mat = np.vstack([input_dict["Q1_(L/h)"], input_dict["Q2_(L/h)"],
                       input_dict["Q3_(L/h)"], input_dict["Q4_(L/h)"]]).T
    t_index = np.array(input_dict["t_index_min"])

    n_iter = Q_mat.shape[0]

    # --- Extract Pareto info ---
    f1_vals = np.array(output_dict["f1_vals"][0])  # raffinate
    f2_vals = np.array(output_dict["f2_vals"][0])  # raffinate
    c1_vals = np.array(output_dict["c1_vals"][0])
    c2_vals = np.array(output_dict["c2_vals"][0])

    pur_mask = find_pareto_front(c1_vals, c2_vals)
    rec_mask = find_pareto_front(f1_vals, f2_vals)

    # --- Colors & markers ---
    flow_colors = ["C0", "C1", "C2", "C3"]
    flow_labels = ["Q1", "Q2", "Q3", "Q4"]
    marker_t = "x"

    fig, axs = plt.subplots(2, 2, figsize=(14, 10))
    axs = axs.flatten()

    for i in range(4):
        ax = axs[i]
        for j in range(4):
            color = flow_colors[j] if i == j else "lightgray"
            for k in range(n_iter):
                line_color = "red" if pur_mask[k] else ("blue" if rec_mask[k] else "gray")
                ax.axvline(k, color=line_color, linestyle="--", alpha=0.5)
            ax.scatter(range(n_iter), Q_mat[:, j], color=color, s=40, label=flow_labels[j] if i==j else "")
        # Indexing time in all plots
        ax.scatter(range(n_iter), t_index, color="k", marker=marker_t, label="t_index")
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Flow (L/h) / t_index (min)")
        ax.set_title(f"Highlighting {flow_labels[i]}")
        ax.yaxis.set_major_formatter(lambda x, _: f"{x:.2g}")

    # --- Legend in separate figure ---
    fig2, ax2 = plt.subplots(figsize=(6, 2))
    for i, lbl in enumerate(flow_labels):
        ax2.scatter([], [], color=flow_colors[i], label=lbl, s=40)
    ax2.scatter([], [], color="k", marker=marker_t, label="t_index")
    ax2.axvline(0, color="red", linestyle="--", label="Purity Optimal")
    ax2.axvline(0, color="blue", linestyle="--", label="Recovery Optimal")
    ax2.axis("off")
    ax2.legend(loc="center", ncol=3)
    plt.show()

input_dummy_dic = generate_dummy_inputs(n_points, V_col, L, A_col, d_col, e, zone_config)
parsed = parse_inputs_dict(input_dummy_dic, include_t_index_as="col")

# plot_flows_and_indexing(input_dummy_dic)
data_dict = generate_dummy_data()
plot_flows_indexing_grid_with_dict(input_dummy_dic, data_dict)
plot_flows_indexing_grid(input_dummy_dic)

print("M_matrix shape:", parsed["M_matrix"].shape)
print("Q_matrix shape:", parsed["Q_matrix"].shape)
print("Scalars:", parsed["scalars"])
print("Vectors:", parsed["vectors"])
# Run the Functions and Visualise
# --- Paretos
# rec_pareto_image_filename = create_recovery_pareto_plot(f1_vals, f2_vals, zone_config, sampling_budget, optimization_budget)
# pur_pareto_image_filename = create_purity_pareto_plot(c1_vals, c2_vals, zone_config, sampling_budget, optimization_budget)
comp_1_name = "Fructose"
comp_2_name = "Glucose"

plot_raff_ext_pareto(c1_vals, c2_vals, f1_vals, f2_vals)

# plot_dual_pareto(
#     ext_pur_comp1, ext_pur_comp2,
#     ext_rec_comp1, ext_rec_comp2,
#     title_prefix="Extract"
# )

# create_purity_vs_rec_pareto_plot(f1_vals, f2_vals, c1_vals, c2_vals, zone_config, sampling_budget, optimization_budget, comp_1_name)
# # create_purity_vs_rec_pareto_plot(f1_vals, f2_vals, c1_vals, c2_vals, zone_config, sampling_budget, optimization_budget, comp_2_name)
# plot_outputs_vs_iterations(f1_vals, f2_vals, c1_vals, c2_vals)
# compare_recovery_pareto_plot(f1_vals_50, f2_vals_50, f1_vals_100, f2_vals_100, c1_vals_50, c2_vals_50, c1_vals_100, c2_vals_100) # "rec" "pur"

# # --- Constraints
# constraints_vs_iterations(c1_vals, c2_vals)
# plot_inputs_vs_iterations(all_inputs, f1_vals, f2_vals)

# # # # ---- Tables
# opt_table_for_outputs_image_filename = create_output_optimization_table(f1_vals, f2_vals, c1_vals, c2_vals, sampling_budget)
# opt_table_for_inputs_image_filename = create_input_optimization_table(all_inputs, V_col, e, sampling_budget, f1_vals, f2_vals, c1_vals, c2_vals)
# # compare_pareto_similarity(f1_vals_100, f2_vals_100, f1_vals_50, f2_vals_50, tolerance_percent=15)


# %%
