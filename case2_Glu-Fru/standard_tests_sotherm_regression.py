import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from matplotlib.table import Table

# Define standard isotherms
def linear(params, c):
    H = params[0]
    return H * c

def langmuir(params, c):
    Q_max, b = params
    return Q_max * b * c / (1 + b * c)

def freundlich(params, c):
    a, b = params
    return b * c ** (1 / a)

standard_isotherms = [linear, langmuir, freundlich]
isotherm_names = ["Linear", "Langmuir", "Freundlich"]

# Objective and fitting function
def objective(params, isotherm_func, c_data, q_data):
    q_model = isotherm_func(params, c_data)
    return np.sum((q_data - q_model) ** 2)

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

# Define synthetic isotherm models
def linear(params, c):
    K = params[0]
    return K * c

def langmuir(params, c):
    qmax, K = params
    return (qmax * K * c) / (1 + K * c)

def freundlich(params, c):
    Kf, n = params
    return Kf * c ** (1/n)

# Objective function for fitting
def objective(params, isotherm_func, c_data, q_data):
    q_model = isotherm_func(params, c_data)
    return np.sum((q_data - q_model) ** 2)

# R-squared computation
def compute_r_squared(params, isotherm_func, c_data, q_data):
    q_model = isotherm_func(params, c_data)
    ss_res = np.sum((q_data - q_model) ** 2)
    ss_tot = np.sum((q_data - np.mean(q_data)) ** 2)
    return 1 - ss_res / ss_tot

# Multi-start optimization
def multi_start_fit(isotherm_func, c_data, q_data, bounds, n_starts=20):
    best_result = None
    best_r2 = -np.inf
    for _ in range(n_starts):
        initial_guess = [np.random.uniform(low, high) for (low, high) in bounds]
        result = minimize(objective, initial_guess, args=(isotherm_func, c_data, q_data), method='L-BFGS-B', bounds=bounds)
        if result.success:
            r2 = compute_r_squared(result.x, isotherm_func, c_data, q_data)
            if r2 > best_r2:
                best_r2 = r2
                best_result = result
    return best_result, best_r2




def fit_isotherm(c_data, q_data, isotherm_func, initial_guess):
    result = minimize(objective, initial_guess, args=(isotherm_func, c_data, q_data), method='L-BFGS-B')
    q_model = isotherm_func(result.x, c_data)

    ss_res = np.sum((q_data - q_model) ** 2)
    ss_tot = np.sum((q_data - np.mean(q_data)) ** 2)
    r_squared = 1 - ss_res / ss_tot

    n = len(q_data)
    p = len(result.x)
    adj_r_squared = 1 - (1 - r_squared) * (n - 1) / (n - p - 1)

    return result, adj_r_squared

# Data
"""
UNITS, UNITS, UNITS !!!!!
IMPORTANT NOTE: Ce is in g/mL that is, grams_of_solute /(per) volume of liquid phase 
IMPORTANT NOTE: Qe is in g/mL that is, grams_of_solute /(per) VOLUME of solid phase 
"""
Names = ['Glucose', 'Fructose']
datasets = [
            np.array([[0.00000, # Ce (glucose)
                        0.00000,
                        0.00073,
                        0.00104,
                        0.00155,
                        0.00214,
                        0.00305
                        ],
                    [0.00248067,  # Qe (glucose)
                        0.00374133,
                        0.00384978,
                        0.00595089,
                        0.00638467,
                        0.00969222,
                        0.01337933
                        ]]),
            np.array([[0.00000, # Ce (fructose)
                        0.00000,
                        0.00040,
                        0.00113,
                        0.00170,
                        0.00246,
                        0.00337
                        ], 
                    [0.00048800,   # Qe (fructose)
                        0.00084044,
                        0.00084044,
                        0.00134878,
                        0.00159730,
                        0.00180289,
                        0.00262074
                        ]])
                        ]

glu_profiles = []
fru_profiles = []
glu_rsq = []
fru_rsq = []

# Main Loop
# for isotherm_func, isotherm_name in zip(standard_isotherms, isotherm_names):
#     for name, data in zip(Names, datasets):
#         c_data, q_data = data[0], data[1]
#         initial_guess = [1.0] if isotherm_name == "Linear" else [1.0, 1.0]

#         result, adj_r2 = fit_isotherm(c_data, q_data, isotherm_func, initial_guess)
#         q_model = isotherm_func(result.x, c_data)

#         if name == 'Glucose':
#             glu_profiles.append((c_data, q_model, isotherm_name))
#             glu_rsq.append(adj_r2)
#         else:
#             fru_profiles.append((c_data, q_model, isotherm_name))
#             fru_rsq.append(adj_r2)



# Plotting the Isotherm Profiles
fig, axs = plt.subplots(1, 2, figsize=(12, 5), sharey=False)
axs[0].set_title("Glucose Isotherms")
axs[1].set_title("Fructose Isotherms")

for c, q, label in glu_profiles:
    axs[0].plot(c, q, label=label)
axs[0].scatter(datasets[0][0], datasets[0][1], color='k', marker='x', label='Data')
axs[0].set_xlabel("C (g/mL)")
axs[0].set_ylabel("q (g/mL)")
axs[0].legend()

for c, q, label in fru_profiles:
    axs[1].plot(c, q, label=label)
axs[1].scatter(datasets[1][0], datasets[1][1], color='k', marker='x', label='Data')
axs[1].set_xlabel("C (g/mL)")
axs[1].legend()

plt.tight_layout()
plt.show()


############################################################
bounds_list = [[(0.01, 10)], [(0.01, 10), (0.01, 10)], [(0.01, 10), (0.01, 10)]]

# Fit and compare
for i, isotherm_func in enumerate(standard_isotherms):
    bounds = bounds_list[i]
    print(f"\n--- {isotherm_func.__name__.capitalize()} Isotherm ---")
    for name, (c_data, q_data) in zip(Names, datasets):
        result, r2 = multi_start_fit(isotherm_func, c_data, q_data, bounds)
        print(f"{name}: Best Fit Params = {result.x}, R² = {r2:.4f}")


# Plotting the Adjusted R² Table
fig, ax = plt.subplots(figsize=(6, 2))
ax.set_axis_off()

table_data = [
    ["Isotherm", "Glucose Adj. R²", "Fructose Adj. R²"]
] + [
    [name, f"{g:.4f}", f"{f:.4f}"]
    for name, g, f in zip(isotherm_names, glu_rsq, fru_rsq)
]

table = ax.table(cellText=table_data, loc='center', cellLoc='center', colWidths=[0.3, 0.3, 0.3])
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1.2, 1.5)

plt.title("Adjusted R² Comparison for Isotherm Models", pad=20)
plt.show()


# Store best-fit parameters
glu_param_table = []
fru_param_table = []

# Re-run loop to collect fitted parameters
for isotherm_func, isotherm_name in zip(standard_isotherms, isotherm_names):
    for name, data in zip(Names, datasets):
        c_data, q_data = data[0], data[1]
        initial_guess = [4] if isotherm_name == "Linear" else [1.0, 1.0]

        result, _ = fit_isotherm(c_data, q_data, isotherm_func, initial_guess)
        fitted_params = result.x

        if name == 'Glucose':
            glu_param_table.append(fitted_params)
        else:
            fru_param_table.append(fitted_params)

# Prepare table data for plotting
def format_params(param_list, model_name):
    if model_name == "Linear":
        return [f"H = {param_list[0]:.4f}"]
    elif model_name == "Langmuir":
        return [f"Qmax = {param_list[0]:.4f}", f"b = {param_list[1]:.4f}"]
    elif model_name == "Freundlich":
        return [f"a = {param_list[0]:.4f}", f"b = {param_list[1]:.4f}"]

# Construct table data
table_data = [["Isotherm", "Glucose Params", "Fructose Params"]]
for i, name in enumerate(isotherm_names):
    glu_str = "\n".join(format_params(glu_param_table[i], name))
    fru_str = "\n".join(format_params(fru_param_table[i], name))
    table_data.append([name, glu_str, fru_str])

# Plot the table
fig, ax = plt.subplots(figsize=(7, 3))
ax.axis('off')

table = ax.table(
    cellText=table_data,
    cellLoc='center',
    loc='center',
    colWidths=[0.3, 0.35, 0.35]
)
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1.2, 1.5)

plt.title("Best-Fit Parameters for Glucose and Fructose", pad=20)
plt.show()
