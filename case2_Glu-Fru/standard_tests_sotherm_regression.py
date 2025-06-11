import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from matplotlib.table import Table
import matplotlib.pyplot as plt
import pandas as pd





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

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

# Define isotherm models
def linear(params, c):
    K = params[0]
    return K * c

def langmuir(params, c):
    Qmax, K = params
    return (Qmax * K * c) / (1 + K * c)

def freundlich(params, c):
    Kf, n = params
    return Kf * c ** (1 / n)

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

# Multi-start fitting function with data collection
def fit_isotherm_multistart(isotherm_func, c_data, q_data, bounds, n_starts=20):
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


Names = ['Glucose', 'Fructose']
standard_isotherms = [linear, langmuir, freundlich]
bounds_list = [[(0.001, 10)], [(0.001, 10), (0.001, 10)], [(0.001, 10), (0.001, 10)]]

# Store profiles and results
glu_profiles = []
fru_profiles = []
fit_results = {}

# Fit and collect profiles
for i, isotherm_func in enumerate(standard_isotherms):
    bounds = bounds_list[i]
    model_name = isotherm_func.__name__.capitalize()
    for name, (c_data, q_data) in zip(Names, datasets):
        result, r2 = fit_isotherm_multistart(isotherm_func, c_data, q_data, bounds)
        q_model = isotherm_func(result.x, c_data)
        label = f"{model_name} (R²={r2:.4f})"
        if name == 'Glucose':
            glu_profiles.append((c_data, q_model, label))
        else:
            fru_profiles.append((c_data, q_model, label))
        fit_results[(model_name, name)] = (result.x, r2)

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

# Plotting the parameter table
import pandas as pd

# Create a DataFrame from the fit_results dictionary
table_data = []
for (model, component), (params, r2) in fit_results.items():
    param_str = ", ".join(f"{p:.4f}" for p in params)
    table_data.append([model, component, param_str, f"{r2:.4f}"])

df = pd.DataFrame(table_data, columns=["Model", "Component", "Fitted Parameters", "R²"])

# Plot the table
fig, ax = plt.subplots(figsize=(8, 2))
ax.axis('off')
table = ax.table(cellText=df.values, colLabels=df.columns, loc='center', cellLoc='center')
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1.2, 1.2)
plt.title("Best-Fit Parameters and R² Values", fontsize=12)
plt.tight_layout()
plt.show()

