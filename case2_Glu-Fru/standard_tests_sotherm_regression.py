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
Names = ['Glucose', 'Fructose']
datasets = [
    np.array([[3.05, 2.14, 1.55, 1.04, 0.73, 0.00, 0.00],
              [0.0021, 0.0015, 0.0011, 0.0007, 0.0007, 0.0004, 0.0005]]),
    np.array([[3.37, 2.46, 1.70, 1.13, 0.40, 0.00, 0.00],
              [0.0021, 0.0015, 0.0011, 0.0007, 0.0007, 0.0004, 0.0005]])
]

glu_profiles = []
fru_profiles = []
glu_rsq = []
fru_rsq = []

# Main Loop
for isotherm_func, isotherm_name in zip(standard_isotherms, isotherm_names):
    for name, data in zip(Names, datasets):
        c_data, q_data = data[0], data[1]
        initial_guess = [0.1] if isotherm_name == "Linear" else [0.1, 1.0]

        result, adj_r2 = fit_isotherm(c_data, q_data, isotherm_func, initial_guess)
        q_model = isotherm_func(result.x, c_data)

        if name == 'Glucose':
            glu_profiles.append((c_data, q_model, isotherm_name))
            glu_rsq.append(adj_r2)
        else:
            fru_profiles.append((c_data, q_model, isotherm_name))
            fru_rsq.append(adj_r2)

# Plotting the Isotherm Profiles
fig, axs = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
axs[0].set_title("Glucose Isotherms")
axs[1].set_title("Fructose Isotherms")

for c, q, label in glu_profiles:
    axs[0].plot(c, q, label=label)
axs[0].scatter(datasets[0][0], datasets[0][1], color='k', marker='x', label='Data')
axs[0].set_xlabel("C (g/cm³)")
axs[0].set_ylabel("q (g/cm³)")
axs[0].legend()

for c, q, label in fru_profiles:
    axs[1].plot(c, q, label=label)
axs[1].scatter(datasets[1][0], datasets[1][1], color='k', marker='x', label='Data')
axs[1].set_xlabel("C (g/cm³)")
axs[1].legend()

plt.tight_layout()
plt.show()

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
