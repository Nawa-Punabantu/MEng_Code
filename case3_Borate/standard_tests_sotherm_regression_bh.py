#%%
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from matplotlib.table import Table
import matplotlib.pyplot as plt
import pandas as pd




#%%
# Data
"""
UNITS, UNITS, UNITS !!!!!
IMPORTANT NOTE: Ce is in g/mL that is, grams_of_solute /(per) volume of liquid phase 
IMPORTANT NOTE: Qe is in g/mL that is, grams_of_solute /(per) VOLUME of solid phase 
"""

datasets = [ 
            np.array([[

                # UBK
                # Ce (borate)
                0,
                0.000168139,
                0.000131666,
                0.000084601,
                0.000069481,
                0.000041908


                        ],
                    [
                # Qe (borate)
                0,
                0.0000306,
                0.0000375,
                0.0000129,
                0.0000021,
                0.0000082

                        ]]),

            # PCR
            np.array([[ # Ce (borate)

                        0.00000000000,
                        0.00004118008,
                        0.00005373190,
                        0.00006891250,
                        0.00009562707,
                        0.00013429414




                        ], 
                    [# Qe (borate)

                        0.0000000,
                        0.00001175742,
                        0.00007533830,
                        0.00008604645,
                        0.00020571744,
                        0.00018851140

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
def fit_isotherm_multistart(isotherm_func, c_data, q_data, bounds, n_starts=10000):
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


Names = ['Borate-UBK', 'Borate-PCR']
standard_isotherms = [linear, langmuir, freundlich]
bounds_list = [[(0.001, 10)], [(0.001, 10), (0.001, 10)], [(0.001, 10), (0.001, 10)]]

# Store profiles and results
glu_profiles = []
fru_profiles = []
fit_results = {}

# Fit and collect profiles
from sklearn.linear_model import LinearRegression
import numpy as np

for i, isotherm_func in enumerate(standard_isotherms):
    bounds = bounds_list[i]
    model_name = isotherm_func.__name__.capitalize()

    for name, (c_data, q_data) in zip(Names, datasets):
        if isotherm_func.__name__.lower() == "linear":
            # --- Linear Fit with Zero Intercept ---
            c_data_reshaped = np.array(c_data).reshape(-1, 1)
            q_data = np.array(q_data)
            linreg = LinearRegression(fit_intercept=False)
            linreg.fit(c_data_reshaped, q_data)
            slope = linreg.coef_[0]
            r2 = linreg.score(c_data_reshaped, q_data)

            c_line = np.linspace(min(c_data), max(c_data), 1000)
            q_model = linreg.predict(c_line.reshape(-1, 1))

            result_params = np.array([slope])  # Only one parameter: slope
        else:
            # --- Use existing multistart optimization ---
            result, r2 = fit_isotherm_multistart(isotherm_func, c_data, q_data, bounds)
            c_line = np.linspace(c_data[0], c_data[-1], 1000)
            q_model = isotherm_func(result.x, c_line)
            result_params = result.x

        label = f"{model_name} (R²={r2:.4f})"
        if name == Names[0]:
            glu_profiles.append((c_line, q_model, label))
        else:
            fru_profiles.append((c_line, q_model, label))

        fit_results[(model_name, name)] = (result_params, r2)



# Plotting the Isotherm Profiles
fig, axs = plt.subplots(1, 2, figsize=(12, 5), sharey=False)
axs[0].set_title(f"{Names[0]} Isotherms")
axs[1].set_title(f"{Names[1]} Isotherms")

for c, q, label in glu_profiles:
    axs[0].plot(c, q, label=label)
axs[0].scatter(datasets[0][0], datasets[0][1], color='k', marker='x', label='Data')
axs[0].set_xlabel("Ce (g/mL)")
axs[0].set_ylabel("qe (g/mL)")
axs[0].legend()

for c, q, label in fru_profiles:
    axs[1].plot(c, q, label=label)
axs[1].scatter(datasets[1][0], datasets[1][1], color='k', marker='x', label='Data')
axs[1].set_xlabel("Ce (g/mL)")
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


# %%
def plot_combined_isotherm_fit(model_name, glu_data, fru_data, fit_results):
    """
    Plot both glucose and fructose data + fits for a single isotherm model.

    Parameters:
    - model_name: str, one of ['Linear', 'Langmuir', 'Freundlich']
    - glu_data, fru_data: np.array, shape (2, N)
    - fit_results: dict, keys = (Model, Component), values = (params, R²)
    """
    fig, ax = plt.subplots(figsize=(7, 5))

    # Unpack data
    c_glu, q_glu = glu_data
    c_fru, q_fru = fru_data

    # Generate smooth c range
    c_fit_glu = np.linspace(c_glu.min(), c_glu.max(), 500)
    c_fit_fru = np.linspace(c_fru.min(), c_fru.max(), 500)

    # Get fitted params and models
    params_glu, r2_glu = fit_results[(model_name, Names[0])]
    params_fru, r2_fru = fit_results[(model_name, Names[1])]

    # Select model function
    model_func = {
        'Linear': linear,
        'Langmuir': langmuir,
        'Freundlich': freundlich
    }[model_name]

    # Generate model fits
    q_fit_glu = model_func(params_glu, c_fit_glu)
    q_fit_fru = model_func(params_fru, c_fit_fru)

    # Plot glucose
    ax.scatter(c_glu, q_glu, color='green', marker='x', label= f"{Names[0]} Data")
    ax.plot(c_fit_glu, q_fit_glu, color='green', linestyle='--',
            label=f"{Names[0]} Fit\nParams: {', '.join([f'{p:.3f}' for p in params_glu])}\nR² = {r2_glu:.4f}")

    # Plot fructose
    ax.scatter(c_fru, q_fru, color='orange', marker='o', facecolors='none', label=f"{Names[1]} Data")
    ax.plot(c_fit_fru, q_fit_fru, color='orange', linestyle='--',
            label=f"{Names[1]} Fit\nParams: {', '.join([f'{p:.3f}' for p in params_fru])}\nR² = {r2_fru:.4f}")

    # Aesthetics
    ax.set_title(f"{model_name} Isotherm: {Names[0]} & {Names[1]}")
    ax.set_xlabel("Ce (g/mL)")
    ax.set_ylabel("Qe (g/mL)")
    ax.legend()
    ax.grid(True)

    # Force the axes to start at (0, 0)
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
    ax.margins(x=0, y=0)

    plt.tight_layout()
    plt.show()

# %%
# Call the function for each isotherm type
plot_combined_isotherm_fit("Linear", datasets[0], datasets[1], fit_results)
plot_combined_isotherm_fit("Langmuir", datasets[0], datasets[1], fit_results)
plot_combined_isotherm_fit("Freundlich", datasets[0], datasets[1], fit_results)
# %%
 