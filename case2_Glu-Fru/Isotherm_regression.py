import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

def linear(params, c):
    """
    Linear isotherm
    """
    K1 = params[0]
    H = K1 # Henry's Constant
    q_star = H*c
    return q_star

def langmuir(params, c):
    """
    Langmuir isotherm
    """
    K1 = params[0]
    K2 = params[1]

    #  2.1 Langmuir  
    Q_max = K1
    b = K2
    #-------------------------------
    q_star = Q_max*b*c/(1 + b*c)

    return q_star

def freundlich(params, c):
    """
    Freundlich isotherm
    """
    K1 = params[0]
    K2 = params[1]

    n = K1
    b = K2
    #-------------------------------
    q_star = b*c**(1/n)
    #-------------------------------

    return q_star


    
def custom_isotherm(params, c):
    """
    Langmuir isotherm function: q = (Q_max * b * c) / (1 + b * c)

    Parameters:
    - params: list or array of Langmuir parameters [Q_max, b]
    - c: array of liquid concentrations

    Returns:
    - q_star: equilibrium solid concentration
    """


    # Uncomment as necessary

    #------------------- 1. Single Parameters Models
    # Linear
    # K1 = params[0]
    # H = K1 # Henry's Constant
    # q_star_1 = H*c

    #------------------- 2. Two-Parameter Models
    K1 = params[0]
    K2 = params[1]

    # #  2.1 Langmuir  
    # Q_max = K1
    # b = K2
    # #-------------------------------
    # q_star_2_1 = Q_max*b*c/(1 + b*c)
    # #-------------------------------

    # # 2.2 Freundlich
    n = K1
    b = K2
    #-------------------------------
    q_star_2_2 = b*c**(1/n)
    #-------------------------------

    #------------------- 3. Three-Parameter models 
    # K1 = params[0]
    # K2 = params[1]
    # K3 = params[2]

    # Linear + Langmuir
    # H = K1
    # Q_max = K2
    # b = K3
    #-------------------------------
    # q_star_3 = H*c + Q_max*b*c/(1 + b*c)
    #-------------------------------
    return q_star_2_2

def regression_error(params, isotherm_func, c_data, q_data):
    """
    Objective function for minimization (Residual Sum of Squares)

    Parameters:
    - params: parameter guesses
    - isotherm_func: function handle for the isotherm model
    - c_data: measured liquid concentrations
    - q_data: measured solid concentrations

    Returns:
    - RSS (Residual Sum of Squares)
    """
    q_pred = isotherm_func(params, c_data)
    rss = np.sum((q_data - q_pred) ** 2)
    return rss

def fit_isotherm(c_data, q_data, isotherm_func, initial_guess):
    """
    Fits the given isotherm model to experimental data.

    Returns:
    - result: Optimization result object
    - r_squared: coefficient of determination
    """
    result = minimize(regression_error, initial_guess, args=(isotherm_func, c_data, q_data), method='L-BFGS-B')

    if not result.success:
        raise RuntimeError(f"Optimization failed: {result.message}")

    fitted_params = result.x
    q_pred = isotherm_func(fitted_params, c_data)
    ss_res = np.sum((q_data - q_pred) ** 2)
    ss_tot = np.sum((q_data - np.mean(q_data)) ** 2)
    r_squared = 1 - (ss_res / ss_tot)

    return result, r_squared

def plot_fit(c_data, q_data, q_model, fitted_params, name, colours):
    """
    Plot experimental vs fitted data.
    """
    if len(q_model) == 1:
        plt.scatter(c_data, q_data, label='Experimental Data')
        plt.plot(c_data, q_model, 'r-', label='Fitted Model')
        plt.title(f'Isotherm Fit - {name} \n Fitted Parameters: {fitted_params}')
        plt.xlabel('Liquid Concentration [g/mL]')
        plt.ylabel('Solid Concentration [g/mL]')
        plt.legend()
        plt.grid(True)
        plt.show()
    else:
        print('Here!')
        # print(f'q_data[0]:{q_data[0]}')
        # print(f'c_data:{c_data}')
        # print(f'len(q_model):{len(q_model)}')
        plt.scatter(c_data[0], q_data[0], label=f' {name[0]} Experimental Data', color=f'{colours[0]}')
        plt.scatter(c_data[1], q_data[1], label=f' {name[1]} Experimental Data', color=f'{colours[1]}')

        plt.plot(c_data[0], q_model[0], '-', label=f'{name[0]} Fitted Model', color=f'{colours[0]}')
        plt.plot(c_data[1], q_model[1], '-', label=f'{name[1]} Fitted Model', color=f'{colours[1]}')



        plt.title(f'Isotherm Fit - {name} \n Fitted Parameters: {fitted_params}')
        plt.xlabel('Liquid Concentration [g/mL]')
        plt.ylabel('Solid Concentration [g/mL]')
        plt.legend()
        plt.grid(True)
        plt.show()


# Main routine
if __name__ == "__main__":
    Names = ['Glucose', 'Fructose']
    colours = ['Green', 'orange']
    q_model_all = []
    q_data_all = []
    fitted_params_all = []
    c_plot = []

    datasets = [
        np.array([[0.00, 0.00, 0.73, 1.04,1.55, 2.14, 3.05],
                  [0.0013, 0.0018, 0.0028, 0.0031, 0.0032, 0.0033, 0.0034]]),
        np.array([[0.00, 0.00, 0.40,1.13, 1.70, 2.46,  3.37],
                  [0.0023, 0.0036, 0.0038, 0.0009, 0.001, 0.0014, 0.002]])
    ]

    for name, data in zip(Names, datasets):
        c_data, q_data = data[0], data[1]

        # Reasonable initial guess for [Q_max, b]
        initial_guess = [0.1, 1.0]
        # initial_guess = [0.1]

        result, r2 = fit_isotherm(c_data, q_data, custom_isotherm, initial_guess)
        fitted_params = result.x

        print(f"\n\nComponent: {name}")
        print(f"Fitted Parameters: {fitted_params}")
        print(f"R²: {r2:.4f}\n")

        
        q_model = custom_isotherm(fitted_params, c_data)
        print(f'q_model: {q_model}')

        q_model_all.append(q_model)
        q_data_all.append(q_data)
        fitted_params_all.append(fitted_params)
        c_plot.append(c_data)
        # plot_fit(c_data, q_data, q_model, fitted_params, name, colours)
    
    plot_fit(c_plot, q_data_all, q_model_all, fitted_params_all, Names, colours)


