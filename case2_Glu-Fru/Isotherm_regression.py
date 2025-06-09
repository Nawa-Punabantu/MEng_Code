# The general form of the advection-diffusion equation with reaction for any number of components, n:

# $$\frac{\partial C_n}{\partial t} = D \frac{\partial^2 C_n}{\partial x^2} - v \frac{\partial C_n}{\partial x} - F\frac{\partial q_n}{\partial t}$$

# $$F = \frac{1-e}{e}$$

# where:
# - $C$ is the liquid concentration of the diffusing substance.
# - $q$ is the solid concentration of the diffusing substance.
# - $F$ is the phase ratio
# - $e$ bed voidage
# - $t$ is time.
# - $D$ is the diffusion coefficient.
# - $v$ is the velocity field (advection speed).
# - $x$ is the spatial coordinate.

# # tips:
# - the Error: "IndexError: index 10 is out of bounds for axis 0 with size 9"
# may be due to a miss-match in size between the initial conditons and c, q in the ode func.

import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import pandas as pd
# Loading the Plotting Libraries
from matplotlib.pyplot import subplots
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from PIL import Image
from scipy import integrate
from scipy.optimize import minimize
import time
# import plotly.graph_objects as go


def cusotom_isotherm_func(cusotom_isotherm_params, c):
    """
    c => liquid concentration of ci
    q_star => solid concentration of ci @ equilibrium
    cusotom_isotherm_params[i] => given parameter set of component, i
    """

    # Uncomment as necessary

    #------------------- 1. Single Parameters Models
    # Linear
    # K1 = cusotom_isotherm_params[0]
    # H = K1 # Henry's Constant
    # q_star_1 = H*c

    #------------------- 2. Two-Parameter Models
    K1 = cusotom_isotherm_params[0]
    K2 = cusotom_isotherm_params[1]

    # #  2.1 Langmuir  
    Q_max = K1
    b = K2
    #-------------------------------
    q_star_2_1 = Q_max*b*c/(1 + b*c)
    #-------------------------------

    # 2.2 Freundlich
    # a = K1
    # b = K2
    # #-------------------------------
    # q_star_2_2 = b*c**(1/a)
    # #-------------------------------

    #------------------- 3. Three-Parameter models 
    # K1 = cusotom_isotherm_params[0]
    # K2 = cusotom_isotherm_params[1]
    # K3 = cusotom_isotherm_params[2]

    # Linear + Langmuir
    # H = K1
    # Q_max = K2
    # b = K3
    #-------------------------------
    # q_star_3 = H*c + Q_max*b*c/(1 + b*c)
    #-------------------------------

    return q_star_2_1 # [qA, ...]


# Define the objective function

def objective(func_params, func, data):
    """
    Returns the RSE and other relevant regression parameters for downstream modelling
    func => The cusotom_isotherm_func selected
    func_params => proposed parameter values
    data = [c_data, q_data]
    c_data => Liquid concentration data
    q_data => Solid concentration data
    
    """
    # Evaluate the concentration profile
    c_data, q_data =  data[0,:], data[1,:] 

    q_model = func(func_params, c_data)

    error = np.sum((q_data - q_model) ** 2)

    return error



if __name__ == "__main__":
    Names = ['Glucose', 'Fructose']
    # NOTE, UNITS!!!
    # c_data => grams of solute/vol of liquid cm^3
    # q_data => grams of solute/vol of resin (cm^3)

    glu_data = np.array([[3.05,2.14,1.55,1.04,0.73,0.00,0.00], [0.0021,0.0015,0.0011,0.0007,0.0007,0.0004, 0.0005]]) # [c_data, q_data]
    fru_data = np.array([[3.37,2.46,1.70,1.13,0.40,0.00,0.00], [0.0021,0.0015,0.0011,0.0007,0.0007,0.0004, 0.0005]] )# [c_data, q_data]
    data_all = [glu_data, fru_data]

    for i in range(len(data_all)):
        data = data_all[i]
        # Initial guess for parameters [slope, intercept]
        initial_guess =[0,0]

        # Minimize the loss
        func = cusotom_isotherm_func
        result = minimize(objective, initial_guess, args=(func, data), method='L-BFGS-B')

        # Extract fitted parameters
        fitted_params = result.x
        print("Fitted parameters:", fitted_params)

        # Plot 
        plt.scatter(data[0,:], data[1,:], label='Data')
        plt.plot(data[0,:], cusotom_isotherm_func(fitted_params, data[0,:]), color='red', label='Fitted Line')
        plt.title(f'{Names[i]}')
        plt.legend()
        plt.show()






    











