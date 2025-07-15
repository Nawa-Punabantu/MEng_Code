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
import time
# import plotly.graph_objects as go


# Function that converts quantitiy schuduels to the values of that quantity at all time points from the ode
#######################################################



def column_func(column_func_inputs):
    
    #### UNPACK INPUT PARAMETERS ########
    iso_type, Names, color, parameter_sets, Da_all, Bm, e, Q_S, Q_inj, t_index, tend_min, nx, L, d_col, cusotom_isotherm_params_all, kav_params_all =  column_func_inputs[0:]
    
    ############## Calculated (Secondary) Input Parameters:
    Ncol_num = 1
    num_comp = len(Names) # Number of components

    ######## Column Specs:
    # Assuming the pipe diameter is 20% of the column diameter:
    F = (1-e)/e 
    d_in = 0.5 # cm 
    A_col = np.pi * d_col *0.25 # cm^2
    V_col = A_col * L # cm^3
    A_in = np.pi * (d_in/2)**2 # cm^2
    alpha = A_in / A_col
    L_total = L*Ncol_num # Total Lenght of all columns
   
     


    
    ########### Time Specs:
    tend = tend_min * 60 # s
    t_span = (0, tend) # +dt)  # from t=0 to t=n

    t_start_inject = []  # s # start time of the injection 
    t_start_inject_all = []

    time_between_inj = t_index  # s # This is the time between the feed time of 2 injcetions
    # NOTE: The system is not currently designed to account for periods of no external flow
    # Thus (currently): time_between_inj = t_index is always the case
    num_of_injections = int(np.round(tend/time_between_inj))
    
    # 't_start_inject_all' is a vecoter containing the times when port swithes occur for each port
    # Rows --> Different Ports
    # Cols --> Different time points
    t_start_inject_all = [[] for _ in range(Ncol_num)]  # One list for each node (including the main list)
    # Calculate start times for injections
    for k in range(num_of_injections):
        t_start_inject = k * time_between_inj
        t_start_inject_all[0].append(t_start_inject)  # Main list
        for node in range(1, Ncol_num):
            t_start_inject_all[node].append(t_start_inject + node * 0)  # all rows in t_start_inject_all are identical

    t_schedule = t_start_inject_all[0]
    # Column Dimensions:

    # Functions:
    # 1.1. Defining the Isotherm Given that it is uncoupled (UNC)
    # UNC
    # NOTE: You need to manually set the equation you want 
    #       - make sure this corresponds to the number of parameters in cusotom_isotherm_params_all
    #       - Default is Linear
    def cusotom_isotherm_func(cusotom_isotherm_params, c):
        """
        c => liquid concentration of ci
        q_star => solid concentration of ci @ equilibrium
        cusotom_isotherm_params[i] => given parameter set of component, i
        """

        # Uncomment as necessary

        #------------------- 1. Single Parameters Models
        # Linear
        K1 = cusotom_isotherm_params[0]
        H = K1 # Henry's Constant
        q_star_1 = H*c

        #------------------- 2. Two-Parameter Models
        # print(f'cusotom_isotherm_params:{cusotom_isotherm_params}')
        # K1 = cusotom_isotherm_params[0]
        # K2 = cusotom_isotherm_params[1]

        # # # # #  2.1 Langmuir  
        # Q_max = K1
        # b = K2
        # #-------------------------------
        # q_star_2_1 = Q_max*b*c/(1 + b*c)
        # #-------------------------------

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

        return q_star_1 # [qA, ...]
    # 1.2. Defining the Isotherm Given that it is COUPLED (CUP)
    # CUP
    # NOTE: You need to manually set the equation you want 
    #       - make sure this corresponds to the number of parameters in cusotom_isotherm_params_all
    #       - Default is Langmuir
        # Mass Transfer (MT) Models:

    def mass_transfer(kav_params, q_star, q ): # already for specific comp
        # kav_params: [kA, kB]

        # 1. Single Parameter Model
        # Unpack Parameters
        kav =  kav_params
        # MT = kav * Bm/(5 + Bm) * (q_star - q)
        MT = kav * (q_star - q)

        # 2. Two Parameter Model
        # Unpack Parameters
        # kav1 =  kav_params[0]
        # kav2 =  kav_params[1]
    
        # MT = kav1* (q_star - q) + kav2* (q_star - q)**2

        return MT
    

    def cusotom_CUP_isotherm_func(cusotom_isotherm_params, c, IDX, comp_idx):
        """
        Returns  solid concentration, q_star vector for given comp_idx
        *****
        Variables:
        cusotom_isotherm_params => parameters for each component [[A's parameters], [B's parameters]]
        NOTE: This function is (currently) structured to assume A and B have 1 parameter each. 
        c => liquid concentration of c all compenents
        IDX => the first row-index in c for respective components
        comp_idx => which of the components we are currently retreiving the solid concentration, q_star for
        q_star => solid concentration of ci @ equilibrium

        """
        # Unpack the component vectors (currently just considers binary case of A and B however, could be more)
        cA = c[IDX[0] + 0: IDX[0] + nx]
        cB = c[IDX[1] + 0: IDX[1] + nx]
        c_i = [cA, cB]
        # Now, different isotherm Models can be built using c_i
        
        # (Uncomment as necessary)

        #------------------- 1. Single Parameter Models

        # cusotom_isotherm_params has linear constants for each comp
        # Unpack respective parameters
        K1 = cusotom_isotherm_params[comp_idx][0] # 1st (and only) parameter of HA or HB
        q_star_lin = K1*c_i[comp_idx]


        #------------------- Two Parameter Models
        # K1 = cusotom_isotherm_params[comp_idx][0] 
        # K2 = cusotom_isotherm_params[comp_idx][1] 

        # 1. Freundlich
        # a = K1
        # b = K2
        # #-------------------------------
        # q_star_fred = b*c_i[comp_idx]**(1/a)
        # #-------------------------------

        # 2. Langmuir
        # Qmax = K1
        # b = K2
        # # #-------------------------------
        # q_star_lang1 = Qmax*b*c_i[comp_idx]/(1 + b*c_i[comp_idx])
        # # #-------------------------------

        # 3. Coupled Langmuir Model
        # cusotom_isotherm_params = [[QmaxA, bA], [QmaxB, bB]]

        # The parameter in the numerator is dynamic, depends on comp_idx i.e. which component we are looking at:
        # Qmax_i =  cusotom_isotherm_params[comp_idx][0]
        # b_i =  cusotom_isotherm_params[comp_idx][1]
        
        # # Fix the sum of parameters in the demoninator:
        # b1 = cusotom_isotherm_params[0][1] # 1st (and only) parameter of HA 
        # b2 = cusotom_isotherm_params[1][1] # 1st (and only) parameter of HB
        
        # q_star_lang2 = Qmax_i*b_i*c_i[comp_idx]/(1+ b1*c_i[0] + b2*c_i[1])



        #------------------- 3. Combined Models
        # The parameter in the numerator is dynamic, depends on comp_idx:
        # K_lin =  cusotom_isotherm_params[comp_idx][0]
        # K_Q = cusotom_isotherm_params[comp_idx][1]
        
        # # Fix the sum of parameters in the demoninator:
        # K1 = cusotom_isotherm_params[0][0] # 1st (and only) parameter of HA 
        # K2 = cusotom_isotherm_params[1][0] # 1st (and only) parameter of HB
        
        # c_sum = K1 + K2
        # linear_part = K_lin*c_i[comp_idx]
        # langmuir_part = K_Q*c_i[comp_idx]/(1+ K1*c_i[0] + K2*c_i[1])

        # q_star_combined =  linear_part + langmuir_part

        return q_star_lin # [qA, ...]

    # 2.
    # Generate Indices for the columns
    def generate_repeated_numbers(n, m):
        result = []
        n = int(n)
        m = int(m)
        for i in range(m):
            result.extend([i] * n)
        return result

    # 3.
    # INCLUDES THE C0 NODE BY DEFAULT !!!
    def set_x(L, Ncol_num, nx_col,dx):
        if nx_col == None:
            x = np.arange(0, L+dx, dx)
            nnx = len(x)
            nnx_col = int(np.round(nnx/Ncol_num))
            nx_BC = Ncol_num - 1 # Number of Nodes (mixing points/boundary conditions) in between columns
            return x, dx, nnx

        elif dx == None:
            nx = Ncol_num * nx_col
            nx_BC = Ncol_num - 1 # Number of Nodes in between columns
            x = np.linspace(0,L_total,nx)
            ddx = x[1] - x[0]

            # Indecies belonging to the mixing points between columns are stored in 'start'

            
            return x, ddx, nx

        # Fucntion to find the values of scheduled quantities 
        # at all t_ode_times points

    def find_indices(t_ode_times, t_schedule):
        """
        t_schedule -> vector of times when (events) port switches happen e.g. at [0,5,10] seconds
        t_ode_times -> vector of times from ODE

        We want to know where in t_ode_times, t_schedule occures
        These iwll be stored as indecies in t_idx
        Returns:np.ndarray: An array of indices in t_ode_times corresponding to each value in t_schedule.
        """
        t_idx = np.searchsorted(t_ode_times, t_schedule)
        t_idx = np.append(t_idx, len(t_ode_times))

        return t_idx

    def get_all_values(X, t_ode_times, t_schedule_times, Name):
        
        """
        X -> Matrix of Quantity at each schedule time. e.g:
        For a 5 seconds index time, at t_schedule_times = [0,5,10] seconds, the state of the feed would have
        concentractions of, X = [C1,C2,C3] g/m^3

        But what about the concentrations between 0 and 5s - or 5 and 10s?

        This function, creates a vector that fills in the gaps based on the size/resolution of the time
        vector from the solve_ivp output.

        """
        # Get index times
        t_idx = find_indices(t_ode_times, t_schedule_times)
        # print('t_idx:\n', t_idx)

        # Initialize:
        if np.shape(X)[0] == len(X):
            nrows = 1
        else:
            nrows = np.shape(X)[0]
        # print('nrows', nrows)

        values = np.zeros((nrows, len(t_ode_times))) # same num of rows, we just extend the times
        # print('np.shape(values):\n',np.shape(values))

        # Modify:
        k = 0

        for i in range(len(t_idx)-1): # during each schedule interval
            # i => counter that goes thourgh t_idx and schedule columns in case of col
            # j => counter that goes thourgh schedule rows in case of SMB
            j = i%nrows

            # # k is a counter that pushes the row index to the RHS every time it loops back up
            # if j == 0 and i == 0:
            #     pass
            # elif j == 0:
            #     k += 1
            
            if np.shape(X)[0] == len(X): # If we are dealing with the single col case
                X_new = np.tile(X[i], (len(t_ode_times[t_idx[i]:t_idx[i+1]]), 1))
            else:
                X_new = np.tile(X[:,j], (len(t_ode_times[t_idx[i]:t_idx[i+1]]), 1))

            values[:, t_idx[i]:t_idx[i+1]] = X_new.T # apply appropriate quantity value at approprite time intrval
            
        # Visualize:
        # # Table
        # print(Name," Values.shape:\n", np.shape(values))
        # print(Name," Values:\n", values)
        # # Plot
        # plt.plot(t_ode_times, values)
        # plt.xlabel('Time (s)')
        # plt.ylabel('X')
        # plt.show()

        return values, t_idx

    # 5. Function to Build Port Schedules:

    # This is done in two functions: (i) repeat_array and (ii) build_matrix_from_vector
    # (i) repeat_array
    # Summary: Creates the schedule for the 1st port, port 0, only. This is the port boadering Z2 & Z3 and always starts as a Feed port at t=0
    # (i) build_matrix_from_vector 
    # Summary: Takes the output from "repeat_array" and creates schedules for all other ports.
    # The "trick" is that the states of each of the, n, ports at t=0, is equal to the first, n, states of port 0.
    # Once we know the states for each port at t=0, we form a loop that adds the next state. 


    def repeat_array(vector, start_time_num):
        # vector = the states of all ports at t=0, vector[0] = is always the Feed port
        # start_time_num = The number of times the state changes == num of port switches == num_injections
        if isinstance(vector, int) or isinstance(vector, float):
            vector = [vector]
        else:
            pass

        repeated_array = np.tile(vector, (start_time_num // len(vector) + 1))
        return repeated_array[:start_time_num]

    def build_matrix_from_vector(vector, start_time_num):
        # vector = the states of all ports at t=0, vector[0] = is always the Feed port
        start_time_num = int(start_time_num)
        vector = np.array(vector)  # Convert the vector to a NumPy array
        n = len(vector) # number of ports
        
        # Initialize the matrix for repeated elements, ''ALL''
        # Rows: Ports
        # Columns: Time points of state change

        ALL = np.zeros((n, start_time_num), dtype=vector.dtype)  # Shape is (n, start_time_num)
        
        # Initialize the roll matrix
        # Rows: Ports
        # Columns: The first n time points of state change
        roll_mat = np.zeros((n, n), dtype=vector.dtype)  # Initialize an empty matrix with the same type as vector elements

        for i in range(n): # for each port
            # Roll the vector to the right by i positions
            row = np.roll(vector, i)
            roll_mat[i, :] = row  # Assign the row to the matrix

        for i in range(n):
            ALL[i, :] = repeat_array(roll_mat[i], start_time_num)
        return ALL

    # For the matrices for the Pulse Concentrations and Flowrates:
    # For one col, we just want the schedule for the 1 feed port
    Q_values = [Q_inj, 0, 0, 0, 0, 0, 0] # cm^3/s
    Q_pulse_all = repeat_array(Q_values, num_of_injections)



    # Make concentration values for each compoents:
    C_pulse_all = []
    for i in range(num_comp):
        print('i', i)
        C_add = repeat_array(parameter_sets[i]["C_feed"], num_of_injections)
        C_add[1:] = np.zeros_like(num_of_injections-1)
        C_pulse_all.append(C_add)

    Q_pulse_all[1:] = np.zeros_like(Q_pulse_all[1:])


    # Spacial Discretization:
    # Info:
    # nx --> Total Number of Nodes (EXCLUDING mixing points b/n nodes)
    x, dx, nx = set_x(L=L_total, Ncol_num = Ncol_num, nx_col = nx, dx = None)


    # DISPLAYING INPUT INFORMATION:
    # print('---------------------------------------------------')
    # print('Number of Components:', num_comp, )
    # print('---------------------------------------------------')
    # print('\nColumn Specs:\n')
    # print('---------------------------------------------------')
    # print('Number of Nodes:', nx, )
    # print('Column Length:', L, 'cm')
    # print('Column Diameter:', d_col, 'cm')
    # print('Column Volume:', V_col, 'cm^3')
    # print('---------------------------------------------------')
    # print('\Time Specs:\n')
    # print('---------------------------------------------------')
    # print(f'Pulse Time: {t_index} s')
    # print('Simulation Time:', tend_min, 'min')
    # # print('Injections happen at t(s) = :', t_start_inject_all[0], 'seconds')
    # print("alpha:", alpha, '(alpha = A_in / A_col)')
    # print('\nPort Schedules:')
    # for i in range(num_comp):
    #     print("Concentration Schedule:\nShape:",np.shape(C_pulse_all[i]),'\n', C_pulse_all[i], "\n")    
    # print("Injection Flowrate Schedule:\nShape:",np.shape(Q_pulse_all),'\n', Q_pulse_all, "\n") 
    # print('--------------------------------------------------------------------------\n\n\n\n')



    # Forming the ODEs:
    # Frist form the coefficeint matrices that correspond to ci, ci-1 & ci+1

    def coeff_matrix_builder(D, u, dx, c_length): 
        # where n = size of nxn matirx to be built (len(c))
        n = c_length
        # From descritization:
        coef_0 = ( D/dx**2 ) - ( u/dx ) # coefficeint of i-1
        coef_1 = ( u/dx ) - (2*D/(dx**2))# coefficeint of i
        coef_2 = (D/(dx**2))    # coefficeint of i+1
        # coef_0 = 1
        # coef_1 = 2
        # coef_2 = 3

        # Initialize coeff_matrix
        coeff_matrix = np.zeros((n,n))
        # Where the 1st (0th) row and col are for c1
        # 
        coeff_matrix[0,0], coeff_matrix[0,1] = coef_1, coef_2
        # for c2:
        coeff_matrix[1,0], coeff_matrix[1,1], coeff_matrix[1,2] = coef_0, coef_1, coef_2

        for i in range(2,n): # from row i=2 onwards
            # np.roll the row entries from the previous row, for all the next rows
            new_row = np.roll(coeff_matrix[i-1,:],1)
            coeff_matrix[i:] = new_row

        coeff_matrix[-1,0] = 0
        coeff_matrix[-1,-1] = coeff_matrix[-1,-1] + coef_2 # coef_1 + coef_2 account for rolling boundary 

        # if iso_type == "CUP": #(check phone for my video on this)
        #     coeff_matrix[nx-1, nx-1] = coeff_matrix[nx-1, nx-1] + coeff_matrix[nx-1, nx]
        #     coeff_matrix[nx-1, nx], coeff_matrix[nx, nx-1]= 0, 0


        #print(coeff_matrix)
        return coeff_matrix, coef_0, coef_1, coef_2



    ###########################################################################################

    ###########################################################################################


    # Soon to be changed to vector operations!



    # FORMING THE ODES



    ###########################################################################################

    # mod1: UNCOUPLED ISOTHERM:
    # Profiles for each component can be solved independently

    ###########################################################################################
    def mod1(t, v, comp_idx, Q_pulse_all, C_pulse_all, t_start_inject_all, Da_all): 
        # call.append("call")
        # print(len(call))
        c = v[:nx]
        q = v[nx:]

        # Initialize the derivatives
        dc_dt = np.zeros(nx)
        dq_dt = np.zeros(nx)

        # Define lambda functions for column velocities and injections
        # The column velocities:
        Q = lambda t, Q: next((Q[j] for j in range(len(Q)) if t_start_inject_all[0][j] <= t < t_start_inject_all[0][j] + t_index), 1/100000000)
        # Define the modified lambda function to check for injections
        c_inj = lambda t, C_inj, comp_idx: next((C_inj[comp_idx][j] for j in range(len(C_inj[comp_idx])) if t_start_inject_all[0][j] <= t < t_start_inject_all[0][j] + t_index), 1/100000000)
        #c_inj = lambda t, C_inj, i: next((C_inj[i][j] for j in range(len(C_inj[i,:])) if t_start_inject_all[i][j] < t < t_start_inject_all[i][j] + t_index), 1/10000000000)


        # Vectorized boundary condition calculations
        # Q_S =  The volumetric flowrate of the feed to the left of the feed port (pure solvent)
        Q_inj = Q(t, Q_pulse_all)
        Q_out_port = Q_S + Q_inj

        W1 = Q_S / Q_out_port 
        W2 = Q_inj / Q_out_port
        
        C_IN = W1 * 0 + W2 * c_inj(t, C_pulse_all, comp_idx) # "0" because feed to the left of the feed port is pure solvent

        #  Velocity in the Column:
        u_superficial = -Q_out_port/A_col
        u = u_superficial/e # intersticial
        
        # Dispersion in the column:
        # print(f'Da is here : {Da}')
        Da = -Da_all[comp_idx]
        # print(f'Da = {Da}')

        beta = 1 / alpha
        gamma = 1 - 3 * Da / (2 * u * dx)


        c_0 = ((beta * alpha) / gamma) * C_IN - ((2 * Da / (u * dx)) / gamma) * c[0] + ((Da / (2 * u * dx)) / gamma) * c[1]
        
        coeff_matrix, coef_0, coef_1, coef_2 = coeff_matrix_builder(Da, u, dx, len(c))

        # Isotherm:
        #########################################################################
        # IF DOING REGRESSION (one component at a time):
        isotherm = cusotom_isotherm_func(cusotom_isotherm_params_all, c)
        # Otherwise if normally simulating - for mulitple components
        # isotherm = cusotom_isotherm_func(cusotom_isotherm_params_all[comp_idx,:], c)

        # Mass Transfer:
        #########################################################################
        MT = mass_transfer(kav_params_all[comp_idx], isotherm, q)
        #print('MT:\n', MT)
        
        vec_add = np.zeros(len(c))
        vec_add[0] = coef_0 * c_0 # Adjusts C1 to account for C0 (BOUNDARRY)

        dc_dt = coeff_matrix @ c + vec_add - F * MT
        dq_dt = MT 

        return np.concatenate([dc_dt, dq_dt])

    ##################################################################################

    # mod2: COUPLED ISOTHERM
    # Profiles for coupled components MUST be solved TOGETHER

    ##################################################################################


    def mod2(t, v, Q_pulse_all, C_pulse_all, t_start_inject_all, Da_all): #
        
        # where, v = [c, q]
        c = v[:num_comp*nx] # c = [cA, cB] | cA = c[:nx], cB = c[nx:]
        q = v[num_comp*nx:] # q = [qA, qB]| qA = q[:nx], qB = q[nx:]

        # Craate Lables so that we know the component assignement in the c vecotor:
        # These labels represent the 1st row-index for the respective component
        A, B = 0*nx, 1*nx # Assume Binary 2*nx, 3*nx, 4*nx, 5*nx
        IDX = [A, B]

        # PLEASE READ:
        # The liquid concentration of the comp B @ the nth spacial position, is = c[B + n]
        # OR Similarly, the solid concentration @ spacial position 10, of component B is = q[B + 10]
        # OR to refer to all A's OR B's liquid concentrations: c[A + 0: A + nx]  c[B + 0: B + nx] 


        # Initialize the derivatives
        dc_dt = np.zeros_like(c)
        dq_dt = np.zeros_like(q)

        # Define lambda functions for column velocities and injections
        # The column velocities:
        Q = lambda t, Q: next((Q[j] for j in range(len(Q)) if t_start_inject_all[0][j] <= t < t_start_inject_all[0][j] + t_index), 1/100000000)
        # Define the modified lambda function to check for injections
        c_inj = lambda t, C_inj, comp_idx: next((C_inj[comp_idx][j] for j in range(len(C_inj[comp_idx])) if t_start_inject_all[0][j] <= t < t_start_inject_all[0][j] + t_index), 1/100000000)
        #c_inj = lambda t, C_inj, i: next((C_inj[i][j] for j in range(len(C_inj[i,:])) if t_start_inject_all[i][j] < t < t_start_inject_all[i][j] + t_index), 1/10000000000)


        # Vectorized boundary condition calculations
        # Q_S =  The volumetric flowrate of the feed to the left of the feed port (pure solvent)
        Q_inj = Q(t, Q_pulse_all)
        Q_out_port = Q_S + Q_inj

        W1 = Q_S / Q_out_port 
        W2 = Q_inj / Q_out_port
        
        # C_IN = W1 * 0 + W2 * c_inj(t, C_pulse_all, comp_idx) # "0" because feed to the left of the feed port is pure solvent

        #  Velocity in the Column:
        u_superficial = -Q_out_port/A_col
        u = u_superficial/e # intersticial




        # Initialize
        vec_add = np.zeros(len(c)) # vectoer to adjust C1 to accountcusotom_CUP_isotherm_func for C0 (at left boundary):
        MT = np.zeros(len(c)) # column vector: MT kinetcis for each comp: MT = [MT_A MT_B] 
        
        # Da = -Da_all[0]

        # coeff_matrix, coef_0, coef_1, coef_2 = coeff_matrix_builder(Da, u, dx, len(c))

        # beta = 1 / alpha
        # gamma = 1 - 3 * Da / (2 * u * dx)
            
        coeff_matrix = np.zeros((num_comp*nx, num_comp*nx))

        for comp_idx in range(num_comp): # for each component
            # Dispersion in the column:
            Da = -Da_all[comp_idx]

            # Lquid Phase Dynamics of Each Component
            comp_coeff_matrix, coef_0, coef_1, coef_2 = coeff_matrix_builder(Da, u, dx, len(c[IDX[comp_idx] : IDX[comp_idx] + nx]))
            
            # Add to coeff matrix:
            coeff_matrix[IDX[comp_idx] : IDX[comp_idx] + nx, IDX[comp_idx] : IDX[comp_idx] + nx] = comp_coeff_matrix
            
            beta = 1 / alpha
            gamma = 1 - 3 * Da / (2 * u * dx)

            # ------- STILL YET TO APPLY CUSTOM ISOTHERM TO THIS !!!!!!!!!!!!!!!!!!!!!!!!
            ######################(i) Isotherm ####################################################################

            # Comment as necessary for required isotherm:
            isotherm = cusotom_CUP_isotherm_func(cusotom_isotherm_params_all, c, IDX, comp_idx)
            # isotherm = iso_cup_langmuir(theta_cup_lang, c, IDX, comp_idx)

            ################### (ii) MT ##########################################################
            MT_comp = mass_transfer(kav_params_all[comp_idx], isotherm, q[IDX[comp_idx] + 0: IDX[comp_idx] + nx ])
            MT[IDX[comp_idx] + 0: IDX[comp_idx] + nx ] = MT_comp
            # [MT_A, MT_B, . . . ] KINETICS FOR EACH COMP
            # 
            
            # Repective Concentrations at left boundary:
            C_IN = W1 * 0 + W2 * c_inj(t, C_pulse_all, comp_idx)

            c_0_comp = ((beta * alpha) / gamma) * C_IN - ((2 * Da / (u * dx)) / gamma) * c[IDX[comp_idx] + 0] + ((Da / (2 * u * dx)) / gamma) * c[IDX[comp_idx] + 1] 

            # Vectoer to adjust C1 to account for C0 (at left boundary):
            #vec_add[0 + comp_idx * nx] = coef_0 * c_0_comp 
            vec_add[IDX[comp_idx] + 0] = coef_0 * c_0_comp 
        
        dc_dt = coeff_matrix @ c + vec_add - F * MT
        dq_dt = MT 

        return np.concatenate([dc_dt, dq_dt])

    ###########################################################################################

    # SOLVING THE ODES
    # creat storage spaces:
    y_matrices = []
    t_sets = []
    t_lengths = []

    print('----------------------------------------------------------------')
    print('Thermos:')
    print('Isotherm: theta_lin:', cusotom_isotherm_params_all)
    print('kav_params_all:', kav_params_all)
    print('Da:', Da_all)
    print('Bm:', Bm)
    print('----------------------------------------------------------------')
    
    print("\n\nSolving the ODEs. . . .")


    if iso_type == "UNC": # UNCOUPLED - solve 1 comp at a time
        for comp_idx in range(num_comp): # for each component
            print(f'Solving comp {comp_idx}. . . .')
            v0 = np.zeros(nx + nx) #  for both c and q
            solution = solve_ivp(mod1, t_span, v0, args=(comp_idx , Q_pulse_all, C_pulse_all, t_start_inject_all, Da_all))
            y_solution, t = solution.y, solution.t
            y_matrices.append(y_solution)
            t_sets.append(t)
            t_lengths.append(len(t))
        t=0
            

    # Assuming only a binary coupled system
    if iso_type == "CUP": # COUPLED - solve 
            v0 = np.zeros(num_comp*(nx + nx)) # for c and , for each comp
            solution = solve_ivp(mod2, t_span, v0, args=( Q_pulse_all, C_pulse_all, t_start_inject_all, Da_all))
            y_solution, t = solution.y, solution.t
            # Convert y_solution from: [cA, cB, qA, qB] ,  TO: [[cA, qA ], [cB, qB]] 
            # Write a function to do that

            def reshape_ysol(x, nx, num_comp):
                # Initialize a list to store the reshaped components
                reshaped_list = []
                
                # Iterate over the number of components
                for i in range(num_comp):
                    # Extract cX and qX submatrices for the i-th component
                    cX = x[i*nx:(i+1)*nx, :]      # Extract cX submatrix
                    qX = x[i*nx + num_comp*nx : (i+1)*nx + num_comp*nx, :]       # Extract qX submatrix
                    concat = np.concatenate([cX, qX])
                    # print('i:', i)
                    # print('cX:\n',cX)
                    # print('qX:\n',qX)
                    # Append the reshaped pair [cX, qX] to the list
                    reshaped_list.append(concat)
                
                # Convert the list to a NumPy array
                result = np.array(reshaped_list)
                
                return result
            
            y_matrices = reshape_ysol(y_solution, nx, num_comp)
            t_sets = 0


    print('----------------------------------------------------------------')
    print('\nSolution Size:')
    for i in range(num_comp):
        print(f'y_matrices[{i}]', y_matrices[i].shape)
    print('----------------------------------------------------------------')
    print('----------------------------------------------------------------')





    # Visualization



    # MASS BALANCE
    def single_col_Expected_Acc(t_ode, y_ode):

        # "UNC" = UNCOUPLED ISOTHERM (comps have respective time points stored in t_sets)
        # "CUP" = COUPLED ISOTHERM (all comps have same time points, t)

        # Storage Spaces:
        m_in = np.zeros(num_comp)   
        m_out = np.zeros(num_comp)


        #  MASS IN:
        # 1. Get every component's CIN vector over time - by converting the respective schedules to "all_values"
        # 2. Multiply by flow into the column to get mass flowrate, F1 (g/s)
        # 3. Integrate F1 to get mass in g
        # 4. Sum for all comp to get total mass in

        for i in range(num_comp): # for each component
            if iso_type == "UNC":
                C_feed_pulse, t_idx = get_all_values(C_pulse_all[i], t_ode[i], t_schedule, "Concentration Schedule")
                Q_inj, t_idx = get_all_values(Q_pulse_all,  t_ode[i], t_schedule, "Flowrate Injection Schedule")
            elif iso_type == "CUP":
                C_feed_pulse, t_idx = get_all_values(C_pulse_all[i], t_ode, t_schedule, "Concentration Schedule")
                Q_inj, t_idx = get_all_values(Q_pulse_all,  t_ode, t_schedule, "Flowrate Injection Schedule")
            
            # Recall the the concentration from the pulse is diluted at the mixing node:
            W = Q_inj/(Q_inj+Q_S) # weighted flowrate into the column

            C_feed = W * C_feed_pulse # g/cm^3

            
            QIN = Q_S + Q_inj # cm^3/s |  volume flow into col | pluse os sufficiently small
            # print('QIN:\n',QIN)
            # print('type(QIN):\n',type(QIN))
            
            F_feed = C_feed * QIN # (g/cm^3 * cm^3/s)  =>  g/s | mass flow into col (for comp, i)
            # print(f'F_feed: {F_feed} g/s')
            # print('type(F_feed):\n',type(F_feed))

            if iso_type == "UNC":
                m_in[i] = integrate.simpson(F_feed, t_ode[i]) # Perform the integration
                # print(f'QIN*C_feed[0]*t_index: {Q_inj[0][0]*parameter_sets[i]["C_feed"]}')
                # m_in[i] = Q_inj[0][0]*parameter_sets[i]["C_feed"] # *t_index*np.ones_like(t_ode[i]) # Perform the integration
            elif iso_type == "CUP":
                m_in[i] = integrate.simpson(F_feed, t_ode) # Perform the integration
        

            #  MASS OUT:

            # Assume:
            # Inlet and outlet tubes have equal dimensions 
            # Inlet and outlet will have equal volumetric flowrates - incompressible fluid
            QOUT = QIN # cm^3/s

            F_out = y_ode[i][nx-1, :] * QOUT # g/cm^3 * cm^3/s => g/s

            if iso_type == "UNC":
                m_out[i] = integrate.simpson(F_out, t_ode[i]) # Perform the integration
            elif iso_type == "CUP":
                m_out[i] = integrate.simpson(F_out, t_ode) # Perform the integration
            
            # print('i',i)
            F_feed = np.concatenate(F_feed)
            F_out = np.concatenate(F_out)

            # print('type(t_ode[i])', type(t_ode[i]))
            # print('np.shape(t_ode[i])', np.shape(t_ode[i]))
            # print('type(F_feed)', type(F_feed))
            # print('np.shape(F_feed)', np.shape(F_feed))
            # print('type(F_out)', type(F_out))
            # print('np.shape(F_out)', np.shape(F_out))


            # plt.plot(t_ode[i], F_feed, 'k', label = 'IN')
            # plt.plot(t_ode[i], F_out, 'r', label = 'OUT')
            # plt.legend()
            # plt.show()
        

        # print(f'm_in:\n{m_in}')
        # print(f'm_out:\n{m_out}')
        
        Expected_Acc = m_in - m_out

        return Expected_Acc, m_in, m_out, C_feed

    def model_acc(y_ode, V_col, e, num_comp):
        
        """
        Func to integrate the concentration profiles at tend and estimate the amount 
        of solute left on the solid and liquid phases
        """
        mass_l = np.zeros(num_comp)
        mass_r = np.zeros(num_comp)

        # c_avg = np.zeros(num_comp)
        # q_avg = np.zeros(num_comp)

        # Model Acc (how much is left in the column)
        for i in range(num_comp): # for each component
            V_l = e * V_col # Liquid Volume cm^3
            V_r = (1-e)* V_col # resin Volume cm^3



            # conc => g/cm^3
            # V => cm^3
            # integrate to get => g

            # # METHOD 1:
            # V_l = np.linspace(0,V_l,nx) # cm^3
            # V_r = np.linspace(0,V_r,nx) # cm^3
            # mass_l[i] = integrate.simpson(y_ode[i][:nx,-1], x)*A_col*e # mass in liq at t=tend
            # mass_r[i] = integrate.simpson(y_ode[i][nx:,-1], x)*A_col*(1-e) # mass in resin at t=tend

            # # METHOD 2:
            V_l = np.linspace(0,V_l,nx) # cm^3
            V_r = np.linspace(0,V_r,nx) # cm^3
            mass_l[i] = integrate.simpson(y_ode[i][:nx,-1], V_l) # mass in liq at t=tend
            mass_r[i] = integrate.simpson(y_ode[i][nx:,-1], V_r) # mass in resin at t=tend


            # METHOD 3: 
            # c_avg = np.average(y_ode[i][:nx,-1]) # Average conc at t=tend
            # q_avg = np.average(y_ode[i][:nx,-1])

            # mass_l[i] = c_avg * V_l
            # mass_r[i] = q_avg * V_r

            
        Model_Acc = mass_l + mass_r # g

        return Model_Acc
    
    if iso_type == "UNC":
        Expected_Acc, m_in, m_out, C_feed = single_col_Expected_Acc(t_sets, y_matrices)

    elif iso_type == "CUP":
        Expected_Acc, m_in, m_out, C_feed = single_col_Expected_Acc(t, y_matrices)
        
    Model_Acc = model_acc(y_matrices, V_col, e, num_comp)

    # ------------------------------------------
    Error = Model_Acc - Expected_Acc

    Error_percent = (sum(Error)/sum(Expected_Acc))*100
    # ------------------------------------------

    ############## TABLES ##################   
        # Define the data for the table
    
    def check_value_for_zeros(input_value):
        for i in range(len(input_value)):
            if input_value[i] <= 1e-5:
                input_value[i] = 0
            return input_value

    # check_value_for_zeros(Expected_Acc)
    # check_value_for_zeros(Model_Acc)

    # Example usage:

    data = {
        'Metric': [
            'Mass In:',
            'Mass out:',
            'Total Expected Acc (IN-OUT)', 
            'Total Model Acc (r+l)', 
            '---------------------------------',
            'Total Error (Mod-Exp)', 
            'Total Error Percent (relative to Exp_Acc)'
        ],
        'Value': [
            f'{sum(m_in)} g',
            f'{sum(m_out)} g',
            f'{sum(Expected_Acc)} g', 
            f'{sum(Model_Acc)} g',
            '------------------------',
            f'{sum(Error)} g', 
            f'{Error_percent} %'
        ]
    }

    # Create a DataFrame
    df = pd.DataFrame(data)

    # Display the DataFrame
    # print(df)

    # Get the elution curves:
    # Initialize:
    col_elution = []
    print('\n\n\n')
    for i in range(num_comp):
        col_elution.append(y_matrices[i][nx-1,:])
        # print(f'np.shape(col_elution[{i}]): {np.shape(col_elution[i])}')

    return col_elution, y_matrices, nx, t, t_sets, t_schedule, C_feed,  m_in, m_out, Model_Acc, Expected_Acc, Error_percent

# # ------ INPUTS

# #######################################################



# #  2D PLOTS


   


# Animating Results

def coupled_animate_profiles(t, title, y, nx, labels, colors, t_start_inject_all, t_index):
    def create_animation(y_profiles, t, concentration_type, filename, labels, colors):
        fig, ax = plt.subplots(1, 1, figsize=(15, 5))

        # Initialize lines for each profile
        lines = []
        x = np.linspace(0, L, np.shape(y_profiles[0])[0])
        for i, y_profile in enumerate(y_profiles):
            line, = ax.plot(x, y_profile[:, 0], label=f"{labels[i]}: H{labels[i]} = {parameter_sets[i]['H']}, kfp{labels[i]} = {kav_params_all[i]}", color=colors[i])
            lines.append(line)

        # Add a text box in the top right corner to display the time
        time_text = ax.text(0.02, 0.95, '', transform=ax.transAxes, fontsize=12, verticalalignment='top')

        # Add black vertical lines at the left edge at all times
        for col_idx in range(Ncol_num + 1):  # +1 to include the last column boundary
            x_pos = col_idx * L
            ax.axvline(x=x_pos, color='k', linestyle='-')

        # Function to add red vertical lines at the injection times
        def add_pulse_lines(t):
            for col in range(len(t_start_inject_all)):
                for start in t_start_inject_all[col]:
                    if start <= t < start + t_index:
                        x_pos = col * L
                        ax.axvline(x=x_pos, color='r', linestyle='-', linewidth=1)

        # Function to update the y data of the lines
        def update(frame):
            for i, y_profile in enumerate(y_profiles):
                lines[i].set_ydata(y_profile[:, frame])
            time_text.set_text(f'Time: {t[frame]:.2f} s')
            
            # Clear existing red lines
            [line.remove() for line in ax.lines if line.get_color() == 'r']
            # Add new red lines
            add_pulse_lines(t[frame])

            return lines + [time_text]

        # Set the limits for the x and y axis
        y_min = np.min([np.min(y_profile) for y_profile in y_profiles])
        y_max = np.max([np.max(y_profile) for y_profile in y_profiles]) + (5 / 10000000)  # c_IN
        ax.set_ylim([y_min, y_max])
        ax.set_xlabel("Column Length, m")
        ax.set_ylabel(f"{title} {concentration_type} ($\mathregular{{g/cm^3}}$)")
        ax.legend()

        # Determine the number of frames based on the length of the time vector
        n_frame = len(t)
        
        # Create the animation
        ani = animation.FuncAnimation(fig, update, frames=range(n_frame), interval=100, blit=True)

        # Set up the writer
        ffmpegWriter = animation.writers['ffmpeg']
        writer = ffmpegWriter(fps=15, metadata=dict(artist='Me'), bitrate=1800)

        print(f"Saving animation to {filename}...")
        ani.save(filename, writer=writer)
        print(f"Animation saved to {filename}.")

        # Display the animation
        plt.show()

    # Separate the y data into liquid and solid concentrations
    liquid_profiles = [y_profile[:nx, :] for y_profile in y]
    solid_profiles = [y_profile[nx:, :] for y_profile in y]

    # Create animations for liquid and solid concentrations
    create_animation(liquid_profiles, t, "Liquid Concentration", f"{title}_liquid.mp4", labels, colors)
    create_animation(solid_profiles, t, "Solid Concentration", f"{title}_solid.mp4", labels, colors)


def animate_profiles(t_sets, title, y, nx, labels, colors, t_start_inject_all, t_index):
    def create_animation(y_profiles, t_profiles, concentration_type, filename, labels, colors):
        fig, ax = plt.subplots(1, 1, figsize=(15, 5))

        # Initialize lines for each profile
        lines = []
        x = np.linspace(0, L, np.shape(y_profiles[0])[0])
        for i, y_profile in enumerate(y_profiles):
            line, = ax.plot(x, y_profile[:, 0], label=f"{labels[i]}: H{labels[i]} = {parameter_sets[i]['H']}, kfp{labels[i]} = {kav_params_all[i]}", color=colors[i])
            lines.append(line)

        # Add a text box in the top right corner to display the time
        time_text = ax.text(0.02, 0.95, '', transform=ax.transAxes, fontsize=12, verticalalignment='top')

        # Add black vertical lines at the left edge at all times
        for col_idx in range(Ncol_num + 1):  # +1 to include the last column boundary
            x_pos = col_idx * L
            ax.axvline(x=x_pos, color='k', linestyle= '-')


        # Function to update the y data of the lines
        def update(frame):
            for i, (y_profile, t_profile) in enumerate(zip(y_profiles, t_profiles)):
                if frame < len(t_profile):
                    lines[i].set_ydata(y_profile[:, frame])
                    time_text.set_text(f'Time: {t_profile[frame]:.2f} s')
            return lines + [time_text]

        # Set the limits for the x and y axis
        y_min = np.min([np.min(y_profile) for y_profile in y_profiles])
        y_max = np.max([np.max(y_profile) for y_profile in y_profiles]) + (5 / 10000000)  # c_IN
        ax.set_ylim([y_min, y_max])
        ax.set_xlabel("Column Length, m")
        ax.set_ylabel(f"{title} {concentration_type} ($\mathregular{{g/cm^3}}$)")
        ax.legend()

        # Determine the maximum number of frames
        n_frame = max(len(t_profile) for t_profile in t_profiles)
        
        # Create the animation
        ani = animation.FuncAnimation(fig, update, frames=range(n_frame), interval=100, blit=True)

        # Set up the writer
        ffmpegWriter = animation.writers['ffmpeg']
        writer = ffmpegWriter(fps=15, metadata=dict(artist='Me'), bitrate=1800)

        print(f"Saving animation to {filename}...")
        ani.save(filename, writer=writer)
        print(f"Animation saved to {filename}.")

        # Display the animation
        plt.show()

    # Separate the y data into liquid and solid concentrations
    liquid_profiles = [y_profile[:nx, :] for y_profile in y]
    solid_profiles = [y_profile[nx:, :] for y_profile in y]

    # Create animations for liquid and solid concentrations
    create_animation(liquid_profiles, t_sets, "Liquid Concentration", f"{title}_liquid.mp4", labels, colors)
    create_animation(solid_profiles, t_sets, "Solid Concentration", f"{title}_solid.mp4", labels, colors)






######################################## FUNCTION EXECUTIONS ########################

###################### PRIMARY INPUTS #########################
# What tpye of isoherm is required?
# Coupled: "CUP"
# Uncoupled: "UNC"
iso_type = "CUP" 
Names = ["glucose", "fructose"] #, "C"]#, "D", "E", "F"]
color = ["red", "green"] #, "b"]#, "r", "purple", "brown"]
num_comp = len(Names)




e = 0.5     # assuming shperical packing, voidage (0,1]
Q_S = 8.4*0.0166666667 # cm^3/s | The volumetric flowrate of the feed to the left of the feed port (pure solvent)
t_index = 70 # s # Index time # How long the SINGLE pulse holds for
slug_vol = 15 #cm^3
Q_inj = slug_vol/t_index # cm^3/s | The volumetric flowrate of the injected concentration slug

Ncol_num = 1
tend_min = 15.6 # min # How long the simulation is for
nx = 50
Bm = 300
###################### COLUMN DIMENTIONS ########################
L = 17.5 # cm
d_col = 2 # cm




# # Uncomment as necessary:
# # Linear 
# cusotom_isotherm_params_all = np.array([[3.2069715], [3.54]]) # H_glu, H_fru 
# # # Langmuir
# # cusotom_isotherm_params_all = [[3,3]]
# cusotom_isotherm_params_all = np.array([[2.51181596, 1.95381598], [2.55314612, 1.65186647]])

# # Linear + Langmuir
# # cusotom_isotherm_params_all = [[0.3, 1, 2]]

# Parameter sets for different components
# Units:
# - Concentrations: g/cm^3
# - kfp: 1/s

# kav_params_all = [[0.4, 0.4], [0.2, 0.5]] # [[A], [B]]
cusotom_isotherm_params_all = np.array([[3.21],[3.54]])
kav_params_all = [[0.467], [0.462]] # [[A], [B]]
parameter_sets = [
    {"C_feed": 0.42},    # Glucose SMB Launch
    {"C_feed": 0.42}] #, # Fructose


Da_all = np.array([3.218e-6, 8.38e-6 ]) 

column_func_inputs = [iso_type,  Names, color, parameter_sets, Da_all, Bm, e, Q_S, Q_inj, t_index, tend_min, nx, L, d_col, cusotom_isotherm_params_all, kav_params_all]
                    #   iso_type,  Names, color, parameter_sets, Da_all, Bm, e, Q_S, Q_inj, t_index, tend_min, nx, L, d_col, cusotom_isotherm_params_all

start = time.time()
col_elution, y_matrices, nx, t, t_sets, t_schedule, C_feed, m_in, m_out, Model_Acc, Expected_Acc, Error_percent = column_func(column_func_inputs) 
end = time.time()
print('---------------------------')
print(f'Computation Time: {end-start} s || {(end-start)/60} min')
print('---------------------------\n\n')

def col_elution_profile(t_vals, col_elution, num_comp, 
                        bor_curves=None, hcl_curves=None, t_data=None):
    """
    Plot elution profiles for simulated and experimental data.

    Parameters:
    - t_vals: array-like or list of arrays. Time values for the model curves.
    - col_elution: list of arrays. Modelled concentration profiles.
    - num_comp: int. Number of components (e.g., 2 for binary).
    - bor_curves, hcl_curves: optional, experimental concentration profiles.
    - t_data: optional, time array for experimental curves (shared between bor/hcl).
    """

    fig, ax = plt.subplots(1, 1, figsize=(15, 5))

    # Plot based on isotherm type
    if iso_type == 'UNC':
        for i in range(num_comp):
            ax.plot(t_vals[i]/60, col_elution[i], color=color[i], 
                    label=f"Model: {Names[i]}")

    elif iso_type == 'CUP':
        for i in range(num_comp):
            ax.plot(t_vals/60, col_elution[i], color=color[i], 
                    label=f"Model: {Names[i]}")
    
    # === Plot experimental curves if provided ===
    if bor_curves is not None and t_data is not None:
        ax.plot(t_data/60, bor_curves, 'k--', linewidth=2, label="Glucose Exp. Data")

    if hcl_curves is not None and t_data is not None:
        ax.plot(t_data/60, hcl_curves, 'gray', linestyle='dotted', linewidth=2, 
                label="Fructose Exp. Data")

    ax.set_xlabel('Time (min)')
    ax.set_ylabel('Concentration (g/mL)')
    ax.set_title(f"Single Column Elution Curves\n{Names}\n"
                 f"Da: {Da_all}, kfp: [{kav_params_all[0]}, {kav_params_all[1]}], "
                 f"Isotherm Params: {cusotom_isotherm_params_all}")
    ax.legend()
    ax.grid(True)
    plt.tight_layout()
    plt.show()

# Glucose and Fructose Curves:
t_exp = np.array([0.00, 0.60, 1.20, 1.80, 2.40, 3.00, 3.60, 4.20, 4.80, 5.40, 6.00, 6.60, 7.20, 7.80, 8.40, 9.00, 9.60, 10.20, 10.80, 11.40, 12.00, 12.60, 13.20, 13.80, 14.40, 15.00, 15.60 ])*60
glu_exp_data = np.array([0, 0,0, 0, 0, 0.0104382, 0.081351, 0.0963738, 0.1582416, 0.1539864, 0.1369116, 0.1092042, 0.0830898, 0.0517428, 0.0267354, 0.0107838, 0.004887, 0.0023166, 0, 0, 0, 0, 0, 0, 0, 0, 0])
fru_exp_data = np.array([0, 0, 0, 0, 0, 0, 0.004266, 0.03348, 0.141291, 0.1682154, 0.2087208, 0.1417824, 0.11043, 0.070011, 0.0423414, 0.0203148, 0.0065718, 0.0019008, 0.0006588, 0, 0, 0, 0, 0, 0, 0, 0])
col_elution_profile(t, col_elution, num_comp, 
                    bor_curves=glu_exp_data, 
                    hcl_curves=fru_exp_data, 
                    t_data=t_exp)