# -*- coding: utf-8 -*-
# # %%


#%%
import numpy as np
# -*- coding: utf-8 -*-
# # %%


#%%
import numpy as np
from scipy.optimize import minimize
from scipy.integrate import solve_ivp
from scipy import integrate
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C, Matern
from scipy.optimize import differential_evolution
from scipy.optimize import minimize, NonlinearConstraint
import json
from scipy.stats import norm
from scipy.integrate import solve_ivp
from scipy import integrate
import warnings
import time

# Import the custom SMB model
def SMB(SMB_inputs):
    iso_type, Names, color, num_comp, nx_per_col, e, D_all, Bm, zone_config, L, d_col, d_in, t_index_min, n_num_cycles, Q_internal, parameter_sets, cusotom_isotherm_params_all, kav_params_all = SMB_inputs[0:]

    ###################### (CALCUALTED) SECONDARY INPUTS #########################

    # Column Dimensions:
    ################################################################
    F = (1-e)/e     # Phase ratio
    t=0
    t_sets = 0
    Ncol_num = np.sum(zone_config) # Total number of columns
    L_total = L*Ncol_num # Total Lenght of all columns
    A_col = np.pi*0.25*d_col**2 # cm^2
    V_col = A_col * L # cm^3
    V_col_total = Ncol_num * V_col # cm^3
    A_in = np.pi * (d_in/2)**2 # cm^2
    alpha = A_in / A_col
    Z1 = zone_config[0]
    Z2 = zone_config[1]
    Z3 = zone_config[2]
    Z4 = zone_config[3]



    # Time Specs:
    ################################################################

    t_index = t_index_min*60 # s #

    # Notes:
    # - Cyclic Steady state typically happens only after 10 cycles (ref: https://doi.org/10.1205/026387603765444500)
    # - The system is not currently designed to account for periods of no external flow

    n_1_cycle = t_index * Ncol_num  # s How long a single cycle takes

    total_cycle_time = n_1_cycle*n_num_cycles # s

    tend = total_cycle_time # s # Final time point in ODE solver

    tend_min = tend/60

    t_span = (0, tend) # +dt)  # from t=0 to t=n

    num_of_injections = int(np.round(tend/t_index)) # number of switching periods

    # 't_start_inject_all' is a vecoter containing the times when port swithes occur for each port
    # Rows --> Different Ports
    # Cols --> Different time points
    t_start_inject_all = [[] for _ in range(Ncol_num)]  # One list for each node (including the main list)

    # Calculate start times for injections
    for k in range(num_of_injections):
        t_start_inject = k * t_index
        t_start_inject_all[0].append(t_start_inject)  # Main list
        for node in range(1, Ncol_num):
            t_start_inject_all[node].append(t_start_inject + node * 0)  # all rows in t_start_inject_all are identical

    t_schedule = t_start_inject_all[0]

    # REQUIRED FUNCTIONS:
    ################################################################

    # 1.
    # Func to Generate Indices for the columns
    # def generate_repeated_numbers(n, m):
    #     result = []
    #     n = int(n)
    #     m = int(m)
    #     for i in range(m):
    #         result.extend([i] * n)
    #     return result

    # 3.
    # Func to divide the column into nodes
    
    def cusotom_isotherm_func(cusotom_isotherm_params, c):
        """
        c => liquid concentration of ci
        q_star => solid concentration of ci @ equilibrium
        cusotom_isotherm_params = cusotom_isotherm_params_all[i] => given parameter set of component, i
        """

        # Uncomment as necessary

        #------------------- 1. Single Parameters Models
        ## Linear
        K1 = cusotom_isotherm_params[0]
        # print(f'H = {K1}')
        H = K1 # Henry's Constant
        q_star_1 = H*c

        # #------------------- 2. Two-Parameter Models
        # K1 = cusotom_isotherm_params[0]
        # K2 = cusotom_isotherm_params[1]

        # # #  Langmuir  
        # Q_max = K1
        # b = K2
        # #-------------------------------
        # q_star_2 = Q_max*b*c/(1 + b*c)
        # #-------------------------------

        #------------------- 3. Three-Parameter models 
        # K1 = cusotom_isotherm_params[0]
        # K2 = cusotom_isotherm_params[1]
        # K3 = cusotom_isotherm_params[2]

        # # Linear + Langmuir
        # H = K1
        # Q_max = K2
        # b = K3
        # ##-------------------------------
        # q_star_3 = H*c + Q_max*b*c/(1 + b*c)
        # ##-------------------------------

        return q_star_1 # [qA, ...]

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

    def cusotom_CUP_isotherm_func(cusotom_isotherm_params_all, c, IDX, comp_idx):
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
        cA = c[IDX[0]: IDX[0] + nx]
        cB = c[IDX[1]: IDX[1] + nx]
        c_i = [cA, cB]
        # Now, different isotherm Models can be built using c_i
        
        # (Uncomment as necessary)

        #------------------- 1. Linear Models

        # cusotom_isotherm_params_all has linear constants for each comp
        # # Unpack respective parameters
        K1 = cusotom_isotherm_params_all[comp_idx][0] # 1st (and only) parameter of HA or HB
        # print(f'H = {K1}')
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

    # DOES NOT INCLUDE THE C0 NODE (BY DEFAULT)
    def set_x(L, Ncol_num,nx_col,dx):
        if nx_col == None:
            x = np.arange(0, L+dx, dx)
            nnx = len(x)
            nnx_col = int(np.round(nnx/Ncol_num))
            nx_BC = Ncol_num - 1 # Number of Nodes (mixing points/boundary conditions) in between columns

            # Indecies belonging to the mixing points between columns are stored in 'start'
            # These can be thought of as the locations of the nx_BC nodes.
            return x, dx, nnx_col,  nnx, nx_BC

        elif dx == None:
            nx = Ncol_num * nx_col
            nx_BC = Ncol_num - 1 # Number of Nodes in between columns
            x = np.linspace(0,L_total,nx)
            ddx = x[1] - x[0]

            # Indecies belonging to the mixing points between columns are stored in 'start'
            # These can be thought of as the locations of the nx_BC nodes.

            return x, ddx, nx_col, nx, nx_BC

    # 4. A func that:
    # (i) Calcualtes the internal flowrates given the external OR (ii) Visa-versa
    def set_flowrate_values(set_Q_int, set_Q_ext, Q_rec):
        if set_Q_ext is None and Q_rec is None:  # Chosen to specify internal/zone flowrates
            Q_I = set_Q_int[0]
            Q_II = set_Q_int[1]
            Q_III = set_Q_int[2]
            Q_IV = set_Q_int[3]

            QX = -(Q_I - Q_II)
            QF = Q_III - Q_II
            QR = -(Q_III - Q_IV)
            QD = -(QF + QX + QR) # OR: Q_I - Q_IV

            Q_ext = np.array([QF, QR, QD, QX]) # cm^3/s

            return Q_ext

        elif set_Q_int is None and Q_rec is not None:  # Chosen to specify external flowrates
            QF = set_Q_ext[0]
            QR = set_Q_ext[1]
            QD = set_Q_ext[2]
            QX = set_Q_ext[3]

            Q_I = Q_rec  # m^3/s
            Q_III = (QX + QF) + Q_I
            Q_IV = (QD - QX) + Q_I  # Fixed Q_II to Q_I as the variable was not defined yet
            Q_II = (QR - QX) + Q_IV
            Q_internal = np.array([Q_I, Q_II, Q_III, Q_IV])
            return Q_internal


    # 5. Function to Build Port Schedules:

    # This is done in two functions: (i) repeat_array and (ii) build_matrix_from_vector
    # (i) repeat_array
    # Summary: Creates the schedule for the 1st port, port 0, only. This is the port boadering Z2 & Z3 and always starts as a Feed port at t=0
    # (i) build_matrix_from_vector
    # Summary: Takes the output from "repeat_array" and creates schedules for all other ports.
    # The "trick" is that the states of each of the, n, ports at t=0, is equal to the first, n, states of port 0.
    # Once we know the states for each port at t=0, we form a loop that adds the next state.

    # 5.1
    def position_maker(schedule_quantity_name, F, R, D, X, Z_config):

        """

        Function that initializes the starting schedueles for a given quantitiy at all positions

        F, R, D and X are the values of the quantiity at the respective feed ports

        """
        # Initialize:
        X_j = np.zeros(Ncol_num)


        # We set each port in the appropriate position, depending on the nuber of col b/n Zones:
        # By default, Position i = 0 (entrance to col,0) is reserved for the feed node.

        # Initialize Positions:
        # Q_position is a vector whose len is = number of mixing points (ports) b/n columns

        X_j[0] = F        # FEED
        X_j[Z_config[2]] = R     # RAFFINATE
        X_j[Z_config[2] + Z_config[3]] = D    # DESORBENT
        X_j[Z_config[2] + Z_config[3]+  Z_config[0]] = X   # EXTRACT

        return X_j

    # 5.2
    def repeat_array(vector, start_time_num):
        # vector = the states of all ports at t=0, vector[0] = is always the Feed port
        # start_time_num = The number of times the state changes == num of port switches == num_injections
        repeated_array = np.tile(vector, (start_time_num // len(vector) + 1))
        return repeated_array[:start_time_num]

    def initial_u_col(Zconfig, Qint):
        """
        Fun that returns the the inital state at t=0 of the volumetric
        flows in all the columns.

        """
        # First row is for col0, which is the feed to zone 3
        Zconfig_roll = np.roll(Zconfig, -2)
        Qint_roll = np.roll(Qint, -2)

        # print(Qint)
        X = np.array([])

        for i in range(len(Qint_roll)):
            X_add = np.ones(Zconfig_roll[i])*Qint_roll[i]
            # print('X_add:\n', X_add)

            X = np.append(X, X_add)
        # X = np.concatenate(X)
        # print('X:\n', X)
        return X


    def build_matrix_from_vector(vector, t_schedule):
        """
        Fun that returns the schedeule given the inital state at t=0
        vector: inital state of given quantity at t=0 at all nodes
        t_schedule: times at which port changes happen

        """
        # vector = the states of all ports at t=0, vector[0] = is always the Feed port
        start_time_num = int(len(t_schedule))
        vector = np.array(vector)  # Convert the vector to a NumPy array
        n = len(vector) # number of ports/columns

        # Initialize the matrix for repeated elements, ''ALL''
        ALL = np.zeros((n, start_time_num), dtype=vector.dtype)  # Shape is (n, start_time_num)

        for i in range(start_time_num):
            # print('i:',i)
            ALL[:, i] = np.roll(vector, i)
        return ALL



    # # Uncomment as necessary depending on specification of either:
    # # (1) Internal OR (2) External flowrates :
    # # (1)
    # Q_internal = np.array([Q_I, Q_II, Q_III, Q_IV])
    Q_external = set_flowrate_values(Q_internal, None, None) # Order: QF, QR, QD, QX
    QF, QR, QD, QX = Q_external[0], Q_external[1], Q_external[2], Q_external[3]
    # print('Q_external:', Q_external)

    # (2)
    # QX, QF, QR = -0.277, 0.315, -0.231  # cm^3/s
    # QD = - (QF + QX + QR)
    # Q_external = np.array([QF, QR, QD, QX])
    # Q_rec = 33.69 # cm^3/s
    # Q_internal = set_flowrate_values(None, Q_external, Q_rec) # Order: QF, QR, QD, Q

    ################################################################################################


    # Make concentration schedules for each component

    Cj_pulse_all = [[] for _ in range(num_comp)]
    for i in range(num_comp):
        Cj_position = []
        Cj_position = position_maker('Feed Conc Schedule:', parameter_sets[i]['C_feed'], 0, 0, 0, zone_config)
        Cj_pulse = build_matrix_from_vector(Cj_position,  t_schedule)
        Cj_pulse_all[i] = Cj_pulse


    Q_position = position_maker('Vol Flow Schedule:', Q_external[0], Q_external[1], Q_external[2], Q_external[3], zone_config)
    Q_pulse_all = build_matrix_from_vector(Q_position,  t_schedule)

    # Spacial Discretization:
    # Info:
    # nx --> Total Number of Nodes (EXCLUDING mixing points b/n nodes)
    # nx_col --> Number of nodes in 1 column
    # nx_BC --> Number of mixing points b/n nodes
    x, dx, nx_col, nx, nx_BC = set_x(L=L_total, Ncol_num = Ncol_num, nx_col = nx_per_col, dx = None)
    start = [i*nx_col for i in range(0,Ncol_num)] # Locations of the BC indecies
    u_col_at_t0 = initial_u_col(zone_config, Q_internal)
    Q_col_all = build_matrix_from_vector(u_col_at_t0, t_schedule)


    # DISPLAYING INPUT INFORMATION:
    # print('---------------------------------------------------')
    # print('Number of Components:', num_comp)
    # print('---------------------------------------------------')
    # print('\nTime Specs:\n')
    # print('---------------------------------------------------')
    # print('Number of Cycles:', n_num_cycles)
    # print('Time Per Cycle:', n_1_cycle/60, "min")
    # print('Simulation Time:', tend_min, 'min')
    # print('Index Time:', t_index, 's OR', t_index/60, 'min' )
    # print('Number of Port Switches:', num_of_injections)
    # print('Injections happen at t(s) = :', t_schedule, 'seconds')
    # print('---------------------------------------------------')
    # print('\nColumn Specs:\n')
    # print('---------------------------------------------------')
    # print('Configuration:', zone_config, '[Z1,Z2,Z3,Z4]')
    # print(f"Number of Columns: {Ncol_num}")
    # print('Column Length:', L, 'cm')
    # print('Column Diameter:', d_col, 'cm')
    # print('Column Volume:', V_col, 'cm^3')
    # print("alpha:", alpha, '(alpha = A_in / A_col)')
    # print("Nodes per Column:",nx_col)
    # print("Boundary Nodes locations,x[i], i =", start)
    # print("Total Number of Nodes (nx):",nx)
    # print('---------------------------------------------------')
    # print('\nFlowrate Specs:\n')
    # print('---------------------------------------------------')
    # print("External Flowrates =", Q_external, '[F,R,D,X] ml/min')
    # print("Ineternal Flowrates =", Q_internal, 'ml/min')
    # print('---------------------------------------------------')
    # print('\nPort Schedules:')
    # for i in range(num_comp):
    #     print(f"Concentration Schedule:\nShape:\n {Names[i]}:\n",np.shape(Cj_pulse_all[i]),'\n', Cj_pulse_all[i], "\n")
    # print("Injection Flowrate Schedule:\nShape:",np.shape(Q_pulse_all),'\n', Q_pulse_all, "\n")
    # print("Respective Column Flowrate Schedule:\nShape:",np.shape(Q_col_all),'\n', Q_col_all, "\n")


    ###########################################################################################

    ###########################################################################################

    
    # print('kav_params:', kav_params)
    # print('----------------------------------------------------------------')
    ###########################################################################################

    # # FORMING THE ODES


    # Form the remaining schedule matrices that are to be searched by the funcs

    # Column velocity schedule:
    u_superficial = -Q_col_all/A_col
    u_col_all = u_superficial/e # interstitial

    # Column Dispersion schedule:
    # Different matrices for each comp, because diff Pe's for each comp
    D_col_all = []
    for i in range(num_comp): # for each comp
        # D_col = -(u_col_all*L)/Pe_all[i] # constant dispersion coeff
        D_col = np.ones_like(u_col_all)*D_all[i]
        D_col_all.append(D_col)


    # print(f'Shape of u_col_all: {np.shape(u_col_all)}')
    # print(f'Shape of D_col_all: {np.shape(D_col_all)}')
    # print(f'u_col_all: {u_col_all}')
    # print(f'\nD_col_all: {D_col_all}')
    # Storage Spaces:

    coef_0 = np.zeros_like(u_col_all)
    coef_1 = np.zeros_like(u_col_all)
    coef_2 = np.zeros_like(u_col_all)

    # coef_0, coef_1, & coef_2 correspond to the coefficents of ci-1, ci & ci+1 respectively
    # These depend on u and so change with time, thus have a schedule

    # From descritization:
    coef_0_all = [[] for _ in range(num_comp)]
    coef_1_all = [[] for _ in range(num_comp)]
    coef_2_all = [[] for _ in range(num_comp)]

    coef_0_all = []
    coef_1_all = []
    coef_2_all = []

    for j in range(num_comp): # for each comp
        for i  in range(Ncol_num): # coefficients for each col
            coef_0[i,:] = ( D_col_all[j][i,:]/dx**2 ) - ( u_col_all[i,:]/dx ) # coefficeint of i-1
            coef_1[i,:] = ( u_col_all[i,:]/dx ) - (2*D_col_all[j][i,:]/(dx**2))# coefficeint of i
            coef_2[i,:] = (D_col_all[j][i,:]/(dx**2))    # coefficeint of i+1
        coef_0_all.append(coef_0)
        coef_1_all.append(coef_1)
        coef_2_all.append(coef_2)
    # print(f'coef_0_all: {np.shape(coef_0_all)}')
    # All shedules:
    # For each shceudle, rows => col idx, columns => Time idx
    # :
    # - Q_pulse_all: Injection flowrates
    # - C_pulse_all: Injection concentrations for each component
    # - Q_col_all:  Flowrates WITHIN each col
    # - u_col_all: Linear velocities WITHIN each col
    # - D_col_all: Dispersion coefficeints WITHIN each col
    # - coef_0, 1 and 2: ci, ci-1 & ci+1 ceofficients

    # print('coef_0:\n',coef_0)
    # print('coef_1:\n',coef_1)
    # print('coef_2:\n',coef_2)
    # print('\nD_col_all:\n',D_col_all)
    # print('Q_col_all:\n',Q_col_all)
    # print('A_col:\n',A_col)
    # print('u_col_all:\n',u_col_all)


    def coeff_matrix_builder_UNC(t, Q_col_all, Q_pulse_all, dx, start, alpha, c, nx_col, comp_idx): # note that c_length must include nx_BC

        # Define the functions that call the appropriate schedule matrices:
        # Because all scheudels are of the same from, only one function is required
        # Calling volumetric flows:
        get_X = lambda t, X_schedule, col_idx: next((X_schedule[col_idx][j] for j in range(len(X_schedule[col_idx])) if t_start_inject_all[col_idx][j] <= t < t_start_inject_all[col_idx][j] + t_index), 1/100000000)
        get_C = lambda t, C_schedule, col_idx, comp_idx: next((C_schedule[comp_idx][col_idx][j] for j in range(len(C_schedule[comp_idx][col_idx])) if t_start_inject_all[col_idx][j] <= t < t_start_inject_all[col_idx][j] + t_index), 1/100000000)

        # tHE MAIN DIFFERENCE BETWEEEN get_X and get_C is that get_C considers teh component level 

        def small_col_matix(nx_col, col_idx):
        # Initialize small_col_coeff ('small' = for 1 col)

            small_col_coeff = np.zeros((int(nx_col),int(nx_col))) #(5,5)

            # Where the 1st (0th) row and col are for c1
            # get_C(t, coef_0_all, k, comp_idx)
            # small_col_coeff[0,0], small_col_coeff[0,1] = get_X(t,coef_1,col_idx), get_X(t,coef_2,col_idx)
            small_col_coeff[0,0], small_col_coeff[0,1] = get_C(t,coef_1_all,col_idx, comp_idx), get_C(t,coef_2_all,col_idx, comp_idx)
            # for c2:
            # small_col_coeff[1,0], small_col_coeff[1,1], small_col_coeff[1,2] = get_X(t,coef_0,col_idx), get_X(t,coef_1,col_idx), get_X(t,coef_2,col_idx)
            small_col_coeff[1,0], small_col_coeff[1,1], small_col_coeff[1,2] = get_C(t,coef_0_all,col_idx, comp_idx), get_C(t, coef_1_all, col_idx, comp_idx), get_C(t, coef_2_all,col_idx, comp_idx)

            for i in range(2,nx_col): # from row i=2 onwards
                # np.roll the row entries from the previous row, for all the next rows
                new_row = np.roll(small_col_coeff[i-1,:],1)
                small_col_coeff[i:] = new_row

            small_col_coeff[-1,0] = 0
            small_col_coeff[-1,-1] = small_col_coeff[-1,-1] +  get_C(t,coef_2_all,col_idx, comp_idx) # coef_1 + coef_2 account for rolling boundary


            return small_col_coeff

        # Initialize:
        component_coeff_matrix = np.zeros((nx,nx)) # ''large'' = for all cols # (20, 20)

        # Add the cols
        for col_idx in range(Ncol_num):

            srt = col_idx*nx_col
            end = (col_idx+1)*nx_col

            component_coeff_matrix[srt:end, srt:end] = small_col_matix(nx_col,col_idx)
        # print('np.shape(larger_coeff_matrix)\n',np.shape(larger_coeff_matrix))

        # vector_add: vector that applies the boundary conditions to each boundary node
        def vector_add(nx, c, start, comp_idx):
            vec_add = np.zeros(nx)
            c_BC = np.zeros(Ncol_num)
            # Indeceis for the boundary nodes are stored in "start"
            # Each boundary node is affected by the form:
            # c_BC = V1 * C_IN - V2 * c[i] + V3 * c[i+1]

            # R1 = ((beta * alpha) / gamma)
            # R2 = ((2 * Da / (u * dx)) / gamma)
            # R3 = ((Da / (2 * u * dx)) / gamma)

            # Where:
            # C_IN is the weighted conc exiting the port facing the column entrance.
            # alpha , bata and gamma depend on the column vecolity and are thus time dependant
            # Instead of forming schedules for alpha , bata and gamma, we calculate them in-line

            for i in range(len(start)):
                #  start[i] => the node at the entrance to the ith col
                # So start[3] is the node representing the 1st node in col 3

                Q_1 = get_X(t, Q_col_all, i-1) # Vol_flow from previous column (which for column 0, is the last column in the chain)
                Q_2 = get_X(t, Q_pulse_all, i) # Vol_flow injected IN port i

                Q_out_port = get_X(t, Q_col_all, i) # Vol_flow OUT of port 0 (Also could have used Q_1 + Q_2)


                W1 = Q_1/Q_out_port # Weighted flowrate to column i
                W2 = Q_2/Q_out_port # Weighted flowrate to column i

                # Calcualte Weighted Concentration:

                c_injection = get_C(t, Cj_pulse_all, i, comp_idx)

                if Q_2 > 0: # Concentration in the next column is only affected for injection flows IN
                    C_IN = W1 * c[i*nx_col-1] + W2 * c_injection
                else:
                    # C_IN = c[i*nx_col-1] # no change in conc during product collection
                    C_IN = c[start[i]-1] # no change in conc during product collection

                # Calcualte alpha, bata and gamma:
                # Da = get_X(t, D_col_all, i)
                Da = get_C(t, D_col_all, i, comp_idx)
                u =  get_X(t, u_col_all, i)
                beta = 1 / alpha
                gamma = 1 - 3 * Da / (2 * u * dx)

                ##
                R1 = ((beta * alpha) / gamma)
                R2 = ((2 * Da / (u * dx)) / gamma)
                R3 = ((Da / (2 * u * dx)) / gamma)
                ##

                # Calcualte the BC effects:
                j = start[i]
                c_BC[i] = R1 * C_IN - R2 * c[j] + R3 * c[j+1] # the boundary concentration for that node

            # print('c_BC:\n', c_BC)

            for k in range(len(c_BC)):
                # vec_add[start[k]]  = get_X(t,coef_0,k)*c_BC[k]
                vec_add[start[k]]  = get_C(t, coef_0_all, k, comp_idx)*c_BC[k]

            return vec_add
            # print('np.shape(vect_add)\n',np.shape(vec_add(nx, c, start)))
        return component_coeff_matrix, vector_add(nx, c, start, comp_idx)
    
    def matrix_builder(M):
        """
        M => Set of matricies describing the dynamics in all columns of each comp [M_A, M_B]
        -------------------------------------
        This func takes M and adds it to M0.
        """
        # M = Matrix to add (small)

        n = len(M) # number of components
        rx = len(M[0][0,:])  # all members of M are square and equal in size, we just want the col num
        nn = int(n * rx)
        # print(f'nn: {nn}')
        
        M0 = np.zeros((nn, nn))
        # M0 = Initial state of the matrix to be added to


        positon_1 = 0
        positon_2 = rx
        positon_3 = 2*rx

        M0[positon_1:positon_2, positon_1:positon_2] = M[0]

        M0[positon_2:positon_3, positon_2:positon_3] = M[1]

        return M0

    def coeff_matrix_builder_CUP(t, Q_col_all, Q_pulse_all, dx, start_CUP, alpha, c, nx_col, IDX): # note that c_length must include nx_BC

        # Define the functions that call the appropriate schedule matrices:

        # Because all scheudels are of the same from, only one function is required
        # Calling volumetric flows:
        get_X = lambda t, X_schedule, col_idx: next((X_schedule[col_idx][j] for j in range(len(X_schedule[col_idx])) if t_start_inject_all[col_idx][j] <= t < t_start_inject_all[col_idx][j] + t_index), 1/100000000)
        get_C = lambda t, C_schedule, col_idx, comp_idx: next((C_schedule[comp_idx][col_idx][j] for j in range(len(C_schedule[comp_idx][col_idx])) if t_start_inject_all[col_idx][j] <= t < t_start_inject_all[col_idx][j] + t_index), 1/100000000)


        # 1. From coefficent "small" matrix for movement of single comp through single col
        # 2. Form  "large" coefficent matrix for movement through one all cols
        # 3. The large  coefficent matrix for each comp will then be combined into Final Matrix

        # 1.
        def small_col_matrix(nx_col, col_idx, comp_idx):

        # Initialize small_col_coeff ('small' = for 1 col)
            small_col_coeff = np.zeros((int(nx_col),int(nx_col))) # (5,5)

            # Where the 1st (0th) row and col are for c1
            # get_C(t, coef_0_all, k, comp_idx)
            # small_col_coeff[0,0], small_col_coeff[0,1] = get_X(t,coef_1,col_idx), get_X(t,coef_2,col_idx)
            small_col_coeff[0,0], small_col_coeff[0,1] = get_C(t,coef_1_all, col_idx, comp_idx), get_C(t,coef_2_all,col_idx, comp_idx)
            # for c2:
            # small_col_coeff[1,0], small_col_coeff[1,1], small_col_coeff[1,2] = get_X(t,coef_0,col_idx), get_X(t,coef_1,col_idx), get_X(t,coef_2,col_idx)
            small_col_coeff[1,0], small_col_coeff[1,1], small_col_coeff[1,2] = get_C(t,coef_0_all,col_idx, comp_idx), get_C(t, coef_1_all, col_idx, comp_idx), get_C(t, coef_2_all,col_idx, comp_idx)

            for i in range(2,nx_col): # from (3rd) row i=2 onwards
                # np.roll the row entries from the previous row, for all the next rows
                new_row = np.roll(small_col_coeff[i-1,:],1)
                small_col_coeff[i:] = new_row

            small_col_coeff[-1,0] = 0
            small_col_coeff[-1,-1] = small_col_coeff[-1,-1] +  get_C(t,coef_2_all,col_idx, comp_idx) # coef_1 + coef_2 account for rolling boundary
            
            return small_col_coeff

        # 2. Func to Build Large Matrix




            # vector_add: vector that applies the boundary conditions to each boundary node
        def vector_add(nx, c, start):
            vec_add = np.zeros(nx*num_comp)
            c_BC = np.zeros(nx*num_comp)

            # Indeceis for the boundary nodes are stored in "start"
            # Each boundary node is affected by the form:
            # c_BC = V1 * C_IN - V2 * c[i] + V3 * c[i+1]

            # R1 = ((beta * alpha) / gamma)
            # R2 = ((2 * Da / (u * dx)) / gamma)
            # R3 = ((Da / (2 * u * dx)) / gamma)

            # Where:
            # C_IN is the weighted conc exiting the port facing the column entrance.
            # alpha , bata and gamma depend on the column vecolity and are thus time dependant
            # Instead of forming schedules for alpha , bata and gamma, we calculate them in-line

            for i in range(len(start)):
                #k = i%len(start) # Recounts columns for B
                Q_1 = get_X(t, Q_col_all, i-1) # Vol_flow from previous column (which for column 0, is the last column in the chain)
                Q_2 = get_X(t, Q_pulse_all, i) # Vol_flow injected IN port i

                Q_out_port = get_X(t, Q_col_all, i) # Vol_flow OUT of port 0 (Also could have used Q_1 + Q_2)


                W1 = Q_1/Q_out_port # Weighted flowrate to column i
                W2 = Q_2/Q_out_port # Weighted flowrate to column i

                # Calcualte Weighted Concentration:
                # Identifiers:
                A = IDX[0]
                B = IDX[1]

                # C_IN_A = W1 * c[A + i*nx_col-1] + W2 * get_X(t, C_pulse_all_A, i) # c[-1] conc out the last col
                # C_IN_B = W1 * c[B + i*nx_col-1] + W2 * get_X(t, C_pulse_all_B, i) # c[-1] conc out the last col

                C_IN_A = W1 * c[A + i*nx_col-1] + W2 * get_X(t, Cj_pulse_all[0], i) # c[-1] conc out the last col
                C_IN_B = W1 * c[B + i*nx_col-1] + W2 * get_X(t, Cj_pulse_all[1], i) # c[-1] conc out the last col


                # All componenets expirence same liquid veclocity...
                u =  get_X(t, u_col_all, i)
                # print(f"u: {np.shape(u)}")
                # print(f'u: {u}')
                
                # All components have respective dispersions....
                # Calcualte alpha, bata and gamma:
                Da_A = get_C(t, D_col_all, i, comp_idx=0)
                beta_A = 1 / alpha
                gamma_A = 1 - 3 * Da_A / (2 * u * dx)
                ##
                R1_A = ((beta_A * alpha) / gamma_A)
                R2_A = ((2 * Da_A / (u * dx)) / gamma_A)
                R3_A = ((Da_A / (2 * u * dx)) / gamma_A)
                # print(f"R1: {np.shape(R1)}")
                # print(f"R3: {np.shape(R3)}")

                # Calcualte alpha, bata and gamma:
                Da_B = get_C(t, D_col_all, i, comp_idx=1)
                beta_B = 1 / alpha
                gamma_B = 1 - 3 * Da_B / (2 * u * dx)
                ##
                R1_B = ((beta_B * alpha) / gamma_B)
                R2_B = ((2 * Da_B / (u * dx)) / gamma_B)
                R3_B = ((Da_B / (2 * u * dx)) / gamma_B)
                # print(f"R1: {np.shape(R1)}")
                # print(f"R3: {np.shape(R3)}")

                ##

                # Calcualte the BC effects:
                j = start[i]
                # -------------------------
                c_BC[i] = R1_A * C_IN_A - R2_A * c[j] + R3_A * c[j+1] # the boundary concentration for that node
                c_BC[B + i] = R1_B * C_IN_B - R2_B * c[B+j] + R3_B * c[B+j+1]
            # print('c_BC:\n', c_BC)
            # print('c_BC.shape:\n', c_BC.shape)

            for k in range(len(start)):
                vec_add[start[k]]  = get_X(t,coef_0_all[0],k) * c_BC[k]
                vec_add[B + start[k]]  = get_X(t,coef_0_all[1],k) * c_BC[B+ k]

            return vec_add
        
        # --------- Setting up the Variables

        # 3. Generate and Store the Large Matrices
        # Storage Space:
        # NOTE: Assuming all components have the same Dispersion coefficients,
        # all components will have the same large_col_matrix
        # Add the cols
        # Inital final matrix:
        #  nx -> all the nodes in the system
        component_coeff_matrix_A = np.zeros((nx,nx)) 
        component_coeff_matrix_B = np.zeros((nx,nx))
        

        for col_idx in range(Ncol_num): # for each column
            srt = col_idx*nx_col
            end = (col_idx+1)*nx_col

            component_coeff_matrix_A[srt:end, srt:end] = small_col_matrix(nx_col,col_idx, comp_idx=0)
            component_coeff_matrix_B[srt:end, srt:end] = small_col_matrix(nx_col,col_idx, comp_idx=1)

        component_coeff_matrix_all = [component_coeff_matrix_A, component_coeff_matrix_B]
        # print('np.shape(larger_coeff_matrix)\n',np.shape(larger_coeff_matrix))




        # ---------------------- Evaluate
        final_matrix = matrix_builder(component_coeff_matrix_all)
        final_vector = vector_add(nx, c, start_CUP)

        return final_matrix, final_vector

    # ###########################################################################################

    # # mod1: UNCOUPLED ISOTHERM:
    # # Profiles for each component can be solved independently

    # ###########################################################################################
    def mod1(t, v, comp_idx, Q_pulse_all):
        # call.append("call")
        # print(len(call))
        c = v[:nx]
        q = v[nx:]

        # Initialize the derivatives
        dc_dt = np.zeros_like(c)
        dq_dt = np.zeros_like(q)
        # print('v size\n',np.shape(v))

        # Isotherm:
        #########################################################################
        isotherm = cusotom_isotherm_func(cusotom_isotherm_params_all[comp_idx],c)
        # isotherm = iso_lin(theta_lin[comp_idx], c)
        #isotherm = iso_langmuir(theta_lang[comp_idx], c, comp_idx)
        #isotherm = iso_freundlich(theta_fre, c)


        # Mass Transfer:
        #########################################################################
        # print('isotherm size\n',np.shape(isotherm))
        MT = mass_transfer(kav_params_all[comp_idx], isotherm, q)
        #print('MT:\n', MT)

        coeff_matrix, vec_add = coeff_matrix_builder_UNC(t, Q_col_all, Q_pulse_all, dx, start, alpha, c, nx_col, comp_idx)
        # print('coeff_matrix:\n',coeff_matrix)
        # print('vec_add:\n',vec_add)
        dc_dt = coeff_matrix @ c + vec_add - F * MT
        dq_dt = MT

        return np.concatenate([dc_dt, dq_dt])

    # ##################################################################################

    def mod2(t, v):

        # where, v = [c, q]
        c = v[:num_comp*nx] # c = [cA, cB] | cA = c[:nx], cB = c[nx:]
        q = v[num_comp*nx:] # q = [qA, qB]| qA = q[:nx], qB = q[nx:]

        # Craate Lables so that we know the component assignement in the c vecotor:
        A, B = 0*nx, 1*nx # Assume Binary 2*nx, 3*nx, 4*nx, 5*nx
        IDX = [A, B]

        # Thus to refer to the liquid concentration of the i = nth row of component B: c[C + n]
        # Or the the solid concentration 10th row of component B: q[B + 10]
        # Or to refer to all A's OR B's liquid concentrations: c[A + 0: A + nx] OR c[B + 0: B + nx]


        # Initialize the derivatives
        dc_dt = np.zeros_like(c)
        dq_dt = np.zeros_like(q)


        # coeff_matrix, vec_add = coeff_matrix_builder_CUP(t, Q_col_all, Q_pulse_all, dx, start, alpha, c, nx_col, IDX)
        # print('coeff_matrix:\n',coeff_matrix)
        # print('vec_add:\n',vec_add)
        coeff_matrix_A, vec_add_A = coeff_matrix_builder_UNC(t, Q_col_all, Q_pulse_all, dx, start, alpha, c[0:nx], nx_col, 0)
        coeff_matrix_B, vec_add_B = coeff_matrix_builder_UNC(t, Q_col_all, Q_pulse_all, dx, start, alpha, c[nx:2*nx], nx_col, 1)

        coeff_matrix  = matrix_builder([coeff_matrix_A, coeff_matrix_B])
        vec_add = np.concatenate([vec_add_A, vec_add_B])

        ####################### Building MT Terms ####################################################################

        # Initialize

        MT = np.zeros(len(c)) # column vector: MT kinetcis for each comp: MT = [MT_A MT_B]

        for comp_idx in range(num_comp): # for each component
            
            

            ######################(ii) Isotherm ####################################################################

            # Comment as necessary for required isotherm:
            # isotherm = iso_bi_langmuir(theta_blang[comp_idx], c, IDX, comp_idx)
            # isotherm = iso_cup_langmuir(theta_cup_lang, c, IDX, comp_idx)
            isotherm = cusotom_CUP_isotherm_func(cusotom_isotherm_params_all, c, IDX, comp_idx)
            # print('qstar:\n', isotherm.shape)
            ################### (ii) MT ##########################################################
            MT_comp = mass_transfer(kav_params_all[comp_idx], isotherm, q[IDX[comp_idx]: IDX[comp_idx] + nx ])
            MT[IDX[comp_idx]: IDX[comp_idx] + nx ] = MT_comp

            # [MT_A, MT_B, . . . ] KINETICS FOR EACH COMP

        dc_dt = coeff_matrix @ c + vec_add - F * MT
        dq_dt = MT

        return np.concatenate([dc_dt, dq_dt])

    # ##################################################################################

    # SOLVING THE ODES
    # creat storage spaces:
    y_matrices = []

    t_sets = []
    t_lengths = []

    c_IN_values_all = []
    F_in_values_all = []
    call = []

    # print('----------------------------------------------------------------')
    # print("\n\nSolving the ODEs. . . .")



    if iso_type == "UNC": # UNCOUPLED - solve 1 comp at a time
        for comp_idx in range(num_comp): # for each component
            print(f'Solving comp {comp_idx}. . . .')
            v0 = np.zeros(Ncol_num* (nx_col + nx_col)) #  for both c and q
            solution = solve_ivp(mod1, t_span, v0, args=(comp_idx , Q_pulse_all))
            y_solution, t = solution.y, solution.t
            y_matrices.append(y_solution)
            t_sets.append(t)
            t_lengths.append(len(t))
            # print(f'y_matrices[{i}]', y_matrices[i].shape)


    # Assuming only a binary coupled system
    if iso_type == "CUP": # COUPLED - solve
            # nx = nx_col*num_comp
            v0 = np.zeros(num_comp*(nx)*2) # for c and 2, for each comp
            solution = solve_ivp(mod2, t_span, v0)
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
            # print('len(t_sets) = ', len(t_sets[0]))
            # print('len(t) = ', len(t))

    # print('----------------------------------------------------------------')
    # print('\nSolution Size:')
    # for i in range(num_comp):
    #     print(f'y_matrices[{i}]', y_matrices[i].shape)
    # print('----------------------------------------------------------------')
    # print('----------------------------------------------------------------')




    # ###########################################################################################

    # VISUALIZATION

    ###########################################################################################




    # MASS BALANCE AND PURITY CURVES
    ###########################################################################################

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

    # Fucntion to find the values of scheduled quantities
    # at all t_ode_times points

    def get_all_values(X, t_ode_times, t_schedule_times, Name):

        """
        X -> Matrix of Quantity at each schedule time. e.g:
        At t_schedule_times = [0,5,10] seconds feed:
        a concentraction of, X = [1,2,3] g/m^3

        """
        # Get index times
        t_idx = find_indices(t_ode_times, t_schedule_times)
        # print('t_idx:\n', t_idx)

        # Initialize:
        nrows = np.shape(X)[0]
        # print('nrows', nrows)

        values = np.zeros((nrows, len(t_ode_times))) # same num of rows, we just extend the times
        # print('np.shape(values):\n',np.shape(values))

        # Modify:
        k = 0

        for i in range(len(t_idx)-1): # during each schedule interval
            j = i%nrows

            # # k is a counter that pushes the row index to the RHS every time it loops back up
            # if j == 0 and i == 0:
            #     pass
            # elif j == 0:
            #     k += 1

            # print('j',j)

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


    # Function that adds row slices from a matrix M into one vector
    def get_X_row(M, row_start, jump, width):

        """
        M  => Matrix whos rows are to be searched and sliced
        row_start => Starting row - the row that the 1st slice comes from
        jump => How far the row index jumps to caputre the next slice
        width => the widths of each slice e.g. slice 1 is M[row, width[0]:width[1]]

        """
        # Quick look at the inpiuts
        # print('M.shape:\n', M.shape)
        # print('width:', width)

        # Initialize
        values = []
        nrows = M.shape[0]

        for i in range(len(width)-1):
            j = i%nrows
            # print('i', i)
            # print('j', j)
            t_start = int(width[i])
            tend = int(width[i+1])

            kk = (row_start+j*jump)%nrows

            MM = M[kk, t_start:tend]

            values.extend(MM)

        return values



    #  MASS INTO SYSMEM

    # - Only the feed port allows material to FLOW IN
    ###########################################################################################

    # Convert the Feed concentration schedule to show feed conc for all time
    # Do this for each component
    # C_feed_all = [[] for _ in range(num_comp)]

    row_start = 0 # iniital feed port row in schedule matrix

    row_start_matrix_raff = nx_col*Z3
    row_start_matrix_ext = (nx_col*(Z3 + Z4 + Z1))

    row_start_schedule_raff = row_start+Z3
    row_start_schedule_ext = row_start+Z3+Z4+Z1

    jump_schedule = 1
    jump_matrix = nx_col


    def feed_profile(t_odes, Cj_pulse_all, t_schedule, row_start, jump):

        """"
        Function that returns :
        (i) The total mass fed of each component
        (ii) Vector of feed conc profiles of each component
        """

        # Storage Locations:
        C_feed_all = []
        t_idx_all = []
        m_feed = []

        C_feed = [[] for _ in range(num_comp)]

        for i in range(num_comp):

            if iso_type == 'UNC':

                C, t_idx = get_all_values(Cj_pulse_all[i], t_odes[i], t_schedule, 'Concentration')
                t_idx_all.append(t_idx)

            elif iso_type == 'CUP':
                C, t_idx_all = get_all_values(Cj_pulse_all[i], t_odes, t_schedule, 'Concentration')

            C_feed_all.append(C)

            # print('t_idx_all:\n', t_idx_all )

        for i in range(num_comp):
            if iso_type == 'UNC':
                C_feed[i] = get_X_row( C_feed_all[i], row_start, jump, t_idx_all[i]) # g/cm^3
            elif iso_type == 'CUP':
                C_feed[i] = get_X_row( C_feed_all[i], row_start, jump, t_idx_all) # g/cm^3
        # print('C_feed[0]:',C_feed[0])

        for i in range(num_comp):
            F_feed = np.array([C_feed[i]]) * QF # (g/cm^3 * cm^3/s)  =>  g/s | mass flow into col (for comp, i)
            F_feed = np.array([F_feed]) # g/s

            if iso_type == 'UNC':
                m_feed_add = integrate.simpson(F_feed, x=t_odes[i]) # g
            if iso_type == 'CUP':
                m_feed_add = integrate.simpson(F_feed, x=t_odes) # g

            m_feed.append(m_feed_add)

        m_feed = np.concatenate(m_feed) # g
        # print(f'm_feed: {m_feed} g')

        return C_feed, m_feed, t_idx_all

    if iso_type == 'UNC':
        C_feed, m_feed, t_idx_all = feed_profile(t_sets, Cj_pulse_all, t_schedule, row_start, jump_schedule)
    elif iso_type == 'CUP':
        C_feed, m_feed, t_idx_all = feed_profile(t, Cj_pulse_all, t_schedule, row_start, jump_schedule)





    def prod_profile(t_odes, y_odes, t_schedule, row_start_matrix, jump_matrix, t_idx_all, row_start_schedule):

        """"
        Function that can be used to return:

        (i) The total mass exited at the Raffinate or Extract ports of each component
        (ii) Vector of Raffinate or Extract mass flow profiles of each component
        (iii) Vector of Raffinate or Extract vol flow profiles of each component

        P = Product either raff or ext
        """
        ######## Storages for the Raffinate #########
        C_P1 = []
        C_P2 = []

        Q_all_flows = [] # Flowrates expirenced by each component
        m_out_P = np.zeros(num_comp)

        P_vflows_1 = []
        P_mflows_1 = []
        m_P_1 = []

        P_vflows_2 = []
        P_mflows_2 = []
        m_P_2 = []
        t_idx_all_Q = []

        P_mprofile = []
        P_cprofile = []
        P_vflow = [[] for _ in range(num_comp)]


        # First get the values of the volumetric flowrates in each column @ all times from 0 to t_ode[-1]
        # Because UNC has different time intervals for each component, we need to get the time vecoters for each component
        if iso_type == 'UNC':
            for i in range(num_comp): # for each component
                Q_all_flows_add, b = get_all_values(Q_col_all, t_odes[i], t_schedule, 'Column Flowrates')
                # print('Q_all_flows_add:\n', Q_all_flows_add)
                Q_all_flows.append(Q_all_flows_add) # cm^3/s
                t_idx_all_Q.append(b)



        elif iso_type == 'CUP':
            Q_all_flows, t_idx_all_Q = get_all_values(Q_col_all, t_odes, t_schedule, 'Column Flowrates')
            # print('Q_all_flows:\n', Q_all_flows)
            # print('Q_all_flows:\n', np.shape(Q_all_flows))
            # print(f't_idx_all_Q: {np.shape(t_idx_all_Q)}')


        for i in range(num_comp):# for each component

            if iso_type == 'UNC':
                # Search the ODE matrix
                C_R1_add = np.array(get_X_row( y_odes[i][:nx,:], row_start_matrix-1, jump_matrix, t_idx_all_Q[i])) # exclude q
                C_R2_add = np.array(get_X_row( y_odes[i][:nx,:], row_start_matrix, jump_matrix, t_idx_all_Q[i]))
                # Search the Flowrate Schedule
                P_vflows_1_add = np.array(get_X_row(Q_all_flows[i], row_start_schedule-1, jump_schedule, t_idx_all_Q[i]))
                P_vflows_2_add = np.array(get_X_row(Q_all_flows[i], row_start_schedule, jump_schedule, t_idx_all_Q[i]))

            elif iso_type == 'CUP':
                C_R1_add = np.array(get_X_row( y_odes[i][:nx,:], row_start_matrix-1, jump_matrix, t_idx_all)) # exclude q
                C_R2_add = np.array(get_X_row( y_odes[i][:nx,:], row_start_matrix, jump_matrix, t_idx_all))
                P_vflows_1_add = np.array(get_X_row(Q_all_flows, row_start_schedule-1, jump_schedule, t_idx_all_Q))
                P_vflows_2_add = np.array(get_X_row(Q_all_flows, row_start_schedule, jump_schedule, t_idx_all_Q))


            # Raffinate Massflow Curves
            # print('C_R1_add.type():\n',type(C_R1_add))
            # print('np.shape(C_R1_add):\n', np.shape(C_R1_add))

            # print('P_vflows_1_add.type():\n',type(P_vflows_1_add))
            # print('np.shape(P_vflows_1_add):\n', np.shape(P_vflows_1_add))

            # Assuming only conc change accross port when (i) adding feed or (ii) desorbent
            C_R2_add = C_R1_add
            # P_mflows_1_add = C_R1_add * P_vflows_1_add  # (g/cm^3 * cm^3/s)  =>  g/s
            # P_mflows_2_add = C_R2_add * P_vflows_2_add  # g/s

            if row_start_matrix == row_start_matrix_raff:
                P_vflows_1_add = -QR*np.ones_like(C_R1_add)
                P_mflows_1_add = C_R1_add * P_vflows_1_add  # (g/cm^3 * cm^3/s)  =>  g/s

            elif row_start_matrix == row_start_matrix_ext:
                P_vflows_1_add = -QX*np.ones_like(C_R1_add)
                P_mflows_1_add = C_R1_add * P_vflows_1_add  # (g/cm^3 * cm^3/s)  =>  g/s




            # Flow profiles:
            # Concentration
            P_cprofile.append(C_R1_add) # g/s
            # Mass g/s
            P_mprofile.append(P_mflows_1_add ) #- P_mflows_2_add) # g/s
            # Volumetric cm^3/s
            P_vflow[i] = P_vflows_1_add #- P_vflows_2_add # cm^3

            # Integrate
            if iso_type == 'UNC':
                m_P_add_1 = integrate.simpson(P_mflows_1_add, x=t_odes[i]) # g
                # m_P_add_2 = integrate.simpson(P_mflows_2_add, x=t_odes[i]) # g

            if iso_type == 'CUP':
                m_P_add_1 = integrate.simpson(P_mflows_1_add, x=t_odes) # g
                # m_P_add_2 = integrate.simpson(P_mflows_2_add, x=t_odes) # g



            # Storage
            C_P1.append(C_R1_add)  # Concentration Profiles
            C_P2.append(C_R2_add)

            P_vflows_1.append(P_vflows_1_add)
            P_vflows_2.append(P_vflows_2_add)

            P_mflows_1.append(P_mflows_1_add)
            # P_mflows_2.append(P_mflows_2_add)

            m_P_1.append(m_P_add_1) # masses of each component
            # m_P_2.append(m_P_add_2) # masses of each component

        # Final Mass Exited
        # Mass out from P and ext
        for i in range(num_comp):
            m_out_P_add = m_P_1[i] #- m_P_2[i]
            # print(f'i:{i}')
            # print(f'm_out_P_add = m_P_1[i] - m_P_2[i]: { m_P_1[i]} - {m_P_2[i]}')
            m_out_P[i] = m_out_P_add # [A, B] g

        return P_cprofile, P_mprofile, m_out_P, P_vflow



    # Evaluating the product flowrates
    #######################################################
    # raff_mprofile, m_out_raff, raff_vflow = prod_profile(t_sets, y_matrices, t_schedule, row_start_R1, row_start_R2, jump_matrix, t_idx_all, row_start+Z3)
    # ext_mprofile, m_out_ext, ext_vflow = prod_profile(t_sets, y_matrices, t_schedule, row_start_X1, row_start_X2, jump_matrix, t_idx_all, row_start+Z3+Z4+Z1)
    if iso_type == 'UNC':
        raff_cprofile, raff_mprofile, m_out_raff, raff_vflow = prod_profile(t_sets, y_matrices, t_schedule, row_start_matrix_raff, jump_matrix, t_idx_all, row_start_schedule_raff)
        ext_cprofile, ext_mprofile, m_out_ext, ext_vflow = prod_profile(t_sets, y_matrices, t_schedule, row_start_matrix_ext, jump_matrix, t_idx_all, row_start_schedule_ext)
    elif iso_type == 'CUP':
        raff_cprofile, raff_mprofile, m_out_raff, raff_vflow = prod_profile(t, y_matrices, t_schedule, row_start_matrix_raff, jump_matrix, t_idx_all, row_start_schedule_raff)
        ext_cprofile, ext_mprofile, m_out_ext, ext_vflow = prod_profile(t, y_matrices, t_schedule, row_start_matrix_ext, jump_matrix, t_idx_all, row_start_schedule_ext)
    #######################################################
    # print(f'raff_vflow: {raff_vflow}')
    # print(f'np.shape(raff_vflow): {np.shape(raff_vflow[0])}')
    # print(f'ext_vflow: {ext_vflow}')
    # print(f'np.shape(ext_vflow): {np.shape(ext_vflow[0])}')









    # MASS BALANCE:
    #######################################################

    # Error = Expected Accumulation - Model Accumulation

    #######################################################

    # Expected Accumulation = Mass In - Mass Out
    # Model Accumulation = Integral in all col at tend (how much is left in col at end of sim)


    # Calculate Expected Accumulation
    #######################################################
    m_out = np.array([m_out_raff]) + np.array([m_out_ext]) # g
    m_out = np.concatenate(m_out)
    m_in = np.concatenate(m_feed) # g
    # ------------------------------------------
    Expected_Acc = m_in - m_out # g
    # ------------------------------------------


    # Calculate Model Accumulation
    #######################################################
    def model_acc(y_ode, V_col_total, e, num_comp):
        """
        Func to integrate the concentration profiles at tend and estimate the amount
        of solute left on the solid and liquid phases
        """
        mass_l = np.zeros(num_comp)
        mass_r = np.zeros(num_comp)

        for i in range(num_comp): # for each component

            V_l = e * V_col_total # Liquid Volume cm^3
            V_r = (1-e)* V_col_total # resin Volume cm^3

            # conc => g/cm^3
            # V => cm^3
            # integrate to get => g

            # # METHOD 1:
            # V_l = np.linspace(0,V_l,nx) # cm^3
            # V_r = np.linspace(0,V_r,nx) # cm^3
            # mass_l[i] = integrate.simpson(y_ode[i][:nx,-1], x=x)*A_col*e # mass in liq at t=tend
            # mass_r[i] = integrate.simpson(y_ode[i][nx:,-1], x=x)*A_col*(1-e) # mass in resin at t=tend

            # METHOD 2:
            V_l = np.linspace(0,V_l,nx) # cm^3
            V_r = np.linspace(0,V_r,nx) # cm^3

            mass_l[i] = integrate.simpson(y_ode[i][:nx,-1], x=V_l) # mass in liq at t=tend
            mass_r[i] = integrate.simpson(y_ode[i][nx:,-1], x=V_r) # mass in resin at t=tend

            # METHOD 3:
            # c_avg[i] = np.average(y_ode[i][:nx,-1]) # Average conc at t=tend
            # q_avg[i] = np.average(y_ode[i][:nx,-1])

            # mass_l = c_avg * V_l
            # mass_r = q_avg * V_r


        Model_Acc = mass_l + mass_r # g

        return Model_Acc

    Model_Acc = model_acc(y_matrices, V_col_total, e, num_comp)

    # ------------------------------------------
    Error = Model_Acc - Expected_Acc

    Error_percent = (sum(Error)/sum(Expected_Acc))*100
    # ------------------------------------------

    # Calculate KEY PERORMANCE PARAMETERS:
    #######################################################
    # 1. Purity
    # 2. Recovery
    # 3. Productivity


    # 1. Purity
    #######################################################
    # 1.1 Instantanoues:
    # raff_in_purity = raff_mprofile/sum(raff_mprofile)
    # ext_insant_purity = ext_mprofile/sum(ext_mprofile)

    # 1.2 Integral:
    raff_intgral_purity = m_out_raff/sum(m_out_raff)
    ext_intgral_purity = m_out_ext/sum(m_out_ext)

    # Final Attained Purity in the Stream
    raff_stream_final_purity = np.zeros(num_comp)
    ext_stream_final_purity = np.zeros(num_comp)

    for i in range(num_comp):
        raff_stream_final_purity[i] = raff_cprofile[i][-1]
        ext_stream_final_purity[i] = ext_cprofile[i][-1]



    # 2. Recovery
    #######################################################
    # 2.1 Instantanoues:

    # raff_in_recovery = raff_mprofile/sum(C_feed*QF)
    # ext_insant_recovery = ext_mprofile/sum(C_feed*QF)

    # 2.2 Integral:
    raff_recov = m_out_raff/m_in
    ext_recov = m_out_ext/m_in

    # 3. Productivity
    #######################################################




    # Visuliization of PERORMANCE PARAMETERS:
    #######################################################

    ############## TABLES ##################



    # Define the data for the table
    data = {
        'Metric': [
            'Total Mass IN',
            'Total Mass OUT',
            'Total Expected Acc (IN-OUT)',
            'Total Model Acc (r+l)',
            'Total Error (Mod-Exp)',
            'Total Error Percent (relative to Exp_Acc)',
            '',
            'Raffinate Purity [A, B,. . ]',
            'Extract Purity [A, B,. . ]',
            # 'Final Raffinate Dimensionless Stream Concentration  [A, B,. . ]',
            # 'Final Extract Dimensionless Stream Concentration  [A, B,. . ]',
            'Raffinate Recovery[A, B,. . ]',
            'Extract Recovery[A, B,. . ]'
        ],
        'Value': [
            f"{m_in} g",
            f"{m_out} g",
            f'{sum(Expected_Acc)} g',
            f'{sum(Model_Acc)} g',
            f'{sum(Error)} g',
            f'{Error_percent} %\n',
            '',
            f'{raff_intgral_purity} %',
            f'{ext_intgral_purity} %',
            # f'{raff_stream_final_purity} g/cm^3',
            # f'{ext_stream_final_purity}',
            f'{raff_recov} %',
            f'{ext_recov} %'
        ]
    }

    # # Create a DataFrame
    # import pandas as pd
    # df = pd.DataFrame(data)

    # # Display the DataFrame
    # print(df)

    return y_matrices, nx, t, t_sets, t_schedule, C_feed, m_in, m_out, raff_cprofile, ext_cprofile, raff_intgral_purity, raff_recov, ext_intgral_purity, ext_recov, raff_vflow, ext_vflow, Model_Acc, Expected_Acc, Error_percent

#%%
#--------------------------------------------------- Functions


# ---------------- sampling

def lhq_sample_mj(m_min, m_max, n_samples, diff=0.1):
    """
    Function that performs Latin Hypercube (LHS) sampling for [m1, m2, m3, m4]
    Note that for all mjs: (m_min < m_j < m_max)
    And that:
          (i)   m4 < m1 - (diff*m1) and m2 < m1 - (diff*m1)
          (ii)  m2 < m3 - (diff*m3)
          (iii) m3 > m4 + (diff*m4)
    Final result is an np.array of size: (n_samples, 4)
    """

    # Initialize the array to store the samples
    samples = np.zeros((n_samples, 4))

    for i in range(n_samples):
        # Sample m1, m2, m3, m4 within bounds and respecting constraints
        m1 = np.random.uniform(m_min, m_max)
        m4 = np.random.uniform(m_min, m1-diff*m1)

        # Sample m2 such that it respects the constraint: m2 < m1 - (diff*m1)
        m2 = np.random.uniform(m_min, m_max)
        while m2 >= m1 - (diff * m1):  # Ensuring m2 < m1 - (diff*m1)
            m2 = np.random.uniform(m_min, m_max)

        # Sample m3 such that it respects the constraint: m3 > m2 + (diff*m2)
        m3 = np.random.uniform(m_min, m_max)
        while m3 <= m2 + (diff * m2):  # Ensuring m3 > m2 + (diff*m2)
            m3 = np.random.uniform(m_min, m_max)

        # Ensure the constraint: m3 > m4 + (diff * m4)
        while m3 <= m4 + (diff * m4):  # Ensuring m3 > m4 + (diff*m4)
            m3 = np.random.uniform(m_min, m_max)

        # Store the sample in the array
        samples[i] = [m1, m2, m3, m4]


    return samples

def fixed_feed_lhq_sample_mj(t_index_min, Q_fixed_feed, m_min, m_max, n_samples, diff=0.1):
    """
    Since the feed is fixed, m3 is caluclated

    Function that performs Latin Hypercube (LHS) sampling for [m1, m2, m3, m4]
    Note that for all mjs: (m_min < m_j < m_max)
    And that:
          (i)   m4 < m1 - (diff*m1) and m2 < m1 - (diff*m1)
          (ii)  m2 < m3 - (diff*m3)
          (iii) m3 > m4 + (diff*m4)
    Final result is an np.array of size: (n_samples, 4)
    """
    # Initialize the array to store the samples
    samples = np.zeros((n_samples, 4))

    for i in range(n_samples):
        # Sample m1, m2, m3, m4 within bounds and respecting constraints
        m1 = np.random.uniform(m_min, m_max)
        m4 = np.random.uniform(m_min, m1-diff*m1)

        # Sample m2 such that it respects the constraint: m2 < m1 - (diff*m1)
        m2 = np.random.uniform(m_min, m_max)
        while m2 >= m1 - (diff * m1):  # Ensuring m2 < m1 - (diff*m1)
            m2 = np.random.uniform(m_min, m_max)

        # Sample m3 such that it respects the constraint: m3 > m2 + (diff*m2)
        m3 = np.random.uniform(m_min, m_max)
        while m3 <= m2 + (diff * m2):  # Ensuring m3 > m2 + (diff*m2)
            m3 = np.random.uniform(m_min, m_max)

        # Ensure the constraint: m3 > m4 + (diff * m4)
        while m3 <= m4 + (diff * m4):  # Ensuring m3 > m4 + (diff*m4)
            m3 = np.random.uniform(m_min, m_max)

        # Store the sample in the array
        samples[i] = [m1, m2, m3, m4]

    return samples

def fixed_m1_and_m4_lhq_sample_mj(m1, m4, m_min, m_max, n_samples, n_m2_div, diff=0.1):
    """
    - Since the feed is fixed, m3 is caluclated AND
    - Since the desorbant is fixed, m1 is caluclated

    Function that performs Latin Hypercube (LHS) sampling for [m1, m2, m3, m4]
    Note that for all mjs: (m_min < m_j < m_max)
    And that:
          (i)   m4 < m1 - (diff*m1) and m2 < m1 - (diff*m1)
          (ii)  m2 < m3 - (diff*m3)
          (iii) m3 > m4 + (diff*m4)
    Final result is an np.array of size: (n_samples, 4)

    [1.78902051 1.10163238 1.75875405 0.20421105], 7.438074624877448
    """
    
    # Initialize the array to store the samples
    samples = np.zeros((n_samples*n_m2_div, 5))
    samples[:,0] = np.ones(n_samples*n_m2_div)*m_max
    samples[:,-2] = np.ones(n_samples*n_m2_div)*2
    # print(f'samples: {samples}')
    nn = int(np.round(n_samples/2))
    num_of_m3_per_m2 = n_samples

    m2_set = np.linspace(m_min, m_max*0.9, n_m2_div)
    # print(f'm2_set: {m2_set}')

    i = np.arange(len(m2_set))
    k = np.repeat(i,num_of_m3_per_m2)

    print(f'k:{k}')
    #Sample from the separation triangle:
    for i in range(len(k)): # for each vertical line
        # print(f'k: {k[i]}')
        m2 = m2_set[k[i]]

        samples[i, 1] = m2

        if i == 0:
          m2 = 3.34
          m3 = m2 + 0.1
          samples[i, 1] = m2
          samples[i, 2] = m3  # apex of trianlge
        else:
          m3 = np.random.uniform(m2, m_max)

        samples[i, 2] = m3
        samples[i,-1] = 0.3
    return samples


# ---------- Objective Function

def mj_to_Qj(mj):
  '''
  Converts flowrate ratios to internal flowrates - flowrates within columns
  '''
  Qj = (mj*V_col*(1-e) + V_col*e)/(t_index_min*60) # cm^3/s
  return Qj

# Define the obj and constraint functions
# All parameteres
def obj_con(X):
  """Feasibility weighted objective; zero if not feasible.

    X = [m1, m2, m3, m4, t_index];
    Objective: WAR = Weighted Average Recovery
    Constraint: WAP = Weighted Average Purity

    Use WAP to calculate the feasibility weights. Which
    will scale teh EI output.

  """
  X = np.array(X)


  # print(f'np.shape(x_new)[0]: {np.shape(X)}')
  if X.ndim == 1:

      Pur = np.zeros(2)
      Rec = np.zeros(2)
      # Unpack and convert to float and np.arrays from torch.tensors:
      m1, m2, m3, m4, t_index_min = float(X[0]), float(X[1]), float(X[2]), float(X[3]), float(X[4])*t_reff

      print(f'[m1, m2, m3, m4]: [{m1}, {m2}, {m3}, {m4}], t_index: {t_index_min}')

      Q_I, Q_II, Q_III, Q_IV = mj_to_Qj(m1), mj_to_Qj(m2), mj_to_Qj(m3), mj_to_Qj(m4)
      Q_internal = np.array([Q_I, Q_II, Q_III, Q_IV]) # cm^3/s
      Qfeed = Q_III - Q_II
      Qraffinate = Q_III - Q_IV
      Qdesorbent = Q_I - Q_IV
      Qextract = Q_I - Q_II
      Q_external = np.array([Qfeed, Qraffinate, Qdesorbent,Qextract])
      print(f'Q_internal: {Q_internal} cm^s/s')
      print(f'Q_internal: {Q_internal*3.6} L/h')

      print(f'----------------------------------')
      print(f'Q_external: {Q_external} cm^s/s')
      print(f'Q_external: {Q_external*3.6} L/h')
      # print(f'Q_internal type: {type(Q_internal)}')

      SMB_inputs[12] = t_index_min  # Update t_index
      SMB_inputs[14] = Q_internal # Update Q_internal

      results = SMB(SMB_inputs)

      # print(f'done solving sample {i+1}')

      raff_purity = results[10]  # [Glu, Fru]
      ext_purity = results[12]  # [Glu, Fru]

      raff_recovery = results[11]  # [Glu, Fru]
      ext_recovery = results[13]  # [Glu, Fru]

      pur1 = raff_purity[0]
      pur2 = ext_purity[1]

      rec1 = raff_recovery[0]
      rec2 = ext_recovery[1]

      # Pack
      # WAP[i] = WAP_add
      # WAR[i] = WAR_add
      Pur[:] = [pur1, pur2]
      Rec[:] = [rec1, rec2]

  elif X.ndim > 1:
      Pur = np.zeros((len(X[:,0]), 2))
      Rec = np.zeros((len(X[:,0]), 2))

      for i in range(len(X[:,0])):

          # Unpack and convert to float and np.arrays from torch.tensors:
          m1, m2, m3, m4, t_index_min = float(X[i,0]), float(X[i,1]), float(X[i,2]), float(X[i,3]), float(X[i,4])*t_reff

          print(f'[m1, m2, m3, m4]: [{m1}, {m2}, {m3}, {m4}], t_index: {t_index_min}')
          Q_I, Q_II, Q_III, Q_IV = mj_to_Qj(m1), mj_to_Qj(m2), mj_to_Qj(m3), mj_to_Qj(m4)
          Q_internal = np.array([Q_I, Q_II, Q_III, Q_IV]) # cm^3/s
          # Calculate and display external flowrates too
          Qfeed = Q_III - Q_II
          Qraffinate = Q_III - Q_IV
          Qdesorbent = Q_I - Q_IV
          Qextract = Q_I - Q_II

          Q_external = np.array([Qfeed, Qraffinate, Qdesorbent,Qextract])
          print(f'Q_internal: {Q_internal} cm^s/s')
          print(f'Q_internal: {Q_internal*3.6} L/h')

          print(f'----------------------------------')
          print(f'Q_external: {Q_external} cm^s/s')
          print(f'Q_external: {Q_external*3.6} L/h')


          # print(f'Q_internal type: {type(Q_internal)}')
          # Update SMB_inputs:
          SMB_inputs[12] = t_index_min  # Update t_index
          SMB_inputs[14] = Q_internal # Update Q_internal

          results = SMB(SMB_inputs)

          # print(f'done solving sample {i+1}')

          raff_purity = results[10]  # [Glu, Fru]
          ext_purity = results[12]  # [Glu, Fru]

          raff_recovery = results[11]  # [Glu, Fru]
          ext_recovery = results[13]  # [Glu, Fru]

          pur1 = raff_purity[0]
          pur2 = ext_purity[1]

          rec1 = raff_recovery[0]
          rec2 = ext_recovery[1]

          # Pack
          # WAP[i] = WAP_add
          # WAR[i] = WAR_add
          Pur[i,:] = [pur1, pur2]
          Rec[i,:] = [rec1, rec2]
          print(f'Pur: {pur1}, {pur2}')
          print(f'Rec: {rec1}, {rec2}\n\n')

  return  Rec, Pur, np.array([m1, m2, m3, m4, t_index_min])

# Fixed index time
def obj_con_fix_t(X):
  """Feasibility weighted objective; zero if not feasible.

    X = [m1, m2, m3, m4];
    Objective: WAR = Weighted Average Recovery
    Constraint: WAP = Weighted Average Purity

    Use WAP to calculate the feasibility weights. Which
    will scale teh EI output.

  """
  X = np.array(X)


  # print(f'np.shape(x_new)[0]: {np.shape(X)}')
  if X.ndim == 1:

      Pur = np.zeros(2)
      Rec = np.zeros(2)
      # Unpack and convert to float and np.arrays from torch.tensors:
      m1, m2, m3, m4 = float(X[0]), float(X[1]), float(X[2]), float(X[3])

      print(f'[m1, m2, m3, m4]: [{m1}, {m2}, {m3}, {m4}], t_index: {t_index_min}')

      Q_I, Q_II, Q_III, Q_IV = mj_to_Qj(m1), mj_to_Qj(m2), mj_to_Qj(m3), mj_to_Qj(m4)
      Q_internal = np.array([Q_I, Q_II, Q_III, Q_IV]) # cm^3/s
      Qfeed = Q_III - Q_II
      Qraffinate = Q_III - Q_IV
      Qdesorbent = Q_I - Q_IV
      Qextract = Q_I - Q_II
      Q_external = np.array([Qfeed, Qraffinate, Qdesorbent,Qextract])
      print(f'Q_internal: {Q_internal} cm^s/s')
      print(f'Q_internal: {Q_internal*3.6} L/h')

      print(f'----------------------------------')
      print(f'Q_external: {Q_external} cm^s/s')
      print(f'Q_external: {Q_external*3.6} L/h')
      # print(f'Q_internal type: {type(Q_internal)}')

      SMB_inputs[12] = t_index_min  # Update t_index
      SMB_inputs[14] = Q_internal # Update Q_internal

      results = SMB(SMB_inputs)

      # print(f'done solving sample {i+1}')

      raff_purity = results[10]  # [Glu, Fru]
      ext_purity = results[12]  # [Glu, Fru]

      raff_recovery = results[11]  # [Glu, Fru]
      ext_recovery = results[13]  # [Glu, Fru]

      pur1 = raff_purity[0]
      pur2 = ext_purity[1]

      rec1 = raff_recovery[0]
      rec2 = ext_recovery[1]

      # Pack
      # WAP[i] = WAP_add
      # WAR[i] = WAR_add
      Pur[:] = [pur1, pur2]
      Rec[:] = [rec1, rec2]

  elif X.ndim > 1:
      Pur = np.zeros((len(X[:,0]), 2))
      Rec = np.zeros((len(X[:,0]), 2))

      for i in range(len(X[:,0])):

          # Unpack and convert to float and np.arrays from torch.tensors:
          m1, m2, m3, m4, t_index_min = float(X[i,0]), float(X[i,1]), float(X[i,2]), float(X[i,3]), float(X[i,4])*t_reff

          print(f'[m1, m2, m3, m4]: [{m1}, {m2}, {m3}, {m4}], t_index: {t_index_min}')
          Q_I, Q_II, Q_III, Q_IV = mj_to_Qj(m1), mj_to_Qj(m2), mj_to_Qj(m3), mj_to_Qj(m4)
          Q_internal = np.array([Q_I, Q_II, Q_III, Q_IV]) # cm^3/s
          # Calculate and display external flowrates too
          Qfeed = Q_III - Q_II
          Qraffinate = Q_III - Q_IV
          Qdesorbent = Q_I - Q_IV
          Qextract = Q_I - Q_II

          Q_external = np.array([Qfeed, Qraffinate, Qdesorbent,Qextract])
          print(f'Q_internal: {Q_internal} cm^s/s')
          print(f'Q_internal: {Q_internal*3.6} L/h')

          print(f'----------------------------------')
          print(f'Q_external: {Q_external} cm^s/s')
          print(f'Q_external: {Q_external*3.6} L/h')


          # print(f'Q_internal type: {type(Q_internal)}')
          # Update SMB_inputs:
          SMB_inputs[12] = t_index_min  # Update t_index
          SMB_inputs[14] = Q_internal # Update Q_internal

          results = SMB(SMB_inputs)

          # print(f'done solving sample {i+1}')

          raff_purity = results[10]  # [Glu, Fru]
          ext_purity = results[12]  # [Glu, Fru]

          raff_recovery = results[11]  # [Glu, Fru]
          ext_recovery = results[13]  # [Glu, Fru]

          pur1 = raff_purity[0]
          pur2 = ext_purity[1]

          rec1 = raff_recovery[0]
          rec2 = ext_recovery[1]

          # Pack
          # WAP[i] = WAP_add
          # WAR[i] = WAR_add
          Pur[i,:] = [pur1, pur2]
          Rec[i,:] = [rec1, rec2]
          print(f'Pur: {pur1}, {pur2}')
          print(f'Rec: {rec1}, {rec2}\n\n')

  return  Rec, Pur, np.array([m1, m2, m3, m4, t_index_min])


# ------ Generate Initial Data
def generate_initial_data(sampling_budget):

    # generate training data
    # print(f'Getting {sampling_budget} Samples')
    # train_x = lhq_sample_mj(0.2, 1.7, n, diff=0.1)
    # train_x = fixed_feed_lhq_sample_mj(t_index_min, Q_fixed_feed, 0.2, 1.7, n, diff=0.1)
    train_all = fixed_m1_and_m4_lhq_sample_mj(m_max, m_min, m_min, m_max, sampling_budget, 1, diff=0.1)
    # print(f'train_all: {train_all}')
    # print(f'Done Getting {sampling_budget} Samples')

    # print(f'Solving Over {sampling_budget} Samples')
    Rec, Pur, mjs = obj_con(train_all)
    # print(f'Rec: {Rec}, Pur: {Pur}')
    # print(f'Done Getting {sampling_budget} Samples')
    all_outputs = np.hstack((Rec, Pur))
    return train_all, all_outputs

# ------------------ BO FUNTIONS
# --- Surrogate model creation ---
def surrogate_model(X_train, y_train):
    X_train = np.atleast_2d(X_train)
    y_train = np.atleast_1d(y_train)

    if y_train.ndim == 2 and y_train.shape[1] == 1:
        y_train = y_train.ravel()

    # kernel = C(1.0, (1e-4, 10.0)) * RBF(1.0, (1e-4, 10.0))
    kernel = Matern(length_scale=1.0, nu=1.5)

    gp = GaussianProcessRegressor(kernel=kernel, alpha=1e-5, normalize_y=True, n_restarts_optimizer=5)

    gp.fit(X_train, y_train)

    return gp

# --- AQ funcs:

# --- AQ func: Expected Constrained Improvement ---
def log_expected_constrained_improvement(x, surrogate_obj_gp, constraint_gps, constraint_thresholds, y_best,job, PF_weight, xi=0.01):
    x = np.asarray(x).reshape(1, -1)

    mu_obj, sigma_obj = surrogate_obj_gp.predict(x, return_std=True)
    # print(f'mu_obj: {mu_obj}, sigma_obj: {sigma_obj} ')



    with np.errstate(divide='warn'):
	# Note that, because we are maximizing, y_best > mu_obj - always,
	# So Z is always positive. And if Z is positive then its on the "right-hand-side" of the mean
	# Because norm.cdf calcualtes the integral from left to right, it will by default calculate-
	# the probability of begin LESS than (or equal to), Z.
	# Since we want to probability of being greater than or equal to Z, we have two options
	# (1.) Calculate Z -> and compute 1-norm.cdf(Z) 
	# (2.) Calculate abs(Z) -> make Z negative, -abs(Z) -> integrate norm.cdf(-abs(Z))
	
	# The (2.) option is more robust - because by simply omitting the negative sign, it will also work for cases where we are minimizing the objective
	# (2.) is used below
	
        if job == 'maximize':
                # print(f'{job}ing')
                # Calulate Z and enforce, -ve
                Z = -np.abs(y_best - mu_obj) / sigma_obj
                #  Note: ei is always positive

                # 1. 
                # use logcdf() to get the log probability for cases where Z is very small
                # Since Z < 0, logcdf() < 0 (not a probability)
                log_cdf_term = norm.logcdf(Z) 
                # recover the actual probabitlty
                cdf_term = np.exp(log_cdf_term)

                # 2. Do the same for the pdf term
                # Since Z < 0, logpdf() < 0 (not a likihood)
                log_pdf_term = norm.logcdf(Z) 
                # recover the actual likihood
                pdf_term = np.exp(log_pdf_term)
                # ---------------------------------------------------
                ei = (y_best - mu_obj) * cdf_term + sigma_obj * pdf_term

        elif job == 'minimize':
                # Calulate Z and enforce, -ve
                Z = np.abs(y_best - mu_obj) / sigma_obj
                #  Note: ei is always positive

                # 1. 
                # use logcdf() to get the log probability for cases where Z is very small
                # Since Z < 0, logcdf() < 0 (not a probability)
                log_cdf_term = norm.logcdf(Z) 
                # recover the actual probabitlty
                cdf_term = np.exp(log_cdf_term)

                # 2. Do the same for the pdf term
                # Since Z < 0, logpdf() < 0 (not a likihood)
                log_pdf_term = norm.logcdf(Z) 
                # recover the actual likihood
                pdf_term = np.exp(log_pdf_term)
                # ---------------------------------------------------
                ei = (mu_obj-y_best)*cdf_term + sigma_obj * pdf_term
    
		# print(f'ei: {ei}')

    # print(f'ei: {ei}')


    # Calcualte the probability of Feasibility, "prob_feas"
    prob_feas = 1.0 # initialize

    for gp_c, lam in zip(constraint_gps, constraint_thresholds):

        mu_c, sigma_c = gp_c.predict(x, return_std=True)

        # lam -> inf = 1- (-inf -> lam)
        prob_that_LESS_than_mu = norm.cdf((lam - mu_c) / sigma_c)

        prob_that_GREATER_than_mu = 1 - prob_that_LESS_than_mu

        pf = prob_that_GREATER_than_mu

        # pf is a vector,
        # We just want the non-zero part
        pf = pf[pf != 0]
        # If theres no, non-zero part, we need to Avoid pf being an empty array:
        if pf.size == 0 or pf < 1e-12:
            pf = 1e-8


        # print(f'pf: {pf}')

        # if we assume that the condtions are independent,
        # then we can "multiply" the weights to get the "joint probability" of feasility
        prob_feas *= pf

    # print(f'ei: {ei}')
    # print(f'prob_feas: {prob_feas}')

    log_eic = np.log(ei) + PF_weight*np.log(prob_feas)
    # print(f'log_eic: {log_eic}')
    # print(f'Convert to float')

    log_eic = float(np.squeeze(log_eic))  # Convert to scalar
    # print(f'log_eic: {log_eic}')
    return -log_eic

# 1. Expected Improvement ---
def expected_improvement(x, surrogate_gp, y_best):
    """
    Computes the Expected Improvement at a point x.
    Scalarizes the surrogate predictions using Tchebycheff, then computes EI.

    Note that the surrogate GP already has the weights applied to it
    """
    x = np.array(x).reshape(1, -1)

    mu, sigma = surrogate_gp.predict(x, return_std=True)

    # print(f'mu: {mu}')
    # print(f'y_best: {y_best}')
    # Compute EI

    xi = 0.2 # the greater the value of xi, the more we encourage exploration
    with np.errstate(divide='warn'):
        Z = ( mu - y_best - xi) / sigma
        ei = np.abs(mu - y_best - xi) * norm.cdf(Z) + sigma * norm.pdf(Z)
        ei[sigma == 0.0] = 0.0

    return -ei[0]  # Negative for minimization

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

    pi = 1 - norm.cdf(z)

    return -pi

# --- ParEGO Main Loop ---
def constrained_BO(optimization_budget, bounds, initial_guess, all_initial_inputs, all_initial_ouputs, job_max_or_min, constraint_thresholds, xi):

    # xi = exploration parameter (the larger it is, the more we explore)

    # Initial values
    # Unpack from: all_initial_ouputs: [GPur, FPur, GRec, FRec]
    # Recovery Objectives
    f1_vals = all_initial_ouputs[:,0]
    f2_vals = all_initial_ouputs[:,1]
    # print(f'f1_vals: {f1_vals}')
    # print(f'f2_vals: {f2_vals}')
    # Purity constraints
    c1_vals  = all_initial_ouputs[:,2]
    c2_vals  = all_initial_ouputs[:,3]
    print(f'c1-size: {np.shape(c1_vals)}')
    print(f'c2-size: {np.shape(c2_vals)}')

    # population = np.delete(all_initial_inputs, 2, axis=1)
    population = all_initial_inputs
    all_inputs = all_initial_inputs # [m1, m2, m3, m4, t_index]
    # print(f'np.shape(all_inputs):{np.shape(all_inputs)}')
    # print(f'np.shape(all_initial_inputs):{np.shape(all_initial_inputs)}')


    # Unpack from: all_initial_inputs

    # print(f'shpae_f1_vals = {np.shape(f1_vals)}')

    # Initialize where we will store solutions
    population_all = []
    all_constraint_1_gps = []
    all_constraint_2_gps = []
    ei_all = []


    for gen in range(optimization_budget):
        # generation = iteration
        print(f"\n\nStarting gen {gen+1}")


        # Generate random weights for scalarization
        lam = np.random.rand()
        weights = [lam, 1 - lam]
        # print(f'weights: {weights}')
        # Note that we generate new weights in each iteration/generation
        # i.e. each time we update the training set

        #SCALARIZE THE OBJECTIVES (BEFORE APPLYING GP)
        phi = 0.05
        scalarized_f_vals = np.maximum(weights[0]*f1_vals, weights[1]*f2_vals)

        scalarized_f_vals += phi*(weights[0]*f1_vals + weights[1]*f2_vals)

        scalarized_f_vals = weights[0]*f1_vals + weights[1]*f2_vals

        # Fit GP to scalarized_surrogate_objective
        # print(f'population { population}, \nscalarized_f_vals {scalarized_f_vals} ')
        scalarized_surrogate_gp = surrogate_model(population, scalarized_f_vals)
        # Pull mean at relevant poputlation points
        # Mean & Varriance
        scalarized_surrogate_gp_mean, scalarized_surrogate_gp_std = scalarized_surrogate_gp.predict(population, return_std=True)
        # The best value so far:
        y_best = np.max(scalarized_surrogate_gp_mean)
        # y_best = 0.60


        # Fit a GP to each constraint:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            # Glu Raff Purity:
            constraint_1_gp = surrogate_model(population, c1_vals)
            all_constraint_1_gps.append(constraint_1_gp)
            # Fru Ext Purity:
            constraint_2_gp = surrogate_model(population, c2_vals)
            all_constraint_2_gps.append(constraint_2_gp)

        # Define the constraint function for the ei optimizer
        # Constraint function with correct shape

        # Define the non-linear constraint functions with dependencies
        # note that each constraint must be written independently
        # --- CONSTANTS ---
        eps = 0.01  # Safety margin (1%)
        small_value = 1e-6  # To avoid division by zero

        # --- Helper: safe t_index ---
        def get_safe_tindex(x):
            return max(x[-1]*60*t_reff, small_value)

        # --- CONSTRAINT FUNCTIONS ---

        # Standard Flow Ratio Constraints:
        def constraint_m1_gt_m2(x): return x[0] - (1 + eps) * x[1]
        def constraint_m1_gt_m4(x): return x[0] - (1 + eps) * x[3]

        def constraint_m2_lt_m1(x): return (1 - eps) * x[0] - x[1]
        def constraint_m2_lt_m3(x): return (1 - eps) * x[2] - x[1]

        def constraint_m3_gt_m2(x): return x[2] - (1 + eps) * x[1]
        def constraint_m3_gt_m4(x): return x[2] - (1 + eps) * x[3]

        def constraint_m4_lt_m1(x): return (1 - eps) * x[0] - x[3]
        def constraint_m4_lt_m3(x): return (1 - eps) * x[2] - x[3]

        # Pump Constraints (flow differences divided by t_index)
        def constraint_feed_pump_upper(x): return m_diff_max - (x[2] - x[1]) / get_safe_tindex(x)
        def constraint_feed_pump_lower(x): return (x[2] - x[1]) / get_safe_tindex(x) - m_diff_min

        def constraint_desorb_pump_upper(x): return m_diff_max - (x[0] - x[3]) / get_safe_tindex(x)
        def constraint_desorb_pump_lower(x): return (x[0] - x[3]) / get_safe_tindex(x) - m_diff_min

        def constraint_raff_pump_upper(x): return m_diff_max - (x[2] - x[3]) / get_safe_tindex(x)
        def constraint_raff_pump_lower(x): return (x[2] - x[3]) / get_safe_tindex(x) - m_diff_min

        def constraint_extract_pump_upper(x): return m_diff_max - (x[0] - x[1]) / get_safe_tindex(x)
        def constraint_extract_pump_lower(x): return (x[0] - x[1]) / get_safe_tindex(x) - m_diff_min

        # Fixed Feed Flow Constraint
        def constraint_fixed_feed(x):
            return (x[2] - x[1]) - (Q_fixed_feed / ((V_col * (1-e)) / get_safe_tindex(x)))

        # --- Nonlinear Constraints Setup ---

        nonlinear_constraints = [
            NonlinearConstraint(constraint_m1_gt_m2, 0, np.inf),
            NonlinearConstraint(constraint_m1_gt_m4, 0, np.inf),
            NonlinearConstraint(constraint_m2_lt_m1, 0, np.inf),
            NonlinearConstraint(constraint_m2_lt_m3, 0, np.inf),
            NonlinearConstraint(constraint_m3_gt_m2, 0, np.inf),
            NonlinearConstraint(constraint_m3_gt_m4, 0, np.inf),
            NonlinearConstraint(constraint_m4_lt_m1, 0, np.inf),
            NonlinearConstraint(constraint_m4_lt_m3, 0, np.inf),
            
            NonlinearConstraint(constraint_feed_pump_upper, 0, np.inf),
            NonlinearConstraint(constraint_feed_pump_lower, 0, np.inf),
            
            NonlinearConstraint(constraint_desorb_pump_upper, 0, np.inf),
            NonlinearConstraint(constraint_desorb_pump_lower, 0, np.inf),
            
            NonlinearConstraint(constraint_raff_pump_upper, 0, np.inf),
            NonlinearConstraint(constraint_raff_pump_lower, 0, np.inf),
            
            NonlinearConstraint(constraint_extract_pump_upper, 0, np.inf),
            NonlinearConstraint(constraint_extract_pump_lower, 0, np.inf),
            
            # Fixed feed constraint (Optional â€” can comment if not needed)
            # NonlinearConstraint(constraint_fixed_feed, -0.001, 0.001)
        ]

        # ? Now you can pass:
        # constraints=nonlinear_constraints
        # into your differential_evolution call


        # --- Run the optimization ---
        print(f'Maxing ECI')
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            result = differential_evolution(
                func=log_expected_constrained_improvement, # probability_of_improvement(x, surrogate_gp, y_best, xi=0.005), expected_improvement(x, surrogate_gp, y_best) | log_expected_constrained_improvement(x, scalarized_surrogate_gp, [constraint_1_gp, constraint_2_gp], constraint_thresholds, y_best, xi)
                bounds=bounds,
                args=(scalarized_surrogate_gp, [constraint_1_gp, constraint_2_gp], constraint_thresholds, y_best, job_max_or_min, PF_weight, xi),
                strategy='best1bin',
                maxiter=200,
                popsize=15,
                disp=False,
                 constraints=(nonlinear_constraints)
                 )

                # Perform the optimization using L-BFGS-B method
        # result = minimize(
        #     expected_improvement,
        #     initial_guess,
        #     args=(scalarized_surrogate_gp, y_best),
        #     method='L-BFGS-B',
        #     bounds=bounds,
        #     options={'maxiter': 100, 'disp': True})

        x_new = result.x # [m1, m2, m3, m4, t_index_min]
        # print(f"x_new: { x_new}")
        f_new, c_new, mj_and_t_new = obj_con(x_new)



        # Add the new row to all_inputs
        all_inputs = np.vstack((all_inputs, mj_and_t_new))

        # Add to population
        population_all.append(population)
        population = np.vstack((population, x_new))

        f1_vals = np.vstack([f1_vals.reshape(-1,1), f_new[0]])
        f2_vals = np.vstack([f2_vals.reshape(-1,1), f_new[1]])
        c1_vals  = np.vstack([c1_vals.reshape(-1,1), c_new[0]])
        c2_vals  = np.vstack([c2_vals.reshape(-1,1), c_new[1]])

        print(f"Gen {gen+1} Status:\n | Sampled Inputs:{x_new[:-1]}, {x_new[-1]*t_reff} [m1, m2, m3, m4, t_index]|\n Outputs: f1: {f_new[0]*100}%, f2: {f_new[1]*100} % | GPur, FPur: {c_new[0]*100}%, {c_new[1]*100}%")

    return population_all, f1_vals, f2_vals, c1_vals , c2_vals , all_inputs




#%%
# --------------- FUNCTION EVALUATION SECTION

# SMB VARIABLES
#######################################################


###################### PRIMARY INPUTS #########################
# Define the names, colors, and parameter sets for 6 components
Names = ["Borate", "HCl"]#, 'C', 'D']#, "C"]#, "D", "E", "F"]
color = ["red", "green"]#, "purple", "brown"]#, "b"]#, "r", "purple", "brown"]
num_comp = len(Names) # Number of components
e = 0.40         # bed voidage
Bm = 300

# Column Dimensions

# How many columns in each Zone?

Z1, Z2, Z3, Z4 = 1,3,3,1 # *3 for smb config
zone_config = np.array([Z1, Z2, Z3, Z4])
nnn = Z1 + Z2 + Z3 + Z4

L = 70 # cm # Length of one column
d_col = 8.66 # cm # column internal diameter

# Calculate the radius
r_col = d_col / 2
# Calculate the area of the base
A_col = np.pi * (r_col ** 2) # cm^2
V_col = A_col*L # cm^3
# Dimensions of the tubing and from each column:
# Assuming the pipe diameter is 20% of the column diameter:
d_in = 0.2 * d_col # cm
nx_per_col = 15


################ Time Specs #################################################################################
t_index_min = 6 # min # Index time # How long the pulse holds before swtiching
n_num_cycles = 12    # Number of Cycles you want the SMB to run for
###############  FLOWRATES   #################################################################################

# Jochen et al:
# Q_P, Q_Q, Q_R, Q_S = 5.21, 4, 5.67, 4.65 # x10-7 m^3/s
conv_fac = 0.1 # x10-7 m^3/s => cm^3/s
# # Q_P, Q_Q, Q_R, Q_S  = Q_P*conv_fac, Q_Q*conv_fac, Q_R*conv_fac, Q_S*conv_fac
# Q_I, Q_II, Q_III, Q_IV  = Q_P, Q_Q, Q_R, Q_S 
# Q_internal = np.array([Q_I, Q_II, Q_III, Q_IV])

# Other flowrates:
Q_I, Q_II, Q_III, Q_IV = 70,  60.3, 69, 53 # L/h
Q_internal = np.array([Q_I, Q_II, Q_III, Q_IV]) # L/h => cm^3/s




# # Parameter Sets for different components
################################################################

# Units:
# - Concentrations: g/cm^3
# - kh: 1/s
# - Da: cm^2/s

# A must have a less affinity to resin that B - FOUND IN EXtract purity
# Parameter sets for different components
# Units:
# - Concentrations: g/cm^3
# - kfp: 1/s



# ISOTHERM PARAMETERS
###########################################################################################
# What tpye of isoherm is required?
# Coupled: "CUP"
# Uncoupled: "UNC"
iso_type = "CUP"

# Uncomment as necessary:


parameter_sets = [
                    {"C_feed": 0.001752},    # Borate g/cm^3
                    {"C_feed": 0.009726}] #, # HCl g/cm^3

# # UBK Linear Isotherm
Da_all = np.array([3.892e-6, 2.99187705e-6]) 
kav_params_all = np.array([[1.07122526], [1.98350338]])
cusotom_isotherm_params_all = np.array([[2.40846686], [1.55994115]]) # [ [H_borate], [H_hcl] ]


# # PCR Linear Isotherm
# Da_all = np.array([5.77e-7, 2.3812e-6]) 
# kav_params_all = np.array([[0.54026], [2.171826]])
# cusotom_isotherm_params_all = np.array([[3.6124333], [2.4640415]]) # [ [H_borate], [H_hcl] ]

# # UBK Coupled Langmuir Isotherm
# Da_all = np.array([8.4800855e-6, 2.65331183e-6]) 
# kav_params_all = np.array([[1.8647], [1.683276]])
# cusotom_isotherm_params_all = np.array([[2.489017, 1.7129145], [1.51775429, 1.3031422]]) # [ [H_borate], [H_hcl] ]

# PCR Coupled Langmuir Isotherm
# Da_all = np.array([5.5746e-6, 9.9e-6]) 
# kav_params_all = np.array([[0.59849], [2.291]])
# cusotom_isotherm_params_all = np.array([[2.768037, 2.61624570], [2.42315394, 1.88643671]]) # [ [H_borate], [H_hcl] ]


# ----------- ILLOVO FEED STREAM

# parameter_sets = [
#                     {"C_feed": 0.001752},    # Borate g/cm^3
#                     {"C_feed": 0.009726}] #, # HCl g/cm^3

# # UBK Linear Isotherm
# Da_all = np.array([3.892e-6, 2.99187705e-6]) 
# kav_params_all = np.array([[1.07122526], [1.98350338]])
# cusotom_isotherm_params_all = np.array([[2.40846686], [1.55994115]]) # [ [H_borate], [H_hcl] ]


# # PCR Linear Isotherm
# Da_all = np.array([5.77e-7, 2.3812e-6]) 
# kav_params_all = np.array([[0.54026], [2.171826]])
# cusotom_isotherm_params_all = np.array([[3.6124333], [2.4640415]]) # [ [H_borate], [H_hcl] ]




# Sub et al = np.array([[0.27], [0.53]])

# # Langmuir, [Q_max, b]
# cusotom_isotherm_params_all = np.array([[2.51181596, 1.95381598], [3.55314612, 1.65186647]])

# Linear + Langmuir, [H, Q_max, b]
# cusotom_isotherm_params_all = np.array([[1, 2.70420148, 1.82568197], [1, 3.4635919, 1.13858329]])

#%%
# STORE/INITALIZE SMB VAIRABLES
SMB_inputs = [iso_type, Names, color, num_comp, nx_per_col, e, Da_all, Bm, zone_config, L, d_col, d_in, t_index_min, n_num_cycles, Q_internal, parameter_sets, cusotom_isotherm_params_all, kav_params_all]

# ---------- SAMPLE RUN IF NECESSARY
start_test = time.time()
results = SMB(SMB_inputs)
# ref:  [y_matrices, nx, t, t_sets, t_schedule, C_feed, m_in, m_out, raff_cprofile, ext_cprofile, raff_intgral_purity, raff_recov, ext_intgral_purity, ext_recov, raff_vflow, ext_vflow, Model_Acc, Expected_Acc, Error_percent]
# STORE
Raffinate_Purity = results[10]
Raffinate_Recovery = results[11]
Extract_Purity = results[12]
Extract_Recovery = results[13]
Mass_Balance_Error_Percent = results[-1]
m_in = results[6]
m_out = results[7]
Model_Acc =  results[-3]
Expected_Acc = results[-2]
raff_cprofile = results[8]
ext_cprofile= results[9]
import matplotlib.pyplot as plt

# Plotting the data
plt.plot(results[2]/60/60, raff_cprofile[0], label='Raff CProfile 0')
plt.plot(results[2]/60/60, raff_cprofile[1], label='Raff CProfile 1')
plt.plot(results[2]/60/60, ext_cprofile[0], label='Ext CProfile 0')
plt.plot(results[2]/60/60, ext_cprofile[1], label='Ext CProfile 1')

# Adding labels and title
plt.xlabel('Time, hrs')
plt.ylabel('g/mL')
plt.title('Comparison of Raff and Ext CProfiles')

# Adding legend
plt.legend()

# Display the plot
plt.grid(True)
plt.tight_layout()
plt.show()


end_test = time.time()
test_duration = end_test-start_test

# DISPLAY
print(f'\n\n TEST RESULTS : \n')
print(f'Time Taken for 1 SMB Run: {test_duration/60} min')
print(f'Model_Acc: {Model_Acc}')
print(f'Expected_Acc: {Expected_Acc}')


print(f'm_in: {m_in} g')
print(f'm_out: {m_out} g ')
print(f'Raffinate_Recovery: {Raffinate_Recovery} ')
print(f'Extract_Recovery:  {Extract_Recovery}')
print(f'Raffinate_Purity: {Raffinate_Purity} ')
print(f'Extract_Purity: {Extract_Purity}')
print(f'Mass_Balance_Error_Percent: {Mass_Balance_Error_Percent}%')

#%%

# ----- MAIN ROUTINE
if __name__ == "__main__":
    # ------- OPTIMIZATION VARIABLES
    # Do you want to maximize or minimize the objective function?
    job_max_or_min = 'maximize'

    # - - - - -
    Q_fixed_feed = 4 # L/h # minimum (25%) flowrate on pump 6 (smallest pump)
    Q_fixed_feed = Q_fixed_feed/3.6 # L/h --> cm^3/s
    # - - - - -
    t_reff = 10 # min
    # - - - - -
    Q_max = 40 # L/h
    Q_min = 1 # L/h


    Q_max = Q_max/3.6 # L/h --> cm^3/s
    Q_min = Q_min/3.6 # L/h --> cm^3/s
    # - - - - -
    m_max = 4
    m_min = 0.27
    # - - - - -
    sampling_budget = 1 #
    optimization_budget = 60
    constraint_threshold = [0.995, 0.995] # [Glu, Fru]
    # - - - - -

    # - - - - -
    m_diff_max = Q_max/(V_col*(1-e))
    m_diff_min = Q_min/(V_col*(1-e))
    m_fixed_feed = Q_fixed_feed/(V_col*(1-e))
    # - - - - -
    PF_weight = 10 # Weight applied to the probability of feasibility
    # - - - - -
    
    bounds = [  
    (1.1, m_max), # m1
    (1, m_max), # m2
    (1.1, m_max), # m3
    (1, 1.5), # m4
    (0.2, 1) # t_index/t_reff (normalized)
    ]


    initial_guess = 0 # min
    print(f'\n\n OPTIMIZATION INPUTS: \n')
    print(f'Column Volume: {V_col} cm^3 | {V_col/1000} L')
    print(f'Column CSA: {A_col} cm^2')
    print(f'Column Length: {L} cm')
    print(f'Column Diameter: {d_col} cm')
    print(f'Optimization Budget: {optimization_budget}')
    print(f'Sampling Budget: {sampling_budget}')
    print(f"bounds:\nm1: ({bounds[0][0]}, {bounds[0][1]})\nm2: ({bounds[1][0]}, {bounds[1][1]})\nm3: ({bounds[2][0]}, {bounds[2][1]})\nm4: ({bounds[3][0]}, {bounds[3][1]})\nt_index: ({bounds[4][0]*t_reff}, {bounds[4][1]*t_reff}) min")
        
#%%
    # generate iniital samples
    all_initial_inputs, all_initial_outputs = generate_initial_data(sampling_budget)
    print(f'all_initial_inputs\n{ all_initial_inputs}')
    print(f'all_initial_outputs\n{ all_initial_outputs}')

#%%
    # OPTIMIZATION
    population_all, f1_vals, f2_vals, c1_vals, c2_vals, all_inputs  = constrained_BO(optimization_budget, bounds, initial_guess, all_initial_inputs, all_initial_outputs, job_max_or_min, constraint_threshold, 0.001)

#%%
    # ----------- SAVE
    # Convert NumPy array to list
    # Inputs:
    all_inputs_list = all_inputs.tolist()
    # Outputs:
    data_dict = {
        "f1_vals": f1_vals.tolist(),
        "f2_vals": f2_vals.tolist(),
        "c1_vals": c1_vals.tolist(),
        "c2_vals": c2_vals.tolist(),
    }



    # SAVE all_inputs to JSON:
    with open("UBK-borhcl-lin-type1_60iter_all_inputs.json", "w") as f:
        json.dump(all_inputs_list, f, indent=4)

    # SAVE recoveries_and_purities to JSON:
    with open("UBK-borhcl-lin-type1_60iter_all_outputs.json", "w") as f:
        json.dump(data_dict, f, indent=4)



#         # -------- VISULAIZATIONS
#     def load_inputs_outputs(inputs_path, outputs_path):
#         """
#         Loads all_inputs and output values (f1, f2, c1, c2) from saved JSON files and reconstructs them as numpy arrays.
        
#         Args:
#             inputs_path (str): Path to 'all_inputs.json'.
#             outputs_path (str): Path to 'all_outputs.json'.

#         Returns:
#             all_inputs (np.ndarray): Loaded inputs array.
#             f1_vals (np.ndarray): Glucose recovery values.
#             f2_vals (np.ndarray): Fructose recovery values.
#             c1_vals (np.ndarray): Glucose purity values.
#             c2_vals (np.ndarray): Fructose purity values.
#         """
#         # Load inputs
#         with open(inputs_path, "r") as f:
#             all_inputs_list = json.load(f)
#         all_inputs = np.array(all_inputs_list)

#         # Load outputs
#         with open(outputs_path, "r") as f:
#             data_dict = json.load(f)
#         f1_vals = np.array(data_dict["f1_vals"])
#         f2_vals = np.array(data_dict["f2_vals"])
#         c1_vals = np.array(data_dict["c1_vals"])
#         c2_vals = np.array(data_dict["c2_vals"])

#         return all_inputs, f1_vals, f2_vals, c1_vals, c2_vals


#     #%%

#     #------------------------------------------------------- 1. Table

#     def create_output_optimization_table(f1_vals, f2_vals, c1_vals, c2_vals, sampling_budget):
#         # Create a data table with recoveries first
#         data = np.column_stack((f1_vals*100, f2_vals*100, c1_vals*100, c2_vals*100))
#         columns = ['Recovery F1 (%)', 'Recovery F2 (%)', 'Purity C1 (%)', 'Purity C2 (%)']
#         rows = [f'Iter {i+1}' for i in range(len(c1_vals))]

#         # Identify "star" entries (where f1_vals, f2_vals > 70 and c1_vals, c2_vals > 90)
#         star_indices = np.where((f1_vals*100 > 50) & (f2_vals*100 > 50) & (c1_vals*100 > 80) & (c2_vals*100 > 80))[0]

#         # Create figure
#         fig, ax = plt.subplots(figsize=(8, len(c1_vals) * 0.4))
#         ax.set_title("Optimization Iterations: Recovery & Purity Table", fontsize=12, fontweight='bold', pad=5)  # Reduced padding
#         ax.axis('tight')
#         ax.axis('off')

#         # Create the table
#         table = ax.table(cellText=data.round(2),
#                         colLabels=columns,
#                         rowLabels=rows,
#                         cellLoc='center',
#                         loc='center')

#         # Adjust font size
#         table.auto_set_font_size(False)
#         table.set_fontsize(10)
#         table.auto_set_column_width(col=list(range(len(columns))))

#         # Apply colors
#         for i in range(len(c1_vals)):
#             for j in range(len(columns)):
#                 cell = table[(i+1, j)]  # (row, column) -> +1 because row labels shift index
#                 if i < sampling_budget:
#                     cell.set_facecolor('lightgray')  # Grey out first 20 rows
#                 if i in star_indices:
#                     cell.set_facecolor('yellow')  # Highlight star entries in yellow

#         # Save the figure as an image
#         image_filename = "output_optimization_table.png"
#         fig.savefig(image_filename, dpi=300, bbox_inches='tight')
#         plt.show()

#         return image_filename



#     def calculate_flowrates(input_array, V_col, e):
#         # Initialize the external flowrate array with the same shape as input_array
#         internal_flowrate = np.zeros_like(input_array[:,:-1])
#         external_flowrate = np.zeros_like(input_array)
        
#         # Reshape the last column to be a 2D array for broadcasting
#         input_last_col = input_array[:, -1]
        
#         for i, t_index in enumerate(input_last_col):
#             # Calculate the flow rates using the provided formula
#             # Fill each row in external_flowrate:
#             print(f't_index: {t_index}')
#             internal_flowrate[i, :] = (input_array[i, :-1] * V_col * (1 - e) + V_col * e) / (t_index * 60)  # cm^3/s
        

#         internal_flowrate = internal_flowrate*3.6 # cm^3/s => L/h
#         print(f'internal_flowrate: {internal_flowrate}')
#         # Calculate Internal FLowtates:
#         Qfeed = internal_flowrate[:,2] - internal_flowrate[:,1] # Q_III - Q_II 
#         Qraffinate = internal_flowrate[:,2] - internal_flowrate[:,3] # Q_III - Q_IV 
#         Qdesorbent = internal_flowrate[:,0] - internal_flowrate[:,3] # Q_I - Q_IV 
#         Qextract = internal_flowrate[:,0] - internal_flowrate[:,1] # Q_I - Q_II

#         external_flowrate[:,0] = Qfeed
#         external_flowrate[:,1] = Qraffinate
#         external_flowrate[:,2] = Qdesorbent
#         external_flowrate[:,3] = Qextract
#         external_flowrate[:,4] = input_last_col

#         return internal_flowrate, external_flowrate

#     def create_input_optimization_table(input_array, V_col, e, sampling_budget, f1_vals, f2_vals, c1_vals, c2_vals):
#         # Calculate flow rates
#         internal_flowrate, external_flowrate = calculate_flowrates(input_array, V_col, e)
#         flowrates = external_flowrate
#         # Create a data table with flow rates
#         data = external_flowrate
#         columns = ['Feed (L/h)', 'Raffinate (L/h)', 'Desorbent (L/h)', 'Extract(L/h)', 'Index Time (min)']
#         rows = [f'Iter {i+1}' for i in range(len(input_array))]

#         # Identify "star" entries (example condition: flowrate > threshold)
#         star_indices = np.where((f1_vals*100 > 50) & (f2_vals*100 > 50) & (c1_vals*100 > 80) & (c2_vals*100 > 80))[0]
#         # Create figure
#         fig, ax = plt.subplots(figsize=(8, len(input_array) * 0.4))
#         ax.set_title("Optimization Iterations: Flowrate Table", fontsize=12, fontweight='bold', pad=5)  # Reduced padding
#         ax.axis('tight')
#         ax.axis('off')

#         # Create the table
#         table = ax.table(cellText=data.round(3),
#                         colLabels=columns,
#                         rowLabels=rows,
#                         cellLoc='center',
#                         loc='center')

#         # Adjust font size
#         table.auto_set_font_size(False)
#         table.set_fontsize(5)
#         table.auto_set_column_width(col=list(range(len(columns))))

#         # Apply colors
#         for i in range(len(input_array)):
#             for j in range(len(columns)):
#                 cell = table[(i+1, j)]  # (row, column) -> +1 because row labels shift index
#                 if i < sampling_budget:
#                     cell.set_facecolor('lightgray')  # Grey out first sampling_budget rows
#                 if i in star_indices:
#                     cell.set_facecolor('yellow')  # Highlight star entries in yellow

#         # Save the figure as an image
#         image_filename = "input_optimization_table.png"
#         fig.savefig(image_filename, dpi=300, bbox_inches='tight')
#         plt.show()

#         return image_filename





#     #------------------------------------------------------- 2. Recovery Pareto

#     def create_recovery_pareto_plot(f1_vals, f2_vals, zone_config, sampling_budget, optimization_budget):
#         # Convert to percentages
#         f1_vals_plot = f1_vals * 100
#         f2_vals_plot = f2_vals * 100

#         # Function to find Pareto front
#         def find_pareto_front(f1, f2):
#             pareto_mask = np.ones(len(f1), dtype=bool)  # Start with all points assumed Pareto-optimal

#             for i in range(len(f1)):
#                 if pareto_mask[i]:  # Check only if not already removed
#                     pareto_mask[i] = not np.any((f1 >= f1[i]) & (f2 >= f2[i]) & ((f1 > f1[i]) | (f2 > f2[i])))

#             return pareto_mask

#         # Identify Pareto-optimal points
#         pareto_mask = find_pareto_front(f1_vals_plot, f2_vals_plot)

#         plt.figure(figsize=(10, 6))

#         # Plot non-Pareto points in blue
#         plt.scatter(f1_vals_plot[~pareto_mask], f2_vals_plot[~pareto_mask], c='blue', marker='o', label='Optimization Iterations')
#         # Plot Pareto-optimal points in red
#         plt.scatter(f1_vals_plot[pareto_mask], f2_vals_plot[pareto_mask], c='red', marker='o', label='Pareto Frontier')

#         # Plot initial samples in grey
#         # plt.scatter(f1_initial, f2_initial, c='grey', marker='o', label='Initial Samples')

#         # Labels and formatting
#         plt.title(f'Pareto Curve of Recoveries of Glu in Raff vs Fru in Ext\n Config: {zone_config} \nInitial Samples: {sampling_budget}, Opt Iterations: {optimization_budget}')
#         plt.xlabel('Glucose Recovery in Raffinate (%)')
#         plt.ylabel('Fructose Recovery in Extract (%)')
#         plt.xlim(0, 100)
#         plt.ylim(0, 100)
#         plt.grid(True)
#         plt.legend()

#         # Save the figure as an image
#         image_filename = "recovery_pareto.png"
#         plt.savefig(image_filename, dpi=300, bbox_inches='tight')
#         plt.show()

#         return image_filename


#     #------------------------------------------------------- 2. Purity Pareto

#     def create_purity_pareto_plot(c1_vals, c2_vals, zone_config, sampling_budget, optimization_budget):
#         # Convert to percentages
#         c1_vals_plot = c1_vals * 100
#         c2_vals_plot = c2_vals * 100

#         # Function to find Pareto front
#         def find_pareto_front(c1, c2):
#             pareto_mask = np.ones(len(c1), dtype=bool)  # Start with all points assumed Pareto-optimal

#             for i in range(len(c1)):
#                 if pareto_mask[i]:  # Check only if not already removed
#                     pareto_mask[i] = not np.any((c1 >= c1[i]) & (c2 >= c2[i]) & ((c1 > c1[i]) | (c2 > c2[i])))

#             return pareto_mask

#         # Identify Pareto-optimal points
#         pareto_mask = find_pareto_front(c1_vals_plot, c2_vals_plot)

#         plt.figure(figsize=(10, 6))

#         # Plot non-Pareto points in blue
#         plt.scatter(c1_vals_plot[~pareto_mask], c2_vals_plot[~pareto_mask], c='blue', marker='o', label='Optimization Iterations')
#         # Plot Pareto-optimal points in red
#         plt.scatter(c1_vals_plot[pareto_mask], c2_vals_plot[pareto_mask], c='red', marker='o', label='Pareto Frontier')

#         # Plot initial samples in grey
#         # plt.scatter(c1_initial, c2_initial, c='grey', marker='o', label='Initial Samples')

#         # Labels and formatting
#         plt.title(f'Pareto Curve of Purities of Glu in Raff vs Fru in Ext\n Config: {zone_config} \nInitial Samples: {sampling_budget}, Opt Iterations: {optimization_budget}')
#         plt.xlabel('Glucose Purity in Raffinate (%)')
#         plt.ylabel('Fructose Purity in Extract (%)')
#         plt.xlim(0, 100)
#         plt.ylim(0, 100)
#         plt.grid(True)
#         plt.legend()

#         # Save the figure as an image
#         image_filename = "purity_pareto.png"
#         plt.savefig(image_filename, dpi=300, bbox_inches='tight')
#         plt.show()

#         return image_filename



#     #------------------------------------------------------- 4. Pareto Outputs Trace
#     def find_pareto_front(f1, f2):
#         pareto_mask = np.ones(len(f1), dtype=bool)  # Start with all points assumed Pareto-optimal

#         for i in range(len(f1)):
#             if pareto_mask[i]:  # Check only if not already removed
#                 pareto_mask[i] = not np.any((f1 >= f1[i]) & (f2 >= f2[i]) & ((f1 > f1[i]) | (f2 > f2[i])))

#         return pareto_mask

#     def plot_inputs_vs_iterations(input_array, f1_vals, f2_vals):
#             input_names = ['Feed (L/h)', 'Raffinate (L/h)', 'Desorbent (L/h)', 'Extract (L/h)', 'Index Time (min)']
#             # Convert to percentages
#             f1_vals_plot = f1_vals * 100
#             f2_vals_plot = f2_vals * 100

#             # Identify Pareto-optimal points
#             pareto_mask = find_pareto_front(f1_vals_plot, f2_vals_plot)

#             # Filter input_array for Pareto-optimal points
#             pareto_inputs = input_array[pareto_mask]

#             # Plot inputs vs iterations for Pareto-optimal points
#             iterations = np.arange(1, len(pareto_inputs) + 1)

#             fig, ax1 = plt.subplots(figsize=(12, 8))

#             # Plot all inputs except the last one
#             for i in range(pareto_inputs.shape[1] - 1):
#                 ax1.plot(iterations, pareto_inputs[:, i], marker='o', label=f'{input_names[i]}')

#             ax1.set_xlabel('Iteration')
#             ax1.set_ylabel('Flowrates (L/h)')
#             ax1.grid(True)
#             ax1.legend(loc='upper left')

#             # Create a second y-axis for the indexing time
#             ax2 = ax1.twinx()
#             ax2.plot(iterations, pareto_inputs[:, -1], marker='o', color='grey', linestyle = "--", label=f'Input {input_names[-1]}')
#             ax2.set_ylabel('Index Time (min)')
#             ax2.legend(loc='upper right')

#             plt.title('Operating Conditions at Pareto-Optimal Operating Points\nInputs=[Flowrates, Indexing Time]')
#             plt.show()

#     #%%
#     # Run the Functions and Visualise
#     # --- Paretos
#     rec_pareto_image_filename = create_recovery_pareto_plot(f1_vals, f2_vals, zone_config, sampling_budget, optimization_budget)
#     pur_pareto_image_filename = create_purity_pareto_plot(c1_vals, c2_vals, zone_config, sampling_budget, optimization_budget)
#     plot_inputs_vs_iterations(all_inputs, f1_vals, f2_vals)

#     # ---- Tables
#     opt_table_for_outputs_image_filename = create_output_optimization_table(f1_vals, f2_vals, c1_vals, c2_vals, sampling_budget)
#     opt_table_for_inputs_image_filename = create_input_optimization_table(all_inputs, V_col, e, sampling_budget, f1_vals, f2_vals, c1_vals, c2_vals)


# # %%
