#%%
import numpy as np
# -*- coding: utf-8 -*-
# # %%

                
            # # for i in range(len(start)):
            # for i, strt in enumerate(start):
            #     #  start[i] => the node at the entrance to the ith col
            #     # So start[3] is the node representing the 1st node in col 3

            #     if status[i] == '0': # IF NOT A SUB-ZONED VECTOR

            #         Q_1 = get_X(t, Q_col_all, i-1) # Vol_flow from previous column (which for column 0, is the last column in the chain)
            #         Q_2 = get_X(t, Q_pulse_all, i) # Vol_flow injected IN port i

            #         Q_out_port = get_X(t, Q_col_all, i) # Vol_flow OUT of port 0 (Also could have used Q_1 + Q_2)


            #         # W1 = Q_1/Q_out_port # Weighted flowrate to column i
            #         # W2 = Q_2/Q_out_port # Weighted flowrate to column i

            #         W1 = Q_1/(Q_1 + Q_2) # Weighted flowrate to column i
            #         W2 = Q_2/(Q_1 + Q_2) # Weighted flowrate to column i
            #         u =  get_X(t, u_col_all, i)


                   
                    
            #         if Q_2 > 0  and Q_2 == Q_external[0]: # Concentration in the next column is only affected for injection flows IN
            #          # Calcualte Weighted Concentration:
            #             c_injection = get_C(t, Cj_pulse_all, i, comp_idx)

            #             C_IN = W1 * c[strt-1] + W2 * c_injection
            #             # print(f'C_IN: {C_IN}')

            #         elif Q_2  == Q_external[2]: # if desorbent
            #             # print(f'strt: {strt}')
            #             # print(f'desorbent')
            #             C_IN = W1 * c[strt-1]
                
            #         else:
            #             # C_IN = c[i*nx_col-1] # no change in conc during product collection
            #             C_IN = c[strt-1] # no change in conc during product collection
                    
                
            #     elif status[i] != '0': # IF IT IS IN A SUBZONE (also if its a single col receiving from a subzone)

            #         if len(status[i]) == 5: # if it is reciving from a single col
            #             # get the rank
            #             rank = get_rank(status[i])
            #             zone_position = get_zone(status[i])

            #             Q_1 = get_X(t, Q_col_all, i-1-rank)
            #             # NOTE:
            #             # WE WANT THE PULSE EXPEIRENCED BY THE COLUM IN RANK 0
            #             Q_2 = get_X(t, Q_pulse_all, i-rank)
                        
            #             c_injection = get_C(t, Cj_pulse_all, i-rank, comp_idx)

            #             if Q_2 > 0:  
            #                 W1 = Q_1/(Q_1 + Q_2) # Weighted flowrate to column i
            #                 W2 = Q_2/(Q_1 + Q_2) # Weighted flowrate to column i
                            
            #                 C_IN = W1 * c[strt - 1 - rank*nx_per_col] + W2 * c_injection
                            
            #                 u_split = get_X(t, u_col_all, i)/sub_zone_config[zone_position-1]
            #                 u = u_split
            #             else:
            #                 C_IN = c[strt - 1 - rank*nx_per_col]
            #                 u_split = get_X(t, u_col_all, i)/sub_zone_config[zone_position-1]
            #                 u = u_split
                        
            #             # print(f'strt: {strt}')
            #             # print(f'status: {status[i]}')
            #             # print(f'rank: {rank}')
            #             # print(f't: {t/60} min')
            #             # print(f'c_inj: {c_injection}')
            #             # print(f'u: {get_X(t, u_col_all, i)} cm/s')
            #             # print(f'u_split: {u_split} cm/s')
            #             # print(f'Q_1: {Q_1}, Q_2: {Q_2}\n\n')

            #         if len(status[i]) > 5: # if it is reciving from a/(another) sub_zone

            #             # print(f'were in len(status[i]) > 5')
            #             pervious_subz_count = get_N_subzone(status[i])
            #             current_subz_count = get_M_subzone(status[i])
                        
            #             # print(f'pervious_subz_count: {pervious_subz_count}')
            #             rank = get_rank(status[i])
            #             zone_position = get_zone(status[i])
            #             # print(f'zone_position: {zone_position}')

            #             j = strt - 1 - rank*nx_per_col

            #             Q_2 = get_X(t, Q_pulse_all, i-rank)
                        
            #             c_injection = get_C(t, Cj_pulse_all, i-rank, comp_idx)

            #             # Flowrates out the the columns of all the previous subzone cols
            #             # '+1' to include the pulse fowrate
            #             Q_pervious_all = np.zeros(pervious_subz_count + 1)

            #             for ai in range(pervious_subz_count):
            #                 Q_1  = get_X(t, Q_col_all, i - (ai + 1 + rank))/pervious_subz_count
            #                 Q_pervious_all[ai] = Q_1
                        
            #             Q_pervious_all[-1] = Q_2
                        

                     

            #             Q_sum = np.sum(Q_pervious_all)

            #             # initalize the concentration sum:
            #             c_sum =  np.zeros(pervious_subz_count + 1)

            #             c_sum[-1] = (Q_2/(Q_sum))*c_injection

            #             for bi in range(pervious_subz_count):
            #                 c_add = (Q_pervious_all[bi]/Q_sum) * c[j - bi*nx_per_col]
            #                 c_sum[bi] = c_add
                        
            #             c_sum = np.sum(c_sum) # sum of all weighted concentrations

            #             C_IN = c_sum
            #             # current_subz_count = 1 if single column
            #             u_split = get_X(t, u_col_all, i)/current_subz_count
            #             u = u_split
                        
            #             # print(f'strt: {strt}')
            #             # print(f'status: {status[i]}')
            #             # print(f'rank: {rank}')
            #             # print(f't: {t/60} min')
            #             # print(f'Q_pervious_all: {Q_pervious_all}')
            #             # print(f'u: {get_X(t, u_col_all, i)} cm/s')
            #             # print(f'u_split: {u_split} cm/s')
            #             # print(f'Q_1: {Q_1}, Q_2: {Q_2}\n\n')
                        

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
# import pandas as pd

# --------------------------------------------------- Functions

# ----- smb

# UNITS:
# All units must conform to:
# Time - s
# Lengths - cm^2
# Volumes - cm^3
# Masses - g
# Concentrations - g
# Volumetric flowrates - cm^3/s


def SMB(SMB_inputs):
    iso_type, Names, color, num_comp, nx_per_col, e, D_all, Bm, zone_config, L, d_col, d_in, t_index_min, n_num_cycles, Q_internal, parameter_sets, cusotom_isotherm_params_all, subzone_set = SMB_inputs[0:]

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


    def get_zone(status_unit):
        """
        Retrieves the zone number from the status_unit string.
        '0' → zone 0
        '4.1.0' or '4.2.N3.1.M3' → zone 4
        """
        try:
            return int(status_unit.strip().split('.')[0])
        except (IndexError, ValueError):
            raise ValueError(f"Invalid status_unit format: '{status_unit}' — cannot retrieve zone.")

    def get_type(status_unit):
        """
        Retrieves the type identifier from the status_unit string.
        '0' → assumed type 1
        '4.1.0' or '4.2.N3.1.M3' → type from 2nd segment
        """
        try:
            segments = status_unit.strip().split('.')
            return int(segments[1]) if len(segments) > 1 else 1
        except ValueError:
            raise ValueError(f"Invalid status_unit format: '{status_unit}' — cannot retrieve type.")

    def get_rank(status_unit):
        """
        Retrieves the rank of the column within its zone.
        - '0' → rank 0
        - '4.1.0' → rank 0
        - '4.2.N3.1.M3' → rank 1
        """
        try:
            segments = status_unit.strip().split('.')
            if segments == ['0']:
                return 0
            # Last number before 'M' (if exists) is rank
            for seg in reversed(segments):
                if seg.startswith('M'):
                    continue
                if seg.startswith('N'):
                    continue
                return int(seg)
            raise ValueError("Could not determine rank.")
        except (IndexError, ValueError):
            raise ValueError(f"Invalid status_unit format: '{status_unit}' — cannot retrieve rank.")

    def get_N_subzone(status_unit):
        """
        Retrieves the number of columns in the upstream sub-zone from the 'N' segment.
        Raises helpful error if not found (likely to be type 1).
        """
        try:
            for seg in status_unit.strip().split('.'):
                if seg.startswith('N'):
                    return int(seg[1:])
            raise ValueError("No 'N' segment found — likely type 1 or not receiving from sub-zone.")
        except ValueError as ve:
            raise ValueError(f"Invalid status_unit format: '{status_unit}' — {str(ve)}")

    def get_M_subzone(status_unit):
        """
        Retrieves the number of columns in the current sub-zone from the 'M' segment.
        Raises helpful error if not found (likely a single-column zone).
        """
        try:
            for seg in status_unit.strip().split('.'):
                if seg.startswith('M'):
                    return int(seg[1:])
            raise ValueError("No 'M' segment found — likely not part of a multi-column sub-zone.")
        except ValueError as ve:
            raise ValueError(f"Invalid status_unit format: '{status_unit}' — {str(ve)}")

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
        q_star_1 = K1*c_i[comp_idx]


        #------------------- 2. Coupled Langmuir Models
        # The parameter in the numerator is dynamic, depends on comp_idx:
        # K =  cusotom_isotherm_params_all[comp_idx][0]
        
        # # Fix the sum of parameters in the demoninator:
        # K1 = cusotom_isotherm_params_all[0][0] # 1st (and only) parameter of HA 
        # K2 = cusotom_isotherm_params_all[1][0] # 1st (and only) parameter of HB
        
        # q_star_2 = K*c_i[comp_idx]/(1+ K1*c_i[0] + K2*c_i[1])

        #------------------- 3. Combined Coupled Models
        # The parameter in the numerator is dynamic, depends on comp_idx:
        # K_lin =  cusotom_isotherm_params_all[comp_idx][0]
        
        # # Fix the sum of parameters in the demoninator:
        # K1 = cusotom_isotherm_params_all[0][0] # 1st (and only) parameter of HA 
        # K2 = cusotom_isotherm_params_all[1][0] # 1st (and only) parameter of HB
        
        # c_sum = K1 + K2
        # linear_part = K_lin*c_i[comp_idx]
        # langmuir_part = K*c_i[comp_idx]/(1+ K1*c_i[0] + K2*c_i[1])

        # q_star_3 =  linear_part + langmuir_part


        return q_star_1 # [qA, ...]

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
    Q_col_at_t0 = initial_u_col(zone_config, Q_internal)
    Q_col_all = build_matrix_from_vector(Q_col_at_t0, t_schedule)

    Bay_matrix = build_matrix_from_vector(np.arange(1,Ncol_num+1), t_schedule)


    # DISPLAYING INPUT INFORMATION:
    print('---------------------------------------------------')
    print('Number of Components:', num_comp)
    print('---------------------------------------------------')
    print('\nTime Specs:\n')
    print('---------------------------------------------------')
    print('Number of Cycles:', n_num_cycles)
    print('Time Per Cycle:', n_1_cycle/60, "min")
    print('Simulation Time:', tend_min, 'min')
    print('Index Time:', t_index, 's OR', t_index/60, 'min' )
    print('Number of Port Switches:', num_of_injections)
    print('Injections happen at t(s) = :', t_schedule, 'seconds')
    print('---------------------------------------------------')
    print('\nColumn Specs:\n')
    print('---------------------------------------------------')
    print('Configuration:', zone_config, '[Z1,Z2,Z3,Z4]')
    print(f"Number of Columns: {Ncol_num}")
    print('Column Length:', L, 'cm')
    print('Column Diameter:', d_col, 'cm')
    print('Column Volume:', V_col, 'cm^3')

    print("alpha:", alpha, '(alpha = A_in / A_col)')
    print("Nodes per Column:",nx_col)
    print("Boundary Nodes locations,x[i], i =", start)
    print("Total Number of Nodes (nx):",nx)
    print('---------------------------------------------------')
    print('\nFlowrate Specs:\n')
    print('---------------------------------------------------')
    print("External Flowrates =", Q_external, '[F,R,D,X] ml/min')
    print("Ineternal Flowrates =", Q_internal, 'ml/min')
    print('---------------------------------------------------')
    print('\nPort Schedules:')
    for i in range(num_comp):
        print(f"Concentration Schedule:\nShape:\n {Names[i]}:\n",np.shape(Cj_pulse_all[i]),'\n', Cj_pulse_all[i], "\n")
    print("Injection Flowrate Schedule:\nShape:",np.shape(Q_pulse_all),'\n', Q_pulse_all, "\n")
    print("Respective Column Flowrate Schedule:\nShape:",np.shape(Q_col_all),'\n', Q_col_all, "\n")
    print("Bay Schedule:\nShape:",np.shape(Bay_matrix),'\n', Bay_matrix, "\n")


    ###########################################################################################

    ###########################################################################################

    # Mass Transfer (MT) Models:

    def mass_transfer(kav, q_star, q): # already for specific comp

        MT = kav * Bm/(5 + Bm) * (q_star - q)
        # MT = kav * (q_star - q)
        return MT

    # MT PARAMETERS
    ###########################################################################################
    # print('np.shape(parameter_sets[:]["kh"]):', np.shape(parameter_sets[3]))
    kav_params = [parameter_sets[i]["kh"] for i in range(num_comp)]  # [kA, kB, kC, kD, kE, kF]
    
    # print('kav_params:', kav_params)
    # print('----------------------------------------------------------------')
    ###########################################################################################

    # # FORMING THE ODES


    # Form the remaining schedule matrices that are to be searched by the funcs

    # Column velocity schedule:
    # 1. Veclocity Scheudle if there were no subzones
    # u_col_all = -Q_col_all/A_col/e
    # print(f'u_col_all: {np.shape(u_col_all)}')
    # print(f'u_col_all: {u_col_all}')

    # 2. Veclocity Scheudle if there were subzones
    def get_u_col_at_t0_adj(u_col_at_t0, subzone_set):
        """
        Adjusts the linear velocities of columns based on subzone configurations.
        
        If a column is located in a bay that is part of a subzone's downstream bays 
        (i.e., subzone[1]), then its velocity is divided by the number of such bays 
        to reflect the shared flow among them.

        Parameters:
            u_col_at_t0 (ndarray): 1D array of linear velocities for each bay at time t0.
            subzone_set (list): List of subzones. Each subzone is a pair of lists: [upstream_bays, downstream_bays].

        Returns:
            u_col_adj_at_t0 (ndarray): Adjusted linear velocity array.
        """
        u_col_adj_at_t0 = np.copy(u_col_at_t0)

        for i in range(len(u_col_at_t0)):
            bay = i + 1  # Bay numbers start from 1
            adjusted = False

            for subzone in subzone_set:
                downstream_bays = subzone[1]

                if bay in downstream_bays:
                    divid_by = len(downstream_bays)
                    u_col_adj_at_t0[i] = u_col_at_t0[i] / divid_by
                    adjusted = True
                    print(f"Bay {bay}: {u_col_at_t0[i]} divided by {divid_by} -> {u_col_adj_at_t0[i]}")
                    break  # Stop after finding the first matching subzone

            if not adjusted:
                u_col_adj_at_t0[i] = u_col_at_t0[i]
                print(f"Bay {bay}: not in subzone, remains {u_col_at_t0[i]}")

        return u_col_adj_at_t0

    

    u_col_at_t0 = initial_u_col(zone_config, -Q_internal/A_col/e)
    Q_col_at_t0 = initial_u_col(zone_config, Q_internal)


    u_col_at_t0_new = get_u_col_at_t0_adj(u_col_at_t0, subzone_set)
    Q_col_at_t0_new = get_u_col_at_t0_adj(Q_col_at_t0, subzone_set)

    print(f'u_col_at_t0:{u_col_at_t0}')
    print(f'u_col_at_t0_new:{u_col_at_t0_new}\n\n')

    u_col_all = build_matrix_from_vector(u_col_at_t0_new, t_schedule)
    Q_col_all = build_matrix_from_vector(Q_col_at_t0_new, t_schedule)


    # print(f'u_col_all_adj: {u_col_all_adj}')
    # print(f'u_col_all: {u_col_all}')



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
    def get_column_idx_for_bae(t, bay, t_schedule, Bay_matrix):
        """
        Returns the column ID (i.e. row index of Bay_matrix) where 'bae' is located at time 't'.

        Parameters:
        - t : float
            Current time
        - bae : int
            Bay number (1-based)
        - t_schedule : list of float
            Time slice boundaries; length should be one more than number of time steps
        - Bay_matrix : 2D array-like
            Shape: (n_columns, n_time_slices)
            Rows = column IDs (0-based)
            Columns = time slices
            Entries = bay numbers (1-based)

        Returns:
        - column_idx : int
            The column ID (row index) where the bay is found at time t.
            Returns None if not found.
        """
        import numpy as np
        # Bay_matrix = np.array(Bay_matrix)

        # Find time slice index
        time_idx = None
        for j in range(len(t_schedule) - 1):
            if t_schedule[j] <= t < t_schedule[j+1]:
                time_idx = j
                # print(f'j: {j}')

                break
        else:
            if t >= t_schedule[-1]:
                time_idx = len(t_schedule) - 1

        if time_idx is not None:
            # Scan the time_idx column across all rows
            for row_idx in range(Bay_matrix.shape[0]):
                if Bay_matrix[row_idx, time_idx] == bay:
                    # print(f'row_idx: {row_idx}')
                    return row_idx  # This is the column ID

        return None





    def coeff_matrix_builder_UNC(t, Q_col_all, Q_pulse_all, dx, start, alpha, c, nx_col, comp_idx, Bay_matrix, subzone_set): # note that c_length must include nx_BC

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
            small_col_coeff[0,0], small_col_coeff[0,1] = get_C(t, coef_1_all,col_idx, comp_idx), get_C(t,coef_2_all,col_idx, comp_idx)
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
        def vector_add(nx, c, start, comp_idx, Bay_matrix, subzone_set):
            vec_add = np.zeros(nx)
            c_BC = np.zeros(Ncol_num)
            # print(f'start: {start}')
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
            # for i in range(len(start)):

            for i, strt in enumerate(start):
                bay = get_X(t, Bay_matrix, i)
                found_subzone = False
                # print(f'time: {t/60} min')
                # print(f'col: {i}')
                # print(f'bay: {bay}\n\n')


                for sz, subzone in enumerate(subzone_set):
                    if bay in subzone[1]:
                        found_subzone = True
                        feed_col_bays = subzone[0]

                        # example for single bay case
                        if len(feed_col_bays) == 1:
                            # print(f'better to come in here!')
                            # print(f'time: {t/60} min')
                            # print(f'feed_col_bays[0]: {feed_col_bays[0]}')
                            col_idx_feed = get_column_idx_for_bae(t, feed_col_bays[0], t_schedule, Bay_matrix)
                            # print(f'col_idx_feed: {col_idx_feed}\n\n')
                            # print(f'time: {t/60} min')
                            # print(f'bay: {bay}')
                            # print(f'col_idx_feed: {col_idx_feed}\n\n')


                            Q_1 = get_X(t, Q_col_all, col_idx_feed)
                            c_1 = c[((col_idx_feed + 1) * nx_per_col) - 1]

                            Q_inj = get_X(t, Q_pulse_all, (col_idx_feed + 1)%Ncol_num)
                            c_inj = get_C(t, Cj_pulse_all, (col_idx_feed + 1)%Ncol_num, comp_idx)

                            Q_previous = [Q_1, Q_inj]
                            c_sum = [c_1, c_inj]

                            
                            Q_sum = sum(Q_previous)

                            if Q_inj > 0:
                                c_sum = [(q / Q_sum) * c for q, c in zip(Q_previous, c_sum)]
                                C_IN = sum(c_sum)
                            else:
                                C_IN = c_1
                            
                           

                            u = get_X(t, u_col_all, i)
                            
                            # n_current_sub_zone = int(len(subzone[1]))

                        
                            # print(f'time: {t/60} min')
                            # print(f'coming from: col: {col_idx_feed} in bay: {feed_col_bays[0]}')
                            # print(f'going to: col: {i} in bay: {bay}')
                            # print(f'Q_previous: {Q_previous} col {col_idx_feed} [flowrate, desorbent_flow]')
                            # print(f'c_sum: {c_sum}\n\n')
                            # print(f'------------------')
                            # print(f'n_current_sub_zone: {len(subzone[1])}')
                            # print(f'Q before split: {Q_sum} cm3/s')
                            # print(f'C_IN: {C_IN}')
                            # print(f'Q (in col {i}) after split: {-1*u*A_col*0.56} cm3/s\n\n')

                        elif len(feed_col_bays) > 1:  # if that subzone has multiple feed bays
                            # print(f'please dont come in here!')
                            c_sum = []
                            Q_previous = []
                            # print(f'feed_col_bays: {feed_col_bays}')
                            for fb, feed_bay in enumerate(feed_col_bays):  # for each feed bay
                                #What is the number (ID) of the feed column of interest?:
                                col_idx_feed = get_column_idx_for_bae(t, feed_bay, t_schedule, Bay_matrix)  # get the column index using bae
                                # Use the column label to get the flowrate of the col
                                # print(f'time: {t/60} min')
                                # print(f'bay:{bay}')
                                # print(f'col_idx_feed: {col_idx_feed}\n\n')
                                Q_1 = get_X(t, Q_col_all, col_idx_feed)
                                # Again use the col_idx_feed; to get the concentration out of that column
                                c_1 = c[(col_idx_feed + 1) * nx_per_col - 1]

                                # Store this information
                                Q_previous.append(Q_1)
                                c_sum.append(c_1)

                                # Is there any flow event (F,R,X,D) happening after the 3 columns in the preceding feed bay
                                # if so, that information will be in the schedule matrix at the row corresponding to
                                # the col that is (one) position ahead of the last in the feed_bay set

                                if fb == len(feed_col_bays)-1:
                                    col_idx_ahead = (col_idx_feed + 1) % Ncol_num 
                                    # print(f'inj schedule from: {col_idx_ahead}')
                                    Q_inj = get_X(t, Q_pulse_all, col_idx_ahead)
                                    c_inj = get_C(t, Cj_pulse_all, col_idx_ahead, comp_idx)
                                    
                                    # Store
                                    if Q_inj > 0:
                                        Q_previous.append(Q_inj)
                                        c_sum.append(c_inj)

                            # print(f'time: {t/60} min')
                            # print(f'coming from: col: {col_idx_feed} in bay:{feed_bay}')
                            # print(f'going to: col: {i} in bay: {bay}')
                            # print(f'Q_previous: {Q_previous} [col {col_idx_feed} flowrate, desorbent]')
                            # print(f'c_sum: {c_sum} [conc out upstream cols, conc from #flow_event]')
                            # print(f'--------------------')

                            # print(f'Q_previous: {Q_previous}')

                            Q_sum = np.sum(Q_previous)

                             
                            weights = Q_previous / Q_sum
                            c_add_them_up = weights * c_sum
                            # print(f'Q_sum: {Q_sum} cm3/s')
                            # print(f'weights: {weights}')
                            # print(f'c_sum: {c_sum}, [perv cols, cinj]')   
                            # print(f'c_add_them_up: {c_add_them_up}')
                            # print(f'--------------------\n\n')

                            C_IN = np.sum(c_add_them_up)
                            


                            n_current_sub_zone = len(subzone[1])

                            u = get_X(t, u_col_all, i)

                            # print(f'Q before manifold: {Q_sum} cm3/s')
                            # print(f'Q (in col {i}) after manifold: {-1*u*A_col*0.56} cm3/s\n\n')

                        break  # done once subzone match is found

                if found_subzone ==  False:
      
                    Q_1 = get_X(t, Q_col_all, i-1)
                    Q_2 = get_X(t, Q_pulse_all, i)
                    W1 = Q_1 / (Q_1 + Q_2)
                    W2 = Q_2 / (Q_1 + Q_2)
                    u = get_X(t, u_col_all, i)
                    c_injection = get_C(t, Cj_pulse_all, i, comp_idx)


                    if Q_2 > 0:
                        C_IN = W1 * c[strt - 1] + W2 * c_injection
                    else:
                        C_IN = c[strt - 1]

                    # print(f'time: {t/60} min')
                    # print(f'From column: {i-1} in bay: {get_X(t, Bay_matrix, i-1)}')
                    # print(f'To column: {i}, in bay {bay}')
                    # print(f'[{Q_1}, {Q_2}]: [col {i} flowrate, feed/raff]\n\n') # get the column index using bae


                # Calcualte alpha, bata and gamma:
                # Da = get_X(t, D_col_all, i)
                Da = get_C(t, D_col_all, i, comp_idx)
                beta = 1 / alpha
                gamma = 1 - 3 * Da / (2 * u * dx)

                ##
                # R1 = ((beta * alpha) / gamma)
                R1 = ((beta *alpha) / gamma)
                R2 = ((2 * Da / (u * dx)) / gamma)
                R3 = ((Da / (2 * u * dx)) / gamma)
                ##

                # Calcualte the BC effects:
                
                c_BC[i] = R1 * C_IN - R2 * c[strt] + R3 * c[strt+1] # the boundary concentration for that node
                
            # print('c_BC:\n', c_BC)

            for k in range(len(c_BC)):
                # vec_add[start[k]]  = get_X(t,coef_0,k)*c_BC[k]
                # print(vec_add)
                vec_add[start[k]]  = get_C(t, coef_0_all, k, comp_idx)*c_BC[k]
                # print(f'vec_add: {vec_add}')

            return vec_add
            # print('np.shape(vect_add)\n',np.shape(vec_add(nx, c, start)))

        return component_coeff_matrix, vector_add(nx, c, start, comp_idx, Bay_matrix, subzone_set)


    
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
        MT = mass_transfer(kav_params[comp_idx], isotherm, q)
        #print('MT:\n', MT)

        coeff_matrix, vec_add = coeff_matrix_builder_UNC(t, Q_col_all, Q_pulse_all, dx, start, alpha, c, nx_col, comp_idx, Bay_matrix, subzone_set)
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
        coeff_matrix_A, vec_add_A = coeff_matrix_builder_UNC(t, Q_col_all, Q_pulse_all, dx, start, alpha, c[0:nx], nx_col, 0, Bay_matrix, subzone_set)
        coeff_matrix_B, vec_add_B = coeff_matrix_builder_UNC(t, Q_col_all, Q_pulse_all, dx, start, alpha, c[nx:2*nx], nx_col, 1, Bay_matrix, subzone_set)

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
            MT_comp = mass_transfer(kav_params[comp_idx], isotherm, q[IDX[comp_idx]: IDX[comp_idx] + nx ])
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
        P_mprofile_smooth = []
        P_cprofile_smooth = []

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
            print('Q_all_flows:\n', Q_all_flows)
            print('Q_all_flows:\n', np.shape(Q_all_flows))
            print(f't_idx_all_Q: {np.shape(t_idx_all_Q)}')


        for i in range(num_comp):# for each component

            if iso_type == 'UNC':
                # Search the ODE matrix
                C_R1_add = np.array(get_X_row( y_odes[i][:nx,:], row_start_matrix-1, jump_matrix, t_idx_all_Q[i])) # exclude q
                C_R2_add = np.array(get_X_row( y_odes[i][:nx,:], row_start_matrix, jump_matrix, t_idx_all_Q[i]))
                # # Search the Flowrate Schedule
                # P_vflows_1_add = np.array(get_X_row(Q_all_flows[i], row_start_schedule-1, jump_schedule, t_idx_all_Q[i]))
                # P_vflows_2_add = np.array(get_X_row(Q_all_flows[i], row_start_schedule, jump_schedule, t_idx_all_Q[i]))

            elif iso_type == 'CUP':

                C_R1_add = np.array(get_X_row( y_odes[i][:nx,:], row_start_matrix-1, jump_matrix, t_idx_all)) # exclude q
                C_R2_add = np.array(get_X_row( y_odes[i][:nx,:], row_start_matrix, jump_matrix, t_idx_all))
                # P_vflows_1_add = np.array(get_X_row(Q_all_flows, row_start_schedule-1, jump_schedule, t_idx_all_Q))
                # P_vflows_2_add = np.array(get_X_row(Q_all_flows, row_start_schedule, jump_schedule, t_idx_all_Q))


            # Raffinate Massflow Curves
            # print('C_R1_add.type():\n',type(C_R1_add))
            # print('np.shape(C_R1_add):\n', np.shape(C_R1_add))

            # print('P_vflows_1_add.type():\n',type(P_vflows_1_add))
            # print('np.shape(P_vflows_1_add):\n', np.shape(P_vflows_1_add))

            # Assuming only conc change accross port when (i) adding feed or (ii) desorbent
            C_R1_add = C_R2_add
            # P_mflows_1_add = C_R1_add * P_vflows_1_add  # (g/cm^3 * cm^3/s)  =>  g/s
            # P_mflows_2_add = C_R2_add * P_vflows_2_add  # g/s

            if row_start_matrix == row_start_matrix_raff:
                P_vflows_1_add = -QR*np.ones_like(C_R1_add)
                P_mflows_1_add = C_R1_add * P_vflows_1_add  # (g/cm^3 * cm^3/s)  =>  g/s

            elif row_start_matrix == row_start_matrix_ext:
                P_vflows_1_add = -QX*np.ones_like(C_R1_add)
                P_mflows_1_add = C_R1_add * P_vflows_1_add  # (g/cm^3 * cm^3/s)  =>  g/s




            # Flow profiles:

            # Volumetric cm^3/s
            P_vflow[i] = P_vflows_1_add #- P_vflows_2_add # cm^3

            # Integrate
            # Define rolling mean function
            def rolling_mean(y, window_size):
                return np.convolve(y, np.ones(window_size)/window_size, mode='same')

            # Apply rolling mean before integration
            window_size = 25  # Try 5, 7, 9 — experiment depending on how spiky the data is

            P_mflows_1_add_smooth = rolling_mean(P_mflows_1_add, window_size)
            C_R1_add_smooth =  rolling_mean(C_R1_add, window_size)
            # P_mflows_2_add_smooth = rolling_mean(P_mflows_2_add, window_size)

            if iso_type == 'UNC':
                m_P_add_1 = integrate.simpson(P_mflows_1, x=t_odes[i]) # g
                # m_P_add_2 = integrate.simpson(P_mflows_2_add, x=t_odes[i]) # g

            if iso_type == 'CUP':
                m_P_add_1 = integrate.simpson(P_mflows_1_add, x=t_odes) # g
                # m_P_add_2 = integrate.simpson(P_mflows_2_add, x=t_odes) # g



            # Storage
            # Concentration
            P_cprofile.append(C_R1_add) # g/s
            P_cprofile_smooth.append(C_R1_add_smooth) # g/s
            # Mass g/s
            P_mprofile.append(P_mflows_1_add) #- P_mflows_2_add) # g/s
            P_mprofile_smooth.append(P_mflows_1_add_smooth) #- P_mflows_2_add) # g/s

            C_P1.append(C_R1_add)  # Concentration Profiles
            C_P2.append(C_R2_add)

            P_vflows_1.append(P_vflows_1_add)
            # P_vflows_2.append(P_vflows_2_add)

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
    Error = abs(Model_Acc) - abs(Expected_Acc)

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




# Plotting Fucntions - if need be
###########################################################################################
# Loading the Plotting Libraries
from matplotlib.pyplot import subplots
import matplotlib.pyplot as plt
import matplotlib.animation as animation
# from PIL import Image
from scipy import integrate
import plotly.graph_objects as go
###########################################
# IMPORTING MY OWN FUNCTIONS
###########################################
# Loading the Plotting Libraries
from matplotlib.pyplot import subplots
import matplotlib.pyplot as plt
import matplotlib.animation as animation
# from PIL import Image
from scipy import integrate
import plotly.graph_objects as go
###########################################
# IMPORTING MY OWN FUNCTIONS
###########################################
def see_prod_curves(t_odes, Y, t_index) :
    # Y = C_feed, C_raff, C_ext
    # X = t_sets
    fig, ax = plt.subplots(1, 3, figsize=(25, 5))
    

    # 0 - Feed Profile
    # 1 - Raffinate Profile
    # 2 - Extract Profile
    t_odes  = t_odes/60/60
    # Concentration Plots
    for i in range(num_comp): # for each component
        if iso_type == "UNC":
            ax[0].plot(t_odes[i], Y[0][i], color = colors[i], label = f"{Names[i]}, {Names[i]}:{cusotom_isotherm_params_all[i]}, kh:{parameter_sets[i]['kh']}")
            ax[1].plot(t_odes[i], Y[1][i], color = colors[i], label = f"{Names[i]}, {Names[i]}:{cusotom_isotherm_params_all[i]}, kh:{parameter_sets[i]['kh']}")
            ax[2].plot(t_odes[i], Y[2][i], color = colors[i], label = f"{Names[i]}, {Names[i]}:{cusotom_isotherm_params_all[i]}, kh:{parameter_sets[i]['kh']}")
        
        elif iso_type == "CUP":    
            ax[0].plot(t_odes, Y[0][i], color = colors[i], label = f"{Names[i]}, {Names[i]}:{cusotom_isotherm_params_all[i]}, kh:{parameter_sets[i]['kh']}")
            ax[1].plot(t_odes, Y[1][i], color = colors[i], label = f"{Names[i]}, {Names[i]}:{cusotom_isotherm_params_all[i]}, kh:{parameter_sets[i]['kh']}")
            ax[2].plot(t_odes, Y[2][i], color = colors[i], label = f"{Names[i]}, {Names[i]}:{cusotom_isotherm_params_all[i]}, kh:{parameter_sets[i]['kh']}")
        
    # Add Accessories
    ax[0].set_xlabel('Time, hrs')
    ax[0].set_ylabel('($\mathregular{g/cm^3}$)')
    ax[0].set_title(f'Feed Concentration Curves\nConfig: {Z1}:{Z2}:{Z3}:{Z4},\nNumber of Cycles:{n_num_cycles}\nIndex Time: {t_index}s')
    # ax[0].legend()

    ax[1].set_xlabel('Time, hrs')
    ax[1].set_ylabel('($\mathregular{g/cm^3}$)')
    ax[1].set_title(f'Raffinate Elution Curves\nConfig: {Z1}:{Z2}:{Z3}:{Z4},\nNumber of Cycles:{n_num_cycles}\nIndex Time: {t_index}s')
    # ax[1].legend()

    ax[2].set_xlabel('Time, hrs')
    ax[2].set_ylabel('($\mathregular{g/cm^3}$)')
    ax[2].set_title(f'Extract Elution Curves\nConfig: {Z1}:{Z2}:{Z3}:{Z4},\nNumber of Cycles:{n_num_cycles}\nIndex Time: {t_index}s')
    # ax[2].legend()


    plt.show()

    # Volumetric Flowrate Plots
    fig, vx = plt.subplots(1, 2, figsize=(25, 5))
    for i in range(num_comp): # for each component
        if iso_type == "UNC":
            
            vx[0].plot(t_odes[i], Y[3][i], color = colors[i], label = f"{Names[i]}, {Names[i]}:{cusotom_isotherm_params_all[i]}, kh:{parameter_sets[i]['kh']}")
            vx[1].plot(t_odes[i], Y[4][i], color = colors[i], label = f"{Names[i]}, {Names[i]}:{cusotom_isotherm_params_all[i]}, kh:{parameter_sets[i]['kh']}")
        
        elif iso_type == "CUP":    
            
            vx[0].plot(t_odes, Y[3][i], color = colors[i], label = f"{Names[i]}, {Names[i]}:{cusotom_isotherm_params_all[i]}, kh:{parameter_sets[i]['kh']}")
            vx[1].plot(t_odes, Y[4][i], color = colors[i], label = f"{Names[i]}, {Names[i]}:{cusotom_isotherm_params_all[i]}, kh:{parameter_sets[i]['kh']}")
        
    # Add Accessories
    vx[0].set_xlabel('Time, hrs')
    vx[0].set_ylabel('($\mathregular{cm^3/s}$)')
    vx[0].set_title(f'Raffinate Volumetric Flowrates')
    vx[0].legend()

    vx[1].set_xlabel('Time, hrs')
    vx[1].set_ylabel('($\mathregular{cm^3/s}$)')
    vx[1].set_title(f'Extract Volumetric Flowrates')
    # vx[1].legend()

    plt.show()

def col_liquid_profile(t, y, Axis_title, c_in, Ncol_num, L_total):
    y_plot = np.copy(y)
    # # Removeing the BC nodes
    # for del_row in start:
    #     y_plot = np.delete(y_plot, del_row, axis=0)
        
    # print('y_plot:', y_plot.shape)
    
    x = np.linspace(0, L_total, np.shape(y_plot[0:nx, :])[0])
    dt = t[1] - t[0]
    

    
    # Start vs End Snapshot
    fig, ax = plt.subplots(1, 2, figsize=(25, 5))

    ax[0].plot(x, y_plot[:, 0], label="t_start")
    ax[0].plot(x, y_plot[:, -1], label="t_end")

    # Add vertical black lines at positions where i % nx_col == 0
    for col_idx in range(Ncol_num + 1):  # +1 to include the last column boundary
        x_pos = col_idx #nx_col + col_idx*nx_col + col_idx #col_idx * ((nx_col) * dx)
        #x_pos = dx * x_pos
        ax[0].axvline(x=x_pos, color='k', linestyle='-')
        ax[1].axvline(x=x_pos, color='k', linestyle='-')

    ax[0].set_xlabel('Column Length, m')
    ax[0].set_ylabel('($\mathregular{g/l}$)')
    ax[0].axhline(y=c_in, color='g', linestyle= '--', linewidth=1, label="Inlet concentration")  # Inlet concentration
    # ax[0].legend()

    # Progressive Change at all ts:
    for j in range(np.shape(y_plot)[1]):
        ax[1].plot(x, y_plot[:, j])
        ax[1].set_xlabel('Column Length, m')
        ax[1].set_ylabel('($\mathregular{g/l}$)')
    plt.show()


def col_solid_profile(t, y, Axis_title, Ncol_num, start, L_total):
    
    # Removeing the BC nodes
    y_plot = np.copy(y)
    # Removeing the BC nodes
    for del_row in start:
        y_plot = np.delete(y_plot, del_row, axis=0)
        
    # print('y_plot:', y_plot.shape)
    
    x = np.linspace(0, L_total, np.shape(y_plot[0:nx, :])[0])
    dt = t[1] - t[0]
    
    # Start vs End Snapshot
    fig, ax = plt.subplots(1, 2, figsize=(25, 5))

    ax[0].plot(x, y_plot[:, 0], label="t_start")
    ax[0].plot(x, y_plot[:, -1], label="t_end")
    # ax[0].plot(x, y_plot[:, len(t) // 2], label="t_middle")

    # Add vertical black lines at positions where i % nx_col == 0
    for col_idx in range(Ncol_num + 1):  # +1 to include the last column boundary
        x_pos = col_idx*L #nx_col + col_idx*nx_col + col_idx #col_idx * ((nx_col) * dx)
        #x_pos = dx * x_pos
        ax[0].axvline(x=x_pos, color='k', linestyle='-')
        ax[1].axvline(x=x_pos, color='k', linestyle='-')

    ax[0].set_xlabel('Column Length, m')
    ax[0].set_ylabel('($\mathregular{g/l}$)')
    ax[0].set_title(f'{Axis_title}')
    ax[0].legend()

    # Progressive Change at all ts:
    for j in range(np.shape(y_plot)[1]):
        ax[1].plot(x, y_plot[:, j])
        ax[1].set_xlabel('Column Length, m')
        ax[1].set_ylabel('($\mathregular{g/l}$)')
        ax[1].set_title(f'{Axis_title}')
    plt.show()  # Display all the figures 







#%%
# --------------- FUNCTION EVALUATION SECTION

# SMB VARIABLES
#######################################################
# What tpye of isoherm is required?
# Coupled: "CUP"
# Uncoupled: "UNC"
iso_type = "CUP"

###################### PRIMARY INPUTS #########################
# Define the names, colors, and parameter sets for 6 components
Names = ["Glucose", "Fructose"]#, 'C', 'D']#, "C"]#, "D", "E", "F"]
colors = ["green", "orange"]    #, "purple", "brown"]#, "b"]#, "r", "purple", "brown"]
num_comp = len(Names) # Number of components
e = 0.56         # bed voidage
Bm = 300

# Column Dimensions

# How many columns in each Zone?

Z1, Z2, Z3, Z4 = 6, 6, 6, 6 # *3 for smb config
zone_config = np.array([Z1, Z2, Z3, Z4])

# sub_zone information - EASIER TO FILL IN IF YOU DRAW THE SYSTEM
# -----------------------------------------------------
# sub_zone_j = [[feed_bays], [reciveinig_bays]]
# -----------------------------------------------------
# feed_bay = the bay(s) that feed the set of reciveing bays in "reciveinig_bays" e.g. [2] or [2,3,4] 
# reciveinig_bays = the set of bayes that recieve material from the feed bay

"""
sub-zones are counted from the feed onwards i.e. sub_zone_1 is the first subzone "seen" by the feed stream. 
Bays are counted in the same way, starting from 1 rather than 0
"""
sub_zone_1 = [[6], [1, 2]] # ---> in subzone 1, there are 2 columns stationed at bay 3 and 4. Bay 3 and 4 recieve feed from bay 1"""
sub_zone_11 = [[1, 2], [3]] 

sub_zone_2 = [[3], [4, 5]] 
sub_zone_22 = [[4, 5], [6]] 
# PACK:
subzone_set = [sub_zone_1, sub_zone_11, sub_zone_2, sub_zone_22]
subzone_set = [] # no subzoning

# PLEASE ASSIGN THE BAYS THAT ARE TO THE IMMEDIATE LEFT OF THE RAFFIANTE AND EXTRACT
# product_bays = [2, 5] # [raff, extract]



L = 30 # cm # Length of one column
d_col = 2 # cm # column internal diameter

# Calculate the radius
r_col = d_col / 2
# Calculate the area of the base
A_col = np.pi * (r_col ** 2) # cm^2
V_col = A_col*L # cm^3
# Dimensions of the tubing and from each column:
# Assuming the pipe diameter is 20% of the column diameter:
d_in = 0.2 * d_col # cm
nx_per_col = 12


################ Time Specs #################################################################################
t_index_min = 3.3 # min # Index time # How long the pulse holds before swtiching
n_num_cycles = 15    # Number of Cycles you want the SMB to run for
###############  FLOWRATES   #################################################################################

# Jochen et al:
Q_P, Q_Q, Q_R, Q_S = 5.21, 4, 5.67, 4.65 # x10-7 m^3/s
conv_fac = 0.1 # x10-7 m^3/s => cm^3/s
Q_P, Q_Q, Q_R, Q_S  = Q_P*conv_fac, Q_Q*conv_fac, Q_R*conv_fac, Q_S*conv_fac

Q_I, Q_II, Q_III, Q_IV = Q_R,  Q_S, Q_P, Q_Q

# Q_I, Q_II, Q_III, Q_IV = 2,1,2,1

Q_internal = np.array([Q_I, Q_II, Q_III, Q_IV])



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
parameter_sets = [
    {"kh": 0.0315, "C_feed": 0.2222},    # Glucose SMB Launch
    {"kh": 0.0217, "C_feed": 0.2222}] #, # Fructose

Da_all = np.array([6.218e-6, 6.38e-6 ]) 

# ISOTHERM PARAMETERS
####################################################################### ####################
# Uncomment as necessary:

# Linear, H
cusotom_isotherm_params_all = np.array([[0.27], [0.53]]) # H_glu, H_fru 
# Sub et al = np.array([[0.27], [0.53]])

# # Langmuir, [Q_max, b]
# cusotom_isotherm_params_all = np.array([[2.51181596, 1.95381598], [3.55314612, 1.65186647]])

# Linear + Langmuir, [H, Q_max, b]
# cusotom_isotherm_params_all = np.array([[1, 2.70420148, 1.82568197], [1, 3.4635919, 1.13858329]])


# STORE/INITALIZE SMB VAIRABLES
SMB_inputs = [iso_type, Names, colors, num_comp, nx_per_col, e, Da_all, Bm, zone_config, L, d_col, d_in, t_index_min, n_num_cycles, Q_internal, parameter_sets, cusotom_isotherm_params_all, subzone_set]

#%% ---------- SAMPLE RUN IF NECESSARY
start_test = time.time()
y_matrices, nx, t, t_sets, t_schedule, C_feed, m_in, m_out, raff_cprofile, ext_cprofile, raff_intgral_purity, raff_recov, ext_intgral_purity, ext_recov, raff_vflow, ext_vflow, Model_Acc, Expected_Acc, Error_percent = SMB(SMB_inputs)
end_test = time.time()

duration = end_test - start_test
# print(f'Simulation Took: {duration/60} min')
# print(f'ext_cprofile: {ext_cprofile}')
# print(f'raff_cprofile: {raff_cprofile}')
#%% Plotting


print("-----------------------------------------------------------")
Y = [C_feed, raff_cprofile, ext_cprofile, raff_vflow, ext_vflow]

# Y = [C_feed, ext_vflow, raff_vflow ]
if iso_type == "UNC":
    see_prod_curves(t_sets, Y, t_index_min*60)
elif iso_type == "CUP":
    see_prod_curves(t, Y, t_index_min*60)

# Define the data for the table
data = {
    'Metric': [
        'Total Expected Acc (IN-OUT)', 
        'Total Model Acc (r+l)', 
        'Total Error Percent (relative to Exp_Acc)', 
        'Mass In',
        'Mass Out',
        
        'Raffinate Purity [A, B,. . ]', 
        'Extract Purity [A, B,. . ]',
        'Raffinate Recovery[A, B,. . ]', 
        'Extract Recovery[A, B,. . ]'
    ],
    'Value': [
        f'{sum(Expected_Acc)} g', 
        f'{sum(Model_Acc)} g', 
        f'{Error_percent} %',

        f'{m_in} g',
        f'{m_out} g', 

        f'{raff_intgral_purity} %', 
        f'{ext_intgral_purity} %', 
        f'{raff_recov} %', 
        f'{ext_recov} %'
    ]
}

import pandas as pd
# Create a DataFrame
df = pd.DataFrame(data)

# Display the DataFrame
print(df)


# Plot the table as a figure
fig, ax = plt.subplots(figsize=(8, 4)) # Adjust figure size as needed
ax.axis('tight')
ax.axis('off')
table = ax.table(cellText=df.values, colLabels=df.columns, cellLoc='center', loc='center')

# Format the table's appearance
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1.5, 1.5)  # Adjust scaling of the table

# Display the table
plt.show()


#%%
print(f'shape(y): {np.shape(y_matrices)}')

#%%


# ANIMATION
###########################################################################################

import matplotlib.pyplot as plt
import matplotlib.animation as animation

def animate_smb_concentration_profiles(y, t, labels, colors, nx_per_col, cols_per_zone, L_col,
                                       t_index, parameter_sets, filename="smb_profiles.mp4"):
    """
    Create an animated visualization of SMB liquid-phase concentration profiles across zones.

    Parameters:
    - y: (n_components, nx_total, time_points) concentration data
    - t: time vector (in seconds)
    - labels: list of component names
    - colors: list of colors per component
    - nx_per_col: number of spatial points per column
    - cols_per_zone: list with number of columns per zone
    - L_col: length of each column (m)
    - t_index: time (s) for 1 indexing shift
    - parameter_sets: list of dicts per component
    - filename: output video file name
    """
    n_components, nx_total, nt = y.shape
    n_zones = len(cols_per_zone)
    n_cols_total = sum(cols_per_zone)
    L_total = n_cols_total * L_col

    # Determine frame indices for animation (≤ 90s total shown if t > 120s)
    if t[-1] > 120:
        t_segment = 30  # seconds
        frames_per_segment = int(t_segment / (t[1] - t[0]))
        first_idx = np.arange(0, frames_per_segment)
        middle_idx = np.arange(nt // 2 - frames_per_segment // 2, nt // 2 + frames_per_segment // 2)
        last_idx = np.arange(nt - frames_per_segment, nt)
        selected_frames = np.concatenate([first_idx, middle_idx, last_idx])
    else:
        selected_frames = np.arange(nt)

    # Calculate column junction positions
    col_boundaries = [i * nx_per_col for i in range(n_cols_total + 1)]
    x_full = np.linspace(0, L_total, nx_total)

    # Initial port positions (index in spatial array), assuming inlet at col 0 (zone 3)
    stream_order = ["Feed", "Extract", "Raffinate", "Desorbent"]
    stream_colors = ["red", "blue", "orange", "purple"]
    stream_zone = [2, 1, 3, 0]  # zone indices: Feed at zone 3 (index 2), etc.

    # Compute starting port positions in terms of column number
    start_ports = np.cumsum([0] + cols_per_zone[:-1])  # column index per zone start

    # Pre-compute port positions over time (indexed every t_index)
    port_positions = {stream: [] for stream in stream_order}
    for time_val in t:
        idx_shift = int(time_val // t_index)
        for i, stream in enumerate(stream_order):
            base_col = start_ports[stream_zone[i]]
            pos = (base_col + idx_shift) % n_cols_total
            port_positions[stream].append(pos * nx_per_col * L_col)  # convert to length

    # Set up figure and axes (4 stacked panels)
    fig, axes = plt.subplots(n_zones, 1, figsize=(8, 10), sharex=True)
    lines = [[] for _ in range(n_zones)]

    for zone_id, ax in enumerate(axes):
        ax.set_xlim(0, L_total)
        ax.set_ylim(0, np.max(y))
        ax.set_ylabel("C (g/L)")
        ax.set_title(f"Zone {zone_id + 1}")

        # Vertical black lines for column boundaries
        col_start = sum(cols_per_zone[:zone_id]) * nx_per_col * L_col
        for i in range(cols_per_zone[zone_id] + 1):
            ax.axvline(x=col_start + i * L_col, color='black', linewidth=0.5)

        # Plot initialization for each component
        for comp_idx in range(n_components):
            # Spatial slice for this zone
            start = sum(cols_per_zone[:zone_id]) * nx_per_col
            end = start + cols_per_zone[zone_id] * nx_per_col
            x = x_full[start:end]
            line, = ax.plot(x, y[comp_idx, start:end, 0], color=colors[comp_idx], label=labels[comp_idx])
            lines[zone_id].append(line)

    # Add time and legend box
    time_text = axes[0].text(0.95, 0.9, '', transform=axes[0].transAxes,
                             ha='right', va='top', fontsize=10, bbox=dict(facecolor='white', alpha=0.7))
    axes[-1].set_xlabel("Column Length (m)")
    axes[0].legend(loc='upper left', bbox_to_anchor=(1, 1))

    # Initialize stream vertical lines
    stream_lines = [axes[0].axvline(0, color=color, linestyle='--', linewidth=1.5) for color in stream_colors]

    # Update function
    def update(frame_idx):
        t_hr = t[frame_idx] / 3600  # convert to hours
        time_text.set_text(f"Time: {t_hr:.2f} h")

        for zone_id, ax in enumerate(axes):
            start = sum(cols_per_zone[:zone_id]) * nx_per_col
            end = start + cols_per_zone[zone_id] * nx_per_col
            for comp_idx in range(n_components):
                lines[zone_id][comp_idx].set_ydata(y[comp_idx, start:end, frame_idx])

        # Update stream lines (positioned in top axis only)
        for i, stream in enumerate(stream_order):
            x_pos = port_positions[stream][frame_idx]
            stream_lines[i].set_xdata(x_pos)

        return [l for sublist in lines for l in sublist] + stream_lines + [time_text]

    # Create animation
    ani = animation.FuncAnimation(fig, update, frames=selected_frames, interval=100, blit=True)

    # Save animation
    writer = animation.FFMpegWriter(fps=15, bitrate=1800)
    ani.save(filename, writer=writer)
    plt.close()
    return filename

# Run it with the simulated data
sample_data_bundle = {
    "y": y_matrices,
    "t": t,
    "labels": Names,
    "colors": colors,
    "nx_per_col": nx_per_col,
    "cols_per_zone": zone_config,
    "L_col": L,
    "t_index": t_index_min,
    "parameter_sets": parameter_sets
}

def plot_all_columns_single_axes(y_matrices, t,indxing_period, time_index,
                                  nx_per_col, L_col, zone_config, 
                                  labels=None, colors=None,
                                  title="Concentration Across Entire Column"):
    """
    Plot all columns together on one continuous axis for each component at a given time index.

    Parameters:
    - y_matrices: array of shape (n_components, nx_total, n_timepoints)
    - time_index: int, index along the time axis
    - nx_per_col: int, spatial points per column
    - L_col: float, physical length of each column
    - labels: list of component names (optional)
    - colors: list of line colors per component (optional)
    - title: overall figure title
    """

    n_components, nx_total, n_timepoints = y_matrices.shape

    n_columns = np.sum(zone_config)
    nx_total = nx_per_col*n_columns # just for the liquid phase

    total_length = L_col * n_columns # cm
    x = np.linspace(0, total_length, nx_total)
    # Rotate so that Z3 is first
    zone_config_rot = np.roll(zone_config, -2)  # => [Z3, Z4, Z1, Z2]

    # Labels accordingly
    zone_labels = ['Z3', 'Z4', 'Z1', 'Z2']

    plt.figure(figsize=(10, 6))
    for i in range(n_components):
        y_vals = y_matrices[i][ 0:nx_total, time_index]
        label = labels[i] if labels else f"Comp {i+1}"
        color = colors[i] if colors else None
        plt.plot(x, y_vals, label=label, color=color)
    x_plot = 0
    for i, x_zone in enumerate(zone_config_rot):
        x_plot += x_zone
        plt.axvline(x=x_plot*L, color='k', linestyle='--', linewidth=2)
        plt.text(x_plot*L, plt.ylim()[1]*0.995, zone_labels[i], ha='right', va='top',
                fontsize=8, color='k')
    for i in range(0,n_columns+1):
        x_plot = i*L
        plt.axvline(x=x_plot, color='grey', linestyle='--', linewidth=1)

    plt.title(f"{title} (Time Index {time_index}) (Time Stamp: {t[time_index]/60} min)\nIndxing_period: {indxing_period} min")
    plt.xlabel("Position along full unit (cm)")
    plt.ylabel("Concentration (g/L)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


#%%
# animate_smb_concentration_profiles(**sample_data_bundle)
plot_all_columns_single_axes(
    y_matrices=y_matrices,
    t = t,
    indxing_period = t_index_min,
    # time_index= int(np.round(np.shape(y_matrices)[2]*0.01)),
    time_index= 100, # 27, 31, 35. 70, 90
    nx_per_col=nx_per_col,
    L_col = L,
    zone_config = zone_config,
    labels=['A', 'B'],
    colors=['green', 'orange']
)




# %%



# %%
