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
from SMB_func_general import SMB

def opt_func(batch, SMB):
    # UNPACK "batch":
    opt_inputs = batch[0]
    SMB_inputs = batch[1]
    
    # UNPACK RESPECTIVE INPUTS
    Description, save_name_inputs, save_name_outputs, job_max_or_min, t_reff, Q_max, Q_min, m_max, m_min, sampling_budget, optimization_budget, constraint_threshold, PF_weight, bounds, triangle_guess = opt_inputs[0:]
    iso_type, Names, color, num_comp, nx_per_col, e, Da_all, Bm, zone_config, L, d_col, d_in, t_index_min, n_num_cycles, Q_internal, parameter_sets, cusotom_isotherm_params_all, kav_params_all, subzone_set, t_simulation_end = SMB_inputs[0:]


    # SECONDARY VARIABLES

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


    # - - - - -
    m_diff_max = Q_max/(V_col*(1-e))
    m_diff_min = Q_min/(V_col*(1-e))
    
    # - - - - -

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
                m2 = 0.8
                m3 = m2 + 0.1
                samples[i, 1] = m2
                samples[i, 2] = m3  # apex of trianlge
            else:
                m3 = np.random.uniform(m2, m_max)   

            samples[i, 2] = m3
            samples[i, -2] = m2 - 0.3
            samples[i,-1] = 0.6
        return samples


    # ---------- Objective Function

    def mj_to_Qj(mj, t_index_min):
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
            m1, m2, m3, m4, t_index_min = float(X[0]), float(X[1]), float(X[2]), float(X[3]), float(X[4])

            print(f'[m1, m2, m3, m4]: [{m1}, {m2}, {m3}, {m4}], t_index: {t_index_min}')

            Q_I, Q_II, Q_III, Q_IV = mj_to_Qj(m1, t_index_min), mj_to_Qj(m2, t_index_min), mj_to_Qj(m3, t_index_min), mj_to_Qj(m4, t_index_min)
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
                m1, m2, m3, m4, t_index_min = float(X[i,0]), float(X[i,1]), float(X[i,2]), float(X[i,3]), float(X[i,4])

                print(f'[m1, m2, m3, m4]: [{m1}, {m2}, {m3}, {m4}], t_index: {t_index_min}')
                Q_I, Q_II, Q_III, Q_IV = mj_to_Qj(m1, t_index_min), mj_to_Qj(m2, t_index_min), mj_to_Qj(m3, t_index_min), mj_to_Qj(m4, t_index_min) 
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
    def generate_initial_data(triangle_guess, sampling_budget=1):

        # generate training data
        # print(f'Getting {sampling_budget} Samples')
        # train_x = lhq_sample_mj(0.2, 1.7, n, diff=0.1)
        # train_x = fixed_feed_lhq_sample_mj(t_index_min, Q_fixed_feed, 0.2, 1.7, n, diff=0.1)
        train_all = fixed_m1_and_m4_lhq_sample_mj(m_max, m_min, m_min, m_max, sampling_budget, 1, diff=0.1)
        # print(f'train_all: {train_all}')
        # print(f'Done Getting {sampling_budget} Samples')

        # print(f'Solving Over {sampling_budget} Samples')
        if len(triangle_guess) == True:
            Rec, Pur, mjs = obj_con(triangle_guess)
            # print(f'Rec: {Rec}, Pur: {Pur}')
            # print(f'Done Getting {sampling_budget} Samples')
            all_outputs = np.hstack((Rec, Pur))
            return triangle_guess, all_outputs
        else:
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
    def constrained_BO(optimization_budget, bounds, all_initial_inputs, all_initial_ouputs, job_max_or_min, constraint_thresholds, xi):

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
            # def constraint_fixed_feed(x):
            #     return (x[2] - x[1]) - (Q_fixed_feed / ((V_col * (1-e)) / get_safe_tindex(x)))

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
            def passes_manual_check(vec):
                # Extract m1–m4
                m1, m2, m3, m4 = vec[0:4]
                # Check your "down–up–down–up" pattern
                return (m1 > m2 and m3 > m2 and m4 < m3 and m4 < m1)
            

            attempt = 0
            x_new = None
            max_attempts = 3

            while attempt < max_attempts:
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

                x_candidate = result.x # [m1, m2, m3, m4, t_index_min]

                if passes_manual_check(x_candidate):
                    x_new = x_candidate
                    break
                else:
                    print(f"Attempt {attempt+1}: Failed manual pattern check.")
                    attempt += 1
                
                # If still invalid after all attempts, tweak vector slightly
                if x_new is None:                    
                    print("Tweaking vector to satisfy pattern...")
                    x_new = x_candidate.copy()
                    m1, m2, m3, m4 = x_new[0:4]

                    # Adjust values to enforce pattern:
                    if not (m1 > m2): m1 = m2 + abs(m2)*0.1 + 1e-6
                    if not (m3 > m2): m3 = m2 + abs(m2)*0.1 + 1e-6
                    if not (m4 < m3): m4 = m3 - abs(m3)*0.1 - 1e-6
                    if not (m4 < m1): m4 = min(m4, m1 - abs(m1)*0.1 - 1e-6)

                    x_new[0:4] = [m1, m2, m3, m4]
            
            
            # print(f"x_new: {x_new}") *t_reff
            x_new[-1] = x_new[-1]*t_reff
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

            print(f"Gen {gen+1} Status:\n | Sampled Inputs:{x_new[:-1]}, {x_new[-1]*t_reff} [m1, m2, m3, m4, t_index]|\n Outputs: G_f1: {f_new[0]*100} %, F_f2: {f_new[1]*100} % | GPur, FPur: {c_new[0]*100}%, {c_new[1]*100}%")

        return f1_vals, f2_vals, c1_vals , c2_vals , all_inputs




    #%%
    # --------------- FUNCTION EVALUATION SECTION
    # ---------- SAMPLE RUN IF NECESSARY
    # start_test = time.time()
    # results = SMB(SMB_inputs)
    # # ref:  [y_matrices, nx, t, t_sets, t_schedule, C_feed, m_in, m_out, raff_cprofile, ext_cprofile, raff_intgral_purity, raff_recov, ext_intgral_purity, ext_recov, raff_vflow, ext_vflow, Model_Acc, Expected_Acc, Error_percent]
    # # STORE
    # Raffinate_Purity = results[10]
    # Raffinate_Recovery = results[11]
    # Extract_Purity = results[12]
    # Extract_Recovery = results[13]
    # Mass_Balance_Error_Percent = results[-1]
    # m_in = results[6]
    # m_out = results[7]
    # Model_Acc =  results[-3]
    # Expected_Acc = results[-2]
    # raff_cprofile = results[8]
    # ext_cprofile= results[9]
    # import matplotlib.pyplot as plt

    # # Plotting the data
    # plt.plot(results[2]/60/60, raff_cprofile[0], label='Raff CProfile 0')
    # plt.plot(results[2]/60/60, raff_cprofile[1], label='Raff CProfile 1')
    # plt.plot(results[2]/60/60, ext_cprofile[0], label='Ext CProfile 0')
    # plt.plot(results[2]/60/60, ext_cprofile[1], label='Ext CProfile 1')

    # # Adding labels and title
    # plt.xlabel('Time, hrs')
    # plt.ylabel('g/mL')
    # plt.title('Comparison of Raff and Ext CProfiles')

    # # Adding legend
    # plt.legend()

    # # Display the plot
    # plt.grid(True)
    # plt.tight_layout()
    # plt.show()


    # end_test = time.time()
    # test_duration = end_test-start_test

    # # DISPLAY
    # print(f'\n\n TEST RESULTS : \n')
    # print(f'Time Taken for 1 SMB Run: {test_duration/60} min')
    # print(f'Model_Acc: {Model_Acc}')
    # print(f'Expected_Acc: {Expected_Acc}')


    # print(f'm_in: {m_in} g')
    # print(f'm_out: {m_out} g ')
    # print(f'Raffinate_Recovery: {Raffinate_Recovery} ')
    # print(f'Extract_Recovery:  {Extract_Recovery}')
    # print(f'Raffinate_Purity: {Raffinate_Purity} ')
    # print(f'Extract_Purity: {Extract_Purity}')
    # print(f'Mass_Balance_Error_Percent: {Mass_Balance_Error_Percent}%')

    #%%

    # ----- MAIN ROUTINE
    if __name__ == "__main__":
        Q_max = Q_max/3.6 # l/h => ml/s
        Q_min = Q_min/3.6 # l/h => ml/s
        # SUMMARY
        print(f'\n\n OPTIMIZATION INPUTS: \n')
        print(f'Column Volume: {V_col} cm^3 | {V_col/1000} L')
        print(f'Column CSA: {A_col} cm^2')
        print(f'Column Length: {L} cm')
        print(f'Column Diameter: {d_col} cm')
        print(f'Optimization Budget: {optimization_budget}')
        print(f'Sampling Budget: {sampling_budget}')
        print(f'[Q_min, Q_max] = [{Q_max*3.6}, {Q_min*3.6}] L/h')
        print(f"bounds:\nm1: ({bounds[0][0]}, {bounds[0][1]})\nm2: ({bounds[1][0]}, {bounds[1][1]})\nm3: ({bounds[2][0]}, {bounds[2][1]})\nm4: ({bounds[3][0]}, {bounds[3][1]})\nt_index: ({bounds[4][0]*t_reff}, {bounds[4][1]*t_reff}) min")
            
    #%%
        # generate iniital samples
        all_initial_inputs, all_initial_outputs = generate_initial_data(triangle_guess, sampling_budget)
        print(f'all_initial_inputs\n{ all_initial_inputs}')
        print(f'all_initial_outputs\n{ all_initial_outputs}')

    #%%
        # OPTIMIZATION
        print(f'running opt')
        f1_vals, f2_vals, c1_vals, c2_vals, all_inputs  = constrained_BO(optimization_budget, bounds, all_initial_inputs, all_initial_outputs, job_max_or_min, constraint_threshold, 0.001)

    #%%
        # ----------- SAVE
        # Convert NumPy array to list

        # Inputs:
        all_inputs_list = all_inputs.tolist()
        # Outputs:
        data_dict = {
            "Description": Description,
            "f1_vals": f1_vals.tolist(),
            "f2_vals": f2_vals.tolist(),
            "c1_vals": c1_vals.tolist(),
            "c2_vals": c2_vals.tolist(),
        }



        # SAVE all_inputs to JSON:
    
        with open(save_name_inputs, "w") as f:  
            json.dump(all_inputs_list, f, indent=4)

        # SAVE recoveries_and_purities to JSON:
        with open(save_name_outputs, "w") as f:
            json.dump(data_dict, f, indent=4)
        
        print(f'Saved Sucessfully')


    # %%
    
    # return f1_vals, f2_vals, c1_vals, c2_vals, all_inputs


# call the respective input files

from opt_input_data_file import opt_batches 

for b_idx, batch in enumerate(opt_batches):
    
    print(f'\n\n----------------------------------------------------------')
    print(f'Optimizring Batch {b_idx +1}/{len(opt_batches)}')
    print(f'----------------------------------------------------------')
    opt_func(batch, SMB)