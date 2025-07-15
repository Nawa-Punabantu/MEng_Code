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
from SMB_func_bh import SMB

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
# Da_all = np.array([3.892e-6, 2.99187705e-6]) 
# kav_params_all = np.array([[1.07122526], [1.98350338]])
# cusotom_isotherm_params_all = np.array([[2.40846686], [1.55994115]]) # [ [H_borate], [H_hcl] ]


# # PCR Linear Isotherm
Da_all = np.array([5.77e-7, 2.3812e-6]) 
kav_params_all = np.array([[0.54026], [2.171826]])
cusotom_isotherm_params_all = np.array([[3.6124333], [2.4640415]]) # [ [H_borate], [H_hcl] ]

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
    with open("PCR-borhcl-lin-type1_60iter_all_inputs.json", "w") as f:
        json.dump(all_inputs_list, f, indent=4)

    # SAVE recoveries_and_purities to JSON:
    with open("PCR-borhcl-lin-type1_60iter_all_outputs.json", "w") as f:
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
