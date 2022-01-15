### Synthetic Control Simulation 

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
np.random.seed(91)

from toolz import partial
from scipy.optimize import fmin_slsqp


def loss_function(w, X1,X0, V):
    
    W = np.matrix(w).T
    matrix_value =  (X1 - X0*W).T * V * (X1 - X0*W)
    return np.array(matrix_value).flatten()[0]


def get_w(X0, X1, V):
    
    w_start = (1/X0.shape[1])*np.ones(X0.shape[1])
    weights = fmin_slsqp(partial(loss_function, X1 = X1, X0 = X0, V = V),
                         w_start,
                         f_eqcons=lambda x: np.sum(x) - 1,
                         bounds=[(0.0, 1.0)]*len(w_start),
                         disp=False)
    return weights

def find_best_V_w(X0, X1, Y_donors, Y1, n_features, search_space = None, search_size = 1000):
    """Find V that gives the best prediction error for pre-treatment periods.
    This is slow because the V algorithm is not sophisticated."""


    if search_space == None:
        search_space = np.random.dirichlet(np.ones(n_features), size = search_size)
    
    # Find V giving best MSPE across random_v candidates
    best_SPE = 10**30

    for v in search_space:
        V_candidate = np.matrix( np.diag( v ) ) # (a,b,Y) x (0,1,2,3) importances

        w_of_V = get_w(X0, X1, V_candidate)
        W_of_V = np.matrix(w_of_V).T

        SPE = np.sum(np.array(Y1 - Y_donors*W_of_V)**2)
        if SPE < best_SPE:
            best_SPE = SPE
            best_v = v
            best_w = w_of_V
            
    V_star = np.matrix(np.diag(best_v))

    return V_star, best_w


def synthetic_control(data,
                        features,
                        treated_unit,
                        units,
                        periods,
                        pretreatment_periods,
                        n_features,
                     search_size = 1000):
    
    pretreatment_cols = []
    for i in list(data):
        i2 = i
        for x in features:
            i2 = i2.lstrip(x)
        if int(i2) in pretreatment_periods:
            pretreatment_cols.append(i)
            
    pretreatment_data = data[pretreatment_cols]
    
    X1 = np.matrix(pretreatment_data.loc[treated_unit]).T
    X0 = np.matrix(pretreatment_data.loc[units]).T
    X0_all_periods = np.matrix(data.loc[units]).T
    
    outcome_cols = [x for x in list(data) if 'Y' in x]
    Y_donors = np.matrix(data.loc[list(units), outcome_cols]).T
    Y1 = np.matrix(data.loc[[treated_unit], outcome_cols]).T

    
    #return X1, X0, pretreatment_data
    V_star, estimated_weights = find_best_V_w(X0, X1, Y_donors, Y1, n_features, search_size = search_size)
    synthetic_unit = X0_all_periods*np.matrix(estimated_weights).T
    
    synth_outcomes = np.array(synthetic_unit[-len(periods):]).flatten()
    true_outcomes = data.loc[treated_unit, outcome_cols].values
    #treated_outcomes = data.loc[treated_unit + 'treat', outcome_cols].values
    
    return synth_outcomes, true_outcomes, V_star, estimated_weights



def rmspe_ratio(synth_outcomes, true_outcomes, treatment_start):
    
    pre_errors = true_outcomes[0:treatment_start] - synth_outcomes[0:treatment_start]
    pre_rmspe = np.sum(pre_errors**2)/treatment_start
    
    post_errors = true_outcomes[treatment_start:] - synth_outcomes[treatment_start:]
    post_rmspe = np.sum(post_errors**2)/ len(post_errors)
    
    #print(post_rmspe, pre_rmspe)
    return post_rmspe / pre_rmspe