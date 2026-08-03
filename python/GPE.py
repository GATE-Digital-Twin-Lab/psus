from uqpylab import sessions, display_util
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

display_util.load_plt_defaults()

mySession = sessions.cloud()
uq = mySession.cli
mySession.reset()

uq.rng(0, 'twister')
np.random.seed(0)

ModelOpts = {
    'Type': 'Model',
    'ModelFun': 'RPrepo.RP14',
}
myModel = uq.createModel(ModelOpts)

InputOpts = {
    "Marginals": [
        {"Name": "x1", "Type": "Uniform", "Parameters": [70, 80]},
        {"Name": "x2", "Type": "Gaussian", "Moments": [39, 0.1]},
        {"Name": "x3", "Type": "Gumbel", "Parameters": [1342, 272.9]},
        {"Name": "x4", "Type": "Gaussian", "Moments": [400, 0.1]},
        {"Name": "x5", "Type": "Gaussian", "Moments": [250_000, 35_000]},
    ]
}
myInput = uq.createInput(InputOpts)

MetaOpts = {
    'Type': 'Metamodel',
    'MetaType': 'Kriging',
    'Input': myInput['Name'],
    'FullModel': myModel['Name'],
    'ExpDesign': {
        'Sampling': 'LHS',
        'NSamples': 50
    },
    'Corr': {
        'Family': 'Matern-5_2'
    }
}

myKriging = uq.createModel(MetaOpts)
uq.print(myKriging)

ED_X = np.array(myKriging['ExpDesign']['X'])
ED_Y = np.array(myKriging['ExpDesign']['Y']).flatten()

initial_theta = np.array(myKriging['Kriging']['theta'][0]).flatten()[0]

def manual_loo_cv(theta_val, X_design, Y_design):
    
    n = len(Y_design)
    loo_means = np.zeros(n)
    loo_vars = np.zeros(n)
    
    theta_list = np.atleast_1d(theta_val).tolist()
    
    for i in range(n):
        train_idx = np.setdiff1d(np.arange(n), i)
        X_train = X_design[train_idx, :]
        Y_train = Y_design[train_idx]
        
        X_val = X_design[i, :].reshape(1, -1)
        
        fold_opts = {
            'Type': 'Metamodel',
            'MetaType': 'Kriging',
            'Input': myInput['Name'],
            'ExpDesign': {
                'X': X_train.tolist(),
                'Y': Y_train.tolist()
            },
            'Corr': {'Family': 'Matern-5_2'},
            'Kriging': {
                'ThetaInit': theta_list  
            }
        }
        
        temp_model = uq.createModel(fold_opts)
        
        mean_eval, var_eval = uq.evalModel(temp_model, X_val)
        
        loo_means[i] = mean_eval.flatten()[0]
        loo_vars[i] = var_eval.flatten()[0]
        
        uq.cmd(f"clear {temp_model['Name']}")
        
    return loo_means, loo_vars

print("Computing Baseline LOO distribution (P0) via Manual Loop...")
mu_init, var_init = manual_loo_cv(initial_theta, ED_X, ED_Y)
sigma_init = np.sqrt(np.maximum(var_init, 1e-16))
print(mu_init, var_init)
