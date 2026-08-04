#%%
from uqpylab import sessions, display_util
import numpy as np
import matplotlib.pyplot as plt
from RPrepo import*
# Load display defaults for UQ[py]Lab plots
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

# InputOpts = {
#     "Marginals": [
#         {"Name": "x1", "Type": "Gaussian", "Moments": [0, 1]},
#         {"Name": "x2", "Type": "Gaussian", "Moments": [0, 1]},
#     ]
# }

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

# #Validation
# #Create a validation sample of size $10^3$:
# Xval = uq.getSample(myInput, 1e3)

# #Evaluate the full model responses at the validation set points:
# Yval = uq.evalModel(myModel, Xval)

# #Evaluate the Kriging predictor mean at the validation set points:
# YKRGmean = uq.evalModel(myKriging, Xval)

# To visually assess the performance of the Kriging metamodel, create a 
# scatter plot of the Kriging predictions (i.e., the mean) vs. the true responses
# on the validation set:
# plt.scatter(Yval, YKRGmean, marker='o',alpha=.7, s=5)
# plt.plot([np.min(Yval), np.max(Yval)], [np.min(Yval), np.max(Yval)], 'k')
# plt.grid('True')
# plt.xlabel('$\\mathrm{Y^{true}}$')
# plt.ylabel('$\\mathrm{\\mu_{\\widehat{Y}}}^{KRG}$')
# plt.tick_params(axis='both')
# plt.xlim(np.min(Yval), np.max(Yval))
# plt.ylim(np.min(Yval), np.max(Yval))
# plt.gca().set_aspect('equal', adjustable='box')
# plt.show()

def kriging_wrapper(x):
    # uq.evalModel for Kriging returns prediction mean and variance
    mean, var = uq.evalModel(myKriging, x, nargout=2)
    
    # if isinstance(res, (list, tuple)) and len(res) >= 2:
    #     y_mean, y_var = res[0], res[1]
    # else:
    #     y_mean = res
    #     y_var = np.zeros_like(y_mean) + 1e-10  # Fallback minimal variance
        
    std = np.sqrt(var)  # Standard deviation (epistemic uncertainty)
    return mean.flatten(), std.flatten()

d = 5
t_star = 0
N = 1000
p0 = 0.1
out_d = 'norm'
inp_d = [
    {'name': 'uniform',  'parameters': [70, 10]},         
    {'name': 'norm',     'parameters': [39, 0.1]},         
    {'name': 'gumbel_r', 'parameters': [1342.0, 272.9]},   
    {'name': 'norm',     'parameters': [400, 0.1]},        
    {'name': 'norm',     'parameters': [250000.0, 35000.0]}
]

# inp_d = [   
#     {'name': 'norm',     'parameters': [0, 1]},        
#     {'name': 'norm',     'parameters': [0, 1]}
# ]

print("Started P-SuS execution using GPE.")

p_F_PSuS, p_F_IPSuS, info = psus(
    func=kriging_wrapper, 
    d=d, 
    t_star=t_star, 
    n=N, 
    p=p0, 
    out_dist=out_d, 
    inp_dist=inp_d,
)

print("Completed P-SuS execution using GPE.")

mySession.quit()



# %%
