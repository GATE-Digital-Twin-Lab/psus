from psus_test import psus
import numpy as np
import scipy.stats as sts

from pnm_surrogate import branin, vectorise_uncertainty

#%%
def tiny_pf_func(x):
    # x in [0,1]^d
    # very small probability of exceeding threshold
    # let output = sum(x) + small noise
    y = np.sum(x, axis=1)
    p1 = y                      # mean parameter
    p2 = 0.05*np.ones_like(y)   # small variance
    return p1, p2
    
#%% Set up simple test
func = tiny_pf_func
d = 2
t_star = 1.95
n = 500          # samples per level
p = 0.1          # conditional probability
out_dist = "norm"
inp_dist = {'name':'uniform', 'parameters':[0,1]}

p_F, dict_out_F = psus(func, d, t_star, n, p, out_dist, inp_dist)

#%% Monte Carlo check
np.random.seed(1)
X = getattr(sts, inp_dist['name'])(*inp_dist['parameters']).rvs((10_000,2))

y, u = tiny_pf_func(X)
p_F_MC = np.mean(y >= t_star)

#%% Prints
print("============== P-SuS ===============")
print("Probability of failure, p_F:", p_F['p_F'])
print("p_F mean:", p_F['mean'])
print("p_F variance:", p_F['var'])

print("\n============== Monte Carlo ===============")
print("Probability of failure, p_F_MC:", p_F_MC)



#%% Set up Branin
d = 2
t_star = 230
N = 1000
p0 = 0.1

out_d = 'norm'
inp_d = {'name': 'uniform', 'parameters':[0,1]}

func = lambda std: lambda x: vectorise_uncertainty(branin, x, std=std)

#%% Run P-SuS with model uncertainty reducing based on computational resources
std = [2,1.5,1,0.5,0.25,0.1,0.05,0.01];

p_F_list = []
info_list = []

for s in std:
    a, b = psus(func(s), d, t_star, N, p0, out_d, inp_d);
    p_F_list.append(a)
    info_list.append(b)
    
    print("Iteration for sd = {s} completed.")
