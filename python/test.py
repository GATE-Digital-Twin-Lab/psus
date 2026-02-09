from pnm_surrogate import (generate_uncertainty, vectorise_uncertainty,
                          plot_uncertainty, branin)
from psus_modified import psus

import numpy as np
import scipy.stats as sts
import matplotlib.pyplot as plt

def tiny_pf_func(x):
    # x in [0,1]^d
    y = np.sum(x, axis=1)
    p1 = y                      
    p2 = 0.05*np.ones_like(y)   
    return p1, p2
    
func = tiny_pf_func
d = 2
t_star = 1.9
n = 2000          # samples per level
p = 0.1          # conditional probability
out_dist = "norm"
inp_dist = {'name':'uniform', 'parameters':[0,1]}

p_F, p_F_i, p_F_p, p_F_pqd, dict_out_F, L = psus(func, d, t_star, n, p, out_dist, inp_dist)
np.random.seed(1)
X = getattr(sts, inp_dist['name'])(*inp_dist['parameters']).rvs((10_000,2))

y, u = tiny_pf_func(X)
p_F_MC = np.mean(y >= t_star)

print("============== P-SuS ===============")
print("Probability of failure, p_F:", p_F['p_F'])
print("p_F mean:", p_F['mean'])
# print("p_F variance:", p_F['var'])
print("p_F_i mean:", p_F_i['mean'])
# print("p_F_i variance:", p_F_i['var'])
print("p_F_p mean:", p_F_p['mean'])
# print("p_F_p variance:", p_F_p['var'])
print("p_F_pqd mean:", p_F_pqd['mean'])
print("\n============== Monte Carlo ===============")
print("Probability of failure, p_F_MC:", p_F_MC)

