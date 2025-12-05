from psus_test import psus
import numpy as np
import scipy.stats as sts
def tiny_pf_func(x):
    # x in [0,1]^d
    # very small probability of exceeding threshold
    # let output = sum(x) + small noise
    y = np.sum(x, axis=1)
    p1 = y                      # mean parameter
    p2 = 0.05*np.ones_like(y)   # small variance
    return p1, p2
    
func = tiny_pf_func
d = 2
t_star = 1.8
n = 500          # samples per level
p = 0.1          # conditional probability
out_dist = "norm"
inp_dist = {'name':'uniform', 'parameters':[0,1]}

p_F, dict_out_F = psus(func, d, t_star, n, p, out_dist, inp_dist)

print("Probability of Failure:", p_F['p_F'])
print("Mean:", p_F['mean'])
print("Variance:", p_F['var'])