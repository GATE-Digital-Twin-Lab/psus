# -*- coding: utf-8 -*-
"""
Created on Mon Nov 10 15:08:15 2025

@author: P.Hristov
"""

import numpy as np
import scipy.stats as sts
import matplotlib.pyplot as plt

import os
os.chdir(os.path.dirname(os.path.realpath(__file__)))
import pickle

from pnm_surrogate import (generate_uncertainty, vectorise_uncertainty,
                          plot_uncertainty, branin)
from psus_modified_slim import psus

#%% Test the PNM surrogate functionality on random numbers
f = -1 + 10*np.random.rand(50)

std = 1
for i, fi in enumerate(f):
    mu = generate_uncertainty(fi, std, seed=int(np.abs(fi*10000)))
    plot_uncertainty(fi, mu, std)
    plt.title(f'{i}, {fi:.4f}')
    plt.xlim(-5,12)
    plt.show()

for i, fi in enumerate(f):
    mu = generate_uncertainty(fi, std, seed=int(np.abs(fi*10000)), calibrated=False)
    plot_uncertainty(fi, mu, std)
    plt.title(f'{i}, {fi:.4f}')
    plt.xlim(-5,12)
    plt.show()


#%% Test the PNM surrogate on the Branin function
x = np.random.rand(10,2)
f = branin(x)

std = 2
for i, fi in enumerate(f):
    mu = generate_uncertainty(fi, std=std, seed=int(np.abs(fi*10000)))
    plot_uncertainty(fi, mu, std)
    plt.title(f'{i}, {fi:.4f}')
    plt.show()

for i, fi in enumerate(f):
    mu = generate_uncertainty(fi, std=std, seed=int(np.abs(fi*10000)), calibrated=False)
    plot_uncertainty(fi, mu, std)
    plt.title(f'{i}, {fi:.4f}')
    plt.show()

#%% Test the PNM surrogate on the Branin function
x = np.random.rand(1000,2)
f = branin(x)

std = 2
mu_cal = []
mu_ncal = []
for i, fi in enumerate(f):
    mu_cal.append(generate_uncertainty(fi, std=std, seed=int(np.abs(fi*10000))))

for i, fi in enumerate(f):
    mu_ncal.append(generate_uncertainty(fi, std=std, seed=int(np.abs(fi*10000)), calibrated=False))
    
mu_cal = np.array(mu_cal)   
mu_ncal = np.array(mu_ncal)  

#%% 
pd = sts.norm
plt.hist(pd(mu_ncal, std).cdf(f), np.arange(0,1,0.01), color='r', label='Uncalibrated simulations')
plt.hist(pd(mu_cal, std).cdf(f), np.arange(0,1,0.01), color='g', label='Calibrated simulations')

plt.legend()
plt.show()

#%% Set up Branin
d = 2
t_star = 230
N = 100
p0 = 0.1
out_d = 'norm'
inp_d = {'name': 'uniform', 'parameters':[0,1]}
func = lambda std: lambda x: vectorise_uncertainty(branin, x, std=std) #Default is all responses are calibrated

#%% Load previously run IP-SuS for Branin
path = os.path.normpath(
    r'C:\Users\petar.hristov\OneDrive - GATE Institute - Sofia University'+\
    r'\Reliable Computational Modelling - Documents\Projects\Internal Projects'+\
    r'\UQ\Code\psus')

with open(os.path.join(path, 'tmp_psus.pkl'), 'rb') as file:
    info_list = pickle.load(file)

#%% Run P-SuS with model uncertainty reducing based on computational resources
# std = [2,1.5,1,0.5,0.25,0.1,0.05,0.01];
std = [0.1,0.05,0.01];
p_F_PSuS_list = []
p_F_IPSuS_list = []
info_list = []

np.random.seed(1)
X = getattr(sts, inp_d['name'])(*inp_d['parameters']).rvs((10_000,2))

for s in std:
    p_F_PSuS, p_F_IPSuS, info = psus(func(s), d, t_star, N, p0, out_d, inp_d);
    p_F_PSuS_list.append(p_F_PSuS)
    p_F_IPSuS_list.append(p_F_IPSuS)
    info_list.append(info)
    
    plt.show()
    print(f"Iteration for sd = {s} completed.")
#%% MC check 
p_F_MC = []
np.random.seed(1)
X = getattr(sts, inp_d['name'])(*inp_d['parameters']).rvs((10_000,2))

for s in std:
    y, _ = func(s)(X)
    p_F_MC.append(np.mean(y >= t_star)) 
    print(f"Iteration for sd = {s} completed.")

#%% True p_F MC check
y = branin(X)
p_F_true = np.mean(y >= t_star)

#%% Plot all levels for all available precisions
for i, exp in enumerate(info_list):
    prod = 1
    _, axs = plt.subplots(1, len(exp), figsize=(30,10), sharey=True)
    _, ax1 = plt.subplots(1,1, figsize=(15,15))
    for j, l in enumerate(exp):
        if j == (len(exp) - 1): key = 'p_F_n'
        else: key = 'p_F_i' #Assuming * is Frechet by default
        
        l[key].plot(ax=axs[j])
        axs[j].set_title(rf'$p_{{F_{j}}}$', fontsize=35)
        axs[j].xaxis.set_tick_params(labelsize=20)
        axs[j].yaxis.set_tick_params(labelsize=20)
        
        prod *= l[key]
    prod.plot(ax=ax1)
    ax1.vlines(p_F_MC[i], 1.0, 'r--') #Failure probability with MC for the PNM
    ax1.vlines(p_F_true, 1.0, 'k') #Failure probability with MC for the true model
    ax1.set_title('$p_F^{IP-SuS}$', fontsize=35)
    ax1.yaxis.set_tick_params(labelsize=20)
    ax1.xaxis.set_tick_params(labelsize=20)

#%% Prints
print("============== P-SuS ===============")
print("Probability of failure, p_F:", p_F_values)
print("\n============== Monte Carlo ===============")
print("Probability of failure, p_F_MC:", p_F_MC)

#%% Set up tiny function
def tiny_pf_func(x):
    # x in [0,1]^d
    y = np.sum(x, axis=1)
    p1 = y                      
    p2 = 0.05*np.ones_like(y)   
    return p1, p2
    
func = tiny_pf_func
d = 2
t_star = 1.9
n = 500          # samples per level
p = 0.1          # conditional probability
out_dist = "norm"
inp_dist = {'name':'uniform', 'parameters':[0,1]}

p_F, dict_out_F, L = psus(func, d, t_star, n, p, out_dist, inp_dist)
np.random.seed(1)
X = getattr(sts, inp_dist['name'])(*inp_dist['parameters']).rvs((10_000,2))

y, u = tiny_pf_func(X)
p_F_MC = np.mean(y >= t_star)

print("============== P-SuS ===============")
print("Probability of failure, p_F:", p_F['p_F'])
print("p_F mean:", p_F['mean'])
print("p_F variance:", p_F['var'])
print("Level:", L)
print("\n============== Monte Carlo ===============")
print("Probability of failure, p_F_MC:", p_F_MC)

# print("Probability of Failure_1:", p_F['p_F'])
# print("Final level:", L)

# def tiny_pf_func2(x):
#     # x in [0,1]^d
#     # y = weighted sum of inputs
#     w = np.array([0.3, 0.5, 0.2])  # weights sum to 1
#     y = x @ w  # matrix multiplication
#     p1 = y                        # mean
#     p2 = 0.02*np.ones_like(y)     # small constant variance
#     return p1, p2

# # Set up parameters
# func = tiny_pf_func2
# d = 3
# t_star = 0.85      # near the upper end of sum(x*w)
# n = 500
# p = 0.1
# out_dist = "norm"
# inp_dist = {'name':'uniform', 'parameters':[0,1]}

# # Run P-SuS
# p_F, dict_out_F = psus(func, d, t_star, n, p, out_dist, inp_dist)

# print("Probability of Failure_2:", p_F['p_F'])

# def pf_func_3d(x):
#     """
#     x in [0,1]^3
#     Failure occurs when weighted sum of inputs exceeds a threshold.
#     """
#     w = np.array([0.2, 0.5, 0.3])  # weights sum to 1
#     y = x @ w                       # weighted sum
#     p1 = y                          # mean
#     p2 = 0.02*np.ones_like(y)       # small variance
#     return p1, p2

# # Parameters
# func = pf_func_3d
# d = 3
# t_star = 0.85     # threshold near upper end of y
# n = 500           # samples per level
# p = 0.1           # conditional probability
# out_dist = "norm"
# inp_dist = {'name':'uniform', 'parameters':[0,1]}

# # Run P-SuS
# p_F, dict_out_F = psus(func, d, t_star, n, p, out_dist, inp_dist)
# print("Probability of Failure_3:", p_F['p_F'])
