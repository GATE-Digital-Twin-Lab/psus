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

from pnm_surrogate import (generate_uncertainty, plot_uncertainty, branin)

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
