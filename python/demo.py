# -*- coding: utf-8 -*-
"""
Created on Mon Nov 10 15:08:15 2025

@author: P.Hristov
"""

from pnm_surrogate import *
from psus import psus

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
# x = np.random.rand(1000,2)
# f = branin(x)

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
 
# %% Set up
d = 2
t_star = 230
N = 1000
p0 = 0.1
# mode = 'neg';

out_d = 'norm'
inp_d = {'name': 'uniform', 'parameters':[0,1]}

# %% Run 1 iter
func = lambda x: vectorise_uncert(branin, x, std=2)
p_F, info = psus(func, d, t_star, N, p0, out_d, inp_d);
    
# %% Run multiple iter's
iter = [1,5,10,15,25,50,75,100,250:250:1000]';

for i in 2:length(iter):
    func = @(x)branin_uncert(x,iter(i),mode); #Shrinking level of uncert
    [~,sOut(i,1)] = psus(func,d,t_star,N,p0,outD,inpD);
    disp(iter(i))
