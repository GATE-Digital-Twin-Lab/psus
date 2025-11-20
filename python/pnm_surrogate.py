# -*- coding: utf-8 -*-
"""
Created on Mon Nov 10 08:58:06 2025

@author: P.Hristov
"""
import numpy as np
import scipy.stats as sts

import matplotlib.pyplot as plt

#%% Functions
def generate_uncertainty(f, std=1, calibrated=True, seed=42):
    ql = 0.17
    qr = 0.83
    
    if calibrated: #Stay within [ql, qr] region
        np.random.seed(seed)
        q = ql + (qr-ql)*np.random.rand()
        add = 0
        
    else: #Go outside of [ql, qr] region
        t = np.array([0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.75]) #Decision thresholds
        adds = np.linspace(2.5, 0, len(t)) * std
        
        np.random.seed(seed)
        u = np.random.rand() #Choose a point in the region
        
        q = ql*u
        # add = 5*std if u < 0.1 else 0 #Randomly push mean 5% of std out - helpful for bounded distros
        add = adds[np.max(np.where(t<u))]
        
        np.random.seed(seed+1000)
        if np.random.rand() < 0.5: #Choose left or right tail - True = right
            q += qr
            add = -add #Set the correct sign
    
        print(u, add)
    mu = sts.norm(f, std).ppf(1-q) + add
    
    return mu

def plot_uncertainty(f, mu, std):
    ql = 0.17
    qr = 0.83
    
    u = np.linspace(0.0001, 0.9999, 1000)
    z = np.zeros(u.shape)
    pd = sts.norm(mu, std)
    x = pd.ppf(u)
    pdf = pd.pdf(x)
    
    xl = pd.ppf(ql)
    xr = pd.ppf(qr)
    
    color = 'green'
    if not ql < pd.cdf(f) < qr: color = 'red'

    plt.plot(x, pdf)
    plt.fill_between(x[xr < x], z[xr < x], pdf[xr < x], facecolor=[0.85]*3)
    plt.fill_between(x[x < xl], z[x < xl], pdf[x < xl], facecolor=[0.85]*3)
    plt.axvline(f, ymax=1, color=color, ls='--')
    plt.ylim(0, plt.ylim()[1])

def vectorise_uncert(f, x, std, seed=None, calibrated=True):
    '''Pass a function f and an array of input point x.'''
    
    n = x.shape[0]
    if not hasattr(calibrated, '__iter__'):
        cal = [calibrated]*n
    
    y = f(x)
    mu = np.zeros((y.shape[0], 1)) #Only 2-par distros for now
    
    if not seed: seed = np.int32(np.abs(y*1e4))
    else: seed = [seed]*n
    
    for i, yi in enumerate(y):
        mu[i,:] = generate_uncertainty(yi, std=std, seed=seed[i], calibrated=cal[i])
        
    return mu, std #This format is expected by psus and is the more general one

#%% Test Functions
def branin(x):
    '''
    x is a 1x2 vector 
    '''
    
    # %% Input scaling
    u1 = x[:,0]; #Uniform scales
    u2 = x[:,1];
    
    x1 = 15*u1-5;
    x2 = 15*u2;
    
    # %% Mean
    a = 1
    b = 5.1/4/np.pi**2
    c = 5/np.pi
    r = 6
    s = 10
    t = 1/(8*np.pi)
    
    y = (a*(x2-b*x1**2+c*x1-r)**2+s*(1-t)*np.cos(x1)+s)+5*x1; #True function

    return y