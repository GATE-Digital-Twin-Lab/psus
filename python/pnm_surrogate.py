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


def vectorise_uncertainty(f, x, std, seed=None, calibrated=True):
    '''Pass a function f and an array of input points x.'''
    
    n = x.shape[0]
    if not hasattr(calibrated, '__iter__'): #Enable some responses to be calibrated and some not
        cal = [calibrated]*n
    
    y = f(x)
    mu = np.zeros(y.shape[0]) #Only 2-par distros for now
    
    if not seed: seed = np.int32(np.abs(y*1e4))
    else: seed = [seed]*n
    
    shp_std = np.asarray(std).shape
    if len(shp_std) < 1: std = np.array([std]*n) #Check if std is a scalar
    elif len(shp_std) == 1: std = np.array(std) #Check if std is a list or 1D array
    
    for i, (yi, sdi) in enumerate(zip(y, std)):
        mu[i] = generate_uncertainty(yi, std=sdi, seed=seed[i], calibrated=cal[i])
    
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
    
    y = (a*(x2-b*x1**2+c*x1-r)**2+s*(1-t)*np.cos(x1)+s)+5*x1 #True function

    return y

def branin_uncert(x, evals=1, mode='neg', get_true_out=False):
    #x is a 1x2 vector 
    #
    
    f = branin(x)
    
    # Uncertainty
    if mode == 'const': #Variance depends only on iterations
        #evals = min(evals,100); #Prevent negative variance
        #u = 50./evals-.5 * ones(size(x,1),1); #For example
        cv = cv_uncert(x);
    else:
        u = np.sqrt(np.exp(-0.01*evals) * np.abs(f)); #SD of a normal deviate

    # Mean of observation
    # scales = np.log(np.abs(f)+1) / 2 #Original c
    # scales = np.log(np.abs(f)+1) / 3 #Modified c for UC
    scales = (np.sin(f)+1) / 2 #Modified c for UC
    # scales = min(log(abs(f)+1) / 2, 1.8) #Modified c for UC
    
    if mode == 'pos':
        m = f + scales * u;
    elif mode == 'neg':
        m = f - scales * u;
    else:
        u = (cv * m);
        m = f #Exact convergence in mean with some bounds
    
    if get_true_out: return m, u, f
    
    return m, u

def cv_uncert(x):
    cv = np.zeros((x.shape[0], 1));
    
    for i in range(len(x)):
        nX = np.linalg.norm(x[i])
        
        if nX <= 0.2:
            cv[i] = 0.0028
        elif 0.2 < nX <= 0.4:
            cv[i] = 0.0049
        elif 0.4 < nX <= 0.6:
            cv[i] = 0.0946
        elif 0.6 < nX <= 0.8:
            cv[i] = 0.0151
        elif 0.8 < nX <= 1:
            cv[i] = 0.0023
        elif 1 < nX <= 1.2:
            cv[i] = 0.0102
        elif 1.2 < nX <= 1.4:
            cv[i] = 0.0291
        elif 1.4 < nX <= 1.6:
            cv[i] = 0.0420
        elif 1.6 < nX <= 1.8:
            cv[i] = 0.0464;
        elif 1.8 < nX <= 2:
            cv[i] = 0.0531;
    return cv