# -*- coding: utf-8 -*-
"""
Created on Wed Nov 12 16:07:40 2025

@author: P.Hristov
"""

import numpy as np
import scipy.stats as sts
from sklearn.datasets import make_spd_matrix

import matplotlib.pyplot as plt

#%% What happens if we let t_i be a distribution - a 1D example
n = 100 #Number of sample locations for plotting
x = np.linspace(0,1,n)

sigma = 2
theta = 0.1

mu = lambda x: (6*x-2)**2*np.sin(12*x-4)
k = lambda x, xp, theta: np.exp(-(x-xp)**2/2/theta**2)

# Sigma = make_spd_matrix(n_dim=n, random_state=42)
Sigma = sigma**2 * (np.array([k(xi,x,theta) for xi in x]) + 1e-6*np.eye(n))

f = sts.multivariate_normal(mu(x), Sigma, seed=42)

#%%
plt.plot(x, mu(x))
plt.plot(x, f.rvs())

#%%
f_rvs = f.rvs(500)

plt.plot(x, f_rvs.T, c='gray', lw=0.5)
plt.plot(x, mu(x))

#%%
plt.rcParams.update({'text.usetex':True, 'text.latex.preamble': r'\usepackage{amsfonts}'})

plt.subplots(1,1,figsize=(15,8))
t_i = 2.5
t_i_up = t_i+1.96*sigma
t_i_lw = t_i-1.96*sigma

x_crit = [0.4, 0.75, 0.92]
uf = 1.96 * np.sqrt(np.diag(Sigma))
ci_f =np.array([mu(x) - uf, mu(x) + uf])

plt.plot(x, mu(x), label='"True" function, $\mu(x)$')
plt.plot(x, ci_f.T, c='gray', ls='--', lw=1, label=['95\% predictive interval of $\mu(x)$', ''])

plt.axhline(t_i, color='r', lw=1, label='Critical level, $t_i$') #Plot t_i
plt.axhline(t_i+1.96*sigma, color='r', lw=1, ls='--', label='95\% predictive interval of $t_i$') #Plot t_i
plt.axhline(t_i-1.96*sigma, color='r', lw=1, ls='--') #Plot t_i

# plt.grid(which='both')

#Plot sideways distributions
pd_ti = sts.norm(t_i, sigma)
label_fill = ['Probability of exceedence, $p_{ij}$','','']
label_fill_lw = [r'Probability of exceedence, $\overline{p}_{ij}$','','']
label_fill_up = [r'Probability of exceedence, $\underline{p}_{ij}$','','']

for i, xc in enumerate(x_crit):
    muc = mu(xc)
    pd = sts.norm(muc, sigma)
    ys = np.linspace(muc+3*sigma, muc-3*sigma, 200)
    # ys = np.linspace(max(muc,t_i)+3*sigma, min(muc,t_i)-3*sigma, 200)
    xs = pd.pdf(ys)
    xs_ti = pd_ti.pdf(ys)
    
    xpdf = 0.75*xs + xc
    xpdf_ti = 0.75*xs_ti + xc
    # exc = xpdf[ys >= t_i]
    exc_up = xpdf[ys >= t_i_up]
    exc_lw = xpdf[ys >= t_i_lw]
    
    plt.plot(xpdf, ys, c='k', lw=1)
    # plt.plot(xpdf_ti, ys, c='r', lw=1)
    plt.axvline(xc, color='k', lw=0.5)
    plt.hlines(muc, xc, max(xpdf), color='k', lw=0.3)
    # plt.fill_between(exc, t_i, ys[ys >= t_i], facecolor='gray', alpha=0.3, label=label_fill[i])
    plt.fill_between(exc_up, t_i_up, ys[ys >= t_i_up], facecolor='r', alpha=0.3, label=label_fill_up[i])
    plt.fill_between(exc_lw, t_i_lw, ys[ys >= t_i_lw], facecolor='b', alpha=0.2, label=label_fill_lw[i])
    # plt.fill_between(np.minimum(xpdf, xpdf_ti), ys, facecolor=[0.85]*3, label=label_fill[i])

plt.xlabel('$x$', fontsize = 25)
plt.ylabel('$y$', fontsize = 25)
plt.gca().tick_params(axis='both', which='major', labelsize=20)
plt.legend(loc='upper left', fontsize=20)

#%% Computing t_i
from pyuncertainnumber import pba

y0 = pba.normal(2,2)
y1 = pba.normal(3,2)
tf = (y0+y1)/2
ti = y0.add(y1, dependency='i')/2
tp = y0.add(y1, dependency='p')/2
to = y0.add(y1, dependency='o')/2

tis = [ti,tp,to,tf]

#%% Compute p_ij
deps = ['i','p','o','f']
y = pba.normal(mu(0.4), sigma)
r = []
R = []

p = []
P = []

p_95 = []
_, axs = plt.subplots(2,2, figsize=(15,15))
for t in tis:
    p_95.append(y.cdf(t.alpha_cut(0.025).left), t.alpha_cut(0.975))
    for d in deps:
        tmp = t.sub(y, dependency=d)
        p = tmp.cdf(0)
        
        r.append(tmp)
                  
#%% Plot t_i under dep's
ax = plt.subplot()
ti.plot(bound_colors=['b']*2, label='Independence', ax=ax, style='simple')
tp.plot(bound_colors=['r']*2, label='Comonotnicity', ax=ax, style='simple')
to.plot(bound_colors=['k']*2, label='Countermonotonicity', ax=ax, style='simple')
tf.plot(bound_colors=['g']*2, label='Frechet', ax=ax, style='simple')

ax.set_ylabel(r'$\mathbb{P}(T_i\leq t_i)$')
ax.set_xlabel('$t_i$')
ax.legend()