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
                          plot_uncertainty, branin, branin_uncert)
from python.ipsus import psus

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
N = 1000
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
# std = [0.1,0.05,0.01];
std = [15]
p_F_PSuS_list = []
p_F_IPSuS_list = []
info_list = []

for s in std:
    p_F_PSuS, p_F_IPSuS, info = psus(func(s), d, t_star, N, p0, out_d, inp_d);
    p_F_PSuS_list.append(p_F_PSuS)
    p_F_IPSuS_list.append(p_F_IPSuS)
    info_list.append(info)
    
    plt.show()
    print(f"Iteration for sd = {s} completed.")
    
#%% Save pickle with experiment
with open(os.path.join(path, f'exp_psus_n={N}.pkl'), 'wb') as file:
    pickle.dump((p_F_IPSuS_list, p_F_PSuS_list, info_list), file)
    
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

#%% Set up Branin - old Matlab uncertainty - now calibrated
d = 2
t_star = 230
N = 1000
p0 = 0.1
out_d = 'norm'
inp_d = {'name': 'uniform', 'parameters':[0,1]}
func = lambda evals: lambda x: branin_uncert(x, evals=evals) #No calibration is explicitly enforced

#%% Run P-SuS with model uncertainty reducing based on computational resources
# Original formulation from paper
# iterations = [1,5,10,15,25,50,75,100,250,500,750,1000]
iterations = [750]
p_F_PSuS_list = []
p_F_IPSuS_list = []
info_list = []

for it in iterations:
    p_F_PSuS, p_F_IPSuS, info = psus(func(it), d, t_star, N, p0, out_d, inp_d);
    p_F_PSuS_list.append(p_F_PSuS)
    p_F_IPSuS_list.append(p_F_IPSuS)
    info_list.append(info)
    
    plt.show()
    print(f"Iteration for sd = {it} completed.")

#%% Save pickle with experiment
with open(os.path.join(path, f'exp_psus_original_branin_calibrated_n={N}.pkl'), 'wb') as file:
    pickle.dump((p_F_IPSuS_list, p_F_PSuS_list, info_list), file)

#%% Load experiment
import os
import pickle
import simple_pbox_dev as pbd

path = os.path.normpath(
    r'C:\Users\petar.hristov\OneDrive - GATE Institute - Sofia University'+\
    r'\Reliable Computational Modelling - Documents\Projects\Internal Projects'+\
    r'\UQ\Code\psus')

with open(os.path.join(path, 'exp_psus_original_branin_n=1000.pkl'), 'rb') as file:
    experiment = pickle.load(file)

p_F_SuS = 0.0066

#%% Use moment information - only one exp for now - lowest precision
pb_mmms = []
exp = experiment[2][0]
exp_mom = exp[1]['p_Fi_imprecise'] #By the silly structure all moments are already in level 1

for i in range(len(exp)):
    _, ax = plt.subplots(1,1)
    
    if i == len(experiment[2][0])-1: #Deal with F
        exp_pb = exp[i]['p_F_n']
    else:
        exp_pb = exp[i+1]['p_F_i']
    pb_mmms.append(
        pbd.mmms(exp_pb.range.lo, exp_pb.range.hi, exp_mom['mn'][i],
                  exp_mom['vr'][i]))
        
    exp_pb.plot(ax=ax)
    pb_mmms[i].plot(ax=ax)

#%% Compute final estimator
p_F_mmms = pb_mmms[0]
for pb in pb_mmms[1:]:
    p_F_mmms = p_F_mmms * pb

ax = plt.subplot()
p_F_mmms.plot(ax=ax)
experiment[0][0]['bounds'].plot(ax=ax)

#%% Define save-fig path
path_fig = os.path.normpath(
    r'C:\Users\petar.hristov\OneDrive - GATE Institute - Sofia University'+\
    r'\Reliable Computational Modelling - Documents\Group papers\REC\P-SuS\Figs')
    
#%% Plot all levels for all available precisions
iterations = [1,500,1000]
inds = [0,-3,-1]
 
# iterations = [500]
# inds = [-3]
 
for ind, it in zip(inds, iterations):
    level_pbox = []
    exp = experiment[2][ind]
    for e in exp:
        if e.get('p_F_i'): level_pbox.append(e['p_F_i'])
        if e.get('p_F_n'): level_pbox.append(e['p_F_n'])
    fig, axs = plt.subplots(1, len(exp), figsize=(30,10), sharey=True)
    fig1, ax1 = plt.subplots(1,1, figsize=(15,15))
    for j, pbox in enumerate(level_pbox): #This really is range(len(exp[1:]+1)) == range(len(exp))
        pbox.plot(ax=axs[j],
                  bound_colors=['k']*2, left_line_kwargs={"linewidth":3},
                  right_line_kwargs={"linewidth":3})
        if j == len(level_pbox)-1: title = '$\hat{{p}}_{F_n}$'
        else: title = rf'$\hat{{p}}_{{F_{j+1}}}$'
        
        ylabel = ''
        if j == 0: ylabel = 'Cumulative probability'
        axs[j].set_xlabel(title, fontsize=35)
        axs[j].set_ylabel(ylabel, fontsize=35)
        if it == 500 and j == 0:
            axs[j].set_xticks([0.093,0.099,0.105])
        if it == 500 and j == 1:
            axs[j].set_xticks([0.07,0.1,0.13])
        if it == 500 and j == 2:
            axs[j].set_xticks([0.3,0.5,0.7])
            axs[j].set_xlim(0.23,0.77)
        axs[j].xaxis.set_tick_params(labelsize=20)
        axs[j].yaxis.set_tick_params(labelsize=20)
                
    experiment[0][ind]['bounds'].plot(ax=ax1, bound_colors=['k']*2,
                                    left_line_kwargs={"linewidth":3},
                                    right_line_kwargs={"linewidth":3})
    ax1.vlines(p_F_SuS, ymin=0, ymax=1.0, color='r', ls='--', lw=3, label='$\hat{p}_F^{SuS}$')
    # ax1.set_title(f'Iteration: {it}', fontsize=35)
    ax1.set_ylabel('')
    ax1.set_xlabel('$\hat{p}_F^{IP-SuS}$', fontsize=50)
    ax1.yaxis.set_tick_params(labelsize=40)
    ax1.xaxis.set_tick_params(labelsize=40)
    if it==1000:
        ax1.legend(fontsize=50)
        ax1.set_xticks(np.linspace(56,70,3)/10000)
    if it==1: ax1.set_ylabel('Cumulative probability', fontsize=40)
    if it != 1: ax1.set_yticks([])
    
    fig.savefig(os.path.join(path_fig, f'pf_levels_it={it}.png'),
                bbox_inches='tight', dpi=600)
    
    fig1.savefig(os.path.join(path_fig, f'pf_ipsus_it={it}.png'),
                bbox_inches='tight', dpi=600)
    
#%% Look for convergence in p_F
iterations = [1,5,10,15,25,50,75,100,250,500,750,1000]

median_l = []
median_r = []
cspi_95_l = []
cspi_95_r = []
for p_F in experiment[0]:
    median_l.append(p_F['bounds'].median.lo)
    median_r.append(p_F['bounds'].median.hi)
    cspi_95_l.append(p_F['bounds'].alpha_cut(0.025).lo)
    cspi_95_r.append(p_F['bounds'].alpha_cut(0.975).hi)

fig, ax = plt.subplots(1,1, figsize=(20,10))

ax.axhline(0.0066, color='r', lw=3) #p_F^SuS - from the paper

ax.semilogy(np.divide(iterations,100), median_l, 'k.-', markersize=18, lw=3, label='Median')
ax.semilogy(np.divide(iterations,100), median_r, 'k.-', markersize=18, lw=3)
ax.semilogy(np.divide(iterations,100), cspi_95_l, 'k--', lw=3, label='Central, 95% probability interval')
ax.semilogy(np.divide(iterations,100), cspi_95_r, 'k--', lw=3)

ax.yaxis.set_tick_params(labelsize=18)
ax.xaxis.set_tick_params(labelsize=18)
ax.set_xlabel('Computational budget, $\ell$', fontsize=25)
ax.set_ylabel('$\hat{p}_F^{IP-SuS}$', fontsize=25)
ax.legend(fontsize=20)

ax.set_ylim(1e-3, 0.25);
ax.grid()

fig.savefig(os.path.join(path_fig, 'convergence_zoom_y.png'),
            bbox_inches='tight', dpi=600)
#%% Investigate variability in a single computational budget response - 750
it = 750
PSuS_repeat = []
IPSuS_repeat = []
info_repeat = []

for i in range(100):
    p_F_PSuS, p_F_IPSuS, info = psus(func(it), d, t_star, N, p0, out_d, inp_d);
    PSuS_repeat.append(p_F_PSuS)
    IPSuS_repeat.append(p_F_IPSuS)
    info_repeat.append(info)
    print(f'======= Done with iteration {i+1} ========')

#%% Save pickle with repetition experiment
with open(os.path.join(path, f'exp_psus_original_branin_calibrated_repeated_it=750_n={N}.pkl'), 'wb') as file:
    pickle.dump((IPSuS_repeat, PSuS_repeat, info_repeat), file)

#%% Load pickle with repetition experiment
#%% Load experiment
import os
import pickle
import simple_pbox_dev as pbd

N = 1000

path = os.path.normpath(
    r'C:\Users\petar.hristov\OneDrive - GATE Institute - Sofia University'+\
    r'\Reliable Computational Modelling - Documents\Projects\Internal Projects'+\
    r'\UQ\Code\psus')

with open(os.path.join(path, f'exp_psus_original_branin_calibrated_repeated_it=750_n={N}.pkl'), 'rb') as file:
    IPSuS_repeat, PSuS_repeat, info_repeat = pickle.load(file)


#%% Plot final estimate medians, means and CSPIs from experiment and compare to the variability in P-SuS mean
import Interval as ival
TMP = []
n = len(IPSuS_repeat)

y = np.linspace(0.05,0.95,n)
# quantities = ['median','mean','cspi_95']
quantities = ['cspi_95']

for qty in quantities:
    out = 0
    fig, ax = plt.subplots(1,1, figsize=(15,15))
    for i, p_F in enumerate(IPSuS_repeat):
        if qty == 'cspi_95':
            tmp = ival.I(p_F['bounds'].alpha_cut(0.025).lo, p_F['bounds'].alpha_cut(0.975).hi)
            # ax.set_title(f'{qty.upper()}', fontsize=20)
        else:
            tmp = ival.I(getattr(p_F['bounds'], qty).to_numpy())
            # ax.set_title(f'{qty.capitalize()}', fontsize=20)

        color = 'k'
        if not ival.inside(0.0066, tmp):
            out += 1
            color = 'r'
        
        TMP.append(tmp.mid())
        ax.errorbar(tmp.mid(), y[i], xerr=tmp.width()/2, 
                    fmt=' ', ecolor=color, capsize=6, lw=3)
        
    ax.axvline(0.0066, color='b') #p_F^SuS - from the paper
    ax.grid()
    
    ax.set_xlabel('$\hat{p}_F^{IP-SUS}$', fontsize=30)
    ax.xaxis.set_tick_params(labelsize=20)
    ax.set_yticks([])
    
    print(out)
    
    # fig.savefig(os.path.join(path_fig, 'pf_ipsus_cspi95_it=750.png'),
    #             bbox_inches='tight', dpi=600)
#%% Plot level medians and CSPIs from experiment and compare to the variability in P-SuS mean
y = np.linspace(0.05,0.95,n)
quantities = ['median','cspi_95']

level = 2
estim = 'p_F_i'
ref = 0.66

for qty in quantities:
    out = 0
    _, ax = plt.subplots(1,1, figsize=(15,15))
    for i, rep in enumerate(info_repeat):
        if len(rep) < level+1: continue 
        if qty == 'cspi_95':
            tmp = ival.I(rep[level][estim].alpha_cut(0.025).lo, rep[level][estim].alpha_cut(0.975).hi)
            ax.set_title(f'{qty.upper()}: level {level}, {estim}', fontsize=20)

        else:
            tmp = ival.I(getattr(rep[level][estim], qty).to_numpy())
            ax.set_title(f'{qty.capitalize()}: level {level}, {estim}', fontsize=20)
        
        color = 'k'
        if not ival.inside(ref, tmp):
            out += 1
            color = 'r'
        
        ax.errorbar(tmp.mid(), y[i], xerr=tmp.width()/2, 
                    fmt=' ', ecolor=color, capsize=5)
    
    ax.axvline(ref, color='b') #p_F^SuS - from the paper
    ax.grid()
    
    if estim == 'p_F_i': xlabel = f'{{F_{level}}}'
    else: xlabel = f'{{F|F_{level}}}'
    ax.set_xlabel(rf'$\hat{{p}}_{xlabel}$', fontsize=30)
    ax.xaxis.set_tick_params(labelsize=20)
    ax.set_yticks([])
    print(out)
    
#%% Plot final estimate mean +- 3std and mean +- 3C std from experiment
# import Interval as ival

# n = len(IPSuS_repeat)

# y = np.linspace(0.05,0.95,n)

# out = 0
# _, ax = plt.subplots(1,1, figsize=(15,15))
# for i, p_F in enumerate(PSuS_repeat):
#     mean = p_F['mean']
#     # std =  np.sqrt(p_F['var'])
#     # ax.set_title('$\mu\pm 3\sigma$ - P-SuS', fontsize=20)
#     std =  np.sqrt(p_F['Cvar'])
#     ax.set_title('$\mu\pm 3C_{max}\sigma$ - P-SuS', fontsize=20)

#     tmp = ival.I(mean - 3*std, mean + 3*std)

#     color = 'k'
#     if not ival.inside(0.0066, tmp):
#         out += 1
#         color = 'r'
    
#     ax.errorbar(tmp.mid(), y[i], xerr=tmp.width()/2, 
#                 fmt=' ', ecolor=color, capsize=5)

# ax.axvline(0.0066, color='b') #p_F^SuS - from the paper
    
# ax.set_xlabel('$\hat{p}_F^{P-SUS}$', fontsize=20)
# ax.grid()

# print(out)

#%% Plot final estimate 95% CSPI via Cantelli constructed bounds
from cantelli_construct import cantelli_cspi
TMP_PSUS = []
n = len(PSuS_repeat)

y = np.linspace(0.05,0.95,n)
out = 0
fig, ax = plt.subplots(1,1, figsize=(15,15))

for i, p_F in enumerate(PSuS_repeat):
    if isinstance(p_F['var'], np.ndarray): p_F_var = p_F['var'][0]
    cspi = cantelli_cspi(p_F['mean'], np.sqrt(p_F_var), 0.05)
    # cspi = cantelli_cspi(p_F['mean'], np.sqrt(p_F['Cvar']), 0.05)
    # ax.set_title('95% CSPI', fontsize=20)

    color = 'k'
    if not ival.inside(0.0066, cspi):
        out += 1
        color = 'r'
    
    TMP_PSUS.append(cspi.mid())
    ax.errorbar(cspi.mid(), y[i], xerr=cspi.width()/2, 
                fmt=' ', ecolor=color, capsize=6, lw=3)

ax.axvline(0.0066, color='b') #p_F^SuS - from the paper
ax.grid()
    
ax.set_xlabel('$\hat{p}_F^{P-SUS}$', fontsize=30)
ax.xaxis.set_tick_params(labelsize=20)
ax.set_yticks([])

print(out)

# fig.savefig(os.path.join(path_fig, 'pf_psus_cspi95_it=750.png'),
#             bbox_inches='tight', dpi=600)