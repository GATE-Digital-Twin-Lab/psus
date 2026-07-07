import pickle
import os
from psus_modified_slim import psus
import Interval as ival

#%%
# Configuration
from RPrepo import*
d = 5
t_star = 0
N = 1000
p0 = 0.1
out_d = 'norm'
inp_d = [
    {'name': 'uniform', 'parameters': [70, 10]},          
    {'name': 'norm',    'parameters': [39, 0.1]},         
    {'name': 'gumbel_r','parameters': [1342.0, 272.9]},   
    {'name': 'norm',    'parameters': [400, 0.1]},        
    {'name': 'norm',    'parameters': [250000.0, 35000.0]}
]

# Assuming RP14 is defined or imported in your environment
func = lambda evals: (lambda x: rp_uncert(x, RP14, evals=evals))

# 1. First Run: Convergence over iterations
iterations = [1, 5, 10, 15, 25, 50, 75, 100, 250, 500, 750, 1000]

p_F_list = []
info_list = []

for it in iterations:
    p_F_PSuS, p_F_IPSuS, info = psus(func(it), d, t_star, N, p0, out_d, inp_d)
    p_F_list.append(p_F_IPSuS)
    info_list.append(info)
    print(f"Iteration {it} completed.")

with open('exp_rp_convergence.pkl', 'wb') as f:
    pickle.dump((p_F_list, info_list), f)

# 2. Second Run: 100 repetitions at it=750
it_rep = 750
IPSuS_repeats = []
for i in range(100):
    _, p_F_IPSuS, _ = psus(func(it_rep), d, t_star, N, p0, out_d, inp_d)
    IPSuS_repeats.append(p_F_IPSuS)
    print(f"Repetition {i+1} at {it_rep} completed.")

with open('exp_rp_repetitions.pkl', 'wb') as f:
    pickle.dump(IPSuS_repeats, f)

# Load from 'exp_rp_convergence.pkl'
median_l = [p['bounds'].median.lo for p in p_F_list]
median_r = [p['bounds'].median.hi for p in p_F_list]
cspi_l = [p['bounds'].alpha_cut(0.025).lo for p in p_F_list]
cspi_r = [p['bounds'].alpha_cut(0.975).hi for p in p_F_list]

plt.figure(figsize=(10, 5))
plt.axhline(0.000752, color='r', lw=2, label='True $p_F$')
plt.semilogy(np.divide(iterations,100), median_l, 'k.-', label='Median')
plt.semilogy(np.divide(iterations,100), median_r, 'k.-')
plt.semilogy(np.divide(iterations,100), cspi_l, 'k--', label='95% CSPI')
plt.semilogy(np.divide(iterations,100), cspi_r, 'k--')
plt.xlabel('Computational budget, $\ell$')
plt.ylabel('$\hat{p}_F^{IP-SuS}$')
plt.title('RP14', fontsize=16)
plt.legend(fontsize=20)
plt.grid(True)
plt.show()

# Load from 'exp_rp_repetitions.pkl'
out = 0
n = len(IPSuS_repeats)
y = np.linspace(0.05, 0.95, n)
TMP = []  

fig, ax = plt.subplots(1, 1, figsize=(15, 15))

# 2. Loop through the 100 simulation runs to build the error bars
for i, p_F in enumerate(IPSuS_repeats):
    # Extract the 95% CSPI lower and upper bounds directly
    lo = p_F['bounds'].alpha_cut(0.025).lo
    hi = p_F['bounds'].alpha_cut(0.975).hi
    tmp = ival.I(lo, hi)
    
    # Check if the true probability falls inside this specific run's interval
    color = 'k'
    if not ival.inside(0.000752, tmp):
        out += 1
        color = 'r'  # Mark red if it misses the true line
    
    TMP.append(tmp.mid())
    
    # Plot the interval line using its midpoint and half-width
    ax.errorbar(tmp.mid(), y[i], xerr=tmp.width()/2, 
                fmt=' ', ecolor=color, capsize=6, lw=3)

# 3. Add formatting, true line, and adjust limits to fit your true p_F
ax.axvline(0.000752, color='b', lw=2) 
ax.grid(True)

# Format axes
ax.set_xlabel('$\\hat{p}_F^{IP-SUS}$', fontsize=30)
ax.xaxis.set_tick_params(labelsize=20)
ax.set_yticks([])
ax.set_title('RP14', fontsize=16)

# Adjust horizontal limits to frame your true p_F perfectly (0.000752)
ax.set_xlim(0, 0.005) 

plt.show()

print(f"Number of intervals that failed to capture the true p_F: {out}")
# %%
# Configuration
from RPrepo import*
d = 2
t_star = 0
N = 1000
p0 = 0.1
out_d = 'norm'
inp_d = [  
    {'name': 'norm',    'parameters': [0, 1]},        
    {'name': 'norm',    'parameters': [0, 1]}
]

# Assuming RP14 is defined or imported in your environment
func = lambda evals: (lambda x: rp_uncert(x, RP22, evals=evals))

# 1. First Run: Convergence over iterations
iterations = [1, 5, 10, 15, 25, 50, 75, 100, 250, 500, 750, 1000]

p_F_list = []
info_list = []

for it in iterations:
    p_F_PSuS, p_F_IPSuS, info = psus(func(it), d, t_star, N, p0, out_d, inp_d)
    p_F_list.append(p_F_IPSuS)
    info_list.append(info)
    print(f"Iteration {it} completed.")

with open('exp_rp_convergence.pkl', 'wb') as f:
    pickle.dump((p_F_list, info_list), f)

# 2. Second Run: 100 repetitions at it=750
it_rep = 750
IPSuS_repeats = []
for i in range(100):
    _, p_F_IPSuS, _ = psus(func(it_rep), d, t_star, N, p0, out_d, inp_d)
    IPSuS_repeats.append(p_F_IPSuS)
    print(f"Repetition {i+1} at {it_rep} completed.")

with open('exp_rp_repetitions.pkl', 'wb') as f:
    pickle.dump(IPSuS_repeats, f)

# Load from 'exp_rp_convergence.pkl'
median_l = [p['bounds'].median.lo for p in p_F_list]
median_r = [p['bounds'].median.hi for p in p_F_list]
cspi_l = [p['bounds'].alpha_cut(0.025).lo for p in p_F_list]
cspi_r = [p['bounds'].alpha_cut(0.975).hi for p in p_F_list]

plt.figure(figsize=(10, 5))
plt.axhline(0.00416, color='r', lw=2, label='True $p_F$')
plt.semilogy(np.divide(iterations,100), median_l, 'k.-', label='Median')
plt.semilogy(np.divide(iterations,100), median_r, 'k.-')
plt.semilogy(np.divide(iterations,100), cspi_l, 'k--', label='95% CSPI')
plt.semilogy(np.divide(iterations,100), cspi_r, 'k--')
plt.xlabel('Computational budget, $\ell$')
plt.ylabel('$\hat{p}_F^{IP-SuS}$')
plt.title('RP22', fontsize=16)
plt.legend(fontsize=20)
plt.grid(True)
plt.show()

# Load from 'exp_rp_repetitions.pkl'
out = 0
n = len(IPSuS_repeats)
y = np.linspace(0.05, 0.95, n)
TMP = []  

fig, ax = plt.subplots(1, 1, figsize=(15, 15))

# 2. Loop through the 100 simulation runs to build the error bars
for i, p_F in enumerate(IPSuS_repeats):
    # Extract the 95% CSPI lower and upper bounds directly
    lo = p_F['bounds'].alpha_cut(0.025).lo
    hi = p_F['bounds'].alpha_cut(0.975).hi
    tmp = ival.I(lo, hi)
    
    # Check if the true probability falls inside this specific run's interval
    color = 'k'
    if not ival.inside(0.00416, tmp):
        out += 1
        color = 'r'  # Mark red if it misses the true line
    
    TMP.append(tmp.mid())
    
    # Plot the interval line using its midpoint and half-width
    ax.errorbar(tmp.mid(), y[i], xerr=tmp.width()/2, 
                fmt=' ', ecolor=color, capsize=6, lw=3)

# 3. Add formatting, true line, and adjust limits to fit your true p_F
ax.axvline(0.00416, color='b', lw=2) 
ax.grid(True)

# Format axes
ax.set_xlabel('$\\hat{p}_F^{IP-SUS}$', fontsize=30)
ax.xaxis.set_tick_params(labelsize=20)
ax.set_yticks([])
ax.set_title('RP14', fontsize=16)

# Adjust horizontal limits to frame your true p_F perfectly (0.000752)
ax.set_xlim(0, 0.02) 

plt.show()

print(f"Number of intervals that failed to capture the true p_F: {out}")

#%%
from RPrepo import*
d = 5
t_star = 0
N = 1000
p0 = 0.1
out_d = 'norm'
inp_d = [
    {'name': 'uniform', 'parameters': [70, 10]},          
    {'name': 'norm',    'parameters': [39, 0.1]},         
    {'name': 'gumbel_r','parameters': [1342.0, 272.9]},   
    {'name': 'norm',    'parameters': [400, 0.1]},        
    {'name': 'norm',    'parameters': [250000.0, 35000.0]}
]

# Assuming RP14 is defined or imported in your environment
func = lambda evals: (lambda x: rp_uncert(x, RP111, evals=evals))

# 1. First Run: Convergence over iterations
iterations = [1, 5, 10, 15, 25, 50, 75, 100, 250, 500, 750, 1000]

p_F_list = []
info_list = []

for it in iterations:
    p_F_PSuS, p_F_IPSuS, info = psus(func(it), d, t_star, N, p0, out_d, inp_d)
    p_F_list.append(p_F_IPSuS)
    info_list.append(info)
    print(f"Iteration {it} completed.")

with open('exp_rp_convergence.pkl', 'wb') as f:
    pickle.dump((p_F_list, info_list), f)

# 2. Second Run: 100 repetitions at it=750
it_rep = 750
IPSuS_repeats = []
for i in range(100):
    _, p_F_IPSuS, _ = psus(func(it_rep), d, t_star, N, p0, out_d, inp_d)
    IPSuS_repeats.append(p_F_IPSuS)
    print(f"Repetition {i+1} at {it_rep} completed.")

with open('exp_rp_repetitions.pkl', 'wb') as f:
    pickle.dump(IPSuS_repeats, f)

# Load from 'exp_rp_convergence.pkl'
median_l = [p['bounds'].median.lo for p in p_F_list]
median_r = [p['bounds'].median.hi for p in p_F_list]
cspi_l = [p['bounds'].alpha_cut(0.025).lo for p in p_F_list]
cspi_r = [p['bounds'].alpha_cut(0.975).hi for p in p_F_list]

plt.figure(figsize=(10, 5))
plt.axhline(0.000000765, color='r', lw=2, label='True $p_F$')
plt.semilogy(np.divide(iterations,100), median_l, 'k.-', label='Median')
plt.semilogy(np.divide(iterations,100), median_r, 'k.-')
plt.semilogy(np.divide(iterations,100), cspi_l, 'k--', label='95% CSPI')
plt.semilogy(np.divide(iterations,100), cspi_r, 'k--')
plt.xlabel('Computational budget, $\ell$')
plt.ylabel('$\hat{p}_F^{IP-SuS}$')
plt.title('RP111', fontsize=16)
plt.legend(fontsize=20)
plt.grid(True)
plt.show()

# Load from 'exp_rp_repetitions.pkl'
out = 0
n = len(IPSuS_repeats)
y = np.linspace(0.05, 0.95, n)
TMP = []  

fig, ax = plt.subplots(1, 1, figsize=(15, 15))

# 2. Loop through the 100 simulation runs to build the error bars
for i, p_F in enumerate(IPSuS_repeats):
    # Extract the 95% CSPI lower and upper bounds directly
    lo = p_F['bounds'].alpha_cut(0.025).lo
    hi = p_F['bounds'].alpha_cut(0.975).hi
    tmp = ival.I(lo, hi)
    
    # Check if the true probability falls inside this specific run's interval
    color = 'k'
    if not ival.inside(0.000000765, tmp):
        out += 1
        color = 'r'  # Mark red if it misses the true line
    
    TMP.append(tmp.mid())
    
    # Plot the interval line using its midpoint and half-width
    ax.errorbar(tmp.mid(), y[i], xerr=tmp.width()/2, 
                fmt=' ', ecolor=color, capsize=6, lw=3)

# 3. Add formatting, true line, and adjust limits to fit your true p_F
ax.axvline(0.000000765, color='b', lw=2) 
ax.grid(True)

# Format axes
ax.set_xlabel('$\\hat{p}_F^{IP-SUS}$', fontsize=30)
ax.xaxis.set_tick_params(labelsize=20)
ax.set_yticks([])
ax.set_title('RP111', fontsize=16)

# Adjust horizontal limits to frame your true p_F perfectly (0.000752)
ax.set_xlim(0, 0.000003) 

plt.show()

print(f"Number of intervals that failed to capture the true p_F: {out}")



# %%
