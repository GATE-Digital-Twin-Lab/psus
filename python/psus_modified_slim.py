from psus import psort, fill_out, generate_out_funcs, pmma, varindepprod, highlight
from mc_mv_final import sum_distributions_frechet, product_distributions_frechet_samples, var_bounds_pbox
import numpy as np
import scipy.stats as sts
from pyuncertainnumber import pba
import Interval as ival

import matplotlib.pyplot as plt

def psus(func, d, t_star, n, p, 
         out_dist,
         inp_dist = {'name':'uniform', 'parameters':[0,1]},
         plot = False):
    '''
    %%% Inputs:
    %%%     func    - single input function handle to the probabilistic code
    %%%     d       - dimensionality of the input space (scalar)
    %%%     t_star  - critical threshold (scalar)
    %%%     n       - target number of samples per P-SuS level (scalar)
    %%%     p       - target level probability (scalar)
    %%%     out_dist - name of the output distribution (char)
    %%%     inp_dist - name and parameter tuple for the input distribution 
    %%%               (1-by-2 cell). If left unspecified, a uniform distro on
    %%%               [0,1] is used.
    %%%
    %%% Outputs:
    %%% 	pF	  - probability of failure structure, with deterministic pF and
    %%% 			information about the pF distribution - 
    %%%				mean and variance under independence and perfect dependence
    %%%				between levels.
    %%% 	sOutF - structure containing full information about the psus run
    %%%
    %%% Only a normal proposal distribution is used for now.
    %%% The function requires Statistics and Machine Learning Toolbox in MATLAB
    %%% to be installed.'''
    
    # PREPARE PRELIMINARIES
    get_mean_var, excd_fun = generate_out_funcs(out_dist); #Check that output distribution is
                                        #available and get moment transform
                                        #and membership functions
    zero_prob = False
    
    logc_acc = lambda pTarg: np.random.rand(len(pTarg), 1) < pTarg.reshape(-1,1) #Acceptance function
    d_func = lambda x, s: np.random.normal(x,s) #Construct proposal distribution
    
    p0N = int(p*n) #Target number of seeds
    
    # PREPARE INPUTS
    # dist_obj = getattr(sts, inp_dist['name'])(*inp_dist['parameters'])
    dist_obj = [
    getattr(sts, dist_info['name'])(*dist_info['parameters'])
    for dist_info in inp_dist
    ]
    
    # Sample the input
    # x = dist_obj.rvs((n,d))
    x = np.column_stack([
    dist_obj[j].rvs(size=n)
    for j in range(d)
    ])
    
    # OBTAIN RESPONSE - Assume 2 parameter location scale for now
    p1, p2 = func(x); #Get output distribution parameters
    
    p1 = np.asarray(p1).reshape(-1, 1)
    p2 = np.asarray(p2).reshape(-1, 1)

    # RANK RESPONSES
    [x_sort, par_sort, y_sort, uncert_sort] = psort(out_dist,[p1,p2],x,50);
    
    # Output
    # inp_par = {'func':func,'outdist':out_dist,'dim':d,'t_star':t_star,
    #               'p_0':p,'N':n}
    info_out = [{}]
    
    # Set loop
    L = 0; #Conditional Level
    n_gen = [n]; #Samples at uncond level
    n_pt = [n]; #To correctly compute pF if no conditional levels are needed

    mn = []
    vr = []
    C_F = []
    C = []
    pF_f = []
    pF_f_mean = []
    pF_f_var = []

    # Run loop
    while True: #n_F < n*p
        # Record failure
        p_excd_F = excd_fun(par_sort[:,0], par_sort[:,1], t_star)
        ind_F = logc_acc(p_excd_F)
        n_F = np.sum(ind_F)
        print(f'\t\t\tLevel {L}', end='')
        
        if n_F > n*p:
            print(' - final level\n-----------------------------------')
            print(f'\tN_F: {n_F}')
            # Compute moments of counting distro
            mn.append(np.sum(p_excd_F)) #Mean - can be used in both Poisson and Gaussian approx.
            vr.append(np.sum( p_excd_F*(1-p_excd_F) )) #Variance - for Gaussian approx.;
    
            # Compute scaling constant - N_F
            C_F = min( mn[L]/3/np.sqrt(vr[L]), (n_gen[L]-mn[L])/3/np.sqrt(vr[L]) )
    
    
            bernoullis = [pba.bernoulli(float(p)) for p in p_excd_F if p > 1e-4] #Passing pun precision 
            N_F_pbox = bernoullis[0]
            
            print(f'\tSumming {len(bernoullis)} distributions...', end='')
            for b in bernoullis[1:]: 
                N_F_pbox = N_F_pbox.add(b, dependency='f')
                
            N_F_pbox = N_F_pbox/len(p_excd_F)
            pF_f.append(N_F_pbox)
    
            pF_pbox_mean_pun = ival.I(np.mean(N_F_pbox.left), np.mean(N_F_pbox.right))
            pF_pbox_var_pun = var_bounds_pbox(N_F_pbox)
    
            pF_pbox_mean_form, pF_pbox_var_form = sum_distributions_frechet(bernoullis)
            pF_pbox_mean_form = pF_pbox_mean_form/len(p_excd_F)
            pF_pbox_var_form = pF_pbox_var_form/len(p_excd_F)**2
            
            pF_f_mean.append(ival.imp(pF_pbox_mean_pun, pF_pbox_mean_form))
            pF_f_var.append(ival.imp(pF_pbox_var_pun, pF_pbox_var_form))
            print('done!')
            if plot:
                N_F_pbox.plot()
                plt.title(f'Level {L} - final')
            
            # Fill in new data
            fill_out(info_out, L, x=x_sort, pars=par_sort, y=y_sort, u=uncert_sort,
                    n_gen=n_gen[L], ind_F=ind_F, p_star=p_excd_F, n_F=n_F,
                    p_F_n=N_F_pbox, mn_F=mn[L], vr_F=vr[L], C_F=C_F)
            break
        
        print('\n-----------------------------------')
        print(f'\tN_F: {n_F}')
        
        fill_out(info_out, L, ind_F=ind_F, p_star=p_excd_F, n_F=n_F) #Fill only what is available - the rest can be computed outside
        
    	# CALCULATE LEVEL
        level = y_sort[p0N];

        # Next level probabilities
        p_in_Fi = excd_fun(par_sort[:,0], par_sort[:,1], level)

        # Moments of counting distribution - standard P-SuS for comparison
        mn.append(np.sum(p_in_Fi)) #Mean - can be used in both Poisson and Gaussian approx.
        vr.append(np.sum( p_in_Fi*(1-p_in_Fi) )) #Variance - for Gaussian approx.;

        # Compute scaling constant - N_C
        C.append(min( mn[L]/3/np.sqrt(vr[L]), (n_gen[L]-mn[L])/3/np.sqrt(vr[L]) ))

        bernoullis_Fi = [pba.bernoulli(float(p)) for p in p_in_Fi if p > 1e-4] #Passing pun precision
        print(f'\tNumber of steps in p-boxes: {bernoullis_Fi[0].steps}.')
        N_Fi_pbox_f = bernoullis_Fi[0]

        print(f'\tSumming {len(bernoullis_Fi)} distributions...', end='')
        for b in bernoullis_Fi[1:]:
            N_Fi_pbox_f = N_Fi_pbox_f.add(b, dependency='f')

        N_Fi_pbox_f = N_Fi_pbox_f/len(p_in_Fi)
        pF_f.append(N_Fi_pbox_f)
        
        pF_pbox_mean_pun_i = ival.I(np.mean(N_Fi_pbox_f.left), np.mean(N_Fi_pbox_f.right))
        pF_pbox_var_pun_i = var_bounds_pbox(N_Fi_pbox_f)

        pF_pbox_mean_form_i, pF_pbox_var_form_i = sum_distributions_frechet(bernoullis_Fi)
        pF_pbox_mean_form_i = pF_pbox_mean_form_i/len(p_in_Fi)
        pF_pbox_var_form_i = pF_pbox_var_form_i/len(p_in_Fi)**2
        
        pF_f_mean.append(ival.imp(pF_pbox_mean_pun_i, pF_pbox_mean_form_i))
        pF_f_var.append(ival.imp(pF_pbox_var_pun_i, pF_pbox_var_form_i))
        print('done!')
        if plot:
            N_Fi_pbox_f.plot()
            plt.title(f'Level {L}')

        # Choose seeds
        print('\tChoosing seeds...', end='')
        ind_Fi = logc_acc(p_in_Fi);
        ind_Fi = np.where(ind_Fi != 0)[0]  # extract indices
        ind_Fi = ind_Fi[:int(np.floor(mn[L]))]  # first floor(mn) indices

        if len(n_pt) <= L:
            n_pt.append(len(ind_Fi))
        else:
            n_pt[L] = len(ind_Fi) #This may now be redundant
        
        seeds = x_sort[ind_Fi,:];
        print('done!')

        # Fill in update
        fill_out(info_out, L, t_i=level, ind_Fi=ind_Fi, p_ij=p_in_Fi, n_C=n_pt[L],
                 p_F_i = N_Fi_pbox_f, mn=mn[L], vr=vr[L], C=C[L])
        
        
        # Use MMA to populate the conditional level
        print('\tSampling conditional level...', end='')
        condSamp, p1, p2, pA = pmma(d_func, excd_fun, dist_obj, func, d, seeds,
                                par_sort[ind_Fi,0], par_sort[ind_Fi,1], level,
                                n, n_pt[L]);
        print('done!')
        
        p1 = p1.flatten();
        p2 = p2.flatten();

        if p1.ndim == 1: p1 = p1[:, np.newaxis]
        if p2.ndim == 1: p2 = p2[:, np.newaxis]

        L += 1;
        
        if condSamp.ndim == 2:
            condSamp = condSamp[:, :, np.newaxis]
      
        condSamp_perm = np.transpose(condSamp, (0, 2, 1))
        rows = condSamp_perm.shape[0] * condSamp_perm.shape[1]
        condSamp = condSamp_perm.reshape(rows, d)

        if len(n_gen) <= L:
            n_gen.append(len(condSamp))
        else:
            n_gen[L] = len(condSamp)
        print('\tSorting samples for next level...', end='')
        [x_sort,par_sort,y_sort,uncert_sort] = psort(out_dist,[p1,p2],condSamp,50);
        print('done!\n')
        
        # Timing
        if L == 17:
            highlight('Probability of failure is zero to machine precision\nExiting...', (255,100,0))
            zero_prob = True
            break
    
        info_out.append({})
        
    # Calculate probability of failure
    p_F = {}
    p_F_ipsus = {}
    print('\tFinal calculations')
    print('----------------------------')
    if not zero_prob:
        n_pt = np.array(n_pt)
        n_gen = np.array(n_gen)
        mn = np.array(mn)
        vr = np.array(vr)

        # P_F, mean and variance
        p_F['p_F'] = np.prod(n_pt/n_gen[:L]) * n_F/n_gen[L]
        p_F['mean'] = np.prod(mn/n_gen)
        p_F['var'] = varindepprod(mn, vr)/np.int64(np.prod(n_gen))**2 #Otherwise type == np.int32 and we get overflow

        # Maximal dependence
        C = [info_out[l]['p_Ci']['C'] for l in range(L)]
        C.append(info_out[L]['p_F']['C'])
        C = np.array(C)
        p_F['Cvar'] = varindepprod(mn, C**2 * vr) / np.int64(np.prod(n_gen))**2 #Otherwise type == np.int32 and we get overflow

        #Frechet
        print(f'Multiplying {len(pF_f)} distributions...', end='')
        p_f = pF_f[0]
        for p in pF_f[1:]:
             p_f = p_f.mul(p, dependency='f')
        p_f.plot()
        plt.title('Product distribution - $p_F^{IP-SuS}$')
        print('done!')
        
        p_F_ipsus['bounds'] = p_f
        print(p_f.lo)
        print(p_f.hi)
        # p_F_ipsus_mean_pun = ival.I(np.mean(p_f.left), np.mean(p_f.right))
        # p_F_ipsus_mean_form, p_F_ipsus_var_form = product_distributions_frechet_samples(pF_f)
        # p_F_ipsus['mean'] = ival.imp(p_F_ipsus_mean_pun, p_F_ipsus_mean_form)

        # p_F_ipsus_var_pun = var_bounds_pbox(p_f)
        # p_F_ipsus['var'] = ival.imp(p_F_ipsus_var_pun, p_F_ipsus_var_form)

        p_F_ipsus_mean_pun = ival.I(np.mean(p_f.left), np.mean(p_f.right))
        p_F_ipsus['mean'] = p_F_ipsus_mean_pun
        p_F_ipsus_var_pun = var_bounds_pbox(p_f)
        p_F_ipsus['var'] = p_F_ipsus_var_pun

    else:
        p_F['p_F'] = 0;
        p_F['mean'] = 0;
        p_F['var'] = np.inf;
    

    return p_F, p_F_ipsus, info_out #, p_F_i, p_F_p, p_F_pqd, info_out, inp_par