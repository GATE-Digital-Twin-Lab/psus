from psus import psort, fill_out, generate_out_funcs, pmma, varindepprod, highlight
import numpy as np
import scipy.stats as sts
import matplotlib.pyplot as plt
sts.trapz = np.trapz
import pyuncertainnumber as pun
from pyuncertainnumber import pba


def psus(func, d, t_star, n, p,
         out_dist,
         inp_dist = {'name':'uniform', 'parameters':[0,1]}):
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
    zero_prob = False;
    logc_acc = lambda pTarg: np.random.rand(len(pTarg), 1) < pTarg.reshape(-1,1) #Acceptance function
    
    d_func = lambda x, s: np.random.normal(x,s) #Construct proposal distribution
    
    p0N = int(p*n) #Target number of seeds
    
    # PREPARE INPUTS
    dist_obj = getattr(sts, inp_dist['name'])(*inp_dist['parameters'])
    
    # Sample the input
    x = dist_obj.rvs((n,d));
    
    # OBTAIN RESPONSE - Assume 2 parameter location scale for now
    p1, p2 = func(x); #Get output distribution parameters
    
    p1 = np.asarray(p1).reshape(-1, 1)
    p2 = np.asarray(p2).reshape(-1, 1)

    # RANK RESPONSES
    [x_sort, par_sort, y_sort, uncert_sort] = psort(out_dist,[p1,p2],x,50);
    
    # Output
    inp_par = {'func':func,'outdist':out_dist,'dim':d,'t_star':t_star,
                  'p_0':p,'N':n}
    # info_out = [{'x':[],'y':[],'u':[],'pars':[],'ind_F':[],'ind_Fi':[],'t_i':[],
    # 	'p_star':[],'p_ij':[],'N_i':[],'N_C':[],'p_Ci':[],'N_F':[],'p_Fi':[]}]
    info_out = [{}]
    # info_out = []
    
    # Set loop
    L = 0; #Conditional Level
    n_gen = [n]; #Samples at uncond level
    n_pt = [n]; #To correctly compute pF if no conditional levels are needed

    mn = []
    vr = []
    C_F = []
    C = []
    mn_i = []
    mn_p = []
    vr_i = []
    vr_p = []
    # Run loop
    while True: #n_F < n*p
        # Record failure
        p_excd_F = excd_fun(par_sort[:,0], par_sort[:,1], t_star)
        ind_F = logc_acc(p_excd_F);
        n_F = np.sum(ind_F);

        # Compute moments of counting distro
        mn.append(np.sum(p_excd_F)) #Mean - can be used in both Poisson and Gaussian approx.
        vr.append(np.sum( p_excd_F*(1-p_excd_F) )) #Variance - for Gaussian approx.;

        # Compute scaling constant - N_F
        C_F.append(min( mn[L]/3/np.sqrt(vr[L]), (n_gen[L]-mn[L])/3/np.sqrt(vr[L]) ));


        bernoullis = [pba.bernoulli(float(p)) for p in p_excd_F]
        N_F_dist_i = bernoullis[0]
        N_F_dist_p = bernoullis[0]
        for b in bernoullis[1:]:
            N_F_dist_i = N_F_dist_i.add(b, dependency='i')
            N_F_dist_p = N_F_dist_p.add(b, dependency='p')

        mn_i.append(float(N_F_dist_i.mean.lo)/len(p_excd_F))
        mn_p.append(float(N_F_dist_p.mean.lo)/len(p_excd_F))
        vr_i.append(float(N_F_dist_i.var.lo)/len(p_excd_F)**2)
        vr_p.append(float(N_F_dist_p.var.lo)/len(p_excd_F)**2)
        # Fill in new data
        fill_out(info_out, L, x=x_sort, pars=par_sort, y=y_sort, u=uncert_sort,
                n_gen=n_gen[L], ind_F=ind_F, p_star=p_excd_F, n_F=n_F,
                mn_F=mn[L], vr_F=vr[L], C_F=C_F[L])
                                           
        if n_F > n*p: break
    	
    	# CALCULATE LEVEL
        level = y_sort[p0N];

        # Next level probabilities
        # p_in_Fi = excd_fun(par_sort, level); #Probability of exceedance
        p_in_Fi = excd_fun(par_sort[:,0], par_sort[:,1], level)

        # Counting distribution moments
        mn[L] = np.sum(p_in_Fi) #Mean - can be used in both Poisson and Gaussian approx.
        vr[L] = np.sum( p_in_Fi*(1-p_in_Fi) ) #Variance - for Gaussian approx.;

        # Compute scaling constant - N_C
        C.append(min( mn[L]/3/np.sqrt(vr[L]), (n_gen[L]-mn[L])/3/np.sqrt(vr[L]) ))


        bernoullis_Fi = [pba.bernoulli(float(p)) for p in p_in_Fi]
        N_Fi_dist_i = bernoullis_Fi[0]
        N_Fi_dist_p = bernoullis_Fi[0]
        for b in bernoullis_Fi[1:]:
            N_Fi_dist_i = N_Fi_dist_i.add(b, dependency='i')
            N_Fi_dist_p = N_Fi_dist_p.add(b, dependency='p')
        mn_i[L] = float(N_F_dist_i.mean.lo)/len(p_in_Fi)
        mn_p[L] = float(N_F_dist_p.mean.lo)/len(p_in_Fi)
        vr_i[L] = float(N_F_dist_i.var.lo)/len(p_in_Fi)**2
        vr_p[L] = float(N_F_dist_p.var.lo)/len(p_in_Fi)**2

        # Choose seeds
        ind_Fi = logc_acc(p_in_Fi);
        ind_Fi = np.where(ind_Fi != 0)[0]  # extract indices
        ind_Fi = ind_Fi[:int(np.floor(mn[L]))]  # first floor(mn) indices

        if len(n_pt) <= L:
            n_pt.append(len(ind_Fi))
        else:
            n_pt[L] = len(ind_Fi)
        
        seeds = x_sort[ind_Fi,:];
        	
        # Fill in update
        fill_out(info_out, L, t_i=level, ind_Fi=ind_Fi, p_ij=p_in_Fi, n_C=n_pt[L],
                mn=mn[L], vr=vr[L], C=C[L])
        
        
        # Use MMA to populate the conditional level
        condSamp, p1, p2, pA = pmma(d_func, excd_fun, dist_obj, func, d, seeds,
                                par_sort[ind_Fi,0], par_sort[ind_Fi,1], level,
                                n, n_pt[L]);
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
        
        [x_sort,par_sort,y_sort,uncert_sort] = psort(out_dist,[p1,p2],condSamp,50);
        
        # Timing
        if L == 17:
            highlight('Probability of failure is zero to machine precision\nExiting...', (255,100,0))
            zero_prob = True
            break
    
        info_out.append({})
        
    # Calculate probability of failure
    p_F = {}
    p_F_i = {}
    p_F_p = {}
    if not zero_prob:
        n_pt = np.array(n_pt)
        n_gen = np.array(n_gen)
        mn = np.array(mn)
        vr = np.array(vr)


        mn_i = np.array(mn_i)
        vr_i = np.array(vr_i)
        mn_p = np.array(mn_p)
        vr_p = np.array(vr_p)

        # # P_F, mean and variance
        p_F['p_F'] = np.prod(n_pt/n_gen[:L]) * n_F/n_gen[L]
        p_F['mean'] = np.prod(mn/n_gen)
        p_F['var'] = varindepprod(mn, vr)/np.prod(n_gen**2)

        # Maximal dependence
        C = [info_out[l]['p_Ci']['C'] for l in range(L)]
        C.append(info_out[L]['p_Fi']['C'])
        C = np.array(C)
        p_F['Cvar'] = varindepprod(mn, C**2 * vr) / np.prod(np.array(n_gen)**2)


        p_F_i['mean'] = np.prod(mn_i)
        p_F_i['var'] = varindepprod(mn_i, vr_i)
        p_F_p['mean'] = np.prod(mn_p)
        p_F_p['var'] = varindepprod(mn_p, vr_p)
    else:
        p_F['p_F'] = 0;
        p_F['mean'] = 0;
        p_F['var'] = np.inf;
    

    return p_F, p_F_i, p_F_p, info_out, inp_par