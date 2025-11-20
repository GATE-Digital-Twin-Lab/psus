# -*- coding: utf-8 -*-
"""
Created on Fri Nov  7 15:55:07 2025

@author: P.Hristov
"""

#%% Import
import numpy as np
import scipy.stats as sts


#%% Definitions
def highlight(msg,textcolor=(246,247,246),background=(25,35,45),): # rgb codes (default for Spyder console)
    s = '\33[38;2;'
    for _ in textcolor : s += str(_) + ';'
    s += '48;2'
    for _ in background : s += ';' + str(_) 
    print(('{0}' + msg).format(s+'m'),'\33[0m')


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
    
    # %% PREPARE PRELIMINARIES
    get_mean_var, excd_fun = generate_out_funcs(out_dist); #Check that output distribution is
                                        #available and get moment transform
                                        #and membership functions
    zero_prob = False;
    logc_acc = lambda pTarg: np.random.rand(len(pTarg), 1) < pTarg #Acceptance function
    
    d_func = lambda x, s: np.random.normal(x,s) #Construct proposal distribution
    
    p0N = p*n #Target number of seeds
    
    # %% PREPARE INPUTS
    dist_obj = getattr(sts, inp_dist['name'])(*inp_dist['parameters'])
    
    # %% Sample the input
    x = dist_obj.rvs((n,d));
    
    # %% OBTAIN RESPONSE - Assume 2 parameter location scale for now
    p1, p2 = func(x); #Get output distribution parameters
    y, u = get_mean_var(p1, p2);
    # y = D(:,1);
    # u = D(:,2);
    
    # %% RANK RESPONSES
    [x_sort, par_sort, y_sort, uncert_sort] = psort(out_dist,[p1,p2],y,u,x,50);
    
    # %% Output
    inp_par = {'func':func,'outdist':out_dist,'dim':d,'t_star':t_star,
                  'p_0':p,'N':n}
    dict_out = {'x':[],'y':[],'u':[],'pars':[],'ind_F':[],'ind_Fi':[],'t_i':[],
    	'p_star':[],'p_ij':[],'N_i':[],'N_C':[],'p_Ci':[],'N_F':[],'p_Fi':[]};
    
    
    # %% Set loop
    L = 0; #Conditional Level
    n_gen = [n]; #Samples at uncond level
    n_pt = [n]; #To correctly compute pF if no conditional levels are needed
    
    mn = []
    vr = []
    C_F = []
    C = []
    # %% Run loop
    while True: #n_F < n*p
        # %% Record failure
        p_excd_F = excd_fun(par_sort,t_star); #Probability of exceeding threshold
        ind_F = logc_acc(p_excd_F);
        n_F = np.sum(ind_F);
        
        # %% Compute moments of counting distro
        mn.append(np.sum(p_excd_F)) #Mean - can be used in both Poisson and Gaussian approx.
        vr.append(np.sum( p_excd_F*(1-p_excd_F) )) #Variance - for Gaussian approx.;
        
        # %% Compute scaling constant - N_F
        C_F.append(min( mn[L]/3/np.sqrt(vr[L]), (n_gen[L]-mn[L])/3/np.sqrt(vr[L]) ));
        
        # %% Fill in new data
        # sOut = fillOut(sOut, L, x_sort, par_sort, y_sort, uncert_sort, ind_F,
        #                None, None, p_excd_F, None, n_gen[L], None, n_F, None,
        #                None, None, mn[L], vr[L], CF[L]);
        dict_out = fillOut(dict_out, L, x_sort, par_sort, y_sort, uncert_sort, ind_F,
                       p_excd_F=p_excd_F, n_gen=n_gen[L], n_F=n_F, mn_F=mn[L],
                       vr_F=vr[L], C_F=C_F[L]);
                                           
        if n_F > n*p: break
    	
    	# %% CALCULATE LEVEL
        level = y_sort[p0N];
        
        # %% Next level probabilities
        p_in_Fi = excd_fun(par_sort, level); #Probability of exceedance
        	
        # %% Counting distribution moments
        mn.append(np.sum(p_in_Fi)) #Mean - can be used in both Poisson and Gaussian approx.
        vr.append(np.sum( p_in_Fi*(1-p_in_Fi) )) #Variance - for Gaussian approx.;
        
        # %% Compute scaling constant - N_C
        C.append(min( mn[L]/3/np.sqrt(vr[L]), (n_gen[L]-mn[L])/3/np.sqrt(vr[L]) ))
        
        # %% Choose seeds
        ind_Fi = logc_acc(p_in_Fi);
        # ind_Fi = find(ind_Fi,floor(mn(L)),'first');
        ind_Fi = np.where(ind_Fi != 0)[np.floor(mn[L])] #Verify that this is equivalent to the above
        
        n_pt.append(len(ind_Fi)) 
        
        seeds = x_sort[ind_Fi,:];
        	
        # %% Fill in update
        dict_out = fillOut(dict_out,L,ind_Fi=ind_Fi,l=level,p_excd_Fi=p_in_Fi,
                           n_C=n_pt[L],mn=mn(L),vr=vr(L),C=C[L]);
        
        # %% Use MMA to populate the conditional level
        condSamp, p1, p2 = pmma(d_func, excd_fun, dist_obj, func, d, seeds,
                                par_sort(ind_Fi,1), par_sort(ind_Fi,2), level,
                                n, n_pt[L]);
        
        # p1 = p1(:);
        # p2 = p2(:);
        p1 = p1.flatten();
        p2 = p2.flatten();
        
        L += 1;
        
        # %% Restructuring 'seeds' for sorting
        rows, _ = condSamp.shape #Find number of rows for reshaping
        condSamp = reshape(permute(condSamp,[1,3,2]),rows,d); #Reshape samples appropriately
        # condSamp = reshape(permute(condSamp,[1,3,2]),rows,d); #Reshape samples appropriately
        
        n_gen.append(len(condSamp));
        
        y, u = get_mean_var(p1, p2);
        
        [x_sort,par_sort,y_sort,uncert_sort] = psort(out_dist,[p1,p2],y,u,
                                                     condSamp,50);
        
        # %% Timing
        if L == 17:
            highlight('Probability of failure is zero to machine precision\nExiting...', (255,100,0))
            zero_prob = True
            break
    
    # %% Calculate probability of failure
    L -= 1; #Adjust to correct number of levels
    
    p_F = {}
    if ~zero_prob:
        # % Independence among Bernoulli's
        p_F['p_F'] = np.prod( n_pt[1:L]/n_gen[1:L] ) * n_F/n_gen[L+1]
        p_F['mean'] = np.prod(mn/n_gen);
        p_F['var'] = varindepprod(mn,vr)/np.prod(n_gen**2);
        
        # % Maximal allowable dependence
        # C = [dict_out(:).p_Ci];
        # C = [[C(:).C].T; sOut(end).p_Fi.C];
        p_F.Cvar = varindepprod(mn, C**2*vr)/np.prod(n_gen**2);
        
    else:
        p_F['p_F'] = 0;
        p_F['mean'] = 0;
        p_F['var'] = np.inf;
    
    dict_out_F = {'Results': dict_out, 'p_F': p_F, 'Inputs': inp_par}
    
    return p_F, dict_out_F


def fillOut(out,L,x=None,pars=None,y=None,u=None,ind_F=None,ind_Fi=None,l=None,
            p_excd_F=None,p_excd_Fi=None,n_gen=None,n_C=None,n_F=None,mn=None,
            vr=None,C=None,mn_F=None,vr_F=None,C_F=None):
    
    out[L]['x'] = x; 
    out[L]['pars'] = pars; 
    out[L]['y'] = y; 
    out[L]['u'] = u; 
       
    out[L]['ind_F'] = ind_F; 
    out[L]['ind_Fi'] = ind_Fi
       
    out[L]['t_i'] = l
    out[L]['p_star'] = p_excd_F
    out[L]['p_ij'] = p_excd_Fi
       
    out[L]['N_i'] = n_gen
    out[L]['N_C'] = n_C
    out[L]['N_F'] = n_F
       
    out[L]['p_Ci']['mn'] = mn
    out[L]['p_Ci']['vr'] = vr
    out[L]['p_Ci']['C'] = C
       
    out[L]['p_Fi']['mn'] = mn_F
    out[L]['p_Fi']['vr'] = vr_F
    out[L]['p_Fi']['C'] = C_F

    return out

def generate_out_funcs(out_dist): #These are loc-scale family only for now: legacy from initial version
    out_dists = ['norm', 'uniform', 'logistic', 'laplace', 't'];
    
    if out_dist.lower() not in out_dists:
        raise(Exception('Unrecognized output distribution.'))
        
    dist_obj = getattr(sts, out_dist)  
    
    get_mean_var = lambda loc, sc: (dist_obj(loc, sc).mean(), dist_obj(loc, sc).var())
    excd_fun = lambda loc, sc, lev: dist_obj(loc, sc).sf(lev)

    return get_mean_var, excd_fun

def pmma(prop_f,excd_f,dist,f,d,seeds,par1,par2,level,n_L,n_C):
    '''
    %%% Modified Metropolis algorithm for distributional data.
    %%% Inputs:
    %%% 	propFunc - proposal pdf
    %%% 	excdFun  - exceedence function
    %%% 	dist 	 - distribution for model inputs
    %%% 	func     - performance function
    %%% 	d        - input dimensionality
    %%%		seeds    - seeds for level
    %%% 	par1	 - first parameter of the distribution
    %%% 	par2	 - second parameter of the distribution
    %%% 	y        - mean of response
    %%% 	u        - variance of response
    %%% 	level    - moments of interim threshold variable
    %%% 	nL       - minimum number of samples per level (keep seeds)
    %%% 	nC       - number of seeds
    %%%
    %%% Outputs:
    %%% 	seeds - seeds for the next level of ps
    %%% 	par1  - first parameter of the distribution of the seeds
    %%% 	par2  - second parameter of the distribution of the seeds
    %%% 	pA	  - probability of acceptance
    '''
    
    n_S = np.ceil( (n_L-n_C)/n_C ); #Number of states per chain when keeping seeds
    
    s = np.repeat(np.std(seeds),len(seeds),1); #Proposal distribution std
    prop_f = lambda x: prop_f(x,s);
    
    p_A = np.zeros((n_S,d));#Initializing array for the probability of acceptance
    
    for k in range(n_S):
        #%% Random walk
        u_rand = np.random.rand(n_C, d); #Uniform numbers to check acceptance
        
        p_star = prop_f(seeds[:,:,k]); #Step for random walk
        
        r = dist.pdf(p_star)/dist.pdf(seeds[:,:,k]); #Uniform pdfs 
        accept = u_rand < r; #Acceptance criterion
        p_A[k,:] = np.mean(accept); #Probability of acceptance
        zeta = seeds[:,:,k]; #Copy to zeta
        zeta[accept] = p_star[accept]; #Replace the old samples with the accepted ones where appropriate
        
        #%% Domain acceptance
        par1[:,k+1] = par1[:,k];
        par2[:,k+1] = par2[:,k];
    
        par1[np.any(accept,axis=1),k+1], par2[np.any(accept,axis=1),k+1] = \
                                            f( zeta[np.any(accept, axis=1),:] ) #Evaluate the objective function at new points
        
        seeds[:,:,k+1] = seeds[:,:,k]; #Populate the next state of the chain with the previous one by default
        
        p_in_Fi = excd_f([par1[:,k+1], par2[:,k+1]], level); #Prob. y is in Fi
        
        u_rand_F = np.random.rand(n_C, 1); #Uniform RV's to check acceptance in F
        in_Fi = u_rand_F < p_in_Fi;
    
    	# %% Selection
        seeds[in_Fi,:,k+1] = zeta[in_Fi,:]; #Replace seeds that have passed with zeta...
        par1[not in_Fi, k+1] = par1[not in_Fi, k] #... replace parameter values too...
        par2[not in_Fi, k+1] = par2[not in_Fi, k] #...
    
    return seeds, par1, par2, p_A

def psort(name,pars,mn,vr,x,n):
    '''
    %%% Distributional sorting with modified first- and second-order stochastic dominance
    %%% with Copeland counting.
    %%% This version works with location-scale distributions only.
    %%%
    %%% [xS,parS,mnS,vrS] = psort(name,pars,mn,vr,x,n) takes the name of the distributions to sort - 
    %%% 'Normal', 'Uniform', 'Logistic', 'Laplace', 'Student' (case insensitive), parameters
    %%% of these distributions, mean and variance of the distributions, input data that generated
    %%% the distributions, and the number of quadrature points for second order stochastic dominance.
    %%% If used as part of psus, the parameters, means and variances come from the probabilistic code.
    %%% The function outputs the input data, parameters, means and variances of the sorted distributions.
    '''
    
    %% Preprocess for speed
    if any(vr < 0), error('Variance must be non-negative.'); end
    
    sd = sqrt(vr);
    [distU,~,rInd] = unique([mn,sd,pars],'rows','stable'); %Find unique distros 
    mnU = distU(:,1);
    sdU = distU(:,2);
    parsU = distU(:,3:end);
    
    len = size(distU,1);
    flag = zeros(len);
    
    %% Work
    for i = 1:len
        for j = i+1:len
            flag(j,i) = mfosd(mnU(j),mnU(i),sdU(j),sdU(i));
            if flag(j,i) == 0
                flag(j,i) = sosd(name,parsU(j,:),parsU(i,:),...
                    mnU(j),mnU(i),sdU(j),sdU(i),n);
            end
        end
    end
    
    %% Build score matrix
    flag = tril(flag) - tril(flag)';
    fRank = sum(flag,2);
    fRank = fRank(rInd); %Assign scores to repeated values
    
    %% Sort
    [~,sortOrd] = sort(fRank,'descend');
    
    xS = x(sortOrd,:);
    mnS = mn(sortOrd);
    vrS = vr(sortOrd);
    parS = pars(sortOrd,:);
    
    return xS,parS,mnS,vrS


def mfosd(mn1,mn2,sd1,sd2):
    '''
    %%% Modified first order stochastic dominance for location-scale
    %%% distributions.
    %%% [flag,dX] = mfosd(mn1,mn2,sd1,sd2) - takes in the mean and standard deviations
    %%% of two location-scale distributions to be compared and returns a flag = 1 if
    %%% the first distribution is dominant, flag = -1 if the second distribution is 
    %%% dominant, or flag = 0 if the two distributions are non-dominated to the first order.
    %%% dX contains the differences that gave rise to the ranking.
    ''' 
    # %% Work
    m = mn2-mn1;
    s = 3*(sd2-sd1);
    
    dX = [m-s,m];
    
    if all(dX > 0): flag = -1
    elif all(dX < 0): flag = 1
    else: flag = 0

    return flag, dX

def sosd(name,loc_sc_1,loc_sc_2,mn1,mn2,sd1,sd2,n):
    '''
    %%% Second order stochastic dominance.
    %%% [flag,dF,intF1,intF2,x] = sosd(name,pars1,pars2,mn1,mn2,sd1,sd2,n) compares
    %%% two distributions via second order stochastic dominance, by taking
    %%% in the name of the two-parameter distributions and their means and variances.
    %%% The number of quadrature points is specified through the input n.
    %%% The function returns the dominance flag - 1, if the first distribution is dominant
    %%% -1, if the second distribution is dominant, and 0 if the two distributions are
    %%% non-dominated. To inspect the result, also returns dF, whcih contains the differences 
    %%% between integrals intF1 and intF2, and the quadrature points in the vector x.
    %%%
    %%% The function currently supports only location-scale distributions in line with the use
    %%% of psus.
    '''
    
    xmin = min(mn1 - 5*sd1, mn2 - 5*sd2);
    xmax = max(mn1 + 5*sd1, mn2 + 5*sd2);
    x = np.linspace(xmin,xmax,n);
    
    int_F1 = np.zeros(n,1);
    int_F2 = np.zeros(n,1);
    
    cdf1 = getattr(sts, name)(loc_sc_1) #Generate appropriate CDFs
    cdf2 = getattr(sts, name)(loc_sc_2)
    
    for i in range(n-1): #1:n-1
        int_F1[i+1] = np.trapz(cdf1(x[:i+1]), x[:i+1])
        int_F2[i+1] = np.trapz(cdf2(x[:i+1]), x[:i+1])
    
    d_F = int_F1 - int_F2;
    s_d_F = np.sum(d_F);
    
    if s_d_F > 0: flag = -1; #Second one dominant
    elif s_d_F < 0: flag = 1; #First one dominant
    else: flag = 0; #Indistinguishable => identical

    return flag, d_F, int_F1, int_F2, x


def varindepprod(mu,var):
    '''
    %%% Variance of the product of two independent distributions.
    %%% Inputs:
    %%% 	mu -  a vector of expectations
    %%% 	var - a vector of variances
    '''
    
    n = len(mu); #Number of RV
    
    if n == 1: varprod = var;
    elif n == 2:
        varprod = np.prod(var) + var[0]*mu[1]**2 + var[1]*mu[0]**2;
    else:
        v = varindepprod(mu[1:n-1], var[1:n-1]);
        varprod = var[-1]*v + v*mu[-1]**2 + var[-1]*np.prod(mu[:-1]**2);
    
    return varprod


