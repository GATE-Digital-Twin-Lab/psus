import numpy as np
import scipy.stats as sts


def mfosd(mn1,mn2,sd1,sd2):

    m = mn2-mn1
    s = 3*(sd2-sd1)
    
    dX = [m-s,m]
    
    if all(x > 0 for x in dX):
        flag = -1
    elif all(x < 0 for x in dX):
        flag = 1
    else: 
        flag = 0
        
    return flag, dX

def sosd(name,loc_sc_1,loc_sc_2,n):
    
    int_F1 = np.zeros((n,1));
    int_F2 = np.zeros((n,1));
    
    loc1, scale1 = loc_sc_1
    loc2, scale2 = loc_sc_2
    
    cdf1 = getattr(sts, name)(loc=loc1, scale=scale1) #Generate appropriate CDFs
    cdf2 = getattr(sts, name)(loc=loc2, scale=scale2)

    x1 = cdf1.ppf([0.0001, 0.9999])
    x2 = cdf2.ppf([0.0001, 0.9999])
    xmin = min(x1[0], x2[0]);
    xmax = max(x1[1], x2[1]);
    x = np.linspace(xmin,xmax,n);
    
    for i in range(n-1): #1:n-1

        int_F1[i+1] = np.trapz(cdf1.cdf(x[:i+1]), x[:i+1])
        int_F2[i+1] = np.trapz(cdf2.cdf(x[:i+1]), x[:i+1])
    
    d_F = int_F1 - int_F2;
    s_d_F = np.sum(d_F);
    
    if s_d_F > 0: flag = -1; #Second one dominant
    elif s_d_F < 0: flag = 1; #First one dominant
    else: flag = 0; #Indistinguishable => identical

    return flag, d_F, int_F1, int_F2, x


def psort(name, pars, x, n):
    pars = np.concatenate(pars, axis=1)
    x    = np.asarray(x)

    dist = getattr(sts, name)

    mn = dist.mean(loc=pars[:, 0], scale=pars[:, 1])
    sd = dist.std(loc=pars[:, 0], scale=pars[:, 1])

    dist_mat = np.column_stack([mn, sd, pars])
    
    distU, rInd = np.unique(dist_mat, axis=0, return_inverse=True)

    mnU  = distU[:, 0]
    sdU  = distU[:, 1]
    parsU = distU[:, 2:]
    lenU = distU.shape[0]
    flag = np.zeros((lenU, lenU))

    for i in range(lenU):
        for j in range(i+1, lenU):

            flag_ji, _ = mfosd(mnU[j], mnU[i], sdU[j], sdU[i])
            flag[j, i] = flag_ji
            
            if flag[j, i] == 0:
                flag_ji, *_ = sosd(name, parsU[j], parsU[i], n)
                flag[j, i] = flag_ji
                
    lower = np.tril(flag)
    flag_skew = lower - lower.T
    fRank_unique = np.sum(flag_skew, axis=1)
    fRank = fRank_unique[rInd]
    sortOrd = np.argsort(-fRank)

    xS   = x[sortOrd]
    parS = pars[sortOrd]
    mnS  = mn[sortOrd]
    sdS  = sd[sortOrd]

    return xS, parS, mnS, sdS


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
    out_dists = ['norm', 'uniform', 'logistic', 'laplace', 't']
    
    if out_dist.lower() not in out_dists:
        raise(Exception('Unrecognized output distribution.'))
        
    dist_obj = getattr(sts, out_dist)  
    
    get_mean_var = lambda loc, sc: (dist_obj(loc, sc).mean(), dist_obj(loc, sc).var())
    excd_fun = lambda loc, sc, lev: dist_obj(loc, sc).sf(lev)

    return get_mean_var, excd_fun


def pmma(propFunc, excdFunc, dist, func, d, seeds, par1, par2, level, nL, nC):
    
    nS = int(np.ceil((nL - nC) / nC))  #Number of states per chain when keeping seeds 
    s = np.std(seeds, axis=0, keepdims=True)  
    s = np.repeat(s, seeds.shape[0], axis=0)
    
    original_propFunc = propFunc
    propFunc = lambda x, k: original_propFunc(x, s[:, :, k])

    pA = np.zeros((nS, d)) #Initializing array for the probability of acceptance
    
    for k in range(nS):
        #Random walk
        urand = np.random.rand(nC, d)  #Uniform numbers to check acceptance
        
        pstar = propFunc(seeds[:, :, k],k)  #Step for random walk

        r = dist.pdf(pstar) / dist.pdf(seeds[:, :, k])  #Uniform pdfs 
        accept = urand < r  #Acceptance criterion                      
        pA[k, :] = np.mean(accept, axis=0)  #Probability of acceptance
        zeta = seeds[:, :, k].copy()  #Copy to zeta
        zeta[accept] = pstar[accept]  #Replace the old samples with the accepted ones where appropriate
        
        #Domain acceptance
        par1[:, k+1] = par1[:, k]
        par2[:, k+1] = par2[:, k]
        
        #Evaluate the objective function at new points
        rows = np.any(accept, axis=1)

        if np.any(rows):
            p1_new, p2_new = func(zeta[rows, :])    
            par1[rows, k+1] = p1_new
            par2[rows, k+1] = p2_new

        seeds[:, :, k+1] = seeds[:, :, k]  #Populate the next state of the chain with the previous one by default

        pInFi = excdFunc(np.column_stack([par1[:, k+1], par2[:, k+1]]), level)  #Prob. y is in Fi
        urandF = np.random.rand(nC, 1)  #Uniform RV's to check acceptance in F
        inFi = urandF < pInFi 

        mask = inFi[:, 0]
        seeds[mask, :, k+1] = zeta[mask, :]  #Replace seeds that have passed with zeta...
        par1[~mask, k+1] = par1[~mask, k]  #... replace parameter values too...
        par2[~mask, k+1] = par2[~mask, k]  #...

    return seeds, par1, par2, pA
  


def varindepprod(mu,var):
    '''
    %%% Variance of the product of two independent distributions.
    %%% Inputs:
    %%% 	mu -  a vector of expectations
    %%% 	var - a vector of variances
    '''
    
    n = len(mu); #Number of RV
    
    if n == 1: varprod = var
    elif n == 2:
        varprod = np.prod(var) + var[0]*mu[1]**2 + var[1]*mu[0]**2
    else:
        v = varindepprod(mu[1:n-1], var[1:n-1]);
        varprod = var[-1]*v + v*mu[-1]**2 + var[-1]*np.prod(mu[:-1]**2)
    
    return varprod

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
    dict_out = {'x':[],'y':[],'u':[],'pars':[],'ind_F':[],'ind_Fi':[],'t_i':[],
    	'p_star':[],'p_ij':[],'N_i':[],'N_C':[],'p_Ci':[],'N_F':[],'p_Fi':[]};
    
    
    # Set loop
    L = 0; #Conditional Level
    n_gen = [n]; #Samples at uncond level
    n_pt = [n]; #To correctly compute pF if no conditional levels are needed
    
    mn = []
    vr = []
    C_F = []
    C = []
    # Run loop
    while True: #n_F < n*p
        # Record failure
        # p_excd_F = excd_fun(par_sort,t_star); #Probability of exceeding threshold
        p_excd_F = excd_fun(par_sort[:,0], par_sort[:,1], t_star)
        ind_F = logc_acc(p_excd_F);
        n_F = np.sum(ind_F);

         
        
        # Compute moments of counting distro
        mn.append(np.sum(p_excd_F)) #Mean - can be used in both Poisson and Gaussian approx.
        vr.append(np.sum( p_excd_F*(1-p_excd_F) )) #Variance - for Gaussian approx.;
        
        # Compute scaling constant - N_F
        C_F.append(min( mn[L]/3/np.sqrt(vr[L]), (n_gen[L]-mn[L])/3/np.sqrt(vr[L]) ));
        
        # Fill in new data
        # sOut = fillOut(sOut, L, x_sort, par_sort, y_sort, uncert_sort, ind_F,
        #                None, None, p_excd_F, None, n_gen[L], None, n_F, None,
        #                None, None, mn[L], vr[L], CF[L]);
        if L not in dict_out:
            dict_out[L] = {'p_Ci': {}, 'p_Fi': {}}           
        dict_out = fillOut(dict_out, L, x_sort, par_sort, y_sort, uncert_sort, ind_F,
                       p_excd_F=p_excd_F, n_gen=n_gen[L], n_F=n_F, mn_F=mn[L],
                       vr_F=vr[L], C_F=C_F[L]);
                                           
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
        
        # Choose seeds
        ind_Fi = logc_acc(p_in_Fi);
        # ind_Fi = find(ind_Fi,floor(mn(L)),'first');
        # ind_Fi = np.where(ind_Fi != 0)[np.floor(mn[L])] #Verify that this is equivalent to the above #TEST
        ind_Fi = np.where(ind_Fi != 0)[0]  # extract indices
        ind_Fi = ind_Fi[:int(np.floor(mn[L]))]  # first floor(mn) indices

    

        n_pt.append(len(ind_Fi)) 
        
        seeds = x_sort[ind_Fi,:];
        	
        # Fill in update
        dict_out = fillOut(dict_out,L,ind_Fi=ind_Fi,l=level,p_excd_Fi=p_in_Fi,
                           n_C=n_pt[L],mn=mn[L],vr=vr[L],C=C[L]);
        
        # Use MMA to populate the conditional level
        condSamp, p1, p2, pA = pmma(d_func, excd_fun, dist_obj, func, d, seeds,
                                par_sort[ind_Fi,0], par_sort[ind_Fi,1], level,
                                n, n_pt[L]);
        
        # p1 = p1(:);
        # p2 = p2(:);
        p1 = p1.flatten();
        p2 = p2.flatten();


        if p1.ndim == 1:
            p1 = p1[:, np.newaxis]
        if p2.ndim == 1:
            p2 = p2[:, np.newaxis]

    
        L += 1;
        
        # Restructuring 'seeds' for sorting
        # rows, _ = condSamp.shape #Find number of rows for reshaping
        # condSamp = reshape(permute(condSamp,[1,3,2]),rows,d); #Reshape samples appropriately
        # condSamp = reshape(permute(condSamp,[1,3,2]),rows,d); #Reshape samples appropriately

        rows = condSamp.shape[0]
        

        print("n_gen next =", rows)
        # --- FIX SHAPE FOR SINGLE-SEED CASE ---
        print("condSamp shape BEFORE fix:", condSamp.shape)
        if condSamp.ndim == 2:
            condSamp = condSamp[:, :, np.newaxis]
        print("condSamp shape AFTER fix:", condSamp.shape)

        condSamp_perm = np.transpose(condSamp, (0, 2, 1))
        condSamp = condSamp_perm.reshape(rows, d)

        n_gen.append(len(condSamp));
        
        [x_sort,par_sort,y_sort,uncert_sort] = psort(out_dist,[p1,p2],condSamp,50);
        
        # Timing
        if L == 17:
            highlight('Probability of failure is zero to machine precision\nExiting...', (255,100,0))
            zero_prob = True
            break
    
    # Calculate probability of failure
    L -= 1; #Adjust to correct number of levels
    
    p_F = {}
    # if ~zero_prob:
    #     # Independence among Bernoulli's
    #     p_F['p_F'] = np.prod( n_pt[1:L]/n_gen[1:L] ) * n_F/n_gen[L+1]
    #     p_F['mean'] = np.prod(mn/n_gen);
    #     p_F['var'] = varindepprod(mn,vr)/np.prod(n_gen**2);
        
    #     # Maximal allowable dependence
    #     # C = [dict_out(:).p_Ci];
    #     # C = [[C(:).C].T; sOut(end).p_Fi.C];
    #     # p_F.Cvar = varindepprod(mn, C**2*vr)/np.prod(n_gen**2);
    #     # Collect C across levels
    #     C_list = [level['p_Ci']['C'] for level in dict_out]   # each is scalar
    #     # Append p_Fi.C from the last level
    #     C_list.append(dict_out[-1]['p_Fi']['C'])
    #     C_vec = np.array(C_list)
    #     p_F['Cvar'] = varindepprod(mn, (C_vec**2) * vr) / np.prod(n_gen**2)
    if not zero_prob:
        n_pt_arr = np.array(n_pt[1:L+1])
        n_gen_arr = np.array(n_gen[1:L+1])
        mn_arr = np.array(mn)
        vr_arr = np.array(vr)

        p_F['p_F'] = np.prod(n_pt_arr/n_gen_arr) * n_F/n_gen[L+1]
        p_F['mean'] = np.prod(mn_arr/n_gen_arr)
        p_F['var'] = varindepprod(mn_arr,vr_arr)/np.prod(np.array(n_gen)**2)




        # Maximal dependence
        C_list = [dict_out[l]['p_Ci']['C'] for l in range(L+1)]
        C_vec = np.array(C_list)
        p_F['Cvar'] = varindepprod(mn_arr, (C_vec**2) * vr_arr) / np.prod(np.array(n_gen)**2)
    else:
        p_F['p_F'] = 0;
        p_F['mean'] = 0;
        p_F['var'] = np.inf;
    
    
    dict_out_F = {'Results': dict_out, 'p_F': p_F, 'Inputs': inp_par}
    
    return p_F, dict_out_F