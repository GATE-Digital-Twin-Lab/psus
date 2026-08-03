import numpy as np
import scipy.stats as sts
from python.ipsus import psus
import matplotlib.pyplot as plt

def rp_objective(x, problem):

    x = np.atleast_2d(x)
    
    return problem(x)

def rp_uncert(x, problem, evals=1, mode='neg', get_true_out=False):
    
    f = rp_objective(x, problem)
    
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
        m = f
        u = (cv * m);
    
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


def RP14(x):

    x = np.atleast_2d(np.array(x, dtype=float))

    nrv_e = 5
    if x.shape[1] != nrv_e:
        raise ValueError(f"RP14 expects {nrv_e} variables, got {x.shape[1]}")

    g = (
        x[:, 0]
        - 32 / (np.pi * x[:, 1] ** 3)
        * np.sqrt((x[:, 2] ** 2 * x[:, 3] ** 2) / 16 + x[:, 4] ** 2)
    )


    if g.size == 1:
        return -g.item()   
    return -g

def RP22(x):

    x = np.atleast_2d(np.array(x, dtype=float))

    nrv_e = 2
    if x.shape[1] != nrv_e:
        raise ValueError(f"RP22 expects {nrv_e} variables, got {x.shape[1]}")

    x1 = x[:, 0]
    x2 = x[:, 1]

    g = (
        2.5
        - (1 / np.sqrt(2)) * (x1 + x2)
        + 0.1 * (x1 - x2) ** 2
    )
    if g.size == 1:
        return -g.item()   
    return -g

def RP111(x):

    x = np.atleast_2d(np.array(x, dtype=float))

    nrv_e = 2
    beta = 5

    if x.shape[1] != nrv_e:
        raise ValueError(f"RP111 expects {nrv_e} variables, got {x.shape[1]}")

    x1 = x[:, 0]
    x2 = x[:, 1]

    g = (beta**2) / 2 - np.abs(x1 * x2)

    if g.size == 1:
        return -g.item()   
    return -g

def RP301(x):

    nrv_e = 12
    g = float('nan')
    msg = 'Ok'
    x = np.array(x, dtype='f')

    n_dim = len(x.shape)
    if n_dim == 1:
        x = np.array(x)[np.newaxis]
    elif n_dim > 2:
        msg = 'Only available for 1D and 2D arrays.'
        return float('nan'), float('nan'), msg

    nrv_p = x.shape[1]
    if nrv_p != nrv_e:
        msg = f'The number of random variables (x, columns) is expected to be {nrv_e} but {nrv_p} is provided!'
    else:
        fcc    = (x[:, 0] - 88.0) / 26.0
        fsy20  = (x[:, 1] - 440.0) / 155.0
        fsur20 = (x[:, 2] - 484.0) / 170.0
        esur20 = (x[:, 3] - 7.05e-02) / 3.15e-02
        fsy6   = (x[:, 4] - 590.5) / 206.5
        fsur6  = (x[:, 5] - 649.0) / 225.0
        esur6  = (x[:, 6] - 7.05e-02) / 3.15e-02

        R_eval = (
            1400.3111111111123
            + 178.54313433106893 * (
                -0.47996197545831665
                + 3.2202141324446765 * (
                    1 + np.sqrt(3 * (
                        0.22828876068479559 * (((fcc - 0.014891593793224708) / 0.65363466455722796) - 0.057738591165263246)**2
                        + 0.46703594474269056 * (((fsy20 - 0.0075279502812227138) / 0.63563352132714701) - 0.07095859659482795)**2
                        + 0.027302060350350831 * (((fsur20 + 0.0062269662126906503) / 0.65961478062944501) + 0.070351362425368158)**2
                        + 0.01 * (((esur20 + 0.00086853225393373644) / 0.66556686290155143) + 0.061332967091213719)**2
                        + 0.01 * (((fsy6 - 0.0054762096422681806) / 0.66810973892056291) + 0.024103565140848895)**2
                        + 0.01 * (((fsur6 - 0.012001999982129828) / 0.69528286069689826) + 0.46722591299744204)**2
                        + 0.01 * (((esur6 + 0.0095208524951042282) / 0.69883254093970915) + 1.0826504541520787)**2
                    ))
                )
            )
        )

        g = x[:, 9] * R_eval - x[:, 10] * x[:, 7] - x[:, 11] * x[:, 8]
  
    return -g    



