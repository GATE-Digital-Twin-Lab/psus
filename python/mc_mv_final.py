import math
import pyuncertainnumber as pun
from pyuncertainnumber import pba
import pyuncertainnumber.pba.operation as op
import scipy.stats as sts
import Interval as ival
import numpy as np



def compute_P_Q_scalar(EX, VX, x_low, x_high):
    term_P = (EX - x_low) ** 2 / VX
    P = 1 / (1 + term_P)

    term_Q = (EX - x_high) ** 2 / VX
    Q = 1 / (1 + term_Q)

    return P, Q

def sample_EX_VX(dist, n_samples=100000):
    mean_low, mean_high = dist.mean.lo, dist.mean.hi
    var_low, var_high = dist.var.lo, dist.var.hi

    EX_samples = np.random.uniform(mean_low, mean_high, n_samples)
    VX_samples = np.random.uniform(var_low, var_high, n_samples)

    return EX_samples, VX_samples

def rowe_t_from_samples(dist, t, EX_samples, VX_samples):

    x_low = float(dist.support.lo)
    x_high = float(dist.support.hi)

    L_samples = []
    R_samples = []

    for EX, VX in zip(EX_samples, VX_samples):

        if abs(EX - x_low) < 1e-12 or abs(EX - x_high) < 1e-12:
            continue

        P, Q = compute_P_Q_scalar(EX, VX, x_low, x_high)

        a = EX + VX / (EX - x_low)
        b = EX + VX / (EX - x_high)

        L = P * t(x_low) + (1 - P) * t(a)
        R = Q * t(x_high) + (1 - Q) * t(b)

        L_samples.append(L)
        R_samples.append(R)

    L_interval = ival.I(min(L_samples), max(L_samples))
    R_interval = ival.I(min(R_samples), max(R_samples))

    return ival.env(L_interval, R_interval)

def rowevar_from_samples(dist, t, t_inv, EX_samples, VX_samples):

    x_low = float(dist.support.lo)
    x_high = float(dist.support.hi)

    L_samples = []
    R_samples = []

    for EX, VX in zip(EX_samples, VX_samples):

        P, Q = compute_P_Q_scalar(EX, VX, x_low, x_high)

        a = EX + VX / (EX - x_low)
        b = EX + VX / (EX - x_high)

        rowe_L = P * t(x_low) + (1 - P) * t(a)
        rowe_R = Q * t(x_high) + (1 - Q) * t(b)

        try:
            nu_lower = t_inv(rowe_L)
            nu_upper = t_inv(rowe_R)
        except:
            continue

        L = (
            (t(nu_lower) - t(x_low))**2 / (nu_lower - x_low)**2
            * (VX + (nu_lower - EX)**2)
        )

        R = (
            (t(nu_upper) - t(x_high))**2 / (nu_upper - x_high)**2
            * (VX + (nu_upper - EX)**2)
        )

        L_samples.append(L)
        R_samples.append(R)

    L_interval = ival.I(min(L_samples), max(L_samples))
    R_interval = ival.I(min(R_samples), max(R_samples))

    return ival.env(L_interval, R_interval)


def sum_distributions_frechet(distributions): 

    # Initialize first distr
    first_dist = distributions[0]
    mean_low = first_dist.mean.lo
    mean_high = first_dist.mean.hi
    var_lower = first_dist.var.lo
    var_upper = first_dist.var.hi
    EX = ival.I(mean_low, mean_high)
    VX = ival.I(var_lower, var_upper)
    
    # Iteratively add remaining distr
    for dist in distributions[1:]:

        EY_low = dist.mean.lo
        EY_high = dist.mean.hi
        EY = ival.I(EY_low, EY_high)
        VY_low = dist.var.lo
        VY_high = dist.var.hi
        VY = ival.I(VY_low, VY_high)

        # Update mean
        EX = EX + EY

        # Update table var
        VX = ival.env(ival.sqrt(VX) - ival.sqrt(VY), ival.sqrt(VX) + ival.sqrt(VY))**2

    return EX, VX 

X = pba.bernoulli(0.8658733829633471)
Y = pba.bernoulli(0.8113785888862199)
# Z = pba.bernoulli(0.6552882106982719)
dist = (X, Y)

EX, VX = sum_distributions_frechet(dist)
print(EX, VX)

def product_distributions_frechet(distributions, n_samples=100000):

    # Initialize first distribution
    first_dist = distributions[0]

    EX = ival.I(first_dist.mean.lo, first_dist.mean.hi)
    VX = ival.I(first_dist.var.lo, first_dist.var.hi)

    for dist in distributions[1:]:

        EY = ival.I(dist.mean.lo, dist.mean.hi)
        VY = ival.I(dist.var.lo, dist.var.hi)

        #Mean of product (Fréchet bounds)
        lower_mean = EX * EY - ival.sqrt(VX * VY)
        upper_mean = EX * EY + ival.sqrt(VX * VY)
        mean_product = ival.env(lower_mean, upper_mean)

        #Second moments
        EX2 = EX**2 + VX
        EY2 = EY**2 + VY

        #Estimate V[X²]
        EX_samples, VX_samples = sample_EX_VX(first_dist, n_samples)

        VZ = rowevar_from_samples(
            first_dist,
            lambda x: x**2,
            lambda x: math.sqrt(x),
            EX_samples,
            VX_samples
        )

        #Estimate V[Y²]
        EY_samples, VY_samples = sample_EX_VX(dist, n_samples)

        VW = rowevar_from_samples(
            dist,
            lambda x: x**2,
            lambda x: math.sqrt(x),
            EY_samples,
            VY_samples
        )

        #Product expectations

        # E[X²Y]
        lower = EX2 * EY - ival.sqrt(VZ * VY)
        upper = EX2 * EY + ival.sqrt(VZ * VY)
        EX2Y = ival.env(lower, upper)
        
        # E[XY²]
        lower = EX * EY2 - ival.sqrt(VX * VW)
        upper = EX * EY2 + ival.sqrt(VX * VW)
        EXY2 = ival.env(lower, upper)

        # E[X²Y²]
        lower = EX2 * EY2 - ival.sqrt(VZ * VW)
        upper = EX2 * EY2 + ival.sqrt(VZ * VW)
        EX2Y2 = ival.env(lower, upper)

        #Higher moments
        E11 = mean_product - EX * EY

        E21 = (
            EX2Y
            - EX2 * EY
            +  2 * (EX**2) * EY
            -  2 * EX * mean_product
        )

        E12 = (
            EXY2
            - EX * EY2
            + 2 * EX * (EY**2)
            - 2 * EY * mean_product
        )

        E22 = (
            -3 * (EX**2) * (EY**2)
            + EX2 * (EY**2)
            + (EX**2) * EY2
            + 4 * EX * EY * mean_product
            - 2 * EY * EX2Y
            - 2 * EX * EXY2
            + EX2Y2
        )

        # Calculate Variance
        var_product = (
            (EX**2) * VY
            + (EY**2) * VX
            + 2 * EX * EY * E11
            + 2 * EX * E12
            + 2 * EY * E21
            + E22
            - E11**2
        )
        
        # Update running product
        EX = mean_product
        VX = var_product
        first_dist = dist

    return EX, VX

class IntervalWrapper:
    def __init__(self, interval):
        self.interval = interval

        # extract endpoints regardless of interval implementation
        if hasattr(interval, "left"):
            self.lo = interval.left() if callable(interval.left) else interval.left
        elif hasattr(interval, "leftval"):
            self.lo = interval.leftval
        else:
            raise ValueError("Cannot determine interval lower bound")

        if hasattr(interval, "right"):
            self.hi = interval.right() if callable(interval.right) else interval.right
        elif hasattr(interval, "rightval"):
            self.hi = interval.rightval
        else:
            raise ValueError("Cannot determine interval upper bound")


class TestDist:
    def __init__(self, lo, hi, mean, var):

        # support interval
        self.support = type("obj", (), {"lo": lo, "hi": hi})

        # mean interval
        if isinstance(mean, tuple):
            interval = ival.I(mean[0], mean[1])
        elif hasattr(mean, "left") or hasattr(mean, "leftval"):
            interval = mean
        else:
            interval = ival.I(mean, mean)

        self.mean = IntervalWrapper(interval)

        # variance interval
        if isinstance(var, tuple):
            interval = ival.I(var[0], var[1])
        elif hasattr(var, "left") or hasattr(var, "leftval"):
            interval = var
        else:
            interval = ival.I(var, var)

        self.var = IntervalWrapper(interval)




X = TestDist(1,10,(2,4),(0.25,4))
Y = TestDist(2,8,(3,6),(1,9)) 
# X = pba.bernoulli(0.8658733829633471)
# Y = pba.bernoulli(0.8113785888862199)
# Z = pba.bernoulli(0.6552882106982719)

dist = (X, Y)
# EX, VX = product_distributions_frechet(dist)
# print(EX, VX)


def product_distributions_frechet_samples(distributions, n_samples=100000):

    first_dist = distributions[0]

    EX = ival.I(first_dist.mean.lo, first_dist.mean.hi)
    VX = ival.I(first_dist.var.lo, first_dist.var.hi)

    for dist in distributions[1:]:

        EY = ival.I(dist.mean.lo, dist.mean.hi)
        VY = ival.I(dist.var.lo, dist.var.hi)

        #Mean
        EXEY = EX * EY
        sqrt_term = ival.sqrt(VX * VY)

        lower_mean = EXEY - sqrt_term
        upper_mean = EXEY + sqrt_term

        mean_product = ival.env(lower_mean, upper_mean)

        #Monte Carlo sampling
        # EX_samples = np.random.uniform(EX.left(), EX.right(), n_samples)
        # VX_samples = np.random.uniform(VX.left(), VX.right(), n_samples)
        # EY_samples = np.random.uniform(EY.left(), EY.right(), n_samples)
        # VY_samples = np.random.uniform(VY.left(), VY.right(), n_samples)
        EX_samples, VX_samples = sample_EX_VX(first_dist, n_samples)
        EY_samples, VY_samples = sample_EX_VX(dist, n_samples)

        #Sample E[XY]
        EXEY_s = EX_samples * EY_samples
        sqrt_term_s = np.sqrt(VX_samples * VY_samples)

        mean_low_s = EXEY_s - sqrt_term_s
        mean_high_s = EXEY_s + sqrt_term_s

        mean_product_s = np.random.uniform(mean_low_s, mean_high_s)

        #Second moments
        EX2_s = EX_samples**2 + VX_samples
        EY2_s = EY_samples**2 + VY_samples

        #Approximate higher moments
        # Estimate V[X²]
        # EX_samples, VX_samples = sample_EX_VX(first_dist, n_samples)
        VZ = rowevar_from_samples(
            first_dist,
            lambda x: x**2,
            lambda x: math.sqrt(x),
            EX_samples,
            VX_samples
        )

        # Estimate V[Y²]
        # EY_samples, VY_samples = sample_EX_VX(dist, n_samples)
        VW = rowevar_from_samples(
            dist,
            lambda x: x**2,
            lambda x: math.sqrt(x),
            EY_samples,
            VY_samples
        )

        #Sample V[X²] and V[Y²]
        VZ_s = np.random.uniform(VZ.left(), VZ.right(), len(EX_samples))
        VW_s = np.random.uniform(VW.left(), VW.right(), len(EX_samples))

        #E[X²Y]
        EX2Y_low = EX2_s * EY_samples - np.sqrt(np.maximum(VZ_s * VY_samples, 0))
        EX2Y_high = EX2_s * EY_samples + np.sqrt(np.maximum(VZ_s * VY_samples, 0))
        EX2Y_s = np.random.uniform(EX2Y_low, EX2Y_high)

        #E[XY²]
        EXY2_low = EX_samples * EY2_s - np.sqrt(np.maximum(VX_samples * VW_s, 0))
        EXY2_high = EX_samples * EY2_s + np.sqrt(np.maximum(VX_samples * VW_s, 0))
        EXY2_s = np.random.uniform(EXY2_low, EXY2_high)

        #E[X²Y²]
        EX2Y2_low = EX2_s * EY2_s - np.sqrt(np.maximum(VZ_s * VW_s, 0))
        EX2Y2_high = EX2_s * EY2_s + np.sqrt(np.maximum(VZ_s * VW_s, 0))
        EX2Y2_s = np.random.uniform(EX2Y2_low, EX2Y2_high)

        #Higher moment terms
        E11_s = mean_product_s - EX_samples * EY_samples

        E21_s = (
            EX2Y_s
            - EX2_s * EY_samples
            + 2 * EX_samples**2 * EY_samples
            - 2 * EX_samples * mean_product_s
        )

        E12_s = (
            EXY2_s
            - EX_samples * EY2_s
            + 2 * EX_samples * EY_samples**2
            - 2 * EY_samples * mean_product_s
        )

        E22_s = (
            -3 * EX_samples**2 * EY_samples**2
            + EX2_s * EY_samples**2
            + EX_samples**2 * EY2_s
            + 4 * EX_samples * EY_samples * mean_product_s
            - 2 * EY_samples * EX2Y_s
            - 2 * EX_samples * EXY2_s
            + EX2Y2_s
        )

        #Variance
        var_product_s = (
            EX_samples**2 * VY_samples
            + EY_samples**2 * VX_samples
            + 2 * EX_samples * EY_samples * E11_s
            + 2 * EX_samples * E12_s
            + 2 * EY_samples * E21_s
            + E22_s
            - E11_s**2
        )

        var_product_s = np.maximum(var_product_s, 0)

        var_product = ival.I(float(var_product_s.min()), float(var_product_s.max()))

        EX = mean_product
        VX = var_product
        first_dist = dist

    return EX, VX

X = pba.bernoulli(0.8658733829633471)
Y = pba.bernoulli(0.8113785888862199)
# Z = pba.bernoulli(0.6552882106982719)
dist = (X, Y)

EX, VX = product_distributions_frechet_samples(dist)
print(EX, VX)

def var_bounds_pbox(N_F_dist_f):
    left = np.asarray(N_F_dist_f.left, float)
    right = np.asarray(N_F_dist_f.right, float)

    upper = -np.inf
    for k in range(len(left) + 1):
        arr = np.concatenate([left[:k], right[k:]])
        upper = max(upper, np.var(arr))

    if np.max(left) <= np.min(right):
        lower = 0.0
    else:
        lower = min(np.var(left), np.var(right))

    VX = ival.I(lower, upper)
    return VX

# var = var_bounds_pbox(X)
# print(var)

