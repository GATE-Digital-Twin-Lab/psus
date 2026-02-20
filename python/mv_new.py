import math
import pyuncertainnumber as pun
from pyuncertainnumber import pba
import pyuncertainnumber.pba.operation as op
import scipy.stats as sts

def sum_distributions(distributions): #This is with scipy.stats (rewrite for pun)

    # Initialize first distr
    first_dist = distributions[0]
    mean_total = first_dist.mean()
    var_lower = first_dist.var()
    var_upper = first_dist.var()

    # Iteratively add remaining distr
    for dist in distributions[1:]:

        EX = mean_total
        VX_lower = var_lower
        VX_upper = var_upper

        EY = dist.mean()
        VY = dist.var()

        # Update mean
        mean_total = EX + EY

        # Update table var
        new_lower = (math.sqrt(VX_lower) - math.sqrt(VY))**2
        new_upper = (math.sqrt(VX_upper) + math.sqrt(VY))**2

        var_lower = new_lower
        var_upper = new_upper

    return mean_total, (var_lower, var_upper)


a = [0.8658733829633471, 0.8113785888862199, 0.6552882106982719, 0.2708352347197745, 0.254246607924439, 0.1620913054181461, 0.12930192394973405, 0.09592682587433315, 0.07007347536116451, 0.07005050401659316, 0.04377266084773723, 0.027842516691901964, 0.024701837448865276, 0.017746864804866216, 0.014996482287031149, 0.00539968413701076, 0.0036923511523816676, 0.0018553462610883462, 0.0012056297191706965, 0.0011802532999955063, 0.0010038297156973134, 0.0001242600977393313, 0.0001177013052237979, 0.00011063807164047832, 2.6589826890201716e-05, 2.5825502387389943e-05, 2.1644934384773898e-05, 1.7703759776902196e-05, 1.5586264592011258e-05, 4.468223969294229e-06, 4.118875072902023e-06, 3.2900000120174445e-06, 2.7579995222809e-06, 2.2745555745059995e-06, 1.0383055488095908e-06, 2.805574567771745e-07, 1.690240210421262e-07, 1.0964275360932514e-07, 9.945726906129124e-08, 7.392328050491916e-08, 5.5951208322893045e-08, 2.4761048728221056e-08, 1.865811183990169e-08, 8.900724277996105e-09, 8.720985702530077e-09, 1.3098580627019384e-09, 1.1860943922767471e-09, 4.0281008651470563e-10, 2.9513576469656775e-10, 2.924096691904197e-10, 2.882634366371165e-10, 2.696432517666102e-10, 2.155868855529873e-10, 1.8319722183729937e-10, 1.5395856525271324e-10, 1.456792460082623e-10, 1.1342932032938687e-10, 4.540649036065433e-11, 1.8988530661375188e-11]
print(len(a))
bernoullis = [sts.bernoulli(float(p)) for p in a]

mean, var_interval = sum_distributions(bernoullis)

print("Mean:", mean)
print("Variance interval:", var_interval)


def compute_P_Q(dist): #This is with pun

    mean = dist.mean
    var = dist.var
    x_low = dist.support.lo
    x_high = dist.support.hi

    term_P = (mean - x_low) ** 2 / var
    P = 1 / (1 + term_P)

    term_Q = (mean - x_high) ** 2 / var
    Q = 1 / (1 + term_Q)

    return P, Q


def rowe_t(dist, t):

    mean = dist.mean
    var = dist.var
    x_low = dist.support.lo
    x_high = dist.support.hi

    P, Q = compute_P_Q(dist)

    a = mean + var / (mean - x_low)
    b = mean + var / (mean - x_high)

    L = P * t(x_low) + (1 - P) * t(a)

    R = Q * t(x_high) + (1 - Q) * t(b)

    lower = min(L.left, R.left)
    upper = max(L.right, R.right)

    return pba.Interval(lower, upper)

t = pba.bernoulli(0.3)
rowe_square = rowe_t(t, lambda x: x**2)

print(rowe_square)