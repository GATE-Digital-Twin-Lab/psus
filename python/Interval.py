# -*- coding: utf-8 -*-
"""
Created on Thu Dec 22 09:23:24 2022
@author: peter
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from typing import Union
#from __future__ import 
# from warnings import  warn

EPS = 1e-16 #Proxy machine precision

class Interval:
    
    ######################################################################
    ## STATIC MEMBERS
    ######################################################################
    
    defaultSignificantDigits = 6;
    intervalOrderErrorCode = 0;
    
    leftval = -9.9e99 #These are public for now, but this leaves the possibility
                   #of having inverted intervals
    rightval = 9.9e99
    
    
    
    ######################################################################
    ## helpers
    ######################################################################
    
    
    def __init__(self, left, right=None, dp=None, degen=True):
        if right is None:        
            # 'Void' constructor
            if isinstance(left, Interval):
                right = left.rightval
                left = left.leftval
            # elif hasattr(left, '__iter__'): #A list of np.array
            #     # if hasattr(left[0], '__iter__') #Are we trying to construct an array of Intervals?
            #     right = left[1]
            #     left = left[0]
            # 'None' constructor - iffy
            elif left is None:
                right = None
                left = None
            
            else:
                if not degen: # 'Decimal places' constructor
                    s = f'{left:g}'
                    ls = s.split('.')
                    if dp is None:
                        if len(ls) == 1: dp = 0
                        else: dp = len(ls[1])
                                        
                    right = left + 0.5*10**-dp
                    left = left - 0.5*10**-dp
                else: # Degenerate interval constructor
                    right = left
        
        # else: # Ordinary constructor - COMMENTED OUT TO ALLOW FOR KAUCHER INTERVALS
        #     if (left > right): 
        #         raise(Exception("Math Problem: inverted interval (left > right)"))
        
        self.leftval = left
        self.rightval = right
        
        
    # TESTED CONSTURCTOR
    # def __init__(self, left, right=None, dp=None, degen=True):
    #     if right is None:        
    #         # 'Void' constructor
    #         if isinstance(left, Interval):
    #             right = left.rightval
    #             left = left.leftval
            
    #         # 'None' constructor - iffy
    #         elif left is None:
    #             right = None
    #             left = None
            
    #         else:
    #             if not degen: # 'Decimal places' constructor
    #                 s = f'{left:g}'
    #                 ls = s.split('.')
    #                 if dp is None:
    #                     if len(ls) == 1: dp = 0
    #                     else: dp = len(ls[1])
                                        
    #                 right = left + 0.5*10**-dp
    #                 left = left - 0.5*10**-dp
    #             else: # Degenerate interval constructor
    #                 right = left
        
    #     # else: # Ordinary constructor - COMMENTED OUT TO ALLOW FOR KAUCHER INTERVALS
    #     #     if (left > right): 
    #     #         raise(Exception("Math Problem: inverted interval (left > right)"))
        
    #     self.leftval = left
    #     self.rightval = right
        
        
    def __str__(self):
        self = outerBound(self)
        return f"[{self.leftval:0.{self.defaultSignificantDigits}g}, {self.rightval:0.{self.defaultSignificantDigits}g}]"
    
    def __repr__(self):
        self = outerBound(self)
        return f"interval([{self.leftval:0.{self.defaultSignificantDigits}g}, {self.rightval:0.{self.defaultSignificantDigits}g}])"
    
    
    #---------------------------------------------------------------
    # access/modify methods - don't use dispatch
    #---------------------------------------------------------------
    def left(self, newleft=None):
        if newleft is None:
            return self.leftval
        else:
            if newleft <= self.rightval: #Silently avoid an inverted interval
                self.leftval = newleft
                return self
        
    def right(self, newright=None):
        if newright is None:
            return self.rightval
        else:
            if newright >= self.leftval: #Silently avoid an inverted interval
                self.rightval = newright
                return self
      
    def mid(self): return (self.leftval + self.rightval) / 2.0
    
    def width(self): return (self.rightval - self.leftval)
    
    def isNeg(self):        
        if self.rightval < 0:
            return True
        return False
    
    def isPos(self):
        if self.leftval > 0:
            return True
        return False
    
    def straddles(self): 
        if (self.leftval < 0) & (self.rightval > 0): #Straddling should not include 0 - PH
            return True
        return False
    
    
    #---------------------------------------------------------------
    # operators
    #---------------------------------------------------------------
    
    def __add__(self, other):
        if not isinstance(other, Interval):
            other = Interval(other, degen=True)
            
        return Interval(self.leftval + other.leftval,\
                        self.rightval + other.rightval)
            
    def __radd__(self, other):
        return self.__add__(other)
    
    def __sub__(self, other):
        if not isinstance(other, Interval):
            other = Interval(other, degen=True)
            
        return Interval(self.leftval - other.rightval,\
                        self.rightval - other.leftval)
    
    def __rsub__(self, other):
        if not isinstance(other, Interval):
            other = Interval(other, degen=True)
            
        return Interval(other.leftval - self.rightval,\
                        other.rightval - self.leftval) 

    def __mul__(self, other):
        # print("In __mul__")
        if not isinstance(other, Interval):
            if isinstance(other, np.ndarray):
                return np.array(self) * other #Cast interval as array to use np' methods
            
            if not hasattr(other, '__iter__'): #Scalars
                other = Interval(other, degen=True)
            
        if self.isPos() and other.isPos():
            return Interval(self.leftval * other.leftval,\
                            self.rightval * other.rightval)
        
        m1 = self.leftval * other.leftval
        m2 = self.leftval * other.rightval
        m3 = self.rightval * other.leftval
        m4 = self.rightval * other.rightval
        
        lft = np.min([m1, m2, m3, m4])
        rgt = np.max([m1, m2, m3, m4])
        
        return Interval(lft, rgt)
    
    def __rmul__(self, other):
        return self.__mul__(other)
    
    def __truediv__(self, other):
        if not isinstance(other, Interval):
            other = Interval(other, degen=True)
            
        if other.straddles():
            raise(Exception('Cannot divide with intervals strddling 0.'))
        
        intDen = Interval(1/other.rightval, 1/other.leftval)
        return self * intDen

    def __rtruediv__(self, other):
        if type(other) == int or type(other) == float:
            other = Interval(other)
        
        return other.__truediv__(self)
    
    def __pow__(self, expo):
        if type(expo) == int:
            return self**Interval(expo)
        
        a = self.left()
        b = self.right()
        c = expo.left()
        d = expo.right()
        
        scalarexp   = c == d
        integralexp = scalarexp & (np.floor(c) == c)
        evenexp     = integralexp & ((c % 2) == 0)
        oddexp      = integralexp & ((c % 2) == 1)

        posbase     = (0.0 <= a)
        negbase     = (b <= 0.0)
        zerobase    = ((a <= 0.0) & (0.0 <= b)) #base straddles

        if (integralexp & (c == 1)): return Interval(a, b)
        if (integralexp & nothing(c) & (not zerobase)): return Interval(1, 1)
          
        if (zerobase & (c<=0) & (0.0 <= d)): #Both base and expo straddle 0
            raise(Exception('Cannot compute powers with both base and exponent straddling 0!'))
          
        if scalarexp & nothing(1/c % 1)\
            & nothing(1/c % 2 - 1.0) & (0.0 < c): 
            return Interval(a**c, b**c);
        if scalarexp & nothing(-1/c % 1)\
            & nothing(-1/c % 2 - 1.0) & (c<0.0) & (not zerobase):
            return 1.0/Interval(a**-c, b**-c);
          
        if zerobase & (c <= 0.0):
          	raise(Exception("Cannot compute negative powers when base straddles 0!"))
        
        if integralexp & negbase & evenexp & (0.0 < c):
            return Interval(b**c, a**c)
        
        if integralexp & negbase & evenexp & (d < 0.0):
            return Interval(-a**c, -b**c)
        
        if integralexp & negbase & oddexp:
            return Interval(a**c, b**c);
          
        if integralexp & zerobase & evenexp and not nothing(c):
            return Interval(0, np.maximum(np.abs(a), np.abs(b))**c)
        
        if integralexp & zerobase & oddexp:
            return Interval(a**c, b**c);
          
        if integralexp & posbase & (0<=c):
            return Interval(a**c, b**c);
        
        if integralexp & (not zerobase) & (d<0):
            return Interval(b**d, a**d);
          
        if (1.0<=a) & (1.0<=c):
            return Interval(a**c, b**d);
          
        if posbase:
            mm = a**c;   mmm = mm;
            m = b**d;   mm = np.minimum(mm,m);  mmm = np.maximum(mmm,m);
            m = a**d;   mm = np.minimum(mm,m);  mmm = np.maximum(mmm,m);
            m = b**c;   mm = np.minimum(mm,m);  mmm = np.maximum(mmm,m);
            return Interval(mm,mmm);
           
        raise(Exception("Problem: could not get power. Check base doesn't straddle 0."))
        
    def sqrt(self):
        return sqrt(self)
    
    # def __and__(self, other)
    # def __or__(self, other)
    
    def __lt__(self, other):
        yo = Interval(0, 1)
        if _check_numeric_(other): other = Interval(other)
        if (self.right() < other.left()): yo = Interval(1, 1);
        if (other.right() <  self.left()): yo = Interval(0, 0);
        return yo;
    
    def __gt__(self, other):
        yo = Interval(0, 1)
        if (self.left() > other.right()): yo = Interval(1, 1);
        if (other.left() >  self.right()): yo = Interval(0, 0);
        return yo;
    
    def __le__(self, other):
        yo = Interval(0, 1)
        if (self.right() <= other.left()): yo = Interval(1, 1);
        if (other.right() <  self.left()): yo = Interval(0, 0);
        return yo;
    
    def __ge__(self, other):
        yo = Interval(0, 1)
        if (self.left() >= other.right()): yo = Interval(1, 1);
        if (other.left() >  self.right()): yo = Interval(0, 0);
        return yo;
    
    def __eq__(self, other):
        return ((self.leftval == other.leftval) &\
                (self.rightval == other.rightval)) |\
          (nothing(self.leftval  - other.leftval) &
            nothing(self.rightval - other.rightval));
          
    def __ne__(self, other):
        return not (self == other)
        
    def __neg__(self):
        return Interval(-self.right(), -self.left())
    
    def __pos__(self):
        return self

    def mirror(self):
        a = np.maximum(np.abs(self.leftval), self.rightval)
        return I(-a, a)
    pm = mirror
    
    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        # print("In __array_ufunc__")
        if ufunc is np.sqrt and method == "__call__":
            return sqrt(self)
        if ufunc is np.add and method == "__call__":
            return inputs[0] + np.array(inputs[1])
        if ufunc is np.multiply and method == "__call__":
            return inputs[0] * np.array(inputs[1]) #This should only be invoked if left is array and right is an interval 
        if ufunc is np.divide and method == "__call__":
            return inputs[0] / np.array(inputs[1])
        if ufunc is np.rad2deg and method == "__call__":
            return I(np.rad2deg(inputs[0].leftval), np.rad2deg(inputs[0].rightval))
        return NotImplemented
    
    def plot(self, raise_up=0.0, ax=None, label=None, **kwargs):
        if ax is None:
            ax = plt.subplot()
        
        if 'color' not in kwargs.keys() and 'c' not in kwargs.keys():
            kwargs['color'] = 'k'
        if 'lw' not in kwargs.keys(): kwargs['lw'] = 2
        
        ax.plot([self.leftval, self.leftval], [0,1+raise_up], label=label, **kwargs)
        ax.plot([self.rightval, self.rightval], [0,1+raise_up], **kwargs)
        ax.plot([self.leftval, self.rightval], [1+raise_up, 1+raise_up], **kwargs)
        ax.plot([self.leftval, self.rightval], [0,0], **kwargs)
        
            
I = Interval
#---------------------------------------------------------------
# functions
#---------------------------------------------------------------

def abs(a):
    if 0.0 <= a.left():
        return a
      
    if (a.rightval <= 0.0):
        return -a;

    neg_a = -a
    tmp = max(a, neg_a);
       
    return Interval(np.maximum(0.0, tmp.leftval), tmp.rightval);

def inside(a, b):
    '''Is the interval a nested in the interval b'''
    if not isinstance(a, Interval): a = Interval(a)
    if not isinstance(b, Interval): b = Interval(b) #Allow to test if scalars are inside an interval
    return (a.leftval  >= b.leftval)  &\
			(a.leftval  <= b.rightval) &\
			(a.rightval  <= b.rightval) &\
			(a.rightval >= b.leftval)
            
def overlap(a, b):
    ''' This function *measures* overlap;  it doesn't test for it.
     If the intervals just touch, overlap will return zero!'''
    if (a.rightval <= b.rightval):
        if (b.leftval < a.rightval):
            return a.rightval - max(b.leftval, a.leftval);
    elif (a.leftval < b.rightval):
        return b.rightval - max(b.leftval, a.leftval);
    return 0.0

def sqrt(a, impose_range=False, preserve_sign=False):
    if impose_range: a = cut(a, 0, 'left')
    if preserve_sign and a.straddles():
        sqrt_pos = Interval(0, a.rightval).sqrt()
        sqrt_neg = Interval(0, -a.leftval).sqrt()
        return Interval(-sqrt_neg.rightval, sqrt_pos.rightval)
    
    if a.isNeg() or a.straddles():
        raise(Exception("Math Problem: square root of an at least partially negative interval"))
	
    return Interval(np.sqrt(a.leftval), np.sqrt(a.rightval));
    
def min(a, b):
    return Interval( np.minimum(a.left(), b.left()), np.minimum(a.right(), b.right()) )

def max(a, b):
    return Interval( np.maximum(a.left(), b.left()), np.maximum(a.right(), b.right()) )
        
def env(a, b):
    '''Interval union'''
    return Interval( np.minimum(a.left(), b.left()), np.maximum(a.right(), b.right()) )

def imp(a, b):
    '''Interval intersection'''
    return Interval(np.maximum(a.left(), b.left()), np.minimum(a.right(), b.right()) )

def cut(ival, cutter, side):
    if side == 'left':
        left = cutter if cutter > ival.leftval and cutter < ival.rightval else ival.leftval
        return Interval(left, ival.rightval)
    if side == 'right':
        right = cutter if cutter < ival.rightval and cutter > ival.leftval else ival.rightval
        return Interval(ival.leftval, right)
    raise Exception("Unrecognized option for 'side'; choose 'left' or 'right'. ")

    
def exp(exponent): #euler
    if not isinstance(exponent, Interval):
        exponent = Interval(exponent)
    return Interval( np.exp(exponent.leftval),\
                        np.exp(exponent.rightval) );
    
def log (x, l=10, impose_range=False):
    '''Take the log of the interval x. Log base 10 is default '''
    if impose_range and x.rightval > 0: x = cut(x, EPS, 'left') #Not imposing range on base for now
    if not x.isPos() | l <= 0.0:
        raise(Exception("Logarithm base and operand must be positive."))
		
    return Interval(np.log(x.leftval)/np.log(l),\
                 np.log(x.rightval)/np.log(l))
            
def ln(x, impose_range=False):
    if impose_range and x.rightval > 0: x = cut(x, EPS, 'left')
    if not x.isPos():
        raise(Exception("Logarithm base must be positive."))
    return Interval(np.log(x.leftval), np.log(x.rightval))
    
def sin(x):
    l = np.sin(x.left())
    r = np.sin(x.right())
    
    yLeft = np.minimum(l, r)
    yRight = np.maximum(l,r)
    
    if x.isPos():
        r1 = (x.left() - 0.5*np.pi) / 2/np.pi
        t1 = (np.ceil(r1) - r1) * 2*np.pi
        
        r2 = (x.left() - 1.5*np.pi) / 2/np.pi
        t2 = (np.ceil(r2) - r2) * 2*np.pi
        
        if t1 <= x.width():
             yRight = 1
        if t2 <= x.width():
            yLeft = -1
    elif x.isNeg():
        return -sin(-x)   
    else: #straddles
        a = x.left()
        b = x.right()
        if (a <= -1.5*np.pi) or (b >= 1.5*np.pi):
            yLeft = -1
            yRight = 1    
        else:
            if a <= -0.5*np.pi:
                yLeft = -1
            if b >= 0.5*np.pi:
                yRight = 1
       
    return Interval(yLeft, yRight)

def cos(x):
    return sin(x + np.pi/2)

def tan(x): #No range can be imposed here w/o making a bunch of assumptions
    a = x.leftval
    b = x.rightval
    hpi = np.pi/2
    
    if a < hpi or b > hpi:
        raise(Exception('Math Problem: Output interval extends to +/- infinty.'))
    
    return Interval(np.tan(a), np.tan(b))

def cot(x):
    return 1/tan(x)
    
def asin(x, impose_range=False):
    if impose_range: x = imp(x, Interval(-1,1))
    
    a = x.leftval
    b = x.rightval
    
    if a < -1 or b > 1:
        raise(Exception(f'Math Problem: ArcSine undefined on interval {x}'))
        
    return Interval(np.asin(a), np.asin(b))

def acos(x, impose_range=False):
    if impose_range: x = imp(x, Interval(-1,1))
    if x.leftval < -1 or x.rightval > 1:
        raise(Exception(f'Math Problem: ArcCosine undefined on interval {x}'))
        
    return -asin(x) + np.pi/2

def atan(x):
    return Interval(np.atan(x.leftval), np.atan(x.rightval))
    
def acot(x):
    return atan(x) + np.pi/2

def nothing(nearZero):
    return np.abs(nearZero) < 1.0e-100;

def dual(x):
    if isinstance(x, Interval):
        return Interval(x.rightval, x.leftval)
    if hasattr(x, '__iter__'): #Can handle list or np.array
        return np.array([dual(intv) for intv in x])

def pm(x):
    if isinstance(x, Interval):
        return x.pm()
    if isinstance(x, int) or isinstance(x, float) or isinstance(x, np.float64):
        return Interval(x).pm()

def dot(a:np.ndarray, b:np.ndarray):
    #This works, but use np's code for robustness
    a = np.array(a)
    b = np.array(b)
    
    dim_a = a.ndim
    dim_b = b.ndim
    
    #DIMENSIONAL ERROR CHECKING
    #Once dimensions are established, only collect important sizes
    if dim_a > 2 or dim_b > 2:
        raise Exception('Arrays with dimensions >2 are not supported.')
    
    if dim_a < 2: a = a.reshape(-1,1)
    if dim_b < 2: b = b.reshape(-1,1)
    
    shp_a = a.shape
    shp_b = b.shape
        
    if shp_a == (1,1) or shp_b == (1,1): result = a*b
    else:
        if shp_a[1] != shp_b[0]:
            if 1 in shp_a:
                a = a.T #Enable dot product of originally 1D arrays of the same size
                shp_a = a.shape
            elif 1 in shp_b:
                b = b.T #Enable dot product of originally 1D arrays of the same size
                shp_b = b.shape
            else: raise ValueError('Shapes of arrays do not conform.')
            
            if shp_a[1] != shp_b[0]:
                raise ValueError('Shapes of arrays do not conform.')
                
        result = np.full((shp_a[0], shp_b[1]), np.nan, dtype=object)
        
        for ra in range(shp_a[0]):
            row_a = a[ra,:]
            for cb in range(shp_b[1]):
                col_b = b[:,cb]
                d = 0 
                for ai, bi in zip(row_a, col_b):
                    d += ai*bi
                
                result[ra, cb] = d
            
    return result.squeeze()


def mid(ivals:Union[list, np.ndarray]) -> np.ndarray:
    return np.array([ival.mid() for ival in ivals])

def width(ivals:Union[list, np.ndarray]) -> np.ndarray:
    return np.array([ival.width() for ival in ivals])
    
def oB(x, left, sigDig):
    strRep = f'{x:0.{sigDig}g}'
    xPrint = float(strRep)
    
    prec = np.floor(np.log10(np.abs(x+1e-100)) + 1.0) - sigDig; #Silence log warning by adding 1e-100 to x
    leastsigdigit = 10.0**prec * .5;
    
    if left:
        if xPrint > x and not nothing(xPrint - x):
            x -= leastsigdigit
    else: #right bound
        if xPrint < x and not nothing(xPrint - x):
            x += leastsigdigit
            
    return np.round(x, sigDig)

def outerBound(x):
    sigDig = Interval.defaultSignificantDigits
    a = oB(x.left(), True, sigDig)
    b = oB(x.right(), False, sigDig)
    
    return Interval(a, b)

def _check_numeric_(a):
    #TODO: Also implement checks for np's scalar types
    return type(a) == int or type(a) == float 

#Plotting

def plot2d(interval_list, ax=None, label=None, **kwargs):
    if ax is None:
        ax = plt.subplot()
    
    kwargs['facecolor'] = 'none'
    if 'color' not in kwargs.keys(): kwargs['edgecolor'] = 'k'
    else: kwargs['edgecolor'] = kwargs['color']; kwargs.pop('color')
    if 'lw' not in kwargs.keys(): kwargs['lw'] = 2
    if label: kwargs['label']=label
    
    #Consider passing a list of lists to plot multiple rectangles
    int1 = interval_list[0]
    int2 = interval_list[1]
    
    ax.add_patch(Rectangle([int1.leftval,int2.leftval], int1.width(), int2.width(), **kwargs))
    ax.autoscale()
    
#     #-------------------------------------------------------------------------
#     #other methods - decide if they need to be implemented
#     #-------------------------------------------------------------------------

#     #-------------------------------------------------------------------------
#     friend Interval probsum( const Interval& a, const Interval& b );
#     #-------------------------------------------------------------------------
#     friend Interval gammafunc( const Interval& a);
#     #-------------------------------------------------------------------------
#     #friend Interval adec( const Interval& a, const Interval& c );
#     #-------------------------------------------------------------------------
#     #friend Interval sdec( const Interval& a, const Interval& c );
#     #-------------------------------------------------------------------------
#     friend Interval mdec( const Interval& a, const Interval& c );
#     #-------------------------------------------------------------------------
#     #friend Interval ddec( const Interval& a, const Interval& c );
#     #-------------------------------------------------------------------------
#     friend Interval map( const Interval &a, double(*functionP)(double));
#     #-------------------------------------------------------------------------
#     friend Interval map( const Interval &a, const Interval &b, double(*functionP)(double,double));
#     #-------------------------------------------------------------------------
#     friend Interval abs (const Interval& a);
#     #------------------------------------------------------------------------
#     #friend Interval gamma (const Interval& a);
#     #------------------------------------------------------------------------
#     friend Interval erf (const Interval& a);
#     #------------------------------------------------------------------------
#     friend Interval lambertw (const Interval& a);
#     #------------------------------------------------------------------------
#     friend Interval probit (const Interval& a);
#     #------------------------------------------------------------------------
#     friend Interval logit (const Interval& a);
#     #------------------------------------------------------------------------
#     friend Interval sign (const Interval& a);
#     #------------------------------------------------------------------------
#     #friend Interval arbmap (const Interval& a);