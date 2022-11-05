# *****************************************************************************
# This code has been developed by Antoine Herrmann to verify the derivation
# of the RKKY response functions, made in one, two and three dimension in the 
# source paper, by a numerical method.
# 
# For any support, help about the code or the calculation, use of this code, or 
# any other necessity to share this code, please contact the author at :
# antoine67.travail@laposte.net
#
# All rights reserved
# 
# Source : Antoine Herrmann, "Temperature dependence of the RKKY response 
# function through Sommerfeld’s first order expansion", 2021
# *****************************************************************************

# Summary :
# 
# 044 - 324  : Fitting package's routines
# 325 - 357  : Analysis routines
# 358 - 395  : Mathematical function ( Fermi-Dirac and SinIntegral )
# 396 - 495  : RKKY functions definitions
# 
# 496        : Main Program
# 
# 510 - 527  : Constant definitions
# 528 - 555  : Integration boxes definitions
# 556 - 631  : Integration
# 632 - 694  : Comparison of Integratd and Analytical functions
# 695 - 727  : Fitting the comparison
# 728 - 771  : Fitting the Validity range
# 729 - 781  : Preparing vectors for plots
# 782 - 919  : Making plots and figures



from math import floor, sqrt, exp, sin, cos, pi
from scipy.special import jv, yv
from scipy.optimize import curve_fit
import numpy as np
import random as rdm
import time
import os
from matplotlib import pyplot as plt


"""
*******************************************************************************
*******************************************************************************
 Fitting package
*******************************************************************************
*******************************************************************************
"""    


def Fit_Curve( Points, fun ):
    # Set bounds and convergence criteria for fit
    # Set bounds and convergence criteria for fit
    if fun.__name__ == 'Rational_Square_Fun':
        Points = Shape_Points_For_Inverse_Square_Fit( Points )
        bounds = Set_Rational_Square_Bounds( Points )
        
    if fun.__name__ == 'Inverse_Fun':
        bounds = Set_Inverse_Bounds( Points )
        
    conv_crit = 1e-6
    
    # satisfaction variable
    sat = False
    
    # set initial parameters and research area delta
    old_para = Set_Initial_Guess( len(bounds[0]), bounds )
    delta = Set_Initial_Delta( bounds )
    
    security = 0
    
    while security < 1000:
        new_para = Generate_New_Para( bounds, delta, old_para )
        new_para, g_rule, n_rule = Select_Parameters( Points, old_para, new_para, delta, fun  )
        
        d_error = Compare_Error( Points, old_para, new_para, fun )
        
        s_rule = Selection_Rule( d_error, conv_crit )
        sat = Crossing( g_rule, s_rule, n_rule )
        
        old_para = new_para
        delta = Select_New_Delta( old_para, new_para, g_rule, delta, d_error )
        
        security += 1
    return new_para
   
    
"""
***************************************************************************
 Selectors
***************************************************************************
""" 


def Select_Parameters( Points, old, new, delta, fun ):
    
    # look if the new parameters are not too close from the previous ones
    d_p = [ old[i] - new[i] for i in range( len( old ) ) ]
    n_d_p = Norm( d_p )
    
    s = True
    if n_d_p < delta / 10:
        s = False
        
    old_error = Chi_2( [ Residual( fun, Points[0][i], Points[1][i], old ) for i in range( len( Points[0] ) ) ] )
    new_error = Chi_2( [ Residual( fun, Points[0][i], Points[1][i], new ) for i in range( len( Points[0] ) ) ] )
    
    
    # If the error is smaller, return the new parameters
    if old_error >= new_error:
        return new, True, s
    # Else, return the new parameters with a given probability (to allow the process to jump out of local minima)
    else:
        r = rdm.random()
        if r > exp( 1 - new_error / old_error ):
            return new, True, s
        else:
            return old, False, s

def Select_New_Delta( old, new, rule, delta, d_error ):
    # If the parameter changed
    if rule == True:
        # delta is reduced
        delta = delta * ( 1 - 0.5 )
    else:
        # delta is enlarged
        delta = delta * ( 1 + 0.5 )
    
    return delta

def Selection_Rule( d_error, conv_crit ):
    if d_error < conv_crit:
        return True
    else:
        return False

def Crossing( bool_1, bool_2, bool_3 ):
    if bool_1 == True and bool_2 == True and bool_3 == True :
        return True
    else:
        return False


"""
***************************************************************************
 Generators
***************************************************************************
""" 


def Shape_Points_For_Inverse_Square_Fit( Points ):
    # shapes the points for the fit by cutting the initial set of points when the derivative becomes null
    length = len( Points[0] )
    der_init = ( Points[1][1] - Points[1][0] ) / ( Points[0][1] - Points[0][0] )
    for i in range( length - 2 ):
        der2 = ( Points[1][length-i-1] - Points[1][length-i-3] ) / ( Points[0][length-i-1] - Points[0][length-i-3] )
        if abs( der2 ) > 0.75 * abs( der_init ) :
            new_x = Points[0][:length-i]
            new_y = Points[1][:length-i]
            break
    new_points = np.zeros( ( 2, len(new_x) ) )
    new_points[0] = new_x
    new_points[1] = new_y
    
    return new_points


def Generate_New_Para( bounds, delta, para ):
    # Generates new parameters randomly belonging to the research area delta
    step_para = np.zeros( len( para ) )
    for i in range( len( para ) ):
        step_para[i] = ( delta / 2 ) * ( rdm.random() - 0.5 )
        
    norm = Norm(step_para)
    if norm > delta:
        step_para = [ step_para[i] / ( delta / norm ) for i in range( len( step_para ) ) ]
        
    new_para = [para[i] + step_para[i] for i in range( len( step_para ) ) ]
    for i in range( len( para ) ):
        if new_para[i] < bounds[0][i]:
            new_para[i] = bounds[0][i]
        if new_para[i] > bounds[1][i]:
            new_para[i] = bounds[1][i]
            
    return new_para
    

"""
***************************************************************************
 Bounds and parameters
***************************************************************************
""" 


def Set_Rational_Square_Bounds( Points ):    
    def Set_ab_Bounds( Points ):
        deriv = np.zeros( len( Points[1] ) - 1 ) 
        for i in range( len( Points[1] ) - 1 ):
            deriv[i] = Points[1][i+1] - Points[1][i]
        mean_deriv = Mean( deriv )
        nor_deriv =np.zeros( len( deriv ) ) 
        for i in range( len( deriv ) ):
            nor_deriv[i] = deriv[i] - mean_deriv
        dmax = max( nor_deriv )
        dmin = min( nor_deriv )
        
        # find the points where the experiments diverges
        imax = 0
        imin = 0
        for i in range( len( deriv ) ):
            if nor_deriv[i] == dmax:
                imax = i
            if nor_deriv[i] == dmin:
                imin = i
    
        b1 = Points[0][imin-1]
        b2 = Points[0][imax+1]            
                
        if imax < imin:
            a1 = 0
            a2 = 10
        else:
            a1 = -10
            a2 = 0
        return a1, a2, b1, b2
    
    def Set_c_Bounds( Points ):
        c = ( Points[1][-1] - Points[1][0] ) / ( Points[0][-1] - Points[0][0] )
        if c <= 0:
            c1 = c * 2
            c2 = c / 2
        else:
            c1 = c / 2
            c2 = c * 2  
        return c1, c2
    
    def Set_d_Bounds( Points ):
        if Points[1][0] > 0:
            d1 = Points[1][0] / 2
            d2 = Points[1][0] * 2
        else:
            d1 = Points[1][0] * 2
            d2 = Points[1][0] / 2
        return d1, d2


    a1, a2, b1, b2 = Set_ab_Bounds( Points )
    c1, c2 = Set_c_Bounds( Points )
    d1, d2 = Set_d_Bounds( Points )
    
    return [ [ a1, b1, c1, d1 ], [ a2, b2, c2, d2 ] ]
     
def Set_Inverse_Bounds( Points ):
    def Set_a_Bounds( Points ):
        deriv = np.zeros( len( Points[1] ) - 1 )
        for i in range( len( Points[1] ) - 1 ):
            deriv[i] = Points[1][i+1] -  Points[1][i]
        m_d = Mean(deriv)
        if m_d <= 0:
            a1 = - 0.5 * Points[0][-1] * Points[1][-1] * m_d / 2
            a2 = - Points[0][-1] * Points[1][-1] * m_d / 2
        else:
            a2 = - 0.5 * Points[0][-1] * Points[1][-1] * m_d / 2
            a1 = - Points[0][-1] * Points[1][-1] * m_d / 2
        
        return a1, a2
    """
    def Set_b_Bounds( Points ):
        y_max = Points[1][-1]
        deriv = np.zeros( len( Points[1] ) - 1 )
        for i in range( len( Points[1] ) - 1 ):
            deriv[i] = Points[1][i+1] -  Points[1][i]
        d_max = Mean(deriv)
        if d_max <= 0:
            b2 = - y_max / d_max
            b1 = y_max / d_max
        else:
            b1 = - y_max / d_max
            b2 = y_max / d_max
        return b1, b2
    """
    a1, a2 = Set_a_Bounds( Points )
    #b1, b2 = Set_b_Bounds( Points )
    return [ [ a1 ], [ a2 ] ] # [ [ a1, b1 ], [ a2, b2 ] ] 

def Set_Initial_Guess(n_parameters, bounds):
    guess = np.zeros(n_parameters)
    for i in range( n_parameters ):
        guess[i] = ( bounds[1][i] + bounds[0][i] ) / 2
    return guess

def Set_Initial_Delta(bounds):
    vect = [ bounds[1][i] - bounds[0][i] for i in range( len( bounds[0] ) ) ]
    n = Norm(vect)
    return n/4

def Set_Satisfaction( error_var, conv_crit ):
    if error_var >= conv_crit:
        return False
    else:
        return True
    
    
"""
***************************************************************************
 Residuals and derivative functions
***************************************************************************
"""


def Self_Made_Function(x, Para):
    a = Para[0]
    b = Para[1]
    c = Para[2]
    
    return ( x**2 * exp( - a * x**2 + b * x ) + 1 ) * exp( - c * x**2 )

def Rational_Square_Fun(x, Para):
    
    a = Para[0]
    b = Para[1]
    c = Para[2]
    d = Para[3]
    
    if x == b:
        x += 1e-3
    return a / (x-b)**2 + c*x + d

def Inverse_Fun(x, Para):
    a = Para[0]
    if x == 0:
        x == 1e-3
    return a / x 

def inverse_fun(x, a):
    return a / x

def Residual( fun, x, y, Para ):
    return fun(x, Para) - y

def Chi_2( residue ):
    return np.matmul( np.transpose( residue ), residue )

def Compare_Error( Points, old, new, fun ):
    d_e = Chi_2( [ Residual( fun, Points[0][i], Points[1][i], old ) for i in range( len( Points[0] ) ) ] ) - Chi_2( [ Residual( fun, Points[0][i], Points[1][i], new ) for i in range( len( Points[0] ) ) ] )
    return d_e


"""
***************************************************************************
 Analysis Routines Definitions
***************************************************************************
"""


def Find_Extrema(vect):
    length = len(vect)
    ext = []
    # Derive the vector
    d = [0 for i in range(length-1)]
    for i in range(length-1):
        d[i] = vect[i+1] - vect[i]
    # look for the zeros of the derivative
    for i in range(length-2):
        if d[i+1]*d[i] <= 0:
            ext.append(i+1)
    return ext

def Norm(vect):
    norm = 0
    for i in range(len(vect)):
        norm += vect[i]**2
    return sqrt(norm)

def Mean( vect ):
    mean = 0
    for i in range( len( vect ) ):
        mean += vect[i]
    return mean / len( vect )
    

"""
*******************************************************************************
*******************************************************************************
 Functions Definitions
*******************************************************************************
*******************************************************************************
"""


""" ---   Fermi-Dirac distribution function   --- """

def Fermi_Dirac(k, T):
    if T == 0:
        if k<=k_F:
            return 1
        else:
            return 0
    else:
        if k <= 0:
            return 1
        else:
            return 1 / ( exp( hbar**2*( k**2 - k_F**2 ) / (2*m*k_B*T) ) + 1)
    
""" ---   Definition of the Sine Integral function   --- """

def Si(x):
    t = 0
    step = 0.00001
    si = 0
    while t < x:
        if t==0:
            si = 0
        else:
            si += sin(t)/t*step
        t += step
    return si


"""
***************************************************************************
 1D RKKY functions
***************************************************************************
"""


""" --- Analytical RKKY 1D Response function ( Sommerfeld ) --- """

def Chi_1_Sommerfeld(r, T):
    pre_fact = 4*pi*m / hbar**2 #/ (2*pi)**2
    T_fact = pi**2/24*( T / T_F )**2
    
    x_F = k_F*r
    if x_F == 0:
        x_F = 1e-8
    
    chi_1_T0 = pi/2 - Si(2*x_F)
    chi_1_T = x_F * cos(2*x_F) - 2 * sin(2*x_F)
    chi_1 = pre_fact*( chi_1_T0 - T_fact * chi_1_T )
    return chi_1

""" --- Analytical RKKY 1D Response function ( Exact ) --- """

def Chi_1_Exact(k,r,T):
    
    if k == 0:
        k = 1e-8
        
    pre_fact = - 4*pi*m / hbar**2 #/ (2*pi)**2
    chi_1 =  Fermi_Dirac(k, T) * sin(2*k*r)/(k)
    return pre_fact * chi_1


"""
***************************************************************************
 2D RKKY functions
***************************************************************************
"""


""" --- Analytical RKKY 2D Response function ( Sommerfeld ) --- """

def Chi_2_Sommerfeld(r, T):
    pre_fact = - 2*pi*m / hbar**2 * (2*pi*k_F)**2 #/ (2*pi)**4
    T_fact = pi**2/12*( T / T_F )**2
    
    x_F = k_F*r
    if x_F == 0:
        x_F = 1e-8
    
    chi_2_T0 = jv(0,x_F) * yv(0,x_F) + jv(1,x_F) * yv(1,x_F)
    chi_2_T = x_F * ( jv(1,x_F) * yv(0,x_F) + jv(0,x_F) * yv(1,x_F) )
    chi_2 = pre_fact*( chi_2_T0 - T_fact * chi_2_T )
    return chi_2

""" --- Analytical RKKY 2D Response function ( Exact ) --- """

def Chi_2_Exact(k,r,T):
    x = k*r
    if x == 0:
        x = 1e-8
        
    pre_fact = - 4*pi*m / hbar**2 * (2*pi)**2 #/ (2*pi)**4
    chi_2 = k * Fermi_Dirac(k, T) * jv(0,x) * yv(0,x)
    return pre_fact * chi_2


"""
***************************************************************************
 3D RKKY functions
***************************************************************************
"""


""" --- Analytical RKKY 3D Response function ( Sommerfeld ) --- """

def Chi_3_Sommerfeld(r, T):
    pre_fact = 8*pi*m / ( hbar**2 ) * (4*pi*k_F)**2 * k_F**2 #/ (2*pi)**6
    T_fact = pi**2/24*(T/T_F)**2
    x_F = k_F*r
    
    if x_F == 0:
        x_F = 1e-9
    
    chi_3 = pre_fact*( ( sin(2*x_F) - 2*x_F*cos(2*x_F) ) / ( 2*x_F )**4 + T_fact*( cos(2*x_F)/(2*x_F) ) )
    return chi_3

""" --- Analytical RKKY 3D Response function ( Exact ) --- """

def Chi_3_Exact(k,r,T):
    
    if r == 0:
        r = 1e-9
        
    pre_fact = 2*pi*m / hbar**2 *( 4*pi/r )**2 #/ (2*pi)**6
    chi_3 = pre_fact * ( k*Fermi_Dirac(k, T)*sin(2*k*r) )
    return chi_3


"""
***************************************************************************
 Integration subroutine
***************************************************************************
"""

def Integrate():
    
    print("Starting program \n")
    
    
    """
    ***************************************************************************
     Definition of constants
    ***************************************************************************
    """
    
    if( os.path.exists( "cst.dat" ) ):
        os.remove( "cst.dat" )
    cstfile = open( "cst.dat", "r" )
    cstfile.write( str( hbar ) + "\n" + str( m ) + "\n" + str( k_B ) + "\n" + str( T_F ) + "\n" + str( k_F ) + "\n" )
    cstfile.close()
    
    print("Units Defined \n")
    
    
    """
    ---------------------------------------------------------------------------
     Initialize boxes for integration
    ---------------------------------------------------------------------------
    """
    

    
    # Set integration boxes
    k_max = 2.8*k_F
    r_max = 40 / k_F
    t_max = 0.1*T_F
        
    nb_k_steps = 10000 
    nb_r_steps = 1000
    nb_t_steps = 10
    
    k_step = k_max / nb_k_steps
    r_step = r_max / nb_r_steps
    t_step = t_max / nb_t_steps
    
    # Take into account of the T = 0 case
    nb_t_steps += 1
        
    print("Integration Boxes Defined \n")
    

    
    """
    ---------------------------------------------------------------------------
     Initialize vectors which contains the result of the integration
    ---------------------------------------------------------------------------
    """
    

    
    # Set list of vectors that will contain the analytics and numerics for each temperature
    Chi_1_Analytics = np.zeros( ( nb_t_steps, nb_r_steps-1 ) )
    Chi_1_Integrated = np.zeros( ( nb_t_steps, nb_r_steps-1 ) )
    Chi_2_Analytics = np.zeros( ( nb_t_steps, nb_r_steps-1 ) )
    Chi_2_Integrated = np.zeros( ( nb_t_steps, nb_r_steps-1 ) )
    Chi_3_Analytics = np.zeros( ( nb_t_steps, nb_r_steps-1 ) )
    Chi_3_Integrated = np.zeros( ( nb_t_steps, nb_r_steps-1 ) )
    

    
    """
    ---------------------------------------------------------------------------
     Integrate exact RKKY response function over k and evaluate the analytic 
     result for all radial steps r and temperatures t
    ---------------------------------------------------------------------------
    """
    
    if( os.path.exists( "1D.dat" ) ):
        os.remove( "1D.dat" )
    if( os.path.exists( "2D.dat" ) ):
        os.remove( "2D.dat" )
    if( os.path.exists( "3D.dat" ) ):
        os.remove( "3D.dat" )
    if( os.path.exists( "T.dat" ) ):
        os.remove( "T.dat" )
    
    write1 = open( "1D.dat", "a" )
    write2 = open( "2D.dat", "a" )
    write3 = open( "3D.dat", "a" )
    writeT = open( "T.dat", "a" )
    
    for t in range(nb_t_steps):
        T = t * t_step
        writeT.write( str( T ) + "\n" )
    writeT.close()

    
    print("Starting Integration \n")
    
    # For all radial steps
    c = 0 # steps counter for prints
    start = time.time()
    for r in range( nb_r_steps-1 ):
        R = (r+1) * r_step
        
        # Prints routine
        if c == 20 or c == 0:
            print("Step = ", r, " / ", nb_r_steps)
            c = 0
        c += 1
        
        write1.write( str( R ) + "   " )
        write2.write( str( R ) + "   " )
        write3.write( str( R ) + "   " )
        
        # For all temperatures
        for t in range(nb_t_steps):
            T = t * t_step
            
            # Compute the analytical Sommerfeld derived RKKY response function 
            Chi_1_Analytics[t][r] = Chi_1_Sommerfeld(R, T)
            Chi_2_Analytics[t][r] = Chi_2_Sommerfeld(R, T)
            Chi_3_Analytics[t][r] = Chi_3_Sommerfeld(R, T)
            
            # Integrate the exct RKKY response function by the trapezoid method
            for k in range(nb_k_steps):
                K = k * k_step
                Chi_1_Integrated[t][r] += k_step * ( Chi_1_Exact( K+k_step, R, T ) + Chi_1_Exact( K, R, T ) ) / 2
                Chi_2_Integrated[t][r] += k_step * ( Chi_2_Exact( K+k_step, R, T ) + Chi_2_Exact( K, R, T ) ) / 2
                Chi_3_Integrated[t][r] += k_step * ( Chi_3_Exact( K+k_step, R, T ) + Chi_3_Exact( K, R, T ) ) / 2
                
            Chi_1_Integrated[t][r] += 2*pi**2*m / hbar**2
        
            # Write outputs datas
            write1.write( str( Chi_1_Analytics[t][r] ) + "   " + str( Chi_1_Integrated[t][r] ) + "   " )
            write2.write( str( Chi_2_Analytics[t][r] ) + "   " + str( Chi_2_Integrated[t][r] ) + "   " )
            write3.write( str( Chi_3_Analytics[t][r] ) + "   " + str( Chi_3_Integrated[t][r] ) + "   " )
        
        write1.write( "\n" )
        write2.write( "\n" )
        write3.write( "\n" )
            
        if( r==10 ):
            end = time.time()
            print( "Estimated time : " + str( nb_r_steps * ( end-start ) * 1.1 / 10 ) + " sec" )

    write1.close()
    write2.close()
    write3.close()
        
    """
    ---------------------------------------------------------------------------
     Add the the negative integration as a constant in the 1D case
    ---------------------------------------------------------------------------
    """
    

    """
    for r in range( nb_r_steps-1 ):
        for t in range( nb_t_steps ):
            Chi_1_Integrated[t][r] += 2*pi**2*m / hbar**2 #/ (2*pi)**2
    
    # At this moment, everything works in the x_F space so the r_step is changed
    r_step = k_F * r_step

    #print("Integration done, starting comparison \n")
    """

    
    """
    ---------------------------------------------------------------------------
     Look for the maximum of the functions for each temperature and compare
     Sommerfeld and Exact integration 
    ---------------------------------------------------------------------------
    """
    

    
    # Search extrema
    Extrema_1_Analytics = [ 0 for i in range( nb_t_steps ) ]
    Extrema_2_Analytics = [ 0 for i in range( nb_t_steps ) ]
    Extrema_3_Analytics = [ 0 for i in range( nb_t_steps ) ]
    Extrema_1_Integrated = [ 0 for i in range( nb_t_steps ) ]
    Extrema_2_Integrated = [ 0 for i in range( nb_t_steps ) ]
    Extrema_3_Integrated = [ 0 for i in range( nb_t_steps ) ]
    
    for t in range(nb_t_steps):
        Extrema_1_Analytics[t] = Find_Extrema(Chi_1_Analytics[t])
        Extrema_1_Integrated[t] = Find_Extrema(Chi_1_Integrated[t])
        Extrema_2_Analytics[t] = Find_Extrema(Chi_2_Analytics[t])
        Extrema_2_Integrated[t] = Find_Extrema(Chi_2_Integrated[t])
        Extrema_3_Analytics[t] = Find_Extrema(Chi_3_Analytics[t])
        Extrema_3_Integrated[t] = Find_Extrema(Chi_3_Integrated[t])
    
    # Compare Exact Sommerfeld functions
    Compare_1 = [ 0 for i in range( nb_t_steps ) ]
    Compare_2 = [ 0 for i in range( nb_t_steps ) ]
    Compare_3 = [ 0 for i in range( nb_t_steps ) ]
    
    for t in range( nb_t_steps ):
        tmp = []
        # tmp.append([0,1])
        # security if the analytic and integrated functions do not have the same number of extrema
        min_range = min( len(Extrema_1_Analytics[t]), len(Extrema_1_Integrated[t]) )
        for i in range( min_range ):
            n = Extrema_1_Analytics[t][i]
            tmp.append( [ n*r_step, abs( Chi_1_Integrated[t][n] / Chi_1_Analytics[t][n] ) ] )
        Compare_1[t] = tmp
        
        tmp = []
        # tmp.append([0,1])
        min_range = min( len(Extrema_2_Analytics[t]), len(Extrema_2_Integrated[t]) )
        for i in range( min_range ):
            n = Extrema_2_Analytics[t][i]
            tmp.append( [ n*r_step, abs( Chi_2_Integrated[t][n] / Chi_2_Analytics[t][n] ) ] )
        Compare_2[t] = tmp
        
        tmp = []
        # tmp.append([0,1])
        min_range = min( len(Extrema_3_Analytics[t]), len(Extrema_3_Integrated[t]) )
        for i in range( min_range ):
            n = Extrema_3_Analytics[t][i]
            tmp.append( [ n*r_step, abs( Chi_3_Integrated[t][n] / Chi_3_Analytics[t][n] ) ] )
        Compare_3[t] = tmp
       
    # transpose Compare[t] for fitting
    for t in range( nb_t_steps ):
        Compare_1[t] = np.transpose( Compare_1[t] )
        Compare_2[t] = np.transpose( Compare_2[t] )
        Compare_3[t] = np.transpose( Compare_3[t] )
    
    
    
    """
    ---------------------------------------------------------------------------
     Fitting the comparisons to find the diverging point (parameter b)
    ---------------------------------------------------------------------------
    """
    
    
    print("Fitting the comparisons \n")
    
    T = np.zeros( nb_t_steps )
    for t in range( nb_t_steps ):
        T[t] = t * t_step / T_F
    
    # Find the fitting parameters of the polynomial that best matches the ratio between analytic and integrated response functions
    
    Para_Fit_Compare_1 = [ 0 for i in range( nb_t_steps ) ]
    Para_Fit_Compare_2 = [ 0 for i in range( nb_t_steps ) ]
    Para_Fit_Compare_3 = [ 0 for i in range( nb_t_steps ) ]
    
    for t in range(nb_t_steps):
        Para_Fit_Compare_1[t] = Fit_Curve( Compare_1[t], Rational_Square_Fun )
        Para_Fit_Compare_2[t] = Fit_Curve( Compare_2[t], Rational_Square_Fun )
        Para_Fit_Compare_3[t] = Fit_Curve( Compare_3[t], Rational_Square_Fun )
    
    Validity_Range_1 = np.zeros( nb_t_steps )
    Validity_Range_2 = np.zeros( nb_t_steps )
    Validity_Range_3 = np.zeros( nb_t_steps )
    for t in range( nb_t_steps ):
        Validity_Range_1[t] = Para_Fit_Compare_1[t][1]
        Validity_Range_2[t] = Para_Fit_Compare_2[t][1]
        Validity_Range_3[t] = Para_Fit_Compare_3[t][1]
    
    
    if( os.path.exists( "Validity_range.dat" ) ):
        os.remove( "Validity_range.dat" )
    
    validity_file = open( "Validity_range.dat", "a" )
    for t in range( 3, nb_t_steps ):
        validity_file.write( str( T[t] ) + "   " + str( Validity_Range_1[t] ) + "   " + str( Validity_Range_2[t] ) + "   " + str( Validity_Range_3[t] ) + "\n"  )
    validity_file.close()
    
    """
    ---------------------------------------------------------------------------
     Fit the validity range
    ---------------------------------------------------------------------------
    """
    
    
    print('Fitting validity range')
    
    Temperature_for_fit = np.zeros( nb_t_steps - 3 )
    Validity_1D = np.zeros( nb_t_steps - 3 )
    Validity_2D = np.zeros( nb_t_steps - 3 )
    Validity_3D = np.zeros( nb_t_steps - 3 )
    
    for t in range( nb_t_steps - 3 ):
        Temperature_for_fit[t] = T[t+3]
        Validity_1D[t] = Validity_Range_1[t+3]
        Validity_2D[t] = Validity_Range_2[t+3]
        Validity_3D[t] = Validity_Range_3[t+3]
    
    # Native routine works here
    Validity_Fit_Para_1, _ = curve_fit( inverse_fun, Temperature_for_fit, Validity_1D )
    Validity_Fit_Para_2, _ = curve_fit( inverse_fun, Temperature_for_fit, Validity_2D )
    Validity_Fit_Para_3, _ = curve_fit( inverse_fun, Temperature_for_fit, Validity_3D )
    
    
    # If native routine do not work, use this
    """
    Validity_Fit_Para_1 = Fit_Curve( [ Temperature_for_fit, Validity_1D ], Inverse_Fun )
    Validity_Fit_Para_2 = Fit_Curve( [ Temperature_for_fit, Validity_2D ], Inverse_Fun )
    Validity_Fit_Para_3 = Fit_Curve( [ Temperature_for_fit, Validity_3D ], Inverse_Fun )
    """
    
    
    Validity_Fit_1 = np.zeros( 10 * nb_t_steps )
    Validity_Fit_2 = np.zeros( 10 * nb_t_steps )
    Validity_Fit_3 = np.zeros( 10 * nb_t_steps )
    Temperature_for_fit = np.zeros( 10 * nb_t_steps )
    for t in range( 10 * nb_t_steps ):
        tmp = (t+1) * t_step / 10 / T_F
        Validity_Fit_1[t] = Inverse_Fun( tmp, Validity_Fit_Para_1 )
        Validity_Fit_2[t] = Inverse_Fun( tmp, Validity_Fit_Para_2 )
        Validity_Fit_3[t] = Inverse_Fun( tmp, Validity_Fit_Para_3 )
        Temperature_for_fit[t] = tmp
    
    if( os.path.exists( "Fit.dat" ) ):
        os.remove( "Fit.dat" )
    
    fitfile = open( "Fit.dat", "a" )
    for t in range( len( Temperature_for_fit ) ):
        fitfile.write( str( Temperature_for_fit[t] ) + "   " + str( Validity_Fit_1[t] ) + "   " + str( Validity_Fit_2[t] ) + "   " + str( Validity_Fit_3[t] ) + "\n" )
    fitfile.close()
    
    """
    ---------------------------------------------------------------------------
     Define plot vectors
    ---------------------------------------------------------------------------
    """
    
    
    X_F = [ (r+1) * r_step for r in range(nb_r_steps-1) ]
    Zero = [0 for r in range(nb_r_steps-1)]
    colors = ['Black', 'darkred', 'maroon', 'firebrick', 'brown', 'indianred', 'lightcoral', 'salmon', 'tomato', 'orangered', 'chocolate', 'darkorange']
    
    if( os.path.exists( "xf.dat" ) ):
        os.remove( "xf.dat" )
    writexf = open( "xf.dat", "a" )
    
    for value in X_F:
        writexf.write( str( value ) + "\n" )
    writexf.close()



"""
***************************************************************************
 Plotting functions
***************************************************************************
"""

def make_plots():
    
    # Gather datas
    
    file1 = open( "1D.dat", "r" )
    file2 = open( "2D.dat", "r" )
    file3 = open( "3D.dat", "r" )
    fileT = open( "T.dat", "r" )
    filexf = open( "xf.dat", "r" )
    validity_file = open( "Validity_range.dat", "r" )
    fitfile = open( "Fit.dat", "r" )
    cstfile = open( "cst.dat", "r" )
    
    datas1 = file1.readlines()
    datas2 = file2.readlines()
    datas3 = file3.readlines()
    validity_datas = validity_file.readlines()
    fit_datas = fitfile.readlines()
    csts = cstfile.readlines()
    
    k_F = float( csts[ len( csts ) - 1 ] )
    
    T = fileT.readlines()
    X_F = filexf.readlines()
    for i in range( len( X_F ) ):
        X_F[i] = k_F * float( X_F[i] )
    Zero = [ 0 for i in range( len( X_F ) ) ]
    
    # colors = ['Black', 'darkred', 'maroon', 'firebrick', 'brown', 'indianred', 'lightcoral', 'salmon', 'tomato', 'orangered', 'chocolate', 'darkorange']
    
    file1.close()
    file2.close()
    file3.close()
    fileT.close()
    filexf.close()
    validity_file.close()
    fitfile.close()
    
    for i in range( len( datas1 ) ):
        datas1[i] = datas1[i].split( "   " )
        datas1[i].pop()
        datas2[i] = datas2[i].split( "   " )
        datas2[i].pop()
        datas3[i] = datas3[i].split( "   " )
        datas3[i].pop()
        if i < len( validity_datas ):
            validity_datas[i] = validity_datas[i].split( "   " )
        if i < len( fit_datas ):
            fit_datas[i] = fit_datas[i].split( "   " )
    
    for i in range( len( datas1 ) ):
        for j in range( len( datas1[i] ) ):
            datas1[i][j] = float( datas1[i][j] )
            datas2[i][j] = float( datas2[i][j] )
            datas3[i][j] = float( datas3[i][j] )
            if i < len( validity_datas ) and j < len( validity_datas[i] ):
                validity_datas[i][j] = float( validity_datas[i][j] )
            if i < len( fit_datas ) and j < len( fit_datas[i] ):
                fit_datas[i][j] = float( fit_datas[i][j] )
    
    datas1 = np.array( datas1 )
    datas2 = np.array( datas2 )
    datas3 = np.array( datas3 )
    validity_datas = np.array( validity_datas )
    fit_datas = np.array( fit_datas )
    
    datas1 = datas1.T
    datas2 = datas2.T
    datas3 = datas3.T
    validity_datas = validity_datas.T
    fit_datas = fit_datas.T
    
    datas1 = datas1.tolist()
    datas2 = datas2.tolist()
    datas3 = datas3.tolist()
    validity_datas = validity_datas.tolist()
    fit_datas = fit_datas.tolist()
    
    
    #####
    # Convert datas to good ones for plotting
    #####
    
    i = 1
    cond = True
    Chi_1_Integrated = []
    Chi_2_Integrated = []
    Chi_3_Integrated = []
    Chi_1_Analytics = []
    Chi_2_Analytics = []
    Chi_3_Analytics = []
    while i < len( datas1 ):
        if cond:
            Chi_1_Analytics.append( datas1[i] )
            Chi_2_Analytics.append( datas2[i] )
            Chi_3_Analytics.append( datas3[i] )
            cond = False
        else:
            Chi_1_Integrated.append( datas1[i] )
            Chi_2_Integrated.append( datas2[i] )
            Chi_3_Integrated.append( datas3[i] )
            cond = True
        i += 1
    
    nb_t_steps = len( Chi_1_Analytics )
    
    Temperature_for_fit  = fit_datas[0]
    Validity_Fit_1 = fit_datas[1]
    Validity_Fit_2 = fit_datas[2]
    Validity_Fit_3 = fit_datas[3]
    
    T = validity_datas[0]
    Validity_Range_1 = validity_datas[1]
    Validity_Range_2 = validity_datas[2]
    Validity_Range_3 = validity_datas[3]
    
    
    
    
    fig10, ax10 = plt.subplots(3,1)
    
    ax10[0].plot(X_F, Chi_1_Integrated[0], color='Black', linewidth= 2, label='T = 0' )
    ax10[0].plot(X_F, Chi_1_Integrated[5], color='Red', linewidth= 2, label='T = 0.05 T_F' )
    ax10[0].plot(X_F, Chi_1_Integrated[nb_t_steps-1], color='Blue', linewidth= 2, label='T = 0.1 T_F' )
    ax10[0].plot(X_F, Zero, color='Black', linewidth= 2 )
    ax10[0].legend(loc = 'lower right', fontsize = 'large')
    ax10[0].set_xlim( [ 0,12 ] )
    ax10[0].set_ylim( [ 1.1*min(Chi_1_Integrated[0]), -0.6*min(Chi_1_Integrated[0]) ] )
    ax10[0].set_xlabel( "x_F" )
    ax10[0].set_ylabel( "X_1( r, T )" )
    ax10[0].axes.yaxis.set_ticklabels([])
    
    ax10[1].plot(X_F, Chi_2_Integrated[0], color='Black', linewidth= 2, label='T = 0' )
    ax10[1].plot(X_F, Chi_2_Integrated[5], color='Red', linewidth= 2, label='T = 0.05 T_F' )
    ax10[1].plot(X_F, Chi_2_Integrated[nb_t_steps-1], color='Blue', linewidth= 2, label='T = 0.1 T_F' )
    ax10[1].plot(X_F, Zero, color='Black', linewidth= 2 )
    ax10[1].legend(loc = 'lower right', fontsize = 'large')
    ax10[1].set_xlim( [ 0,12 ] )
    ax10[1].set_ylim( [ 1.1*min(Chi_2_Integrated[0]), -0.4*min(Chi_2_Integrated[0]) ] )
    ax10[1].set_xlabel( "x_F" )
    ax10[1].set_ylabel( "X_2( r, T )" )
    ax10[1].axes.yaxis.set_ticklabels([])
    
    ax10[2].plot(X_F, Chi_3_Integrated[0], color='Black', linewidth= 2, label='T = 0' )
    ax10[2].plot(X_F, Chi_3_Integrated[5], color='Red', linewidth= 2, label='T = 0.05 T_F' )
    ax10[2].plot(X_F, Chi_3_Integrated[nb_t_steps-1], color='Blue', linewidth= 2, label='T = 0.1 T_F' )
    ax10[2].plot(X_F, Zero, color='Black', linewidth= 2 )
    ax10[2].legend(loc = 'lower right', fontsize = 'large')
    ax10[2].set_xlim( [ 0,12 ] )
    ax10[2].set_ylim( [ 1.1*min(Chi_3_Integrated[0]), -0.4*min(Chi_3_Integrated[0]) ] )
    ax10[2].set_xlabel( "x_F" )
    ax10[2].set_ylabel( "X_3( r, T )" )
    ax10[2].axes.yaxis.set_ticklabels([])
    
    # plt.suptitle('Comparison of 1D, 2D, 3D Integrated response functions for T = 0, 5%, 10% T_F \n')
    
    fig10.set_size_inches( 16, 9 )
    
    plt.show()
    fig10.savefig('Comparison_Exact_Integrations_T=0_0-05_0-1.pdf', bbox_inches='tight')
    
    
    
    fig11, ax11 = plt.subplots(3,1)
    
    ax11[0].plot(X_F, Chi_1_Integrated[0], color='Black', linewidth= 2, label='T = 0' )
    # ax11[0].plot(X_F, Chi_1_Integrated[5], color='Red', linewidth= 2, label='T = 0.05 T_F' )
    # ax11[0].plot(X_F, Chi_1_Analytics[5], color='salmon', linewidth= 2, label='T = 0.05 T_F (Sommerfeld)' )
    ax11[0].plot(X_F, Chi_1_Integrated[nb_t_steps-1], color='Blue', linewidth= 2, label='T = 0.1 T_F' )
    ax11[0].plot(X_F, Chi_1_Analytics[nb_t_steps-1], color='lightskyblue', linewidth= 2, label='T = 0.1 T_F (Sommerfeld)' )
    ax11[0].plot(X_F, Zero, color='Black', linewidth= 2 )
    ax11[0].legend(loc = 'lower right', fontsize = 'large')
    ax11[0].set_xlim( [ 0,12 ] )
    ax11[0].set_ylim( [ 1.1*min(Chi_1_Integrated[0]), -0.6*min(Chi_1_Integrated[0]) ] )
    ax11[0].set_xlabel( "x_F" )
    ax11[0].set_ylabel( "X_1( r, T )" )
    ax11[0].axes.yaxis.set_ticklabels([])
    
    ax11[1].plot(X_F, Chi_2_Integrated[0], color='Black', linewidth= 2, label='T = 0' )
    # ax11[1].plot(X_F, Chi_2_Integrated[5], color='Red', linewidth= 2, label='T = 0.05 T_F' )
    # ax11[1].plot(X_F, Chi_2_Analytics[5], color='salmon', linewidth= 2, label='T = 0.05 T_F (Sommerfeld)' )
    ax11[1].plot(X_F, Chi_2_Integrated[nb_t_steps-1], color='Blue', linewidth= 2, label='T = 0.1 T_F' )
    ax11[1].plot(X_F, Chi_2_Analytics[nb_t_steps-1], color='lightskyblue', linewidth= 2, label='T = 0.1 T_F (Sommerfeld)' )
    ax11[1].plot(X_F, Zero, color='Black', linewidth= 2 )
    ax11[1].legend(loc = 'lower right', fontsize = 'large')
    ax11[1].set_xlim( [ 0,12 ] )
    ax11[1].set_ylim( [ 1.1*min(Chi_2_Integrated[0]), -0.4*min(Chi_2_Integrated[0]) ] )
    ax11[1].set_xlabel( "x_F" )
    ax11[1].set_ylabel( "X_2( r, T )" )
    ax11[1].axes.yaxis.set_ticklabels([])
    
    ax11[2].plot(X_F, Chi_3_Integrated[0], color='Black', linewidth= 2, label='T = 0' )
    # ax11[2].plot(X_F, Chi_3_Integrated[5], color='Red', linewidth= 2, label='T = 0.05 T_F' )
    # ax11[2].plot(X_F, Chi_3_Analytics[5], color='salmon', linewidth= 2, label='T = 0.05 T_F (Sommerfeld)' )
    ax11[2].plot(X_F, Chi_3_Integrated[nb_t_steps-1], color='Blue', linewidth= 2, label='T = 0.1 T_F' )
    ax11[2].plot(X_F, Chi_3_Analytics[nb_t_steps-1], color='lightskyblue', linewidth= 2, label='T = 0.1 T_F (Sommerfeld)' )
    ax11[2].plot(X_F, Zero, color='Black', linewidth= 2 )
    ax11[2].legend(loc = 'lower right', fontsize = 'large')
    ax11[2].set_xlim( [ 0,12 ] )
    ax11[2].set_ylim( [ 1.1*min(Chi_3_Integrated[0]), -0.4*min(Chi_3_Integrated[0]) ] )
    ax11[2].set_xlabel( "x_F" )
    ax11[2].set_ylabel( "X_3( r, T )" )
    ax11[2].axes.yaxis.set_ticklabels([])
    
    # plt.suptitle('Comparison of 1D, 2D, 3D Integrated and Analytical response functions for T =  0, 10% T_F \n')
    
    fig11.set_size_inches( 16, 9 )
    
    plt.show()
    
    fig11.savefig('Comparison_Exact_and_Analytical_Integrations_T=0_0-05_0-1.pdf', bbox_inches='tight')
    
    
    
    fig7, ax7 = plt.subplots(1,1)
    ax7.plot(T, Validity_Range_1, 'o', color='Black', linewidth= 2, label='1D' )
    ax7.plot(T, Validity_Range_2, 'o', color='Red', linewidth= 2, label='2D' )
    ax7.plot(T, Validity_Range_3, 'o', color='Blue', linewidth= 2, label='3D' )
    ax7.plot(Temperature_for_fit, Validity_Fit_1, '--', color='Black', linewidth= 2, label='1D Fit' )
    ax7.plot(Temperature_for_fit, Validity_Fit_2, '--', color='Red', linewidth= 2, label='1D Fit' )
    ax7.plot(Temperature_for_fit, Validity_Fit_3, '--', color='Blue', linewidth= 2, label='1D Fit' )
    # plt.xlim(0,15)
    plt.ylim( 0, 1.2 * max( max( Validity_Range_1 ), max( Validity_Range_2 ), max( Validity_Range_3 ) ) )
    plt.legend(fontsize = 'large')
    plt.xlabel("T / TF")
    plt.ylabel("x_F")
    # plt.suptitle('Validity Range of Sommerfeld Expansion with Respect to Temperature \n')
    # plt.title('  ----------  ')
    plt.show()
    fig7.savefig('Validity_Range_of_Sommerfeld_Expansion_with_Respect_to_Temperature.pdf', bbox_inches='tight')



"""
*******************************************************************************
*******************************************************************************
 Main program
*******************************************************************************
*******************************************************************************
"""

"""
***************************************************************************
 Definition of constants
***************************************************************************
"""


hbar = 1 # 1.054571818e-34                      # Planck reduced constant
m = 1 # 9.10938356e-31                          # electron mass
k_B = 1 # 1.38064852e-23                        # Boltzmann constant
T_F = 28000                                     # Fermi temperature (for Nickel)
k_F = sqrt(2*m*k_B*T_F / hbar**2 )              # Fermi vector

# Threshold = 1 / 100                           # Valitidy threshold for the Sommerfeld expansion
"""
if( os.path.exists( "cst.dat" ) ):
    os.remove( "cst.dat" )
cstfile = open( "cst.dat", "a" )
cstfile.write( str( hbar ) + "\n" + str( m ) + "\n" + str( k_B ) + "\n" + str( T_F ) + "\n" + str( k_F ) + "\n" )
cstfile.close()

print("Units Defined \n")
"""

if __name__ == "__main__":

    # Integrate()
    make_plots()
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    