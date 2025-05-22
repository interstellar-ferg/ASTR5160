import emcee
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
import numpy as np
import argparse
import sys
from astropy.table import QTable

from scipy.optimize import curve_fit
from scipy.optimize import minimize

import corner

# AJF import a chat-gpt co-written code that auto-writes docstrings with variables included
from master_scripts.docstring_wrapper import log_sphinx_io as ds
# AJF note: @docstring is a wrapper that auto-writes docstrings for the function directly below it
# AJF see master_scripts/docstring_wrapper for more details

# AJF created 5/18/25
# AJF last modified 5/21/25

# example command line command to run:
# python final_proj.py /d/scratch/ASTR5160/final/ dataxy.fits 2500


#@ds
def load(fullpath):
    """
    Load in the datatable and extract relevant columns as arrays of variables values
    
    Parameters
    ----------
    fullpath : :class: str
        path to the datatable to load in
        
    Returns
    ----------
    :class: numpy.ndarray
        the x data from table
    :class: numpy.ndarray
        the y data from the table
    :class: numpy.ndarray
        the y-error data from the table

    """
    # AJF read in data as an astropy table
    data = QTable.read(fullpath)
    
    # AJF extract columns of relevant data
    x, y, yerr = data['x'], data['y'], data['yerr']
    
    # AJF convert these columsn to arrays so they are easier to work with
    x, y, yerr = np.array(x), np.array(y), np.array(yerr)    
    
    return x, y, yerr
    





#@ds
def lin(x_lin, m, b):
    """
    Acquire y values for corresponsing x values plugged into a linear function
    
    Parameters
    ----------
    x_lin : :class: numpy.ndarray
        an array of x values to use in the model function
    m : :class: numpy.float64
        the slope of a linear function
    b : :class: numpy.float64
        the y-intercept of the function
        
    Returns
    ----------
    :class: numpy.ndarray
        an array of y-values derived from plugging in the x_lin into the model function
        
    """
    
    # AJF define a linear function
    line = m*x_lin + b
    
    return line






#@ds
def linfit(x, y, yerr, p0):
    """
    Fit data with a linear function and extract the best-fit parameter values
    
    Parameters
    ----------
    x : :class: numpy.ndarray
        the input x data from the table
    y : :class: numpy.ndarray
        the input y data from the table
    yerr : :class: numpy.ndarray
        the input y error data from the table
    p0 : :class: tuple
        the initial guesses, derived from inspecting the first basic plot of the data
        
    Returns
    ----------
    :class: numpy.ndarray
        an array of the parameters best-fit values; slope and y intercept
    
    """
    # AJF set up the curve_fit function to fit the data using input guesses derived from basic plot; use scipy curve_fit
    (mf, bf), ccf = curve_fit(lin, x, y, sigma = yerr, p0 = p0)
    
    # AJF create array of parameter best fit values; do not need covariance matrix for this code, although could be added easily
    fits = np.array([mf, bf])
    
    return fits
 




   

#@ds
def quad(x_lin, a2, a1, a0):
    """
    Acquire y values for corresponsing x values plugged into a quadratic function
    
    Parameters
    ----------
    x_lin : :class: numpy.ndarray
        an array of x values to use in the model function
    a2 : :class: numpy.float64
        the coefficient for the x^2 term
    a1 : :class: numpy.float64
        the coefficient for the x term
    a0 : :class: numpy.float64
        the constant in a quadratic function
        
    Returns
    ----------
    :class: numpy.ndarray
        an array of y-values derived from plugging in the x_lin into the model function

    
    """
    # AJF define a quadratic function
    quadd = a2*(x_lin)**2 + a1*x_lin + a0
    
    return quadd








#@ds
def quadfit(x, y, yerr, p0):
    """
    Fit data with a quadratic function and extract the best-fit parameter values
    
    Parameters
    ----------
    x : :class: numpy.ndarray
        the input x data from the table
    y : :class: numpy.ndarray
        the input y data from the table
    yerr : :class: numpy.ndarray
        the input y error data from the table
    p0 : :class: tuple
        guesses for parameter values derived from inspecting the first vasic plot
        
    Returns
    ----------
    :class: numpy.ndarray
        an array of the parameter best fit values; a2, a1, a0
    
    """
    # AJF set up the curve_fit function to fit the data using input guesses derived from basic plot; use scipy curve_fit    
    (a2, a1, a0), ccf = curve_fit(quad, x, y, sigma=yerr, p0 = p0)
    
    # AJF create array of parameter best fit values; do not need covariance matrix for this code, although could be added easily
    fits = np.array([a2, a1, a0])
    
    return fits





#@ds
def basic_errplot(x, y, yerr):
    """
    plot the input data; used to find initial curve_fit p0 guesses
    
    Parameters
    ----------
    x : :class: numpy.ndarray
        the input x data from the table
    y : :class: numpy.ndarray
        the input y data from the table
    yerr : :class: numpy.ndarray
        the input y error data from the table
        
        
    Returns
    ----------
    None - plots and saves figure
    
    """
    # AJF setup plot
    fig, ax = plt.subplots(1, figsize = (15,20))

    # AJF simple plot cents with means and variances to get a look at what slope and int might be
    ax.errorbar(x, y, yerr = yerr, fmt = '.', label = 'Data with y-errors')

    # AJF add auto-minor ticks (4 per section), increase major tick frequnecy, add major and minor grid
    ax.xaxis.set_minor_locator(AutoMinorLocator(5))
    ax.yaxis.set_minor_locator(AutoMinorLocator(5))
    ax.locator_params(axis='both', nbins=15)
    ax.legend(markerscale=4)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    plt.suptitle('Input Data', y = 0.92)
    
    # Major grid
    ax.grid(True, which='major', linewidth=0.5, color='gray', alpha=0.5)

    # Minor grid
    ax.grid(True, which='minor', linewidth=0.3, color='gray', alpha=0.3)
    #plt.close()
    plt.savefig(f'input_data.png', format = 'png')
    plt.show()






#@ds
def log_lik(fits, x, y, yerr, fitdim):
    """
    Define the log of he liklihood function in Bayesian stats; used for final mcmc sampling when combined with log(prior)
    also used for fitting curves with minimize
    
    Parameters
    ----------
    fits : :class: numpy.ndarray
        the best-fit parameters of whichever model function you are using 
    x : :class: numpy.ndarray
        the input x data from the table
    y : :class: numpy.ndarray
        the input y data from the table
    yerr : :class: numpy.ndarray
        the input y error data from the table      
    fitdim : :class: str
        which model function - linear or quadratic?
        
    Returns
    ----------
    :class: numpy.float64
        the value of the log liklihood for the given input
    
    """
    # AJF determine if input parameters (fits) are for linear or quadratic function, then apply proper model
    if fitdim == 'lin':
        m, b = fits[0], fits[1]
        model = lin(x, m, b)
    elif fitdim == 'quad':
        a2, a1, a0 = fits[0], fits[1], fits[2]
        model = quad(x, a2, a1, a0)
    else:
        print(f'\nYour given fitting dimension was {fitdim} when it should be either "lin" or "quad". Please edit code accordingly.\n')
        sys.exit()
    
    # AJF formula for log liklihood function; can use any model function here; does not include any f factor for variance perturbations
    llf = -0.5 * np.sum((((y - model)**2)/yerr**2) + np.log(yerr**2))
    
    return llf







#@ds
def ll_func(fits, x, y, yerr, fitdim):
    """
    Function that gets passed to minimize - is exactly log(liklihood) function, but negative, so that
    minimize actually does minimize the log (want maximum value since log)

    Parameters
    ----------
    fits : :class: numpy.ndarray
        the best-fit parameters of whichever model function you are using 
    x : :class: numpy.ndarray
        the input x data from the table
    y : :class: numpy.ndarray
        the input y data from the table
    yerr : :class: numpy.ndarray
        the input y error data from the table      
    fitdim : :class: str
        which model function - linear or quadratic?
        
    Returns
    ----------
    :class: numpy.float64
        the value of the log liklihood for the given input, multiplied by a negative so that it can be minimized properly
        

    """
    
    # AJF make sure function is negative so that maximum liklihood becomes negative (log)
    final_func = -log_lik(fits, x, y, yerr, fitdim)
    
    return final_func
    





#@ds
def minimize_func(fits, x, y, yerr, fitdim):
    """
    Find the maximum liklihood values by minimizing the negative of the log(liklihood) function
    extract best fit parameters from this
    
    Parameters
    ----------
    fits : :class: numpy.ndarray
        the best-fit parameters of whichever model function you are using 
    x : :class: numpy.ndarray
        the input x data from the table
    y : :class: numpy.ndarray
        the input y data from the table
    yerr : :class: numpy.ndarray
        the input y error data from the table      
    fitdim : :class: str
        which model function - linear or quadratic?
        
    Returns
    ----------
    :class: scipy.optimize._optimize.OptimizeResult
        the scipy minimize solution data - extract fit parameters from here
    
    
    """
    # AJF set up fitdim integer
    if fitdim == 'lin':
        dim = 2
    elif fitdim == 'quad':
        dim = 3
    else:
        print(f'\nYour given fitting dimension was {fitdim} when it should be either "lin" or "quad". Please edit code accordingly.\n')
        sys.exit()
    
    # AJF perform linear minimize - set seed (as 42! lol) if necessary so that random numbers generated are different every time
    np.random.seed(42)
   
    # AJF perturbate the initial guesses so minimize does not get stuck
    pert_init = fits + 0.1 * np.random.rand(dim)
    
    # AJF perform minimize on linear fits
    solution = minimize(ll_func, pert_init, args = (x, y, yerr, fitdim))
    
    return solution







#@ds
def log_prior(fits, fitdim):
    """
    The prior distribution for use in finding the posterior distribution - represents offset to log(liklihood) function 
    if used in log space and if uniform (since liklihood*prior --> log(liklihood) + log(prior) ) and log(prior) is number if prior is uniform (number) 

    Parameters
    ----------
    fits : :class: numpy.ndarray
        the 
    fitdim : :class: str
        
        
    Returns
    ----------
    :class: float

    
    """
    # AJF set up fitdim integer and correct function
    if fitdim == 'lin':
        m, b = fits
        # AJF set uniform prior limits - uniform b/c integers
        if -3 < m < 1 and 0 < b < 8:
            # AJF return 0.0 instead of an integer (i.e. log(int) = int) because this prior 'offset' to log liklihood func is sort of 'absorbed' by P(Data) 
            # AJF can also think of it as being an 'unormalized' distribution (I think)
            return 0.0
        else:
            # AJF if parameter is outside prior range, give ln(0) = neg. inf
            return -np.inf
    
    elif fitdim == 'quad':
        a2, a1, a0 = fits
        # AJF set uniform prior limits - uniform b/c integers
        if 0 < a2 < 0.5 and -3 < a1 < -1 and 6 < a0 < 10:
            # AJF return 0.0 instead of an integer (i.e. log(int) = int) because this prior 'offset' to log liklihood func is sort of 'absorbed' by P(Data) 
            # AJF can also think of it as being an 'unormalized' distribution (I think)
            return 0.0
        else:
            # AJF if parameter is outside prior range, give ln(0) = neg. inf
            return -np.inf       
        
    else:
        print(f'\nYour given fitting dimension was {fitdim} when it should be either "lin" or "quad". Please edit code accordingly.\n')
        sys.exit()
    







    
#@ds
def log_post_prob(fits, x, y, yerr, fitdim):
    """
    Define the posterior distribution function - liklihood * prior, or, in log, 
    log(liklihood) + log(prior)

    Parameters
    ----------
    fits : :class: numpy.ndarray
        maximum-liklihood values of parameters (derived from minimize liklihood func)
    x : :class: numpy.ndarray
        input x data from table
    y : :class: numpy.ndarray
        input y data from table
    yerr : :class: numpy.ndarray
        input y error data from table
    fitdim : :class: str
        which fit - linear or quadratic?
        
    Returns
    ----------
    :class: numpy.float64
        value of posterior probability for the given fit parameters and data; used for mcmc

    """
    # AJF ensure prior is not negative infinity; if it is, return negative inf (since -inf times anything is neg inf)
    if not np.isfinite(log_prior(fits, fitdim)):
        return -np.inf
        
    # AJF if log_prior exists (is zero) then post is read; instead of prior * liklihood, in log spacr they add
    post = log_prior(fits, fitdim) + log_lik(fits, x, y, yerr, fitdim)
    
    return post

   






#@ds
def basic_23fitplot(lincf, quadcf, linll, quadll, x, y, yerr):
    """
    Used to compare curve_fit derived model fits and log-liklihood minimize derived fits
    
    Parameters
    ----------
    lincf : :class: numpy.ndarray
        curve_fit derived best-fit parameter values for linear function
    quadcf : :class: numpy.ndarray
        curve_fit derived best-fit parameter values for quadratic function
    linll : :class: scipy.optimize._optimize.OptimizeResult
        liklihood minimize derived best-fit parameter values for linear function
    quadll : :class: scipy.optimize._optimize.OptimizeResult
        liklihood minimize derived best-fit parameter values for quadratic function
    x : :class: numpy.ndarray
        input x data from table
    y : :class: numpy.ndarray
        input y data from table
    yerr : :class: numpy.ndarray
        input y error data from table

    Returns
    ----------
    None - plots and saves figure
    
    """
    # AJF make array of x values to use with models
    x_lin = np.linspace(min(x), max(x), 1000)
    
    # AJF unpack params and make y-values for models from curvefit
    # AJF linear
    m, b = lincf[0], lincf[1]
    mlincf = lin(x_lin, m, b)    
    # AJF quadratic
    a2, a1, a0 = quadcf[0], quadcf[1], quadcf[2]
    mquadcf = quad(x_lin, a2, a1, a0)   
    
    # AJF unpack params and make y-values for models from log-liklihood minimize (ll)
    # AJF linear
    m, b = linll.x
    mlinll = lin(x_lin, m, b)
    # AJF quadratic
    a2, a1, a0 = quadll.x
    mquadll = quad(x_lin, a2, a1, a0)   
    
    # AJF setup plot
    fig, ax = plt.subplots(2, figsize = (15,20))

    # AJF simple plot x and y data with error
    ax[0].errorbar(x, y, yerr = yerr, fmt = '.', color = 'k', label = 'Data with y-errors')
    ax[1].errorbar(x, y, yerr = yerr, fmt = '.', color = 'k', label = 'Data with y-errors')
    
    # AJF plot linear models
    ax[0].plot(x_lin, mlinll, 'b--', label = 'Linear Model Fit from LL-Minimize')
    ax[0].plot(x_lin, mlincf, 'r-', label = 'Linear Model Fit from Curve_Fit')
    ax[0].set_title('Linear Fits', loc = 'left')
   
    # AJF plot quadratic models 
    ax[1].plot(x_lin, mquadll, 'b--', label = 'Quadratic Model Fit from LL-Minimize')  
    ax[1].plot(x_lin, mquadcf, 'r-', label = 'Quadratic Model Fit from Curve_Fit') 
    ax[1].set_title('Quadratic Fits', loc = 'left')

    # AJF add auto-minor ticks (4 per section), increase major tick frequnecy, add major and minor grid
    for a in ax:
        a.xaxis.set_minor_locator(AutoMinorLocator(5))
        a.yaxis.set_minor_locator(AutoMinorLocator(5))
        a.locator_params(axis='both', nbins=15)
        a.legend(markerscale=1)
        
        a.set_xlabel('x')
        a.set_ylabel('y')

        # Major grid
        a.grid(True, which='major', linewidth=0.5, color='gray', alpha=0.5)

        # Minor grid
        a.grid(True, which='minor', linewidth=0.3, color='gray', alpha=0.3)
        
    plt.suptitle('Model Fits', y=0.91, fontweight = 800, fontsize = 16)
    plt.savefig(f'model_fits_ll_curve.png', format = 'png')
    #plt.close()
    plt.show()   






#@ds
def MC(fits, x, y, yerr, fitdim, walks):
    """
    Do an MCMC walk, starting from perturbated initial guesses derived from the log(liklihood) minimize results,
    to find the best parameters to fit the input function

    Parameters
    ----------
    fits : :class: scipy.optimize._optimize.OptimizeResult
        the log-liklihood minimize results - best fit parameters
    x : :class: numpy.ndarray
        input x data from table
    y : :class: numpy.ndarray
        input y data from table
    yerr : :class: numpy.ndarray
        input y error data from table
    fitdim : :class: str
        which fit - linear or quadratic?       
    walks : :class: int
        number of steps should each walker do

    Returns
    ----------
    :class: emcee.ensemble.EnsembleSampler
        sampler results
    :class: int
        the number of parameters that were fitted (2 for lin, 3 for quad)
    
    """
    
    # AJF set up fitdim integer and correct function
    if fitdim == 'lin':
        dim = 2
    elif fitdim == 'quad':
        dim = 3            
    else:
        print(f'\nYour given fitting dimension was {fitdim} when it should be either "lin" or "quad". Please edit code accordingly.\n')
        sys.exit()   

    # AJF create initial conditions for sampler - perturbate the liklihood minimize results slightly to prevent stickin (what the example calls gaussian ball)
    # AJF set number of walkers to 40; ndim is set to 2 for linear and 3 for quadratic (number of parameters to fit / number of parameters to explore)
    nwalkers = 40
    init = fits.x + 1e-4*np.random.rand(nwalkers, dim)
    
    # AJF initialize sampler
    sampler = emcee.EnsembleSampler(nwalkers, dim, log_post_prob, args = (x, y, yerr, fitdim))
    print('\nMCMC sampler running...\n')
    
    # AJF start sampler run
    sampler.run_mcmc(init, walks, progress = True)

    return sampler, dim







#@ds    
def plot_walks(sampler, fitdim, frac):
    """
    Plot the MCMC random walks for each parameter of the provided fit

    Parameters
    ----------
    sampler : :class: emcee.ensemble.EnsembleSampler
        the results of the mcmc sampler
    fitdim : :class: str
        the type of function - lin for linear, quad for quadratic
    frac : :class: int
        the x_lim factor to shorten the x axis by; ie if frac = 10 and steps = 5000, x_lim max will be 500 (inspect burn in)
    
    Returns
    ----------
    None - plots and saves figure
    
    
    """
    # AJF ensure correct labels and dimensions are used for each (2 for linear, 3 for quadratic; equals number of params to fit)
    if fitdim == 'lin':
        dim = 2
        labels = ['m', 'b']
    if fitdim == 'quad':
        dim = 3
        labels = ['a2', 'a1', 'a0']    

    # AJF start plot
    fig, ax = plt.subplots(dim, figsize=(12, 8), sharex=True)
    
    # AJF get the chain of samples (i.e. posterior distribution for each walker)
    samples = sampler.get_chain()
    
    # AJF plot over each parameter; i.e. first iter will run through m for lin or a2 for quad, etc.
    for i in range(dim):
        a = ax[i]
        a.plot(samples[:, :, i], "k", alpha=0.3)
        # AJF add toggle frac to display only first frac % of plot xlim (inspect burn in)
        a.set_xlim(0, len(samples)/frac)
        a.set_ylabel(labels[i])
        a.yaxis.set_label_coords(-0.1, 0.5)

    # AJF set titles, labels, save figure
    ax[-1].set_xlabel("Step Number")
    plt.suptitle('MCMC Walks')
    plt.savefig(f'walker_{fitdim}.png', format = 'png')
    
    plt.show()







#@ds
def print_params(flat, fitdim):
    """
    Print out the results of the MCMC walk in a fancy format (print to screen)
    
    Parameters
    ----------
    flat : :class: numpy.ndarray
        the flattened MCMC results chain - posterior chain
    fitdim : :class: str
        the type of fucntion - linear or quadratic
        
    Returns
    ----------
    :class: dict
        a dictionary containing the model parameters and their median values, and their 1-sigma values (50 to 84th percentile and 50 to 16 percentile)
    :class: list
        a list of the parameter names
    
    """
    # AJF filter input data to be linear or wuad fit
    if fitdim == 'lin':
        params = ['m', 'b']
        print(f'MCMC best-fit parameters for linear fit y = m*x + b\n')
    if fitdim == 'quad':
        params = ['a2', 'a1', 'a0']
        print(f'MCMC best-fit parameters for quadratic fit a2*x^2 + a1*x + a0\n')  
    
    # AJF create dict to store val_tup in for each parameter
    vals = {}
    
    # AJF run through loop of finding and printing best fit value for each param for the provided fit (lin or quad)
    for i in range(flat.shape[1]):
        # AJF find the percentile values for the input flat distribution
        mcmc_vals = np.percentile(flat[:, i], [16, 50, 84])
        
        # AJf calculate the differences between these percentiles to acquire 1 sigma deviations from median
        unc = np.diff(mcmc_vals)
        
        # AJF format a tuple as (median 50th perc., difference between median (50th) and lower (16th) percentiles, difference between median (50th) and upper (84th) percentiles
        # AJF i.e., absolute value of unc[0] is the number you would subtract from median to get to 16th percentile; abs(unc[1]) is what you would add to 50th perc to get to 84th
        # AJF represents 1 sigma distance from median in either direction
        # AJF tuple: (median, - 1 sigma, + 1 sigma) or (median, 50th-16th, 84th-50th)
        val_tup = (mcmc_vals[1], unc[0], unc[1])
        
        # AJF find length of the median value for print formatting
        med_str_len = len('{:.6f}'.format(val_tup[0]))+2
        space = ' '*med_str_len
        
        # AJF print each variable's best fit value with 1-sigma uncertainty in both directions; best fit derived from mcmc
        print(f'{space}+{val_tup[2]:.6f}\n{params[i]} = {val_tup[0]:.6f}\n{space}-{val_tup[1]:.6f}\n\n')
        
        # AJF store val_tup in dictionary with note
        vals[params[i]] = val_tup
        
        # AJF add a dummy note variable in dictionary to make sure user formats it correctly if print to screen
        vals['note'] = '(median, - 1 sigma (50th to 16th), + 1 sigma (84th to 50th))'
        
    return vals, params




    


#@ds
def final_plot(lincf, quadcf, linll, quadll, linmc, quadmc, x, y, yerr):
    """
    Used to compare curve_fit derived model fits, log-liklihood minimize derived fits, and mcmc derived fits

    Parameters
    -----------
    lincf : :class: numpy.ndarray
        curve_fit derived best-fit parameter values for linear function
    quadcf : :class: numpy.ndarray
        curve_fit derived best-fit parameter values for quadratic function
    linll : :class: scipy.optimize._optimize.OptimizeResult
        liklihood minimize derived best-fit parameter values for linear function
    quadll : :class: scipy.optimize._optimize.OptimizeResult
        liklihood minimize derived best-fit parameter values for quadratic function
    linmc : :class: dict
        a dictionary containing the mcmc-derived linear-best fit parameter values/1-sigma uncertainties
    quadmc : :class: dict
        a dictionary containing the mcmc-derived quadratic-best fit parameter values/1-sigma uncertainties
    x : :class: numpy.ndarray
        input x data from table
    y : :class: numpy.ndarray
        input y data from table
    yerr : :class: numpy.ndarray
        input y error data from table

    Returns
    ----------
    None - plots and saves figure
    
    """
    
    # AJF make array of x values to use with models
    x_lin = np.linspace(min(x), max(x), 1000)
    
    # AJF unpack params and make y-values for models from curvefit
    # AJF linear
    m, b = lincf[0], lincf[1]
    mlincf = lin(x_lin, m, b)    
    # AJF quadratic
    a2, a1, a0 = quadcf[0], quadcf[1], quadcf[2]
    mquadcf = quad(x_lin, a2, a1, a0)   
    
    # AJF unpack params and make y-values for models from log-liklihood minimize (ll)
    # AJF linear
    m, b = linll.x
    mlinll = lin(x_lin, m, b)
    # AJF quadratic
    a2, a1, a0 = quadll.x
    mquadll = quad(x_lin, a2, a1, a0)   
    
    # AJF unpack params and make y-values for models from mcmc (mc)
    # AJF linear
    m, b = linmc['m'][0], linmc['b'][0]
    mlinmc = lin(x_lin, m, b)
    # AJF quadratic
    a2, a1, a0 = quadmc['a2'][0], quadmc['a1'][0], quadmc['a0'][0]
    mquadmc = quad(x_lin, a2, a1, a0)      
       
    
    # AJF setup plot
    fig, ax = plt.subplots(3, figsize = (15,20))

    # AJF simple plot x and y data with error
    ax[0].errorbar(x, y, yerr = yerr, fmt = '.', color = 'k', label = 'Data with y-errors')
    ax[1].errorbar(x, y, yerr = yerr, fmt = '.', color = 'k', label = 'Data with y-errors')
    ax[2].errorbar(x, y, yerr = yerr, fmt = '.', color = 'k', label = 'Data with y-errors')
    
    # AJF plot linear models
    ax[0].plot(x_lin, mlinmc, 'g-', label = 'Linear Model Fit from MCMC', linewidth = 4)
    ax[0].plot(x_lin, mlinll, 'b-', label = 'Linear Model Fit from LL-Minimize')
    ax[0].plot(x_lin, mlincf, 'r--', label = 'Linear Model Fit from Curve_Fit')
    ax[0].set_title('Linear Fits', loc = 'left')
   
    # AJF plot quadratic models 
    ax[1].plot(x_lin, mquadmc, 'g-', label = 'Quadratic Model Fit from MCMC', linewidth = 4)
    ax[1].plot(x_lin, mquadll, 'b--', label = 'Quadratic Model Fit from LL-Minimize')  
    ax[1].plot(x_lin, mquadcf, 'r-', label = 'Quadratic Model Fit from Curve_Fit')    
    ax[1].set_title('Quadratic Fits', loc = 'left')

    # AJF plot just data with MCMC fits as final plot
    ax[2].plot(x_lin, mlinmc, 'c-', label = 'Linear Model Fit from MCMC')
    ax[2].plot(x_lin, mquadmc, linestyle = '-', color = 'brown', label = 'Quadratic Model Fit from MCMC')
    ax[2].set_title('MCMC Fits to Data', loc = 'left')


    # AJF add auto-minor ticks (4 per section), increase major tick frequnecy, add major and minor grid
    for a in ax:
        a.xaxis.set_minor_locator(AutoMinorLocator(5))
        a.yaxis.set_minor_locator(AutoMinorLocator(5))
        a.locator_params(axis='both', nbins=15)
        a.legend(markerscale=1)
        
        a.set_xlabel('x')
        a.set_ylabel('y')

        # Major grid
        a.grid(True, which='major', linewidth=0.5, color='gray', alpha=0.5)

        # Minor grid
        a.grid(True, which='minor', linewidth=0.3, color='gray', alpha=0.3)
        
    plt.suptitle('Model Fits', y=0.91, fontweight = 800, fontsize = 16)
    #plt.close()
    plt.savefig('final_fits_plot_mcmc.png', format = 'png')
    plt.show() 





#@ds
def corner_plot(flat, params):
    """
    Create the corner plot - histograms and contour maps of posterior chains for each parameter 
    
    Parameters
    ----------
    flat : :class: numpy.ndarray
        the flattened MCMC results chain - posterior chain        
    params : :class: list  
        a list of parameter names; length is 2 for linear (m and b), 3 for quadratic (a2, a1, a0)
    
    Returns
    ----------
    None - plots and saves figure
    
    """
    
    # AJF do the corner plot with labels added to axes, titles added and size set
    fig = corner.corner(flat, labels = params, show_titles = True, title_fmt= '.6f', figsize = (20, 20))
    
    # AJF change title based on parameter number
    if len(params) == 2:
        fig.text(0.55, 0.75, 'A Corner Plot for\nMCMC Linear Best Fit Paramaters:\nDisplays Posterior Distribution\nin Parameter Space', fontsize=9, fontweight = 600)
        plt.savefig('corner_lin.png', format = 'png')
    if len(params)==3:
        fig.text(0.45, 0.85, 'A Corner Plot for\nMCMC Quadratic Best Fit Paramaters:\nDisplays Posterior Distribution\nin Parameter Space', fontsize=12, fontweight = 600)
        plt.savefig('corner_quad.png', format = 'png')
    
    # AJF show plot
    plt.show()





def main():
    # AJF start code here
    par = argparse.ArgumentParser(description='Using Bayes Theory Framework and MCMC sampling (from the emcee package), fits a linear and a quadratic model to an input x, y, yerr dataset')
    par.add_argument("path", type = str, help = 'path to file where reference data file is located; try /d/scratch/ASTR5160/final')
    par.add_argument("file", type = str, help = 'filename for the x, y, yerr data that needs to be input; try dataxy.fits')
    par.add_argument("walker_num", type = int, help = 'Number of steps that each walker should take in the MCMC sampling')

    arg = par.parse_args()
    
    path = arg.path
    filee = arg.file
    wn = arg.walker_num
    
    # AJF ensure that the full path is correctly formatted
    if path[-1] != '/':
        path = path + '/'       
    fullpath = path+filee
    
    # AJF load in data
    x,y,yerr = load(fullpath)
    
    # AJF find initial guesses for curvefit linear and quadratic fits by visually inspecting basic plot
    basic_errplot(x, y, yerr)
    
    # AJF initial guesses based on above plot
    p0l = (-1, 3)
    p0q = (0.5, -1, 8)
    
    # AJF perform curevfitting using the initial guesses found by visual inspection; model params derived from curve_fit will be used as log-liklihood minimize initial vals
    # AJF returns fitting function, best fit parameters in array form
    linfits = linfit(x, y, yerr, p0l)
    quadfits = quadfit(x, y, yerr, p0q)
    
    # AJF print curve_fit best params
    print(f'\nBest fitting parameters from curve_fit for linear: m = {linfits[0]}, b = {linfits[1]}')
    print(f'Best fitting parameters from curve_fit for quadratic: a2 = {quadfits[0]}, a1 = {quadfits[1]}, a0 = {quadfits[2]}\n')
    
    # AJF perform log-liklihood minimization to get maximum-liklihood parameters to feed into MCMC 
    # AJF note that minimize function does exactly that - minimizes. to find maximum liklihood, do minimize to negative log liklihood 
    sol_lin = minimize_func(linfits, x, y, yerr, 'lin')
    sol_quad = minimize_func(quadfits, x, y, yerr, 'quad')

    # AJF print log-liklihood minimize best params
    print(f'\nBest fitting parameters from LL-minimize for linear: m = {(sol_lin.x)[0]}, b = {(sol_lin.x)[1]}')
    print(f'Best fitting parameters from LL-minimize for quadratic: a2 = {(sol_quad.x)[0]}, a1 = {(sol_quad.x)[1]}, a0 = {(sol_quad.x)[2]}\n\n')
    
    # AJF plot the curvefit line and the ll line and the data to visually see which ones fit better
    basic_23fitplot(linfits, quadfits, sol_lin, sol_quad, x, y, yerr)

    # AJF perform mcmc sampling to find linear fit params
    samplerl, diml = MC(sol_lin, x, y, yerr, 'lin', wn)
    
    # AJF plot the MCMC walks of linear fit for inspection on burn in and any thinning; change last input to limit upper x_lim in plot
    plot_walks(samplerl, 'lin', 1)
    
    # AJF perform mcmc sampling to find fit parameters for quadratic fit
    samplerq, dimq = MC(sol_quad, x, y, yerr, 'quad', wn)

    # AJF plot the MCMC walks of quad fit for inspection on burn in and any thinning; change last input to limit upper x_lim in plot
    plot_walks(samplerq, 'quad', 1)
    
    # AJF comments - burn in for both linear and quad fits occured after between 20 and 50 steps - to be safe, just cut out first 100 samples
    # AJF can check this more quantitatively to confirm - 
    taul, tauq = samplerl.get_autocorr_time(), samplerq.get_autocorr_time()
    print(f'\nAutocorrelation times for linear fit m and b and quadratic fit a2, a1, and a0, respectively: {taul} steps and {tauq} steps.\n')
    
    # AJF autocorr times are less than 50 for all - let's cut out first 150 to be super cautious
    # AJF only keep every 10 points to reduce array sizes, then smash down into one dimensional array
    flatl = samplerl.get_chain(discard = 150, thin = 10, flat = True)
    flatq = samplerq.get_chain(discard = 150, thin = 10, flat = True)

    # AJf print the best fit parameters to screen and return their values and 1 sigma uncertainties: # AJF tuple: (median, - 1 sigma, + 1 sigma) or (median, 50th-16th, 84th-50th)
    linmc, linp = print_params(flatl, 'lin')
    print(f'\n\n\n\n')
    quadmc, quadp = print_params(flatq, 'quad')

    # AJF do final fitting plot
    final_plot(linfits, quadfits, sol_lin, sol_quad, linmc, quadmc, x, y, yerr)

    # AJF do corner plots
    corner_plot(flatl, linp)
    corner_plot(flatq, quadp)

    # AJF final comments on linear vs. quadratic fits
    print(f'\nIt appears that, because the 1-sigma confidence (i.e., the data between the 16th and the 84th percentile)')
    print(f'indicates values for a2 between ~ 0.043 and ~ 0.080, I believe that the quadratic fit is justified; if the a2 parameter')
    print(f'were to have some sort of meaningful probability to be zero, then the argument could be made that the linear fit')
    print(f'works just fine. However, a2 = 0 is on the very, very low left side of the posterior distribution, and essentially')
    print(f'has no probabilty of being the true value for a2, thus indicating that a quadratic fit is more probably necessary.\n')

if __name__=='__main__':
    main() 
    
