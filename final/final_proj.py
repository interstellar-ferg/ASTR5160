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


def load(fullpath):
    """
    """
    data = QTable.read(fullpath)
    x, y, yerr = data['x'], data['y'], data['yerr']
    x, y, yerr = np.array(x), np.array(y), np.array(yerr)    
    return x, y, yerr
    






def lin(x_lin, m, b):
    """
    """
    
    line = m*x_lin + b
    return line







def linfit(x, y, yerr, p0):
    """
    """
    x_lin = np.linspace(min(x), max(x), 1000)
    (mf, bf), ccf = curve_fit(lin, x, y, sigma = yerr, p0 = p0)
    fits = np.array([mf, bf])
    
    return fits
 




   


def quad(x_lin, a2, a1, a0):
    """
    """
    
    quadd = a2*(x_lin)**2 + a1*x_lin + a0
    return quadd









def quadfit(x, y, yerr, p0):
    """
    """
    x_lin = np.linspace(min(x), max(x), 1000)    
    (a2, a1, a0), ccf = curve_fit(quad, x, y, sigma=yerr, p0 = p0)
    fits = np.array([a2, a1, a0])
    
    return fits






def basic_errplot(x, y, yerr):
    """
    Used to find initial curve_fit p0 guesses
    
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







def log_lik(fits, x, y, yerr, fitdim):
    """
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
    
    llf = -0.5 * np.sum((((y - model)**2)/yerr**2) + np.log(yerr**2))
    return llf








def ll_func(fits, x, y, yerr, fitdim):
    """
    """
    
    final_func = -log_lik(fits, x, y, yerr, fitdim)
    
    return final_func
    






def minimize_func(fits, x, y, yerr, fitdim):
    """
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








def log_prior(fits, fitdim):
    """
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
    







    

def log_post_prob(fits, x, y, yerr, fitdim):
    """
    """
    # AJF ensure prior is not negative infinity; if it is, return negative inf (since -inf times anything is neg inf)
    if not np.isfinite(log_prior(fits, fitdim)):
        return -np.inf
        
    # AJF if log_prior exists (is zero) then post is read; instead of prior * liklihood, in log spacr they add
    post = log_prior(fits, fitdim) + log_lik(fits, x, y, yerr, fitdim)
    
    return post

   







def basic_23fitplot(lincf, quadcf, linll, quadll, x, y, yerr):
    """
    Used to compare curve_fit derived model fits and log-liklihood minimize derived fits
    
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







def MC(fits, x, y, yerr, fitdim, walks):
    """
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
    
    # AJF start sampler run
    sampler.run_mcmc(init, walks, progress = True)

    return sampler, dim







        
def plot_walks(sampler, fitdim, frac):
    """
    """
    if fitdim == 'lin':
        dim = 2
        labels = ['m', 'b']
    if fitdim == 'quad':
        dim = 3
        labels = ['a2', 'a1', 'a0']    

    fig, ax = plt.subplots(dim, figsize=(12, 8), sharex=True)
    
    samples = sampler.get_chain()
    
    for i in range(dim):
        a = ax[i]
        a.plot(samples[:, :, i], "k", alpha=0.3)
        a.set_xlim(0, len(samples)/frac)
        a.set_ylabel(labels[i])
        a.yaxis.set_label_coords(-0.1, 0.5)

    ax[-1].set_xlabel("Step Number")
    plt.suptitle('MCMC Walks')
    plt.savefig(f'walker_{fitdim}.png', format = 'png')
    
    plt.show()








def print_params(flat, fitdim):
    """
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




    



def final_plot(lincf, quadcf, linll, quadll, linmc, quadmc, x, y, yerr):
    """
    Used to compare curve_fit derived model fits, log-liklihood minimize derived fits, and mcmc derived fits
    
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





def corner_plot(flat, params):
    """
    """
    
    # AJF do the corner plot with labels added to axes, titles added and size set
    fig = corner.corner(flat, labels = params, show_titles = True, title_fmt= '.6f', figsize = (10, 10))
    
    # AJF change title based on parameter number
    if len(params) == 2:
        fig.suptitle('A Corner Plot for MCMC Linear Best Fit Paramaters: Displays Posterior Distribution in Parameter Space')
        plt.savefig('corner_lin.png', format = 'png')
    if len(params)==3:
        fig.suptitle('A Corner Plot for MCMC Quadratic Best Fit Paramaters: Displays Posterior Distribution in Parameter Space')
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



if __name__=='__main__':
    main() 
    
