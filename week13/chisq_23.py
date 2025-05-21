import numpy as np
import argparse
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
from scipy.optimize import curve_fit

# AJF import a chat-gpt co-written code that auto-writes docstrings with variables included
from master_scripts.docstring_wrapper import log_sphinx_io as docstring

# AJF create on 5/20/25
# AJF last modified 5/20/25

# example command line to run:
# python chisq_23.py /d/scratch/ASTR5160/week13/line.data 100

# AJF note: @docstring is a wrapper that auto-writes docstrings for the function directly below it
# AJF see master_scripts/docstring_wrapper for more details

#@docstring
def mean_var(data):
    """ 
    find the mean and variance of an input array's columns
    
    Parameters
    ----------
    data : :class: numpy.ndarray
        input array of x bins and y data 
        
    Returns
    ----------
    :class: numpy.ndarray
        
    :class: numpy.ndarray

    """
    # AJF initialize empty lists to append values to 
    means = []
    variances = []
    
    # AJF data is in 20 rows, 10 columns, so transpose to make for loop run over columns instead of rows
    for c in data.T:
        # AJF compute mean and variance - variance usually needs ddof = 1, but I think in this example data it does not?
        # AJF from lecture: Note that for real-world measurements you will need to 
        # pass ddof = 1 to np.var, as the mean of the y values 
        # has already been estimated once from the data 
        # AJF I guess pass it and see difference?
        # AJF gave no difference in final m/b values, so I guess keep ddof=1 in?
        mean = np.mean(c)
        var = np.var(c, ddof = 1)
        
        # AJF append to lists
        means.append(mean)
        variances.append(var)

    # AJF convert to arrays
    means = np.array(means)
    variances = np.array(variances)
    
    return means, variances





#@docstring
def test_plot(mean, var, x_cent):
    """
    Find out which slopes and intercepts fit the test data well - uses curve_fit
    
    Parameters
    ----------
    mean : :class: numpy.ndarray
        the means of y data within an x-bin
    var : :class: numpy.ndarray
        the variances of y data from within x-bins
    x_cent : :class: numpy.ndarray
        
    Returns
    ----------
    :class: numpy.float64
        curve_fit derived slope of best fit line
    :class: numpy.float64
        curve_fit derived y-int of best fit line

    """
    
    # AJF need linspace of x values for fitting function results
    x = np.linspace(min(x_cent), max(x_cent), 100)
    
    #  AJF create model linear function
    def lin(x, m, b):
        line = m*x + b
        return line
    
    # AJF fit data
    (mf, bf), ccf = curve_fit(lin, x_cent, mean, p0 = (1,0))   
    
    # AJF setup plot
    fig, ax = plt.subplots(1, figsize = (15,20))
    
    # AJF simple plot cents with means and variances to get a look at what slope and int might be
    ax.errorbar(x_cent, mean, yerr = np.sqrt(var), fmt = '.', label = 'average data in center of bins')
    ax.plot(x_cent, mf*x_cent+bf, label = 'fit')

    # AJF add auto-minor ticks (4 per section), increase major tick frequnecy, add major and minor grid
    ax.xaxis.set_minor_locator(AutoMinorLocator(5))
    ax.yaxis.set_minor_locator(AutoMinorLocator(5))
    ax.locator_params(axis='both', nbins=15)
    ax.legend(markerscale=4)
    
    # Major grid
    ax.grid(True, which='major', linewidth=0.5, color='gray', alpha=0.5)

    # Minor grid
    ax.grid(True, which='minor', linewidth=0.3, color='gray', alpha=0.3)
    plt.close()
    #plt.show()
    
    return mf, bf




#@docstring
def final_plots(ems, bees, csgrid, m, b, x_cent, means, vs):
    """
    Plot the final best-fit line and the 3d contour plot of chisq
    
    Parameters
    ----------
    ems : :class: numpy.ndarray
        array of possibe slope values to be used in chisq formula
    bees : :class: numpy.ndarray
        array of possible y-int values to be used in chisq formula
    csgrid : :class: numpy.ndarray
        grid of ems and bees - looped over for chisq
    m : :class: numpy.float64
        curve_fit derived slope of best fit line
    b : :class: numpy.float64
        curve_fit derived y-int of best fit line
    x_cent : :class: numpy.ndarray
        array of average x coordinates (middle of each bin)
    means : :class: numpy.ndarray
        the means of the y data in test data
    vs : :class: numpy.ndarray
        the variances of the y data in test data
    Returns
    ----------
    None - plots the chi-sq data
    
    """
    
    # AJF need linspace of x values for plotting best chisq fitting function results
    x = np.linspace(0, 10, 100)
    
    # AJF make best fit function
    y = m*x+b
    
    # AJF setup plot
    fig, ax = plt.subplots(2, figsize = (15,20))
    fig.subplots_adjust(hspace=0.2)
    
    # AJF simple plot best fit with the data
    ax[0].plot(x, y, 'r-', label = f'Chi-Sq Best Fit with m = {m:.4f}, b = {b:.4f}')
    ax[0].errorbar(x_cent, means, yerr = np.sqrt(vs), fmt = '.', label = 'Average data in center of bins')

    # AJF plot the chi square grid with super cool colors - visual inspection of chisq minimum and maximum values
    cm = ax[1].contourf(bees, ems, csgrid, cmap = 'inferno', levels = 100)

    # AJF plot star where min chisq occurs
    ax[1].plot(b, m, 'c*', label = 'Chi-Sq Min', markersize = 10)

    # AJF add auto-minor ticks (4 per section), increase major tick frequnecy, add major and minor grid
    for a in ax:
        a.xaxis.set_minor_locator(AutoMinorLocator(5))
        a.yaxis.set_minor_locator(AutoMinorLocator(5))
        a.locator_params(axis='both', nbins=15)
        
    
        # Major grid
        a.grid(True, which='major', linewidth=0.5, color='gray', alpha=0.5)

        # Minor grid
        a.grid(True, which='minor', linewidth=0.3, color='gray', alpha=0.3)
    
        # AJF colorbar and legend
        a.legend(markerscale=1)

    plt.colorbar(cm, ax = ax[1])
    
    # AJF set titles and axis titled
    ax[0].set_title("Best-Fit Line vs Data")
    ax[1].set_title("ChiSq Grid (m vs b)")
    ax[1].set_xlabel("b")
    ax[1].set_ylabel("m")

    plt.savefig('chisq_plot.png', format = 'png')
    plt.show()  
    



#@docstring
def chisq(x_cent, means, var, ems, bees):
    """
    Find the minimum chisq and the corresponding slope and y-int
    
    Parameters
    ----------
    x_cent : :class: numpy.ndarray
        
    means : :class: numpy.ndarray
        
    var : :class: numpy.ndarray
        
    ems : :class: numpy.ndarray
        
    bees : :class: numpy.ndarray
        
    Returns
    ----------
    :class: numpy.float64
        the slope variable corresponding to minimum chisq
    :class: numpy.float64
        the y-int variable corresponding to minimum chisq
    :class: numpy.ndarray
        the final filled-out grid of calculated chisqs
    
    """
    
    # AJF make empty grid to fill with chisq values
    csgrid = np.zeros((len(ems), len(bees)))
    
    # AJF run through all ems and bees
    for i, m in enumerate(ems):
        for j, b in enumerate(bees):
            # AJF calculate expected value - I think just use center of bins for x as a 'mean' value?
            E = m * x_cent + b
            O = means
            # AJF calculate chisq and assign it to proper spot in grid
            chisq = np.sum( (O - E)**2 / var)
            csgrid[i, j] = chisq
    
    # AJF find minimum chisq value
    min_chisq = csgrid.min()
    
    # AJF find where chisq is minimum and find associated m and b values
    loc = np.where(csgrid == min_chisq)
    best_m = ems[loc[0][0]]
    best_b = bees[loc[0][0]]
    
    return best_m, best_b, csgrid
    





def main():
    # AJF include decsiption
    par = argparse.ArgumentParser(description='Practice chi-square best fits to data')
    par.add_argument("path", type = str, help = 'path to file where reference data is located; try /d/scratch/ASTR5160/week13/line.data')
    par.add_argument('num', type = int, help = 'Number of slopes and intercepts you want to run through for chisq - 100 or 1000 is good')

    arg = par.parse_args()
    path = arg.path
    num = arg.num
    
    # AJF load in data and find means/varainces of each column
    data = np.loadtxt(path)   
    mean, var = mean_var(data)

    # AJF create x bin array from center as approximation
    x_cent = np.arange(0.5, 10, 1)
    
    # AJF plot the data to get an idea of what range m and b should be - fit with linear function
    mf, bf = test_plot(mean, var, x_cent)
    
    print(f'\nSlope from testing plot is {mf:.4f} and int from plot is {bf:.4f}.\n')    
    
    # AJF create arrays of possible slopes and ints
    ems, bees = np.linspace(round(mf)-3, round(mf)+3, num), np.linspace(round(bf)-5, round(bf)+5, num)
    
    # AJF compute chisq for each combo of slope and int
    m, b, csgrid = chisq(x_cent, mean, var, ems, bees)
    
    print(f'\nFinal slope from chisq is {m:.4f} and int from chisq is {b:.4f}.\n') 
    
    # AJF plot the final chisq grid to visualize it; also plot final slope/int fit to center data
    final_plots(ems, bees, csgrid, m, b, x_cent, mean, var)
    
    print(f'\n\n')
    
    

    
if __name__=='__main__':
    main() 
    
