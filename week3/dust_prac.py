import matplotlib.pyplot as plt
from astropy import units as u
from astropy.coordinates import SkyCoord
from matplotlib.ticker import AutoMinorLocator
import dustmaps
from dustmaps.config import config
from dustmaps.sfd import SFDQuery
import numpy as np
import argparse

# created Feb 7 2025 by AJF
# last editing done: Feb. 9 2025


def red_ext(): 
    """ Calculates the reddening of one target and its extinction in the ugriz bands

    Parameters:
    ----------
    None

    Returns:
    ----------
    None
    """
    
    
    # AJF establish coordinates of target
    coord1 = SkyCoord(ra = '12h42m30s', dec = '+41d12m00s')
    
    # set default directory for maps to this one
    dustdir = '/d/scratch/ASTR5160/data/dust/v0_1/maps'
    config['data_dir'] = dustdir
    
    # AJF fund the reddening value from the data located in the directory listed above
    sfd = SFDQuery()
    ebv = sfd(coord1)
    print(f'\nThis is E(B-V) for part 1: {ebv}')
    
    # SDSS filter constants (in swin.edu link, value is 3.2 for V-band (i think)) - reddening
    ugriz = np.array([4.239,3.303,2.285,1.698,1.263])
    
    # this equation is like the one derived at swin.edu link; finds gal. extinction for the SDSS filters (u-band, g-band, etc.)
    # multiplies reddening (E(B-V)) by filter constants to get extinction
    A = ebv*ugriz





def quas_compare(radec): 
    """ Compares two quasar's color before and after accounting for dust extinction

    Parameters:
    ----------
    radec : :class: list
        list of the ra's and dec's of the two objects; formatted as ra1 dec1 ra2 dec2
    Returns:
    ----------
    None - plots the quasar color diagram
    """
    
    
    # AJF initialize ra and dec values for the two quasars
    ra1, dec1, ra2, dec2 = radec[0], radec[1], radec[2], radec[3]

    # AJF put these into coordinates
    coord_q1 = SkyCoord(ra = ra1*u.degree, dec = dec1*u.degree)
    coord_q2 = SkyCoord(ra = ra2*u.degree, dec = dec2*u.degree)

    # AJF quasar 1 has ra of ~247, quasar 2 has ra of ~ 237
    # AJF initlialize g-band, r-band, and i-band magnitudes of each quasar and their errors
    g1, r1, i1 = 18.81, 18.73, 18.82
    g1_err, r1_err, i1_err = 0.01, 0.01, 0.01
    g2, r2, i2 = 19.1, 18.79, 18.73
    g2_err, r2_err, i2_err = 0.01, 0.01, 0.01

    # AJF all errors are 0.01, so subtracting errors value is same (combining uncertainties)
    comb_err = np.sqrt(2*(0.01**2))
    
    # AJF plot initialize
    fig, ax = plt.subplots(2, figsize = (15,15), sharex=True, sharey=True)
    fig.subplots_adjust(hspace=0.03)
    
    
    # AJF plot original color space diagram
    ax[0].errorbar( (r1-i1), (g1-r1), yerr = comb_err, xerr = comb_err, fmt = 'r.', label = 'Quasar1 (a ~ 247, d ~ 41)')
    ax[0].errorbar( (r2-i2), (g2-r2), yerr = comb_err, xerr = comb_err,fmt =  'b.', label = 'Quasar2 (a ~ 237, d ~ 2)')
    
    
    # AJF correct for dust
    # AJF using only gri corrections 
    gri = np.array([4.239, 3.03, 2.285])
    sfd = SFDQuery()
    
    # AJF find E(B-V) reddening
    ebv1 = sfd(coord_q1)
    ebv2 = sfd(coord_q2)
    print(f'\nThis is reddening for Quasar1: {ebv1:.5} and for Quasar2: {ebv2:.5}.\n')
    
    # AJF calculate extinction in each band
    ext1 = ebv1*gri
    ext2 = ebv2*gri
    
    # AJF initialize an array of magntiudes; same step as just setting their values, but makes subtracting extinction values easier
    # AJF order is g, r, i
    mags1 = np.array([18.81, 18.73, 18.82])
    mags2 = np.array([19.1, 18.79, 18.73])
    
    # AJF subtract off extinction to get corrected magnitudes in each band
    # AJF ext is array of g, r, i extinctions and mags is original magnitudes
    mags1_cor = mags1-ext1
    mags2_cor = mags2-ext2
    
    # AJF now plot the corrected color space diagram 
    ax[1].errorbar( (mags1_cor[1]-mags1_cor[2]), (mags1_cor[0]-mags1_cor[1]), yerr = comb_err, xerr = comb_err, fmt = 'r.', label = 'Quasar1 (a ~ 247, d ~ 41)')
    ax[1].errorbar( (mags2_cor[1]-mags2_cor[2]), (mags2_cor[0]-mags2_cor[1]), yerr = comb_err, xerr = comb_err, fmt = 'b.', label = 'Quasar2 (a ~ 237, d ~ 2)')


    # AJF add auto-minor ticks (4 per section), increase major tick frequnecy, add grid, add legends
    for a in ax:
        a.xaxis.set_minor_locator(AutoMinorLocator(5))
        a.yaxis.set_minor_locator(AutoMinorLocator(5))
        a.locator_params(axis='both', nbins=15)
        a.grid(True, alpha = 0.25)
        a.legend(loc = 'center right')

    # AJF assign x and y labels and title
    ax[1].set_xlabel(r'$\text{r - i}$', fontsize = 14)
    ax[1].set_ylabel(r'$\text{g - r}$', fontsize = 14)
    fig.suptitle('Color Diagrams of Two Quasars W/ and W/Out Dust Extinction Corrections', fontsize = 14, y = 0.91)

    # AJF add text to each plot
    plt.figtext(0.45, 0.45, 'Corrected g-r vs r-i',fontsize = 14)
    plt.figtext(0.45, 0.85, 'Original g-r vs r-i',fontsize = 14)
    
    # AJF move the y-label up to the middle of the plots
    ax[1].yaxis.set_label_coords(-0.04, 1.02)
    
    # AJF save and plot
    plt.savefig('quasar_color_compare.png', format = 'png')
    plt.show()   
    
    
    
    
    
def dust_map(radec): 
    """ Produces a reddening map in regions around the given targets

    Parameters:
    ----------
    radec : :class: list
        list of the ra's and dec's of the two objects; formatted as ra1 dec1 ra2 dec2

    Returns:
    ----------
    None - plots reddening map around the two targets
    """
   
    # AJF initialize ra and dec values
    ra1, dec1, ra2, dec2 = radec[0], radec[1], radec[2], radec[3]
    
    # AJF create arrays for meshgrid centered at target's locations
    ra_q1 = np.linspace(ra1-(50*0.1), ra1+(50*0.1), 100)
    ra_q2 = np.linspace(ra2-(50*0.13), ra2+(50*0.13), 100)
    dec_q1 = np.linspace(dec1-(50*0.1), dec1+(50*0.1), 100)
    dec_q2 = np.linspace(dec2-(50*0.1), dec2+(50*0.1), 100)

    # create the meshgrid
    ra1v, dec1v = np.meshgrid(ra_q1, dec_q1)
    ra2v, dec2v = np.meshgrid(ra_q2, dec_q2)
    
    """
    # AJF quick checks on central value of meshgrid for q1 and q2
    plt.plot(ra1v, dec1v, 'ko', markersize = 4)
    plt.hlines(dec1, ra1-(50*0.1), ra1+(50*0.1))
    plt.vlines(ra1, dec1-(50*0.1), dec1+(50*0.1))
    plt.plot(ra1, dec1, 'r*', markersize = 12)
    plt.show()

    plt.plot(ra2v, dec2v, 'ko', markersize = 4)
    plt.hlines(dec2, ra2-(50*0.13), ra2+(50*0.13))
    plt.vlines(ra2, dec2-(50*0.1), dec2+(50*0.1))
    plt.plot(ra2, dec2, 'r*', markersize = 12)
    plt.show()
    """
    
    # AJF correct for dust
    # AJF using only gri corrections 
    gri = np.array([4.239, 3.03, 2.285])
    sfd = SFDQuery()
    
    # AJF create array of coordinates around each quasar based on meshgrid
    q1_coord_arr = SkyCoord(ra = ra1v*u.degree, dec = dec1v*u.degree)  
    q2_coord_arr = SkyCoord(ra = ra2v*u.degree, dec = dec2v*u.degree) 
    
    # AJF find E(B-V) reddening for full area in sky
    ebv_arr_q1 = sfd(q1_coord_arr)
    ebv_arr_q2 = sfd(q2_coord_arr)


    # AJF plot contours
    # AJF plot initialize
    fig, ax = plt.subplots(2, figsize = (15,15))
    fig.subplots_adjust(hspace=0.13)
    
    # AJF plot quasar1 region reddening map
    cont_q1 = ax[0].contourf(ra1v, dec1v, ebv_arr_q1, cmap = 'gist_earth')
    ax[0].plot(ra1, dec1, 'r*', markersize = 12, label = 'Quasar1')
    
    # AJF plot quasar2 region reddening map
    cont_q2 = ax[1].contourf(ra2v, dec2v, ebv_arr_q2, cmap = 'gist_earth')
    ax[1].plot(ra2, dec2, 'r*', markersize = 12, label = 'Quasar2')
    
    # AJF plot colorbars and move them closer to plot
    bar1 = plt.colorbar(cont_q1, label = 'E(B-V) Reddening Value', pad = 0.02)
    bar2 = plt.colorbar(cont_q2, label = 'E(B-V) Reddening Value', pad = 0.02)
    
    
    # AJF add auto-minor ticks (4 per section), increase major tick frequnecy, add grid, add legends
    for a in ax:
        a.xaxis.set_minor_locator(AutoMinorLocator(5))
        a.yaxis.set_minor_locator(AutoMinorLocator(5))
        a.locator_params(axis='both', nbins=15)
        a.grid(True, alpha = 0.15)
        a.legend(loc = 'center right', framealpha = 0.1)
   
   
    # Format x and y axis labels and title
    ax[1].set_xlabel(r'Right Ascension ($^\circ$)', fontsize = 12)
    ax[1].set_ylabel(r'Declination ($^\circ$)', fontsize = 12)
    ax[1].yaxis.set_label_coords(-0.04, 1.02)
    fig.suptitle('Reddening in Regions Around Quasars', fontsize = 16, y = 0.91)
    ax[0].set_title('Reddening Around Quasar 1', loc = 'left', fontsize = 10)
    ax[1].set_title('Reddening Around Quasar 2', loc = 'left', fontsize = 10)
    
    # AJF save and show
    plt.savefig('contour_plot_reddening.png', format = 'png')
    plt.show()
    



def main(): # AJF executes this section first (highest 'shell' of code)
    parser = argparse.ArgumentParser(description='Reads in datafile and pl')
    
    # AJF add user-defined ra and dec for two objects
    parser.add_argument('-radec', metavar = '--radec', type = float, nargs = '+', help = 'ra, dec of objects; input like ra1 dec1 ra2 dec2')
    arg = parser.parse_args()
    radec = arg.radec
    
    # AJF execute all functions
    red_ext()
    quas_compare(radec) 
    dust_map(radec)




if __name__=='__main__':
    main() 
