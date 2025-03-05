import numpy as np
from numpy.random import random

import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator

import argparse
import healpy as hp

# AJF created 2/17/25
# last modified by AJF on 2/17/25

def area(): 
    """ Create a set of 1,000,000 points of (ra, dec) evenly distributed on a spherical surface, then bin these points
    into HEALpixels and check to see they are approx. evenly distibuted across each equal-area pixel

    Parameters:
    ----------
    None

    Returns:
    ----------
    ra : :class: numpy.ndarray
        array of randomly-generated ra coordinates in degrees evenly distributed on surface of sphere
        
    dec : :class: numpy.ndarray
        array of randomly-generated dec coordinates in degrees evenly distributed on surface of sphere
        
    pixels : :class: numpy.ndarray
        array of length equal to ra and dec which indicates the HEALpix level 1 (base) pixel number that that index
        coordinate pair belongs to (i.e., bins coordinates into HEALpix level 1 pixels)
        
    """


    # AJF create array of ra and dec and plot on sphere surface
    # AJF create 10000 points between 0 and 1, then scale up to 0 to 360 (degrees)
    ra = 360*(random(1000000))
    
    # AJF shift (0,1) to (0,2) range, then subtract from 1 so that range is (1, -1), then take arcsin of this to get values
    # ... ranging from -pi/2 to pi/2, the take np.degrees to get degrees
    # ...(depends on sine, so is uniform area across sphere, not cartesian)
    dec = np.degrees(np.arcsin(1.-random(1000000)*2))
    
    # AJF bin ra and dec into heal-pixels; lonlat = true means angle inputs given are in ra/dec format; nside = 1
    # AJF n-side refers to # of divisions of one side of original base-pixel side (see notes in black notebook)
    pixels = hp.ang2pix(1, ra, dec, lonlat = True)
    
    # AJF find area of one pixel at nside = 1 level of HEALPixel (i.e. area of one pixel at base level)
    pix_area_deg = hp.nside2pixarea(1, degrees=True)
    pix_area = hp.nside2pixarea(1, degrees=False)
    print(f'\nThis is area of one HEALpixel at the nside = 1 (base) level in radians: {pix_area} and in degrees: {pix_area_deg}\n')

    # AJF find out how many ra/dec pairs (points) are in each HEALpixel
    num = np.unique(pixels, return_counts=True)
    print(f'This are the pixel bins: {num[0]+1} and the number of points in each pixel\n(i.e., first number is number of points in pixel 1, second number is # in pixel 2, etc.):\n{num[1]}.\n')
    
    # AJF each pixel holds about 83,000 points, which makes sense for equal area since 1,000,000 / 12 ~ 83,333 and ra,dec are
    # ... evenly distributed across sphere
    
    return ra, dec, pixels




def plot_pix(ra, dec, pixels): 
    """ Plot the created ra/dec coordinates and their level 1 binning into pixels 2, 5, and 8; display these HEALPixel bins
    on the plot by overplotting the binned ra/dec of each of these pixels. Then, find which level 2 pixels exist as 
    daughter pixels of level 1, pixel 5. 

    Parameters:
    ----------
    ra : :class: numpy.ndarray
        array of randomly-generated ra coordinates in degrees evenly distributed on surface of sphere
        
    dec : :class: numpy.ndarray
        array of randomly-generated dec coordinates in degrees evenly distributed on surface of sphere

    pixels : :class: numpy.ndarray
        array of length equal to ra and dec which indicates the HEALpix level 1 (base) pixel number that that index
        coordinate pair belongs to (i.e., bins coordinates into HEALpix level 1 pixels)        
    
    Returns:
    ----------
    None - plots figure showing all data and binned.level 1 pixels 2, 5, and 8 overlayed
    
    """
    
    # AJF create Boolean arrays so that ra and dec can be indexed based on which pixel they should be binned in
    ii = pixels == 2
    v = pixels == 5
    viii = pixels == 8
    
    # AJF create level 2 pixel-binning array; same as parameter pixels, but is now at level 2 of HEALpix bisections
    pix2 = hp.ang2pix(2, ra, dec, lonlat = True)
    
    # AJF use pix2[v] to index the level 2 pixel-binning array according to the true/false values of pixels being in 
    # ... level 1, pixel 5, then use np.unique to find each unique level 2 pixel value that exists inside level 1,
    # ... pixel 5
    pix2_in_pix1_5 = np.unique(pix2[v])
    print(f'These are the level 2 pixels that exist inside level 1 pixel 5: {pix2_in_pix1_5}.\n')

    # AJF plot ra, dec, and binned ra/dec coordinates according to ii, v, and viii (HEALPixels)
    fig, ax = plt.subplots(1, figsize = (15,10) )
    
    ax.scatter(ra, dec, c = 'k', marker = '.', s = 0.7, label = 'All RA/Dec Coords')
    ax.scatter(ra[ii], dec[ii], c = 'b', marker = 'D', alpha = 0.75, s = 0.7, label = 'Level 1, Pixel 2')
    ax.scatter(ra[v], dec[v], c = 'r', marker = 'D', alpha = 0.75, s = 0.7, label = 'Level 1, Pixel 5')
    ax.scatter(ra[viii], dec[viii], c = 'g', marker = 'D', alpha = 0.75, s = 0.7, label = 'Level 1, Pixel 8')

    #############################################################################################
    # AJF set up plot details as always
    ax.xaxis.set_minor_locator(AutoMinorLocator(5))
    ax.yaxis.set_minor_locator(AutoMinorLocator(5))
    ax.set_yticks(np.linspace(-90, 90, 13))
    ax.locator_params(axis='both', nbins=15)
    ax.legend(framealpha = 0.1, bbox_to_anchor = (0.965,0.75), loc = 'lower left', fontsize = 10, markerscale = 8)
    ax.grid(alpha = 0.5)
    
    # AJF set labels    
    ax.set_xlabel(r'Degrees $\degree$', fontsize = 14)
    ax.set_ylabel(r'Degrees $\degree$', fontsize = 14)
    
    # allow for minor and major ticks to exist on top of plot as well
    ax.tick_params(top=True, right = True, which = 'both')
    
    # AJF set titles
    fig.suptitle("HEALPixel Binning", fontsize = 16, weight = 550, y = 0.91)
    
    plt.savefig('pixel_binning.png', format = 'png')
    plt.show()



def main(): # AJF executes this section first (highest 'shell' of code)
    parser = argparse.ArgumentParser(description='Practice binning coordinates into certain HEALPixels at different levels of division')
    # AJF execute all functions
    ra, dec, pixels = area()
    plot_pix(ra, dec, pixels)



if __name__=='__main__':
    main() 
