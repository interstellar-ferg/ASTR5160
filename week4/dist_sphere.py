import numpy as np

import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
import matplotlib.patches as pat

import argparse

import astropy.coordinates as coord
from astropy.coordinates import SkyCoord
from astropy import units as u


# created Feb 14 2025 by AJF
# last editing done: Feb. 16 2025

def angle():

    """ Finds the angle between two points on a spherical surface (i.e. the sky) manually as well as using skycoord separation

    Parameters:
    ----------
    NOT IN USE: radec : :class: list
        list of the ra's and dec's of the two objects; formatted as ra1 dec1 ra2 dec2

    Returns:
    ----------
    None - prints angle between set of coordinates
    
    """
    # AJF initialize ra and dec values for the two quasars
    #ra1, dec1, ra2, dec2 = radec[0], radec[1], radec[2], radec[3] 

    # AJF initialize coordinates 
    #c1 = SkyCoord(ra = ra1*u.degree, dec = dec1*u.degree, frame = 'icrs')
    #c2 = SkyCoord(ra = ra2*u.degree, dec = dec2*u.degree, frame = 'icrs')

    # AJF initialize coordinates 
    c1 = SkyCoord(ra = 263.75*u.degree, dec = -17.9*u.degree, frame = 'icrs')
    c2 = SkyCoord(ra = '20h24m59.9s', dec = 10*u.degree + 6*u.arcmin, frame = 'icrs')    
    
    
    # convert to cartesian but leave as skycoord
    c1.representation_type = coord.CartesianRepresentation
    c2.representation_type = coord.CartesianRepresentation
    
    # AJF find dot product manually
    dot = (c1.x*c2.x) + (c1.y*c2.y) + (c1.z*c2.z)
    
    # AJF find magnitude manually
    mag1 = np.sqrt(c1.x**2+c1.y**2+c1.z**2)
    mag2 = np.sqrt(c2.x**2+c2.y**2+c2.z**2)
    
    # divide dot by mag product to find cos(angle); take cos^-1, multiply by 180/pi for degrees
    angle = np.degrees(np.arccos( dot / (mag1*mag2) ))
    print(f'\nThis is angle computed manually: {angle}')
    
    # AJF check answer with separation - find separation FROM c1 TO c2 (i.e. 'origin' is c1, locaiton that vector points TO is c2)
    sep = c1.separation(c2)
    print(f'\nThis is angle computed via separation method: {sep}\n')
    print(f'This is difference between the two: {np.round(angle-sep, 9)}\n')
    

def search_dist():

    """ Calculate and plot a) pairs of datapoints that are within 10 arcminutes of each other and b) datapoints within a specified circle of given radius in the sky 
    (i.e., a spectroscopic plate)

    Parameters:
    ----------
    None

    Returns:
    ----------
    None - plots, on a single plot, steps 1-3 of Task 8 (subplot 1), and steps 4-5 of Task 8 (subplot 2)
    
    """
    
    # AJF create arrays of ra and dec and plot on sphere surface
    # AJF scale range from 0 to 1, to 0 to pi/12, then shift up by pi/6 so that it runs from pi/6 to pi/4 
    # AJF need between ra 2 hours, 3 hours, which is between 30 and 45 degrees or pi/6 and pi/4 rads
    ra0 = np.degrees(((np.pi/12)*np.random.random(100))+(np.pi/6))
    ra1 = np.degrees(((np.pi/12)*np.random.random(100))+(np.pi/6))
    
    # AJF shift (0,1) to (0,2) range, then subtract from 1 so that range is (1, -1), then take arcsin of this to get values
    # ... ranging from -pi/2 to pi/2 (depends on sine, so is uniform area across sphere, not cartesian)
    dec0 = np.degrees(np.arcsin( (np.pi/90)* (1.-np.random.random(100)*2) ))
    dec1 = np.degrees(np.arcsin( (np.pi/90)* (1.-np.random.random(100)*2) ))

    c0 = SkyCoord(ra = ra0*u.degree, dec = dec0*u.degree, frame = 'icrs')
    c1 = SkyCoord(ra = ra1*u.degree, dec = dec1*u.degree, frame = 'icrs')
    
    
    """
    # AJF quick check that ra and dec are correct
    
    # AJF need radians for aitoff projection
    ra2 = (((np.pi/12)*np.random.random(1000))+(np.pi/6))  
    dec2 = (np.arcsin( (np.pi/90)* (1.-np.random.random(1000)*2) ))
    
    fig = plt.figure(figsize = (10,10))
    ax0 = plt.subplot(211)
    ax0.plot(ra0, dec0, 'r.', markersize = 4)
    
    # AJF aitoff
    ax1 = plt.subplot(212, projection = 'aitoff')
    ax1.plot(ra2, dec2, 'r.', markersize = 2)
    
    # add grid and show plot
    ax1.grid()
    plt.show()
    
    """
    
    # AJF use search around sky using 10 armins (10 * 1/60)
    id1, id2, d1, d2 = c1.search_around_sky(c0, ( 10 * (1/60) ) *u.degree)
    
    # AJF index each ra, dec by these points witin 10 arcmins of each other and combine into pairs
    ra0_ind, dec0_ind = ra0[id1], dec0[id1]
    ra1_ind, dec1_ind = ra1[id2], dec1[id2]
    ra_ic = np.concatenate([ra0_ind, ra1_ind])
    dec_ic = np.concatenate([dec0_ind, dec1_ind])
    
    #############################################################################################
    # AJF unsure how to zoom in on aitoff projection / change y limits, so will plot coords in cartesian space
    # AJF plot ra and dec of forst 3 steps (in red) in instructions; ra, ded sets and where they are within 10 arcmin from each other (search_around_sky)
    fig, ax = plt.subplots(2, figsize = (15,15), sharex = True)
    fig.subplots_adjust(hspace=0.1)
    
    p1 = ax[0].scatter(ra0, dec0, color = 'r', label = 'Dataset 1', s = 20)
    p2 = ax[0].scatter(ra1, dec1, color = 'k', marker = '*', label = 'Dataset 2', s = 40)
    p3 = ax[0].scatter(ra_ic, dec_ic, color = 'g', marker = 'D', label = f"Search Around Sky\nfor Pairs Within 10'", alpha = 0.3, s = 40)
    
    #############################################################################################
    # AJF combine two ra and dec sets together into big ra/dec sets and plot them on second graph
    ra_c = np.concatenate([ra0, ra1])
    dec_c = np.concatenate([dec0, dec1])
    coord_c = SkyCoord(ra = ra_c*u.degree, dec = dec_c*u.degree, frame = 'icrs')
    
    p4 = ax[1].scatter(ra_c, dec_c, c = 'maroon', s = 40, marker = '*', label = 'Comb. RA & Dec')
    
    #############################################################################################
    # AJF plot circle of radius 1.8 degrees and also plot data within this radius as separate set of data
    ra_cent, dec_cent = '2h20m5s', (- 6 *u.arcmin + 12 * u.arcsec)
    cent_coord = SkyCoord(ra = ra_cent, dec = dec_cent, frame = 'icrs')
    ii = coord_c.separation(cent_coord) < 1.8 * u.degree
    ra_circ, dec_circ = ra_c[ii], dec_c[ii]

    # AJF plot circle
    circle = pat.Circle( (cent_coord.ra.degree, cent_coord.dec.degree), radius = 1.8, alpha = 0.2, color='yellow', label = r'Disc w/ Radius = 1.8$\degree$')
    ax[1].add_patch(circle)
    
    # plot data in circle as different dataset
    p5 = ax[1].scatter(ra_circ, dec_circ, c = 'orange', s = 40, marker = 'D', label = 'Data in Spec. Plate', alpha = 0.5)
    
    #############################################################################################
    # AJF set up plot details as always
    for a in ax:
        a.xaxis.set_minor_locator(AutoMinorLocator(5))
        a.yaxis.set_minor_locator(AutoMinorLocator(5))
        a.locator_params(axis='both', nbins=15)
        a.legend(framealpha = 0.1, bbox_to_anchor = (0.99,0.75), loc = 'lower left', fontsize = 8)
        a.grid(alpha = 0.5)
        
    ax[1].set_xlabel(r'Degrees $\degree$', fontsize = 14)
    ax[1].set_ylabel(r'Degrees $\degree$', fontsize = 14, y = 1.1)
    
    # allow for minor and major ticks to exist on top of plot as well
    ax[0].tick_params(top=True, labeltop=True, which = 'both')
    
    # AJF set titles
    ax[0].set_title('Finding Distances Between Mock Data', loc = 'left', fontsize = 11)
    ax[1].set_title('Finding Data Within Radius Centered at Specified Coordinates', loc = 'left', fontsize = 11)
    fig.suptitle("Distances Between Locations on a Sphere's Surface", fontsize = 16, weight = 550, y = 0.95, x = 0.5)
    
    plt.savefig('distances_sphere.png', format = 'png')
    plt.show()
   
   
def main(): # AJF executes this section first (highest 'shell' of code)
    parser = argparse.ArgumentParser(description='Find points within certain distances of each other other points on a spherical surface')
    
    # AJF execute all functions
    angle()
    search_dist()



if __name__=='__main__':
    main()     
