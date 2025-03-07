import numpy as np
from numpy.random import random

import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator

import astropy.coordinates as coord
from astropy.coordinates import SkyCoord
from astropy import units as u
import pymangle as pym
import healpy as hp

import argparse
from math import radians
import time

# created Feb. 27 2025 by AJF
# last editing done: Mar. 3 2025


def bound_caps():
    """ Create sphereical caps for use in masks

    Parameters:
    ----------
    None

    Returns:
    ----------
    cap1a : :class: list
        form (x, y, z, 1) for cap bound by ra
        
    cap2a : :class: list
        form (x, y, z, 1) for cap bound by ra
        
    cap3a : :class: list
        form (0, 0, 1, 1-np.sin(dec)) for cap bound by dec
        
    cap4a : :class: list
        form (0, 0, 1, 1-np.sin(dec)) for cap bound by dec
        
    cap1b : :class: list
        form (x, y, z, 1) for cap bound by ra
        
    cap2b : :class: list
        form (x, y, z, 1) for cap bound by ra
        
    cap3b : :class: list
        form (0, 0, 1, 1-np.sin(dec)) for cap bound by dec
        
    cap4b : :class: list
        form (0, 0, 1, 1-np.sin(dec)) for cap bound by dec
        
    poly_a_area : :class: numpy.float64
        area of polygon a lat-long 'rectangle'
        
    poly_b_area : :class: numpy.float64
        area of polygon b lat-long 'rectangle'        
        
    """
    
    # AJF initialize coordinates ; add 90 degrees to RA coordinate; area is 1 - cos(theta), where theta is 90 - dec
    ra1a = 5*u.hourangle
    ra2a = 6*u.hourangle
    ra1b = 11*u.hourangle
    ra2b = 12*u.hourangle
    
    c1a = SkyCoord(ra = (ra1a.to(u.deg) + 90*u.deg), dec = 0*u.deg)
    c2a = SkyCoord(ra = (ra2a.to(u.deg) + 90*u.deg), dec = 0*u.deg)
    
    c1b = SkyCoord(ra = (ra1b.to(u.deg) + 90*u.deg), dec = 0*u.deg)
    c2b = SkyCoord(ra = (ra2b.to(u.deg) + 90*u.deg), dec = 0*u.deg)
    
    # AJF intitialize coordinates; for dec-bound, ra = 0 and dec = 90 for all; constraint is 1-sin(dec)
    c3 = SkyCoord(ra = 0*u.deg, dec = 90*u.deg)
    
    # ra and decs in degrees
    ra1ar = ra1a.to(u.rad).value
    ra2ar = ra2a.to(u.rad).value

    ra1br = ra1b.to(u.rad).value
    ra2br = ra2b.to(u.rad).value  
    print(ra1ar, ra2ar, ra1br, ra2br)    
    # AJF change to cartesian x y z
    c1a.representation_type = coord.CartesianRepresentation
    c2a.representation_type = coord.CartesianRepresentation
    c1b.representation_type = coord.CartesianRepresentation
    c2b.representation_type = coord.CartesianRepresentation
    c3.representation_type = coord.CartesianRepresentation
    
    # AJF create cap list with proper formatting; ra-caps
    # polygon a ra-caps
    cap1a = list([ c1a.x.value, c1a.y.value, (c1a.z.value), 1  ])
    cap2a = list([ c2a.x.value, c2a.y.value, (c2a.z.value), 1  ])
    
    # AJF polygon b ra-caps
    cap1b = list([ c1b.x.value, c1b.y.value, (c1b.z.value), 1  ])
    cap2b = list([ c2b.x.value, c2b.y.value, (c2b.z.value), 1  ])
    
    # AJF for dec-caps
    # AJF polygon a dec-caps
    cap3a = list([ int(c3.x.value), int(c3.y.value), int(c3.z.value), 1-np.sin(radians(30)) ])
    cap4a = list([ int(c3.x.value), int(c3.y.value), int(c3.z.value), 1-np.sin(radians(40)) ])
    
    # AJF polygon b dec-caps
    cap3b = list([ int(c3.x.value), int(c3.y.value), int(c3.z.value), 1-np.sin(radians(60)) ])
    cap4b = list([ int(c3.x.value), int(c3.y.value), int(c3.z.value), 1-np.sin(radians(70)) ])
    
    # calculate areas
    poly_a_area = ( ra2ar - ra1ar ) * ( np.sin(radians(40)) - np.sin(radians(30)) )
    poly_b_area = ( ra2br - ra1br ) * ( np.sin(radians(70)) - np.sin(radians(60)) )
    print(poly_a_area, poly_b_area)
    
    return cap1a, cap2a, cap3a, cap4a, cap1b, cap2b, cap3b, cap4b, poly_a_area, poly_b_area




def write(c1a, c2a, c3a, c4a, c1b, c2b, c3b, c4b, pa, pb):
    """ Write the sphereical cap information into .ply files for use in mangle

    Parameters:
    ----------
    c1a : :class: list
        form (x, y, z, 1) for cap bound by ra
        
    c2a : :class: list
        form (x, y, z, 1) for cap bound by ra
        
    c3a : :class: list
        form (0, 0, 1, 1-np.sin(dec)) for cap bound by dec
        
    c4a : :class: list
        form (0, 0, 1, 1-np.sin(dec)) for cap bound by dec
        
    c1b : :class: list
        form (x, y, z, 1) for cap bound by ra
        
    c2b : :class: list
        form (x, y, z, 1) for cap bound by ra
        
    c3b : :class: list
        form (0, 0, 1, 1-np.sin(dec)) for cap bound by dec
        
    c4b : :class: list
        form (0, 0, 1, 1-np.sin(dec)) for cap bound by dec

    pa : :class: numpy.float64
        area of polygon a lat-long 'rectangle'
        
    pb : :class: numpy.float64
        area of polygon b lat-long 'rectangle'
        
    Returns:
    ----------
    None - writes text files with final results in proper format to use in masks
    
    """

    #####################################################
    # AJF write second file with both caps in diff polygons; works like 'both caps' in other file, includes UNION of two polygons
    
    # AJF open a text file in write mode in cwd
    out = open('allcaps.ply', 'w')
    
    # AJF write opening lines, start first polygon
    out.write('2 polygons\npolygon 1 ( 4 caps, 1 weight, 0 pixel, '+str(pa)+' str):\n')
    
    # AJF write values of sph. cap lists in order; ensure no spaces at end of line
    # AJF do polygon a
    for i in range(len(c1a)):
        if i<(len(c1a)-1):
            out.write(str(c1a[i])+' ')
        else:
            out.write(str(c1a[i]))
    out.write('\n')

    for i in range(len(c2a)):
        if i<(len(c2a)-1):
            out.write(str(c2a[i])+' ')
        else:
            out.write(str(c2a[i]))
    out.write('\n')

    for i in range(len(c3a)):
        if i<(len(c3a)-1):
            out.write(str(c3a[i])+' ')
        else:
            out.write(str(c3a[i]))
    out.write('\n')
    
    for i in range(len(c4a)):
        if i<(len(c4a)-1):
            out.write(str(c4a[i])+' ')
        else:
            out.write(str(c4a[i]))
    out.write('\n')      
    
    # start second polygon
    out.write('polygon 2 ( 4 caps, 1 weight, 0 pixel, '+str(pb)+' str):\n')
    
    # AJF write values of sph. cap lists in order; ensure no spaces at end of line
    # AJF do polygon b
    for i in range(len(c1b)):
        if i<(len(c1b)-1):
            out.write(str(c1b[i])+' ')
        else:
            out.write(str(c1b[i]))
    out.write('\n')

    for i in range(len(c2b)):
        if i<(len(c2b)-1):
            out.write(str(c2b[i])+' ')
        else:
            out.write(str(c2b[i]))
    out.write('\n')

    for i in range(len(c3b)):
        if i<(len(c3b)-1):
            out.write(str(c3b[i])+' ')
        else:
            out.write(str(c3b[i]))
    out.write('\n')
    
    for i in range(len(c4b)):
        if i<(len(c4b)-1):
            out.write(str(c4b[i])+' ')
        else:
            out.write(str(c4b[i]))
    out.write('\n') 
        
    # AJF close out and save text file 
    out.close()



    #####################################################
    # AJF write second and third file with both caps in diff polygons to visually see output
    
    # AJF open a text file in write mode in cwd
    out = open('polya.ply', 'w')
    
    # AJF write opening lines, start first polygon
    out.write('1 polygons\npolygon 1 ( 4 caps, 1 weight, 0 pixel, '+str(pa)+' str):\n')
    
    # AJF write values of sph. cap lists in order; ensure no spaces at end of line
    # AJF do polygon a
    for i in range(len(c1a)):
        if i<(len(c1a)-1):
            out.write(str(c1a[i])+' ')
        else:
            out.write(str(c1a[i]))
    out.write('\n')

    for i in range(len(c2a)):
        if i<(len(c2a)-1):
            out.write(str(c2a[i])+' ')
        else:
            out.write(str(c2a[i]))
    out.write('\n')

    for i in range(len(c3a)):
        if i<(len(c3a)-1):
            out.write(str(c3a[i])+' ')
        else:
            out.write(str(c3a[i]))
    out.write('\n')
    
    for i in range(len(c4a)):
        if i<(len(c4a)-1):
            out.write(str(c4a[i])+' ')
        else:
            out.write(str(c4a[i]))
     
    out.close
    
    # AJF open a text file in write mode in cwd
    out = open('polyb.ply', 'w')
    
    # AJF write opening lines, start first polygon
    out.write('1 polygons\npolygon 1 ( 4 caps, 1 weight, 0 pixel, '+str(pb)+' str):\n')
    
    # AJF write values of sph. cap lists in order; ensure no spaces at end of line
    # AJF do polygon b
    for i in range(len(c1b)):
        if i<(len(c1b)-1):
            out.write(str(c1b[i])+' ')
        else:
            out.write(str(c1b[i]))
    out.write('\n')

    for i in range(len(c2b)):
        if i<(len(c2b)-1):
            out.write(str(c2b[i])+' ')
        else:
            out.write(str(c2b[i]))
    out.write('\n')

    for i in range(len(c3b)):
        if i<(len(c3b)-1):
            out.write(str(c3b[i])+' ')
        else:
            out.write(str(c3b[i]))
    out.write('\n')
    
    for i in range(len(c4b)):
        if i<(len(c4b)-1):
            out.write(str(c4b[i])+' ')
        else:
            out.write(str(c4b[i]))
    out.write('\n') 
        
    # AJF close out and save text file 
    out.close()



def masks(pa, pb):
    """ Creates and plots masks derived from spherical caps (generated by bound_caps() ) and from written ply files from write() and tests area of each

    Parameters:
    ----------
    pa : :class: numpy.float64
        area of polygon a lat-long 'rectangle'
        
    pb : :class: numpy.float64
        area of polygon b lat-long 'rectangle'

    Returns:
    ----------
    None - creates png image of all masks drawn from ply files written in write()
    
    """
    
    # AJF create array of ra and dec and plot on sphere surface
    # AJF create 10000 points between 0 and 1, then scale up to 0 to 360 (degrees)
    ra = 360*(random(1000000))
    
    # AJF shift (0,1) to (0,2) range, then subtract from 1 so that range is (1, -1), then take arcsin of this to get values
    # ... ranging from -pi/2 to pi/2, the take np.degrees to get degrees
    # ...(depends on sine, so is uniform area across sphere, not cartesian)
    dec = np.degrees(np.arcsin(1.-random(1000000)*2))
           
    # AJF set up masks - these are basically collections of polygons; polygons are the intersecting surfaces created by 1 or more spherical caps
    mall = pym.Mangle('allcaps.ply')
    ma = pym.Mangle('polya.ply')
    mb = pym.Mangle('polyb.ply')
    
    # AJF check ra and dec within mask
    s = time.perf_counter()
    good = mall.contains(ra, dec)
    e = time.perf_counter()
    time_contains = e - s
    gooda = ma.contains(ra,dec)
    goodb = mb.contains(ra,dec)

    # AJF generate random points for all masks so they can be plotted and visualized
    s = time.perf_counter()
    ra_mall, dec_mall = mall.genrand(10000)
    e = time.perf_counter()
    time_genrand = e - s
    
    # AJF compare time of functions to work
    print(f'\nThis is time m.contains takes: {time_contains} and time genrand takes: {time_genrand}\n')
    
    # AJF find number of points in each rectangular section (polygon a and polygon b)
    print(f'\nNumber of points in poly a (6H - 5H, 40 - 30): {np.sum(gooda)}') 
    print(f'Number of points in poly b (12H - 11H, 70 - 60): {np.sum(goodb)}\n')    
    
    # AJF area calcs
    # AJF whole sphere surface is 4pi str, so find frac of (area_of_poly)/(4pi str) * (1,000,000) and compare to above values
    val_num_a = (pa / (4*np.pi)) * 10**6
    val_num_b = (pb / (4*np.pi)) * 10**6
    print(f'\nThis is area of poly a: {pa} and the number of points in polygon a should be, roughly: {val_num_a}')
    print(f'This is area of poly b: {pb} and the number of points in polygon b should be, roughly: {val_num_b}\n')
    
    # AJF plot initialize
    fig, ax = plt.subplots(1, figsize = (20,15))
    fig.subplots_adjust(hspace=0.1)

    # AJF plot each subplot with relevant data
    ax.plot(ra, dec, 'r.', label = 'All', markersize = 2, alpha = 0.5)
    ax.plot(ra[gooda], dec[gooda], 'g.', label = 'Poly A', markersize = 2)
    ax.plot(ra[goodb], dec[goodb], 'c.', label = 'Poly B', markersize = 2)
    #ax.plot(ra[good], dec[good], 'b.', label = 'Mask', markersize = 2, alpha = .5)
    #ax.plot(ra_mall, dec_mall, 'g.', label = 'Mask Gen_Rand', markersize = 1, alpha = 0.5)
    
    # AJF set x and y ticks on bottom right plot for full degree range
    ax.set_xticks(np.arange(-15, 375, 5))
    ax.set_yticks(np.arange(-105, 105, 5))
    
    # AJF add auto-minor ticks (4 per section), increase major tick frequnecy, add grid, add legends
    ax.xaxis.set_minor_locator(AutoMinorLocator(5))
    ax.yaxis.set_minor_locator(AutoMinorLocator(5))
    ax.locator_params(axis='both', nbins=15)
    ax.grid(True, alpha = 0.75)
    ax.legend(loc = 'upper right', bbox_to_anchor = (1,1), fontsize = 10, markerscale = 7)
    
    # AJF set title of plot and axis titles
    fig.suptitle('Lat-Long Rectangular Regions (Polygons) from Masks', weight = 600, fontsize = 16, y = 0.93)
    ax.set_xlabel(r'Right Ascension ($^\circ$)', fontsize = 12)
    #ax.xaxis.set_label_coords(1.09, -0.09)
    ax.set_ylabel(r'Declination ($^\circ$)', fontsize = 12)
    #ax.yaxis.set_label_coords(-0.08, 1.06)
    
    # AJF save and plot
    plt.savefig('mask_regions_area.png', format = 'png')
    plt.show()
    

def main(): # AJF executes this section first (highest 'shell' of code)
    # AJF add description
    parser = argparse.ArgumentParser(description='Find values for specific spherical caps, output them to text files, then plot these lat-long rectangles and compare areas')
    
    # AJF execute main functions
    c1a, c2a, c3a, c4a, c1b, c2b, c3b, c4b, pa, pb = bound_caps()
    write(c1a, c2a, c3a, c4a, c1b, c2b, c3b, c4b, pa, pb)
    masks(pa, pb)


if __name__=='__main__':
    main() 
