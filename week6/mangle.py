import numpy as np

import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator

import astropy.coordinates as coord
from astropy.coordinates import SkyCoord
from astropy import units as u
import pymangle as pym

import argparse
from math import radians

# created Feb. 27 2025 by AJF
# last editing done: Mar. 2 2025


def circ_caps():
    """ Create sphereical caps for use in masks

    Parameters:
    ----------
    None

    Returns:
    ----------
    cap1 : :class: list
        form (x, y, z, 1-cos(theta)) for cap bound by ra and dec; theta is 'radius'
        
    cap2 : :class: list
        form (x, y, z, 1-cos(theta)) for cap bound by ra and dec; theta is 'radius'
    """
    
    # AJF initialize coordinates; since cap is bound in ra and dec != 0/90, then keep as values; area is 1-cos(theta), where theta is "radius"
    c1 = SkyCoord(ra = 76*u.deg, dec = 36*u.deg)
    c2 = SkyCoord(ra = 75*u.deg, dec = 35*u.deg)
    rad = 5
    
    # AJF change to cartesian x y z
    c1.representation_type = coord.CartesianRepresentation
    c2.representation_type = coord.CartesianRepresentation
    
    # AJF create list with proper formatting
    cap1 = list([ c1.x.value, c1.y.value, c1.z.value, 1-np.cos(radians(rad)) ])
    cap2 = list([ c2.x.value, c2.y.value, c2.z.value, 1-np.cos(radians(rad)) ])
    print(f'\nThis is cap1: {cap1}\n')
    print(f'This is cap2: {cap2}\n')
    
    return cap1, cap2




def write(c1, c2):
    """ Write the sphereical cap information into .ply files for use in mangle

    Parameters:
    ----------
    cap1 : :class: list
        form (x, y, z, 1-cos(theta)) for cap bound by ra and dec; theta is 'radius'
        
    cap2 : :class: list
        form (x, y, z, 1-cos(theta)) for cap bound by ra and dec; theta is 'radius'

    Returns:
    ----------
    None - writes text files with final results in proper format to use in masks
    
    """
    
    #####################################################
    # AJF write first file with both caps in same polygon
    
    # AJF open a text file in write mode in cwd
    out = open('intersection.ply', 'w')
    
    # AJF write opening lines
    out.write('1 polygons\npolygon 1 ( 2 caps, 1 weight, 0 pixel, 0 str):\n')
    
    # AJF write values of sph. cap lists in order; ensure no spaces at end of line
    for i in range(len(c1)):
        if i<(len(c1)-1):
            out.write(str(c1[i])+' ')
        else:
            out.write(str(c1[i]))
    out.write('\n')
    for i in range(len(c2)):
        if i<(len(c2)-1):
            out.write(str(c2[i])+' ')
        else:
            out.write(str(c2[i]))
        
    # AJF close out and save text file 
    out.close()

    #####################################################
    # AJF write second file with both caps in diff polygons
    
    # AJF open a text file in write mode in cwd
    out = open('bothcaps.ply', 'w')
    
    # AJF write opening lines, start first polygon
    out.write('2 polygons\npolygon 1 ( 1 caps, 1 weight, 0 pixel, 0 str):\n')
    
    # AJF write values of sph. cap lists in order; ensure no spaces at end of line
    for i in range(len(c1)):
        if i<(len(c1)-1):
            out.write(str(c1[i])+' ')
        else:
            out.write(str(c1[i]))
    out.write('\n')
    
    # start second polygon
    out.write('polygon 2 ( 1 caps, 1 weight, 0 pixel, 0 str):\n')
    
    # AJF write values of sph. cap lists in order; ensure no spaces at end of line
    for i in range(len(c2)):
        if i<(len(c2)-1):
            out.write(str(c2[i])+' ')
        else:
            out.write(str(c2[i]))
        
    # AJF close out and save text file 
    out.close()

    #####################################################
    # AJF write third and fourth files with both caps in diff polygons and seperate files so can plot in two diff colors
    
    # AJF open a text file in write mode in cwd
    out = open('cap1.ply', 'w')
    
    # AJF write opening lines, start first polygon
    out.write('1 polygons\npolygon 1 ( 1 caps, 1 weight, 0 pixel, 0 str):\n')
    
    # AJF write values of sph. cap lists in order; ensure no spaces at end of line
    for i in range(len(c1)):
        if i<(len(c1)-1):
            out.write(str(c1[i])+' ')
        else:
            out.write(str(c1[i]))
    
    # AJF close out and save text file 
    out.close()
    
    # start second polygon file
    out = open('cap2.ply', 'w')
    out.write('polygon 1 ( 1 caps, 1 weight, 0 pixel, 0 str):\n')
    
    # AJF write values of sph. cap lists in order; ensure no spaces at end of line
    for i in range(len(c2)):
        if i<(len(c2)-1):
            out.write(str(c2[i])+' ')
        else:
            out.write(str(c2[i]))
        
    # AJF close out and save text file 
    out.close()

    #####################################################
    # AJF write fifth file ; same as intersection, but with cap 1 constraint flipped (negative)
    
    # AJF open a text file in write mode in cwd
    out = open('intersection_flip1.ply', 'w')
    
    # AJF write opening lines
    out.write('1 polygons\npolygon 1 ( 2 caps, 1 weight, 0 pixel, 0 str):\n')
    
    # AJF write values of sph. cap lists in order; ensure no spaces at end of line
    for i in range(len(c1)):
        if i<(len(c1)-1):
            out.write(str(c1[i])+' ')
        else:
            out.write(str(-1*c1[i]))
    out.write('\n')
    for i in range(len(c2)):
        if i<(len(c2)-1):
            out.write(str(c2[i])+' ')
        else:
            out.write(str(c2[i]))
        
    # AJF close out and save text file 
    out.close()

    #####################################################
    # AJF write sixth file ; same as intersection, but with cap 2 constraint flipped (negative)
    
    # AJF open a text file in write mode in cwd
    out = open('intersection_flip2.ply', 'w')
    
    # AJF write opening lines
    out.write('1 polygons\npolygon 1 ( 2 caps, 1 weight, 0 pixel, 0 str):\n')
    
    # AJF write values of sph. cap lists in order; ensure no spaces at end of line
    for i in range(len(c1)):
        if i<(len(c1)-1):
            out.write(str(c1[i])+' ')
        else:
            out.write(str(c1[i]))
    out.write('\n')
    for i in range(len(c2)):
        if i<(len(c2)-1):
            out.write(str(c2[i])+' ')
        else:
            out.write(str(-1*c2[i]))
        
    # AJF close out and save text file 
    out.close()

    #####################################################
    # AJF write seventh file ; same as intersection, but with caps 1 AND 2 constraint flipped (negative)
    
    # AJF open a text file in write mode in cwd
    out = open('intersection_flip_12.ply', 'w')
    
    # AJF write opening lines
    out.write('1 polygons\npolygon 1 ( 2 caps, 1 weight, 0 pixel, 0 str):\n')
    
    # AJF write values of sph. cap lists in order; ensure no spaces at end of line
    for i in range(len(c1)):
        if i<(len(c1)-1):
            out.write(str(c1[i])+' ')
        else:
            out.write(str(-1*c1[i]))
    out.write('\n')
    for i in range(len(c2)):
        if i<(len(c2)-1):
            out.write(str(c2[i])+' ')
        else:
            out.write(str(-1*c2[i]))
        
    # AJF close out and save text file 
    out.close()




def masks():
    """ Creates and plots masks derived from spherical caps (generated by circ_caps() ) and from written ply files from write()

    Parameters:
    ----------
    None

    Returns:
    ----------
    None - creates png image of all masks drawn from ply files written in write()
    
    """
    
    # AJF set up masks - these are basically collections of polygons; polygons are the intersecting surfaces created by 1 or more spherical caps
    # AJF intersection is a 'normal' mask; it is the intersection of 2 caps comprising one polygon
    minter = pym.Mangle('intersection.ply')
    
    # AJF bothcaps is the union of polygon 1 (containing one cap) and polygon 2 (containing one cap); this union contains ALL unique elements (i.e. union of...
    # ... (1 2 3) and (3 4 5) is (1 2 3 4 5), while intersection (above) is just (3)
    mboth = pym.Mangle('bothcaps.ply')
    
    # AJF create mask of just single caps to ensure what 'bothcaps' vs. 'intersection' is
    mcap1 = pym.Mangle('cap1.ply')
    mcap2 = pym.Mangle('cap2.ply')
    
    # AJF flip the sign of the constraint (i.e. becomes NOT IN) for cap1 in intersection, then cap2 in intersection, then both caps in intersection
    # AJF miflip12 will be the inverse (opposite) of minter since both constraints have been changed to NOT IN (will be ALL coordinated across full sky EXCEPT intersection)
    miflip1 = pym.Mangle('intersection_flip1.ply')
    miflip2 = pym.Mangle('intersection_flip2.ply')
    miflip12 = pym.Mangle('intersection_flip_12.ply')
    
    # AJF generate random points for all masks so they can be plotted and visualized; do 1,000,000 for miflip12 since this plots nearly full sky (save for...
    # ... small shape centered around (75.5ish, 35.5ish) that is INTERSECTION mask
    ra_inter, dec_inter = minter.genrand(10000)
    ra_both, dec_both = mboth.genrand(10000)
    ra1, dec1 = mcap1.genrand(10000)
    ra2, dec2 = mcap2.genrand(10000)
    raif1, decif1 = miflip1.genrand(10000)
    raif2, decif2 = miflip2.genrand(10000)
    raif12, decif12 = miflip12.genrand(1000000)
    
    # AJF plot initialize
    fig, ax = plt.subplots(2,2, figsize = (20,15))
    fig.subplots_adjust(hspace=0.1)

    # AJF plot each subplot with relevant data
    ax[0,0].plot(ra_inter, dec_inter, 'r.', label = 'Intersection', markersize = 2)
    ax[0,0].plot(ra_both, dec_both, 'b.', label = 'Both', markersize = 2)
    ax[0,1].plot(ra1, dec1, 'g.', label = 'Cap 1 Only', markersize = 2)
    ax[0,1].plot(ra2, dec2, 'm.', label = 'Cap 2 Only', markersize = 2)
    
    ax[1,0].plot(ra_inter, dec_inter, 'r.', label = 'Intersection', markersize = 2)
    ax[1,0].plot(raif1, decif1, 'c.', label = 'Inter. w/ Cap 1 Negative', markersize = 2)
    ax[1,0].plot(raif2, decif2, 'y.', label =  'Inter. w/ Cap 2 Negative', markersize = 2)
    
    ax[1,1].plot(raif12, decif12, 'k.', label = f'Intersection w/ Cap 1\nand 2 Negative', markersize = 2)
    
    # AJF set x ticks on bottom right plot for full degree range
    ax[1,1].set_xticks(np.arange(-15, 375, 15))
    
    # AJF add auto-minor ticks (4 per section), increase major tick frequnecy, add grid, add legends
    for a in ax.flat:
        a.xaxis.set_minor_locator(AutoMinorLocator(5))
        a.yaxis.set_minor_locator(AutoMinorLocator(5))
        a.locator_params(axis='both', nbins=15)
        a.grid(True, alpha = 0.25)
        a.legend(loc = 'upper right', bbox_to_anchor = (1.1,1), fontsize = 8, markerscale = 5)
    
    # AJF set title of plot and axis titles
    fig.suptitle('Masks: Please see legends', weight = 600, fontsize = 16, y = 0.93)
    ax[1,0].set_xlabel(r'Right Ascension ($^\circ$)', fontsize = 12)
    ax[1,0].xaxis.set_label_coords(1.09, -0.09)
    ax[1,0].set_ylabel(r'Declination ($^\circ$)', fontsize = 12)
    ax[1,0].yaxis.set_label_coords(-0.08, 1.06)
    
    # AJF save and plot
    plt.savefig('masks_plotted.png', format = 'png')
    plt.show()


def main(): # AJF executes this section first (highest 'shell' of code)
    # AJF add description
    parser = argparse.ArgumentParser(description='Find values for specific spherical caps, output them to text files, then plot and compare these masks in various ways')
    
    # AJF execute main functions
    c1, c2 = circ_caps()
    write(c1, c2)
    masks()


if __name__=='__main__':
    main() 
