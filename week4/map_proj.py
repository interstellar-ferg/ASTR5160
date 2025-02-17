import numpy as np
from numpy.random import random

import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator

import argparse
from astropy.coordinates import SkyCoord
from astropy import units as u
from dustmaps.config import config
from dustmaps.sfd import SFDQuery

# created Feb 14 2025 by AJF
# last editing done: Feb. 16 2025

  
def compare_proj(): 
    """ Creates a set of 10,000 points randomly selected from a spherical surface distribution; one that
    creates an equal liklihood of the spherical surface being evenly populated with this sample as N 
    approaches infinity 
    - plots these points on a cartesian plane, an aitoff projection, and a lambert projection

    Parameters:
    ----------
    None

    Returns:
    ----------
    None - plots cartesian, aitoff, and lambert images of the distribution
    """

    # AJF create array of ra and dec and plot on sphere surface
    # AJF shift the range (0,1) down by 0.5 so that it is at -0.5, 0.5, then scale by 2pi so it is at -pi, pi
    ra = 2*np.pi*(random(10000)-0.5)
    
    # AJF shift (0,1) to (0,2) range, then subtract from 1 so that range is (1, -1), then take arcsin of this to get values
    # ... ranging from -pi/2 to pi/2 (depends on sine, so is uniform area across sphere, not cartesian)
    dec = np.arcsin(1.-random(10000)*2)
    
    # AJF set up plot 
    fig = plt.figure(figsize = (15, 15))
    fig.subplots_adjust(hspace=0.1)
    
    # set up each subplot, make cartesian coord one take up all upper row
    ax0 = plt.subplot(2, 3, (1,3))
    # create aitoff proj
    ax1 = plt.subplot(2,3,(4,5), projection = 'aitoff')
    # create lambert proj
    ax2 = plt.subplot(2,3,6, projection = 'lambert')
    
    # AJF ax0 below
    # AJF ax0 add auto-minor ticks (4 per section), increase major tick frequnecy, add grid
    ax0.xaxis.set_minor_locator(AutoMinorLocator(5))
    ax0.yaxis.set_minor_locator(AutoMinorLocator(5))
    ax0.locator_params(axis='both', nbins=15)
    ax0.grid(True, alpha = 0.5, color = 'b', linewidth = 1)
    # AJF add x and y labels ax0
    xlabs_rad = np.linspace(-1*np.pi, np.pi, 9)
    xlabs = [r'$-\pi$', r'$-3\pi/4$', r'$-\pi/2$', r'$-\pi/4$', '0', r'$\pi/4$', r'$\pi/2$', r'$3\pi/4$', r'$\pi$']
    ylabs_rad = np.linspace(-1/2*np.pi, 1/2*np.pi, 9)
    ylabs = [r'$-\pi/2$', r'$-3\pi/8$', r'$-\pi/4$', r'$-\pi/8$', '0', r'$\pi/8$', r'$\pi/4$', r'$3\pi/8$', r'$\pi/2$']
    ax0.set_xticks(xlabs_rad)
    ax0.set_xticklabels(xlabs, fontsize = 10)
    ax0.set_yticks(ylabs_rad)
    ax0.set_yticklabels(ylabs, fontsize = 10)
    
    # AJF ax1 below, aitoff
    # AJF add grid, x, y labels to aitoff
    ax1.grid(True, alpha = 0.5, color = 'b', linewidth = 1)
    xlabs = ['14h', '16h', '18h', '20h', '22h', '24/0h', '2h', '4h', '6h', '8h', '10h']
    ylabs = [r'$-75\degree$', r'$-60\degree$', r'$-45\degree$', r'$-30\degree$', r'$-15\degree$', r'$0\degree$', r'$15\degree$', r'$30\degree$', r'$45\degree$', r'$60\degree$', r'$75\degree$']
    ax1.set_xticklabels(xlabs, fontsize = 10, weight = 800)
    ax1.set_yticklabels(ylabs, fontsize = 10, weight = 800)
    ax1.tick_params(labelright = True)
    
    # AJF ax2 below, lambert
    # AJF add grid, smaller x labels to lambert
    ax2.grid(True, alpha = 0.5, color = 'b', linewidth = 1)
    xlabs = ['14h', '16h', '18h', '20h', '22h', '24/0h', '2h', '4h', '6h', '8h', '10h']
    ax2.set_xticklabels(xlabs, fontsize = 8, weight = 800)   
    
    # AJF add labels/titles to axes/plots
    fig.suptitle('Cartesian, Aitoff, and Lambert Projections of Equal-Area-Distributed Points on a Sphere', fontsize = 15, weight = 600, y = 0.93)
    ax0.set_title('Cartesian', loc = 'left', fontsize = 11, weight = 550)
    ax1.set_title('Aitoff', loc = 'center', fontsize = 11, weight = 550)
    ax2.set_title('Lambert', loc = 'center', fontsize = 11, weight = 550)
    
    # AJF plot all axes
    ax0.plot(ra, dec, 'r.', markersize = 1.5, alpha = 0.75)
    ax1.plot(ra, dec, 'r.', markersize = 0.7, alpha = 0.75)
    ax2.plot(ra, dec, 'r.', markersize = 0.5, alpha = 0.75)
    
    # AJF show and save
    plt.savefig('projections.png', format = 'png')
    plt.show()
    
    print(f'\nCartesian plot is less dense near poles due to points being equally distributed acorss surface of sphere -\nconverting spherical to cartesian coordinates is not a linear transformation, so expect less near poles.\n')
    
      

def example_dust_aitoff():
    """ Produces a reddening map in regions around the given targets

    Parameters:
    ----------
    radec : :class: list
        list of the ra's and dec's of the two objects; formatted as ra1 dec1 ra2 dec2

    Returns:
    ----------
    None - plots reddening map around the two targets
    """

    ra = np.linspace(0.5, 359.5, 360)
    dec = np.linspace(-89.5, 89.5, 180)

    RA, DEC = np.meshgrid(ra, dec)
    coordinates = SkyCoord(ra=RA*u.deg, dec= DEC*u.deg)

    dustdir = '/d/scratch/ASTR5160/data/dust/v0_1/maps'
    config["data_dir"] = dustdir
    sfd = SFDQuery()

    ebmv = sfd(coordinates)

    fig = plt.figure(figsize=(15,8))

    plt.xlabel(r'$RA  (\degree)$')
    plt.ylabel(r'$Dec  (\degree)$')
    plt.title('Arc of the Milky Way')
    
    plt.contourf(ra, dec, ebmv, levels=1000, cmap="terrain")
    plt.colorbar(label = 'reddening')
    
    plt.show()
    


def main(): # AJF executes this section first (highest 'shell' of code)
    parser = argparse.ArgumentParser(description='Plot RA and Dec on different projections (Aitoff, Lambert, and Cartesian)')
     
    # AJF execute all functions
    compare_proj()
    
    # AJF apparently there are memory issues with part 2 of this classwork (galactic dust in aitoff proj.)
    # ... so copied Adam's emailed code below as a demonstration
    
    #example_dust_aitoff()




if __name__=='__main__':
    main() 
