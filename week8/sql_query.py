from astropy.table import QTable
import matplotlib.pyplot as plt
import os
import argparse

from master_scripts.subplot_master import sub_plots as sub_plots

# created by AJF 3/17/25
# last edited by AJF 3/17/25


def read():
    """ Reads in all csv files as recarrays and extracts ra and dec form each
    
    Parameters:
    -----------
    None
    
    Returns:
    -----------
    ra_list : :class: list
        list of all ra-lists
        
    dec_lists : :class: list
        list of all dec lists
        
    g_list : :class: list
        list of all g-band mag lists
    
    """
    # AJF get pwd
    pwd = str(os.getcwd())
    
    # AJF use list comprehension to find all csv files
    csv = [f for f in os.listdir(pwd) if f.endswith('.csv')]
    
    # AJF make all csv files into recarrays
    tab_list = [QTable.read(c) for c in csv]
    
    # AJF extract ra, dec, and g-band mag from all recarrays
    ra_list = [f['ra'] for f in tab_list]
    dec_list = [f['dec'] for f in tab_list] 
    g_list = [f['g'] for f in tab_list]
        
    return ra_list, dec_list, g_list


def plot(ra, dec, g):
    """
    Parameters: uses master plotting script to set up and then plot ra/dec pairs with g-band mag
    as size indicator
    
    -----------
    ra : :class: list
        list of all ra-lists
        
    dec : :class: list
        list of all dec lists
        
    g : :class: list
        list of all g-band mag lists
    
    Returns:
    -----------
    None - plots each ra/dec pair with size of points related to g mag
    """

    # AJF use master plotting script to set up subplots
    fig, rows, cols, length, position = sub_plots(ra)
     
    # AJF create all necessary subplots for any number of csv files 
    for i in range(length):
        ax = fig.add_subplot(rows, cols, position[i])
        
        # AJF make sccatterplots; change size of marker scaled by cubing g index....
        # ... (increase spread) then scale size evenly by mult. by 0.01
        ax.scatter(ra[i], dec[i], s = ((g[i]**3)*0.01) ,marker = 'o', color = 'r', label = 'G-Band Mag')
        ax.set_xlabel(r'RA ($^\circ$)')
        ax.set_ylabel(r'Dec ($^\circ$)')
        
        # AJF ensures no offset value shown (i.e. if True value used, displays 3e2 on the side)
        ax.ticklabel_format(useOffset=False, style='plain')  
        
        # AJF legend
        ax.legend()
        
    plt.show()
    
def main():# AJF executes this section first (highest 'shell' of code)
    # AJF add description
    parser = argparse.ArgumentParser(description='Use downloaded SDSS SQL Query results to plot ra/dec pairs and g-band magnitudes of these objects; create a "realistic" image of objects')
    ra, dec, g = read()
    plot(ra, dec, g)



if __name__=='__main__':
    main() 

