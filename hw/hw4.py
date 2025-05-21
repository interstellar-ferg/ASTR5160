import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator

from astropy.table import QTable, hstack, vstack, Column, MaskedColumn
from astropy.coordinates import SkyCoord
from astropy import units as u

import argparse
from tqdm import tqdm
from halo import Halo
import sys
import time

from week8.cross_match import leg_query_list as lql
from hw.hw3 import neg_flux as nf
from hw.hw3 import convert_to_mag as f_to_m

# AJF use chatgpt to ignore unit warnings
import warnings
from astropy.utils.exceptions import AstropyWarning

# AJF created code 5/15/25
# AJF last modified 5/20/25

# AJF import a chat-gpt co-written code that auto-writes docstrings with variables included
from master_scripts.docstring_wrapper import log_sphinx_io as ds
# AJF note: @docstring is a wrapper that auto-writes docstrings for the function directly below it
# AJF see master_scripts/docstring_wrapper for more details

# example command to run:
# python hw4.py /d/scratch/ASTR5160/week10/qsos-ra180-dec30-rad3.fits /d/scratch/ASTR5160/data/legacysurvey/dr9/south/sweep/9.0 sweep-170p025-180p030.fits 1


#@ds
def read(q_file):
    """
    Reads in a provided filepath; slices into table of quasars with their RA and decs
    
    Parameters
    ----------
    q_file : :class: str
        filepath to a fits file containing all quasars ra, dec, redshift, and other info
        
    Returns
    ----------
    :class: astropy.table.table.QTable
        table of objects (likely quasars) 
    :class: astropy.table.column.Column
        the ra column of the table located at filepath
    :class: astropy.table.column.Column
        the dec column of the table located at filepath
    :class: astropy.table.column.Column
        column of redshift values for objects in table at filepath
    :class: astropy.table.column.Column
        column of g-magnitudes for objects in table at filepath
    
    """
    
    quas = QTable.read(q_file)
    raq = quas['RA']
    decq = quas['DEC']
    zq = quas['ZEM']
    gq = quas['GMAG']

    return quas, raq, decq, zq, gq
    



#@ds
def leg_query(uf, path, quas, rai, deci, radius, flux_cols):
    """ Using the filelist generated in leg_query_list, actually find the cross-matched objects in legacy
    and write both the survey (quasar) objects and their legacy cross-matches to tables


    Parameters
    ----------    
    uf : :class: numpy.ndarray
        list of fits files that the input survey could have cross-matches in (usually list of sweep files)
    path : :class: str
        the path to the directory where the uf files are located
    quas : :class: astropy.table.table.QTable
        the input table; in current application, probably a table of quasars
    rai : :class: astropy.table.column.Column
        the input table's RAs
    deci : :class: astropy.table.column.Column
        the input table's decs
    radius : :class: float
        the radius in which to cross match; usually 1 arcsecond - provided by command line
    flux_cols : :class: list
        a list of all the relevant columns containing fluxes in nanomaggy units
    
    Returns
    ----------
    :class: astropy.table.table.QTable
        the full cross-matched table between the input survey quas and the other survey located at path/uf
    :class: astropy.table.table.QTable
        only the path/uf portion of the full table, indexed to include only cross matched objects
    :class: astropy.table.table.QTable
        only the quas portion of the full table, indexed to include only cross matched objects
    :class: numpy.ndarray
         the mask for the paht/uf portion of the full table; apply this mask to the full table to get the 2nd returned variable
         
    """ 
    
    # AJF run through unique file list; this way, not loading in each legacy file for all n rows of input survey; finds ra and dec of all...
    # ... objects cross-matched in one unique legacy file and adds this cross-matched table to a list, of which each sub-table is joined together after loop
    print(f'\n\nLocal Legacy cross-matching progress below:\n')
    
    # AJF create empty list to append each cross-matched table to for storage and empty list to append percent of input objects that were matched
    table_list = []
    perc_list = []
    num_list = [] 
    
    # AJF define all unique units that will be used
    nmg = u.def_unit('nanomaggy') 
    
    # AJF create another list that is all neccessary columns for hw; omit any other columns to avoid vstack error of combining tables with 'UnrecognizedUnit' (like 1/deg^2)
    # all_cols = ['RA', 'DEC'] + flux_cols

    # AJF initialize input survey coordinates before for loop 
    c_input = SkyCoord(ra=rai*u.deg, dec=deci*u.deg, frame='icrs') 

    # AJF go through each unique legacy file only once 
    # AJF create empty list to append all tables to 
    all_tabs = []
    
    print(f'Reading in Legacy fits tables from sweep files...\n')
    
    # AJF run through all legacy files and load in all relevant columns from these tables all at omce
    for f in tqdm(uf):
        
        # AJF read in legacy sweep file
        tab = QTable.read(path + '/' + f)
      
        # AJF filter data by rband before finding cross-matches and filtering rest of data; should reduce quantity of data considerably too
        tab = nf(tab, 'FLUX_R')
        tab['FLUX_R'] = (f_to_m(tab['FLUX_R'].value))*u.mag
        tab.rename_column('FLUX_R', 'R')

        # AJF mask based on max magnitude in r band
        tabr, iir = max_mag_cut(tab, 'R', 19)

        # AJF create new fluxes to run over witout r band
        mod_fc = [flux for flux in flux_cols if flux!='FLUX_R']

        # AJF make sure all negative fluxes are set to NaN, then convert flux to mag for all flux columns and rename them (r-band has already been done)
        for band in mod_fc:
            tabr = nf(tabr, band)
            tabr[band] = (f_to_m(tabr[band].value))*u.mag
            if band[-2] == 'W':
                tabr.rename_column(band, band[-2:])
            else:
                tabr.rename_column(band, band[-1:])
        
        # AJF keep track of column names and units; units list will ensure table after concatenate has proper units
        colnams = tabr.colnames
        units = [tabr[col].unit for col in colnams]
     
        # AJF add table to list of tables
        all_tabs.append(tabr)
    
    # AJF start spinner
    print('\n')
    spinner = Halo(text='Matching to input survey data with search_around_sky...', spinner = 'pong', color = 'green')    
    spinner.start()

    # AJF combine all tables together
    arrs_leg = np.concatenate(all_tabs)
    
    # AJF back to table
    tab_leg = QTable(arrs_leg)
    
    # AJF save full table as test
    full_table_before_r = tab
    
    # AJF put units back onto all columns
    for col, un in zip(colnams, units):
        tab_leg[col].unit = un
    
    # AJF read in ra and dec from legacy for cross-match
    ra_leg, dec_leg = tab_leg['RA'], tab_leg['DEC']

    # AJF match coordinates in arg.radius circumference ONLY ONCE to ALL loaded-in legacy tables
    c_leg = SkyCoord(ra=ra_leg, dec=dec_leg, frame='icrs')
    id_leg, idf, extra1, extra2= c_input.search_around_sky(c_leg, radius*u.arcsec)

    # AJF index legacy table to find out which ones are definitely quasars
    leg_quas = tab_leg[id_leg]  
    
    # AJF add redshift to legacy table from qso file
    zem = quas['ZEM'][idf]
    leg_quas['ZEM'] = zem

    # AJF create a mask that's True for all rows, lenght of the final table
    mask = np.ones(len(tab_leg), dtype=bool)

    # AJF set the matched indices to False for not-confirmed-quasars
    mask[id_leg] = False

    # AJF apply the negative mask to get the unmatched entries
    maybe_not_quas = tab_leg[mask]

    # print(f'final tab_leg:\n\n\n{tab_leg}\n')
    # print(f'final leg_quas:\n\n{leg_quas}\n')

    # AJF stop spinner
    spinner.stop()
    
    return tab_leg, leg_quas, maybe_not_quas, id_leg
        


#@ds
def plot(xq, xo, yq, yo, x_name, y_name, x_min, x_max, y_min, y_max):
    """
    Plot the specified color indexes to see what values would be beneficial to cut at - a visual way to determine color cuts
    
    Parameters
    ----------
    xq : :class: astropy.units.quantity.Quantity
        the color index to plot on the x axis for the directly-input survey of objects (quasars) - i.e., g-z    
    xo : :class: astropy.units.quantity.Quantity
        the color index to plot on the x axis for the path/uf files (legacy sweep files) - i.e., r-w1
    yq : :class: astropy.units.quantity.Quantity
        the color index to plot on the y axis for the directly-input survey of objects (quasars) - i.e., g-z 
    yo : :class: astropy.units.quantity.Quantity
        the color index to plot on the y axis for the path/uf files (legacy sweep files) - i.e., r-w1
    x_name : :class: str
        the name of the x-axis color index - i.e., literally 'G-Z' for label/axes titles
    y_name : :class: str
        the name of the y-axis color index - i.e., literally 'R-W1' for label/axes titles
    x_min : :class: int
        added after inspecting the plots initially - this is the vertical line representing the lower
        cutoff of the x-axis color index
    x_max : :class: float
        added after inspecting the plots initially - this is the vertical line representing the upper
        cutoff of the x-axis color index
    y_min : :class: float
        added after inspecting the plots initially - this is the horizontal line representing the lower
        cutoff of the y-axis color index
    y_max : :class: int
        added after inspecting the plots initially - this is the horizontal line representing the upper
        cutoff of the y-axis color index
        
    Returns
    ----------
    None - just plots

    
    """
    # AJF setup plot
    fig, ax = plt.subplots(1, figsize = (15,20))
    
    # AJF plot scatters
    ax.scatter(xo, yo, color = 'b', s = 1, label = 'non-qsos data')
    ax.scatter(xq, yq, color = 'r', s = 8, marker = '*', label = 'quasars')
    
    # AJF plot horizontal lines
    x = [x_min, x_max]
    y_low = [y_min, y_min]
    y_high = [y_max, y_max]
    # AJF plot lower then upper
    ax.plot(x, y_low, 'k-', linewidth = 2, label = 'color-cut box')
    ax.plot(x, y_high, 'k-', linewidth = 2)
    
    # AJF plot vertical lines
    y = [y_min, y_max]
    x_low = [x_min, x_min]
    x_high = [x_max, x_max]
    # AJF plot lower then upper
    ax.plot(x_low, y, 'k-', linewidth = 2)
    ax.plot(x_high, y, 'k-', linewidth = 2)
    
    # AJF add auto-minor ticks (4 per section), increase major tick frequnecy, add major and minor grid
    ax.xaxis.set_minor_locator(AutoMinorLocator(5))
    ax.yaxis.set_minor_locator(AutoMinorLocator(5))
    ax.locator_params(axis='both', nbins=15)
    ax.legend(markerscale=4)
    
    # Major grid
    ax.grid(True, which='major', linewidth=0.5, color='gray', alpha=0.5)

    # Minor grid
    ax.grid(True, which='minor', linewidth=0.3, color='gray', alpha=0.3)
    
    # AJF set title of plot and axis titles
    fig.suptitle('Color Cut', weight = 600, fontsize = 16, y = 0.93)
    ax.set_xlabel(x_name, fontsize = 12)
    ax.set_ylabel(y_name, fontsize = 12)
    
    # AJF plot
    plt.show()   





#@ds
def check_wise(w1q, w2q, diff):
    """
    Check to make sure that the WISE color index in Stern et al. actually holds true and does not filter too many of the table (quasar) data
    
    Parameters
    ----------
    w1q : :class: astropy.units.quantity.Quantity
        a compilation of the direct-input survey (quasar) WISE1 magnitudes 
    w2q : :class: astropy.units.quantity.Quantity
        a compilation of the direct-input survey (quasar) WISE2 magnitudes
    diff : :class: float
        the cutoff difference between WISE1 and WISE2 magnitudes, converted to AB from Vega (0.8)

    Returns
    ----------
    None - just prints a statement

    """
    
    # AJF mask based on if w1 - w2 >= 0.16 (AB mags) as in Stern 2012 paper - Stern used Vega mags, which gives 0.8 as threshold value
    iiw = ( (w1q.value-w2q.value) >= diff) 
    sum_iiw = sum(iiw)
    len_iiw = len(iiw)
    frac = sum_iiw/len_iiw
    print(f'\n\nPercent of confirmed-quasars recognized as such by W1 - W2 > {diff} cut: {(frac*100):.2f} %\n')




#@ds
def flag_cut(flag, bits, tab, typ):
    """
    Filter an input table by applying a flag cut
    
    Parameters
    ----------
    flag : :class: str
        the name of the flag catagory - foir example, MASKBITS
    bits : :class: list
        for flags like MASKBITS, need a list of the input bit values to filter out
    tab : :class: astropy.table.table.QTable
        the input table to apply the cuts to - probably the full crossmatched table
    typ : :class: str
        the type of object this cut is being applied to - for example, 'confirmed-quasar' could be input
        
    Returns
    ----------
    :class: astropy.table.table.QTable
        the input table tab, but now filtered by the flag cut
    :class: numpy.ndarray
        the actual flag cut mask; can be applied to another table of same length to filter same index objects

    
    """
    
    bit = 0
    # AJF calculate bit total
    for b in bits:
        bit += 2**b
    
    # AJF use the flag and bits to mask based on certain flag cuts; do == 0 so that we get objects where this flag is NOT true (i.e. ones we want to keep)
    flag_mask = (tab[flag] & bit) == 0
        
    # AJF find length before masking
    lenq = len(tab)   
    
    # AJF mask the tables
    tab = tab[flag_mask]
    
    # AJF calculate percent of table retained after cut
    perc_q = ((len(tab))/lenq)*100
    
    # AJF print results
    print(f'Percent of {typ} kept after doing {flag} mask with bits {bits}: {perc_q:.2f} %')
    print(f'In other words, reduced number of {typ} from {lenq} to {len(tab)}.')

    return tab, flag_mask



#@ds
def max_mag_cut(tab, band, mag):
    """
    Filter an input table by a maximum magnitude value (i.e. only objects brighter than given value mag)
    
    Parameters
    ----------
    tab : :class: astropy.table.table.QTable
        input table in which to filter magnitudes - usually full cross-matched table
    band : :class: str
        the flux band in which to filter - maybe R, G, Z, WISE1, etc.
    mag : :class: int
        the value of magnitude to filter objects - dimmest magnitude, keep objects that are brighter than this
    
    Returns
    ----------
    :class: astropy.table.table.QTable
        the input table, but now filtered by magnitude
    :class: numpy.ndarray
        the mask that was used to filter the input table - boolean array

    """
    
    # AJF filter data based on brightness (minimum brightness, i.e. less than certain mag)
    mask = (tab[band].value < mag)
    
    # AJF apply cut
    tab = tab[mask]

    # AJF return table and mask
    return tab, mask



#@ds
def color_cut(tab, band_tup, typ):
    """
    Filter an input table via a color cut - color index has to be greater than/less than certain values
    
    Parameters
    ----------
    tab : :class: astropy.table.table.QTable
        input table in which to filter based on color index - i.e., table to apply color cut filter to
    band_tup : :class: tuple
        a 4-tuple containing info about the bands for the color cut - organized like (band1, band2, minimum cutoff for color index,
        maximum value for color index)
    typ : :class: str
        what type of object is being filtered - ex: 'confirmed-quasars'

    Returns
    ----------
    :class: astropy.table.table.QTable
        the input table, but now filtered to only include objects that follow the input color cut criteria (band_tup)
    :class: numpy.ndarray
        the mask used to filter the input table
    
    """
    # AJF unpack the tuple - band1, band2 (for band1 - band2), cut minimum value, cut maximum value
    b1 = band_tup[0]
    b2 = band_tup[1]
    bmin = band_tup[2]
    bmax = band_tup[3]

    # AJF create a color cut mask to cut the table by
    cut_mask = (tab[b1].value - tab[b2].value > bmin) & (tab[b1].value - tab[b2].value < bmax)

    # AJF find length before masking
    lenq = len(tab)   
    
    # AJF apply mask to see preliminary results
    tab = tab[cut_mask]
    
    # AJF calculate percent of table retained after cut
    perc_q = ((len(tab))/lenq)*100
    
    # AJF print results
    print(f'Percent of {typ} kept after doing {b1} - {b2} color cut: {perc_q:.2f} %')
    print(f'In other words, reduced number of {typ} from {lenq} to {len(tab)}.')

    # AJF return the color-cut mask to compile final Boolean array
    return tab, cut_mask






#@ds
def splendid_function(sweep_tab):
    """
    The compiled function that applies flag and color cuts to an input table in order to filter out all objects
    that are not quasars - essentially, attempts to classify all objects in the input table as either a quasar or not
    via color cuts and flag cuts and magnitude cuts    

    Parameters
    ----------
    sweep_tab : :class: astropy.table.table.QTable
        the input table in which to classify objects as either quasars or not quasars

    Returns
    ----------
    :class: numpy.ndarray
        the final mask that should be applied to the input table to filter out non-quasars
    
    """
    # AJF make copy of sweep_tab unaltered
    original_st = sweep_tab
    
    # AJF assign an empty dictionary to fill with masks
    allmasks = {}
    
    # AJF make list of flux columns (without r, w3 and w4) - r done separately, and w3/w4 not used in this context
    fc = ['FLUX_G', 'FLUX_Z', 'FLUX_W1', 'FLUX_W2'] # , 'FLUX_W3', 'FLUX_W4'

    # AJF start spinner
    print('\n')
    spinner = Halo(text='Performing r-band flux to mag and r-band < 19 cut to reduce number of data points...', spinner = 'pong', color = 'yellow')    
    spinner.start()
    
    # AJF rename r band
    sweep_tab.rename_column('FLUX_R', 'R')
    
    # AJF convert positive flux to mag
    sweep_tab = nf(sweep_tab, 'R')        
    sweep_tab['R'] = (f_to_m(sweep_tab['R'].value))*u.mag   

    # AJF get r-band mask - sweep_tab will now have way less rows
    sweep_tab, iir = max_mag_cut(sweep_tab, 'R', 19)    
    spinner.stop()
    
    # AJF rename columns except for r which has already been done
    print(f'Renaming flux columns and converting fluxes to magnitudes...\n')
    for band in tqdm(fc):
        if band[-2] == 'W':
            sweep_tab.rename_column(band, band[-2:])
            band = band[-2:]
        else:
            sweep_tab.rename_column(band, band[-1])   
            band = band[-1]
        
        # AJF remove negative fluxes by setting them to NaN  
        sweep_tab = nf(sweep_tab, band)
            
        # AJF convert each flux to magnitude
        sweep_tab[band] = (f_to_m(sweep_tab[band].value))*u.mag

    # AJF put the r mask in dictionary - r mask has length = to original table original_st
    allmasks['R'] = iir
    
    # AJF define flag cuts
    flags = ['ANYMASK_G', 'ANYMASK_R', 'ANYMASK_Z', 'WISEMASK_W1', 'WISEMASK_W2', 'MASKBITS']
    bitss = [[1], [1], [1], [0, 1, 2, 3, 4, 5, 6, 7], [0, 1, 2, 3, 4, 5, 6, 7], [1, 2, 3, 4, 5, 6, 7,8, 9, 12]]

    # AJF define color cuts
    w1w2_cut_min = 0.16
    w1w2_cut_max = 1.25
    gz_cut_min = -1
    gz_cut_max = 2.5
    rw1_cut_min = -0.05
    rw1_cut_max = 3

    # AJF inform user which cuts are being used
    print(f'\nThe chosen color cuts were:')
    print(f' {w1w2_cut_max} > W1 - W2 > {w1w2_cut_min}\n {gz_cut_max} > G - Z > {gz_cut_min}\n {rw1_cut_max} > R - W1 > {rw1_cut_min}\n')
    print(f'The chosen flag cuts were:')
    print(f' TYPE = PSF\n ANYMASK_G = [1]\n ANYMASK_R = [1]\n ANYYMASK_Z = [1]\n WISEMASK_W1 = [0, 1, 2, 3, 4, 5, 6, 7]\n WISEMASK_W2 = [0, 1, 2, 3, 4, 5, 6, 7]\n MASKBITS = [1, 2, 3, 4, 5, 6, 7,8, 9, 12]\n')    

    # AJF create all-false array for use in next section - see notes below
    sub_master_mask = np.zeros(len(original_st), dtype = bool)

    # AJF get flag masks - perform flag cuts on sweep_tab, which is r-band < 19 
    for f, bits in zip(flags, bitss):
        st_f, iif = flag_cut(f, bits, sweep_tab, 'objects')
        print(f'\n')
     
        # AJF iif mask is of length of sweep_tab, which is the r-band<19 table; convert this mask back to full original_st length by changing values in a full False array
        # AJF note; could also combine all r-band<19 derived masks at the end, then convert that mask back into a full length mask, but doing it the above way...
        # ... will make more sense for users when viewing the dictionary of arrays
        
        # AJF make iif len=original_st...
        sub_master_mask[iir] = iif        
        
        # AJF put mask in dictionary
        allmasks[f] = sub_master_mask
        
        # AJF create all-false array for use in next section - see notes above
        sub_master_mask = np.zeros(len(original_st), dtype = bool)
        
    # AJF get color cut masks; create unique tuples, and put them in a list
    w1w2_tup = ('W1', 'W2', w1w2_cut_min, w1w2_cut_max)
    gz_tup = ('G', 'Z', gz_cut_min, gz_cut_max)
    rw1_tup = ('R', 'W1', rw1_cut_min, rw1_cut_max)
    tups = [w1w2_tup, gz_tup, rw1_tup]
    
    # AJF perform color cut for all tuples in list
    for tup in tups:
        st_cc, iicc = color_cut(sweep_tab, tup, 'objects')       
        print(f'\n')
           
        # AJF make iicc length = original_st...
        sub_master_mask[iir] = iicc
        
        # AJF put mask in dicitonary
        allmasks[tup[0]+'-'+tup[1]] = sub_master_mask
        
        # AJF create all-false array for use in next section - see notes above
        sub_master_mask = np.zeros(len(original_st), dtype = bool)        
         
    # AJF get type==psf mask and put it in dictionary
    iipsf = (sweep_tab['TYPE'] == 'PSF')
    
    # AJF make iipsf length of original_st...
    sub_master_mask[iir] = iipsf
    
    # AJF put mask in dictionary
    allmasks['PSF'] = sub_master_mask
    
    # AJF apply all masks to table; create list of Trues of length = masks in allmasks, then use dictionary to start switching the relevant Trues to False in the master mask
    master_mask = np.ones(len(allmasks['PSF']), dtype = bool)
    for mask in allmasks.values():
        master_mask &= mask
        
    # AJF apply master mask
    clean_sweep = original_st[master_mask]
    
    # AJF print both tables to cross-examine
    #print(f'\nOriginal sweeps file as a table:\n{original_st}')
    #print(f'\nFinal, flag/color-cut sweeps file, which should result in just quasars:\n{clean_sweep}\n')
    
    return master_mask
    
    
    
    
    
    
    
    

def main():
    # AJF add description
    par = argparse.ArgumentParser(description='Use cuts to determine which objects in an input fits/sweep file are quasars')
    par.add_argument("qpath", type = str, help = 'path to file where reference quasars are located; try /d/scratch/ASTR5160/week10/qsos-ra180-dec30-rad3.fits')
    par.add_argument("path2", type = str, help = 'path to file where input sweep/fits file to check number of quasars; will read in this data as astropy table; try /d/scratch/ASTR5160/data/legacysurvey/dr9/south/sweep/9.0')
    par.add_argument("sweep", type = str, help = 'name of the sweep file (located at the path given as the 2nd command-line argument) that you want to find qsos in for main hw4 color-cut application - used sweep-170p025-180p030.fits as a test')
    par.add_argument("radius", type = float, help = 'radius in arcseconds to attempt to match coordinates; used in search_around_sky, ideal to set to 1 arcsecond')    
 
    arg = par.parse_args()

    q_file = arg.qpath
    path2 = arg.path2
    sweep = arg.sweep
    radius = arg.radius    

    # AJF ignore astropy warnings, make printing prettier (lots of units in legacy files are not defined astropy units which makes printing to terminal ugly)
    warnings.simplefilter('ignore', category=AstropyWarning)
    
    # AJF find out what process user wants to see/use
    print(f'\nChoose your destiny...')
    print(f'\n y will first run the user through the code used to actually determine the cuts, then go through splendid_function;\n n will use splendid_function to just apply the chosen cuts to the users sweep file as input in the 3rd command-line argument.')
    check = input(f'Would you like to run through the code the author used to determine relevant color and flag cuts (y or n)?\n')
    
    # AJF do full determination process if user chooses y
    if check in ['y', 'yes', 'Yes', 'Y']:   
        quas, raq, decq, zq, gq = read(q_file)
    	
        # AJF find list of sweep files that contain final input survey coordinates
        uf = lql(raq, decq, path2, len(raq))
        
        ############################################################################################################################################
        # ...splendid function should find a number of quasars that equals this check function confirmed quasars + objects classified as quasars
        # AJF for example, when using sweep file sweep-170p025-180p030.fits, this determination function found 282 'objects' that could be quasars...
        # ... amd 47 out of 57 confirmed-quasars as quasars ... 282 + 47 = 329, which is exactly the number that splendid_function found!
        
        # AJF change uf to just the input sweep file to confirm splendid_function works
        #uf = [sweep]
        ############################################################################################################################################
        
        # AJF create list of flux columns
        leg_flux_cols = ['FLUX_G', 'FLUX_R', 'FLUX_Z', 'FLUX_W1', 'FLUX_W2', 'FLUX_W3', 'FLUX_W4']

        # AJF do cross-match
        tab_leg, leg_quas, maybe_not_quas, id_leg = leg_query(uf, path2, quas, raq, decq, radius, leg_flux_cols)
        
        # AJF find out which color cuts are the best - from color_cut_18.py, try g-z (x) vs. r-w1 (y) first, then from...
        # ... Stern et al. 2012, try W1-W2 > 0.8 --> W1-W2 > 0.16 if using AB mags, which legacy does
        
        # AJF find out how many quasars have redshift above 3 in the qsos objects - if not many, then Stern 2012 color cut is okay (z<3)
        zem3 = sum(leg_quas['ZEM'] > 3)
        total = len(leg_quas['ZEM'])
        print(f'\nPercent of Quasars in QSOS comparative sample that have redshift over z=3 is {((zem3/total)*100):.2f} %.')
        check = input(f'Stern 2012 uses the W1-W2 (vega) > 0.8 criteria for z<=3 in general. Is this an okay sample to use the Stern color cut on? (y or n)\n')
        if check not in ['y', 'Y', 'yes', 'Yes']:
            print(f'Consider using a different color cut, then. Adjust the code as needed. Exiting...\n')
            sys.exit()
        
        # AJF simple names for columns to make color cut tests easier to code
        gq, rq, zq, w1q, w2q, w3q, w4q = leg_quas['G'], leg_quas['R'], leg_quas['Z'], leg_quas['W1'], leg_quas['W2'], leg_quas['W3'], leg_quas['W4'] 
        go, ro, zo, w1o, w2o, w3o, w4o = maybe_not_quas['G'], maybe_not_quas['R'], maybe_not_quas['Z'], maybe_not_quas['W1'], maybe_not_quas['W2'], maybe_not_quas['W3'], maybe_not_quas['W4']

        # AJF define color cuts after inspecting plots; being conservative
        w1w2_cut_min = 0.16
        w1w2_cut_max = 1.25
        gz_cut_min = -1
        gz_cut_max = 2.5
        rw1_cut_min = -0.05
        rw1_cut_max = 3

        # AJF plot g-z vs r-w1, like in class code
        gzq = gq - zq
        rw1q = rq - w1q
        gzo = go-zo
        rw1o = ro - w1o
        plot(gzq, gzo, rw1q, rw1o, 'G-Z', 'R-W1', gz_cut_min, gz_cut_max, rw1_cut_min, rw1_cut_max)

        # AJF check wise criteria to see if W1-W2>0.16 (in AB mags) actually works on the qsos reference data - should return majority 'true'; i.e. should print a percent close to 90 or 100
        diff = 0.16
        check_wise(w1q, w2q, diff)
        
        # AJF ask user if happy with wise mags
        check = input(f'Continue with the W1-W2 cut after seeing the percent of confirmed quasars it verified? (y or n)\n')
        if check not in ['y', 'Y', 'yes', 'Yes']:
            print(f'Consider using a different color cut, then. Adjust the code as needed. Exiting...\n')
            sys.exit()
        
        # AJF perform the band subtraction (colors)...   
        w1w2q = w1q - w2q
        w1w2o = w1o - w2o

        # AJF plot the WISE color cut
        plot(w1w2q, w1w2o, rw1q, rw1o, 'W1-W2', 'R-W1', w1w2_cut_min, w1w2_cut_max, rw1_cut_min, rw1_cut_max)    

        # AJF now need to actually figure out numbers for the color cuts, plot them on graphs, then test different flag cuts
        # print(f'\nAfter looking at the plots, it was decided to use the simplest color cuts (i.e., not fitting lines, etc) and just use comparison (i.e. W1-W2 > 0.16)')
        # print(f'Color Cuts:\n {w1w2_cut_max} > W1 - W2 > {w1w2_cut_min}\n {gz_cut_max} > G - Z > {gz_cut_min}\n {rw1_cut_max} > R - W1 > {rw1_cut_min}\n')

        # AJF below is code snippet used to test different flag cuts 
        # AJF test type = psf 
        len_nqso = len(maybe_not_quas)
        len_qso = len(leg_quas)

        # AJF filter by type
        leg_q_mod = leg_quas[leg_quas['TYPE'] == 'PSF']
        non_q_mod = maybe_not_quas[maybe_not_quas['TYPE'] == 'PSF']

        # AJF find new length and find percentage filtered
        perc_quas = ( (len(leg_q_mod))/len_qso )*100
        perc_non_q = ( (len(non_q_mod))/len_nqso) * 100
        print(f'Percent of Quasars kept after doing PSF flag mask: {perc_quas:.2f} %\nPercent of other objects kept after doing PSF filter: {perc_non_q:.2f} %')

        # AJF test individual different flag cuts for the original, non-color-cut data 
        flag = 'WISEMASK_W2'
        bits = [0, 1, 2, 3, 4, 5, 6, 7]
        flag_cut(flag, bits, leg_quas, 'confirmed-quasars')
        flag_cut(flag, bits, maybe_not_quas, 'objects')
        print(f'\n')
        
        ##############################################################################

        # AJF notes:
        # maskbits 2, 3, 4 keeps all quasars, removes 20% other
        # maskbits 12 keeps all quasars, removes another 3% or so of other
        # maskbits 11 keeps about 98.5% of quasars, removes another 4% or so of other
        # maskbits 1, 8 keeps all quasars, removes 1% each of other
        # maskbit 5, 6, 7, 9 removes tiny other %
        # other maskbits affect quasars - final MASKBITS bits = [1, 2, 3, 4, 5, 6, 7,8, 9, 12]

        # anymask_g = 9, 8, 7, 6, 4, 2 doesnt really do anything, 11 takes 2% off both, 
        # 1 does a lot! none taken off quasar, takes off 8% of other
        # anymask_r does too!
        # and anymask_z!
        # so definetely do anymask_g, r, z and bit = 1

        # both WISEMASK_W1 and W2 remove a few % each from other, and none from quasar, so include:
        # mask = WISEMASK_W1, W2, bits = [0, 1, 2, 3, 4, 5, 6, 7]

        # type = PSF flag removes about 10% of quasars, but removes about 35% of other... is it worth it? yeah probably

        ##############################################################################
       
        # AJF get length of original color-cut/flag-cut free tables to compare removal of other data and quasar data
        lenqo = len(leg_quas)
        lennqo = len(maybe_not_quas)

        # AJF copy quasar and other tables so they are not overwritten yet
        quasars = leg_quas
        others = maybe_not_quas

        # AJF write final outputs
        print(f'\nThe chosen color cuts were:')
        print(f' {w1w2_cut_max} > W1 - W2 > {w1w2_cut_min}\n {gz_cut_max} > G - Z > {gz_cut_min}\n {rw1_cut_max} > R - W1 > {rw1_cut_min}\n')
        print(f'The chosen flag cuts were:')
        print(f' TYPE = PSF\n ANYMASK_G = [1]\n ANYMASK_R = [1]\n ANYYMASK_Z = [1]\n WISEMASK_W1 = [0, 1, 2, 3, 4, 5, 6, 7]\n WISEMASK_W2 = [0, 1, 2, 3, 4, 5, 6, 7]\n MASKBITS = [1, 2, 3, 4, 5, 6, 7,8, 9, 12]\n')    

        # AJF do all above flags in one go   
        flags = ['ANYMASK_G', 'ANYMASK_R', 'ANYMASK_Z', 'WISEMASK_W1', 'WISEMASK_W2', 'MASKBITS']
        bitss = [[1], [1], [1], [0, 1, 2, 3, 4, 5, 6, 7], [0, 1, 2, 3, 4, 5, 6, 7], [1, 2, 3, 4, 5, 6, 7,8, 9, 12]]
        for f, bits in zip(flags, bitss):
            quasars, mask_q = flag_cut(f, bits, quasars, 'confirmed-quasars')
            others, mask_o = flag_cut(f, bits, others, 'objects')
            print(f'\n')

        # AJF apply color-cuts
        w1w2_tup = ('W1', 'W2', w1w2_cut_min, w1w2_cut_max)
        gz_tup = ('G', 'Z', gz_cut_min, gz_cut_max)
        rw1_tup = ('R', 'W1', rw1_cut_min, rw1_cut_max)
        
        q_final, wmask_q = color_cut(quasars, w1w2_tup, 'confirmed-quasars')
        nq_final, wmask_nq = color_cut(others, w1w2_tup, 'objects')
        print(f'\n')
        q_final, gzmask_q = color_cut(q_final, gz_tup, 'confirmed-quasars')
        nq_final, gzmask_nq = color_cut(nq_final, gz_tup, 'objects')
        print(f'\n')
        q_final, rw1mask_q = color_cut(q_final, rw1_tup, 'confirmed-quasars')
        nq_final, rw1mask_nq = color_cut(nq_final, rw1_tup, 'objects')
        print(f'\n')

        # AJF length of tables before PSF
        lenq_psf = len(q_final)
        lennq_psf = len(nq_final)

        # AJF do PSF in addition to other flag cuts 
        q_final = q_final[q_final['TYPE'] == 'PSF']
        nq_final = nq_final[nq_final['TYPE'] == 'PSF']    

        # AJF calculate percent of each left after PSF cut
        perc_q_psf = (len(q_final)/lenq_psf)*100
        perc_nq_psf = (len(nq_final)/lennq_psf)*100

        # AJF print similar notes as flag cuts
        print(f'Percent of confirmed-quasars kept after doing PSF mask: {perc_q_psf:.2f} %\nPercent of other objects kept after doing PSF mask (after any previous cuts): {perc_nq_psf:.2f} %')
        print(f'In other words, reduced number of confirmed-quasars from {lenq_psf} to {len(q_final)} and number of other objects from {lennq_psf} to {len(nq_final)}.')
        print(f'Are the {lennq_psf - len(nq_final)} ({(100-perc_nq_psf):.2f} %) of other objects removed worth the {lenq_psf-len(q_final)} quasars removed? i.e., is PSF cut worth it? I think so, but not entirely sure...\n')
       
        # AJF length of tables after color cut
        lenqf = len(q_final)
        lennqf = len(nq_final)

        # AJF calculate TOTAL percentage (from before all cuts to after all cuts) and print it   
        percq = (lenqf/lenqo)*100
        percnq = (lennqf/lennqo)*100
        print(f'Final percentage of confirmed-quasars left after peforming all flag cuts and color cuts mentioned above: {percq:.2f} % which is {lenqf} out of {lenqo}.')
        print(f'Final percentage of other objects left after peforming all flag cuts and color cuts mentioned above: {percnq:.2f} % which is {lennqf} out of {lennqo}.\n')
        print('-'*150)
        print(f'Finished with example code for determining color and flag cut criteria')
        print('-'*150)
        print(f'\n')

    # AJF just apply the determined color-cuts to an input sweeps file - splendid_function!
    start = time.time()
    spinner = Halo(text='Reading sweeps file into astropy table...', spinner = 'pong', color = 'red') 
    spinner.start()
    swept = QTable.read(path2 + '/' + sweep)
    spinner.stop()
    quas_bool = splendid_function(swept)
    total_quas = sum(quas_bool)
    num_swept = len(quas_bool)
    print('-'*150)
    print(f'Performed splendid_function and, using color and flag cuts, classified {total_quas} objects in the provided sweeps file {sweep} as quasars.')
    print(f'This means that ~ {((total_quas/num_swept)*100):.6f} % of the objects in the sweeps file were classified by the color/flag cut code as quasars.\n')
    end = time.time()
    tt = end - start
    print(f'Total time reading in sweep file and performing splendid_function took was {int(tt//60)} min {tt%60:.3f} sec')
    print(f'Shortest time taken for the above was 13.890 sec.')
    print(f'Note that the sections that took the longest were a) reading the sweep file into a table and b) performing the r<19 cut.\nIf one or both of these were excluded from the total time, the code would run in under 10 seconds.')
    print('-'*150)
    print(f'\n')
    
        
    
    
if __name__=='__main__':
    main() 
    
    
