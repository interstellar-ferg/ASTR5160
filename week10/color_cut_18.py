import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
from scipy.optimize import curve_fit

import argparse
import os
from datetime import datetime

from astropy.table import QTable
import astropy.coordinates as coord
from astropy.coordinates import SkyCoord
from astropy import units as u 
# AJF use chatgpt to ignore unit warnings
import warnings
from astropy.utils.exceptions import AstropyWarning

from halo import Halo

from week8.cross_match import leg_query_list as lql
from week8.cross_match import leg_query as lq

from collections import Counter

# AJF created 3/24
# AJF last edited 3/28

# comments:



def read1(path):
    """ Which object (quasar or star, in the original creations case) files are read; reads and returns ras, decs, and redshifts (zem)
    
    Parameters:
    ----------
    path : :class: string
        linux directory path to where stars and quasars are located; will read in this data as astropy table; try /d/scratch/ASTR5160/week10

    Returns:
    ----------
    ra_s : :class: astropy.table.column.Column
        stars ra values pulled directly from fits file
    dec_s : :class: astropy.table.column.Column
        stars dec values pulled directly from fits file   
    z_s : class: astropy.table.column.Column
        stars redshift value pulled directly from fits file
    ra_q : :class: astropy.table.column.Column
        quasars ra values pulled directly from fits file
    dec_q : :class: astropy.table.column.Column
        quasars dec values pulled directly from fits file   
    z_q : class: astropy.table.column.Column
        quasars redshift value pulled directly from fits file    

    """
    
    # AJF define file names
    qf = 'qsos-ra180-dec30-rad3.fits'
    sf = 'stars-ra180-dec30-rad3.fits'
    
    # AJF load in data and slice into ra, dec, redshift
    tab_s = QTable.read(path + '/' + sf)
    tab_q = QTable.read(path + '/' + qf)
    ra_s, dec_s, z_s = tab_s['RA'], tab_s['DEC'], tab_s['ZEM']
    ra_q, dec_q, z_q = tab_q['RA'], tab_q['DEC'], tab_q['ZEM']
    
    
    return ra_s, dec_s, z_s, ra_q, dec_q, z_q






def dust_correct(tab_q2, tab_s2):
    """ Corrects two tables' objects' fluxes for dust extinction

    Parameters:
    ----------
    tab_q2 : :class: astropy.table.table.QTable
        object type 1 (quasars, in original case) cross-matched table after cross-matching with legacy fits file objects
    tab_s2 : :class: astropy.table.table.QTable
        object type 2 (stars, in original case) cross-matched table after cross-matching with legacy fits file objects

    Returns:
    ----------
    tab_q2 : :class: astropy.table.table.QTable
        dust-corrected object type 1 (quasars, in original case) cross-matched table
    tab_s2 : :class: astropy.table.table.QTable
        dust-corrected object type 1 (stars, in original case) cross-matched table
        
        
    """
    # AJF complicated list comp to build matching FLUX_# with MW_TRANSMISSION_#
    # AJF make tuple pairs of column names (c,d) if the last two characters of string equal each other (_G, _G, for example) AND if first two characters of one ...
    # ... comparison column is MW (for MW_TRANSMISSION column) AND if the comparison columns don't match (so MW_TRANSMISSION is not matched with MW_TRANSMISSION) 
    list_col = [ (c,d) for c in tab_q2.colnames for d in tab_q2.colnames if d[-2:] == c[-2:] and c[0:2] == 'MW' and d!=c]
    
    # AJF for each tuple pair of FLUX matched with MW_TRANSMISSION, divide the original flux by the MW_TRANSMISSION value to correct for dust
    for cn in list_col:
        tab_q2[f'{cn[1]}_DC'] = tab_q2[cn[1]]/tab_q2[cn[0]]
        tab_s2[f'{cn[1]}_DC'] = tab_s2[cn[1]]/tab_s2[cn[0]]
        
    return tab_q2, tab_s2






def convert_to_mag(final_fc, tab_q, tab_s):
    """ Converts two tables' objects' fluxes to magnitudes

    Parameters:
    ----------
    final_fc : :class: list
        list of command-line argument Legacy survey fits file columns that are also flux columns; i.e., columns like BRICKID are excluded from final_fc, even if they are a command-line argument   
    tab_q : :class: astropy.table.table.QTable
        object type 1 (quasars, in original case) cross-matched table after cross-matching with legacy fits file objects
    tab_s : :class: astropy.table.table.QTable
        object type 2 (stars, in original case) cross-matched table after cross-matching with legacy fits file objects

    Returns:
    ----------
    tab_q : :class: astropy.table.table.QTable
        dust-corrected object type 1 (quasars, in original case) cross-matched table
    tab_s : :class: astropy.table.table.QTable
        dust-corrected object type 1 (stars, in original case) cross-matched table    

    """
    # AJF loop over all command-line argument fluxes (i.e. FLUX_G, FLUX_R, etc.)
    for cn in final_fc:
        # AJF use only value since using units returns error sometime
        cn_val = tab_q[cn].value
        
        # AJF use list comprehesnion to make negative flux values into NaN so np.log10 does not return error (can't find a log of a negative number)
        # AJF note: negative flux values indicate no flux was detected in that band
        cn_val = [c if c>0 else np.nan for c in cn_val]
        
        # AJF use end of FLUX column names in magntiude column names; convert fluxes to mags with formula
        tab_q['MAG'+cn[4:]] = (22.5 - 2.5*np.log10(cn_val))*u.mag
        
        # AJF do same with other table
        cn_val = tab_s[cn].value
        cn_val = [c if c>0 else np.nan for c in cn_val]
        tab_s['MAG'+cn[4:]] = (22.5 - 2.5*np.log10(cn_val))*u.mag
    
    return tab_q, tab_s




    

def color_cut(tab_q, tab_s):
    """ Calculates relevant color indexes, then fits these color indexes with linear model functions. Finds the average of the two functions
    and uses this average as a proxy of the 'separation line' between them, which acts as the 'color cut' line. Returns the color indexes and 
    fit parameters for use in plotting function.
    
    Parameters:
    ----------   
    tab_q : :class: astropy.table.table.QTable
        dust-corrected object type 1 (quasars, in original case) cross-matched table with magnitudes and fluxes
    tab_s : :class: astropy.table.table.QTable
        dust-corrected object type 1 (stars, in original case) cross-matched table with magnitudes and fluxes

    Returns:
    ----------
    gz_q1 : :class: astropy.units.quantity.Quantity
        G-Z color index for all targets for object type 1 (quasar, in original application)
    rw1_q1 : :class: astropy.units.quantity.Quantity
        R-W1 color index for all targets for object type 1 (quasar, in original application)
    gz_s1 : :class: astropy.units.quantity.Quantity
        G-Z color index for all targets for object type 2 (star, in original application)
    rw1_s1 : :class: astropy.units.quantity.Quantity
        R-W1 color index for all targets for object type 2 (star, in original application)     
    lin : :class: function
        model linear function
    avg_m : :class: numpy.float64
        slope of 'separation/color-cut' line - found by averaging slopes of G-Z and R-W1 color index data
    avg_b : :class: numpy.float64
        y-intercept of 'separation/color-cut' line - found by averaging y-int of G-Z and R-W1 color index data for each object type
    mq : :class: numpy.float64
        slope of G-Z (x) vs. R-W1 (y) data for object type 1 (quasar, in original application)
    bq : :class: numpy.float64
        y-intercept of G-Z (x) vs. R-W1 (y) data for object type 1 (quasar, in original application)
    ms : :class: numpy.float64
        slope of G-Z (x) vs. R-W1 (y) data for object type 2 (star, in original application)   
    bs : :class: numpy.float64
        y-intercept of G-Z (x) vs. R-W1 (y) data for object type 1 (quasar, in original application)        
           
    
    """
    # AJF find color indexes
    rw1_q1 = tab_q['MAG_R_DC'] - tab_q['MAG_W1_DC']
    gz_q1 = tab_q['MAG_G_DC'] - tab_q['MAG_Z_DC']    
    rw1_s1 = tab_s['MAG_R_DC'] - tab_s['MAG_W1_DC']
    gz_s1 = tab_s['MAG_G_DC'] - tab_s['MAG_Z_DC'] 
    
    # AJF some rows have NaN values, which can't be subtracted, so just remove those rows in each star/quasar table
    # use np.isnan function for each color index, then use ~ for NOT operator to keep all non-NaN values
    q_nonnan = ~np.isnan(gz_q1) & ~np.isnan(rw1_q1)
    s_nonnan = ~np.isnan(gz_s1) & ~np.isnan(rw1_s1)

    # AJF only keep non-NaN rows in each color index column; curve_fit cannot handle NaN   
    gz_q = gz_q1[q_nonnan]
    rw1_q = rw1_q1[q_nonnan]
    gz_s = gz_s1[s_nonnan]
    rw1_s = rw1_s1[s_nonnan]
    
    
    # AJF need linspace of x values for fitting function results
    x = np.linspace(min(min(gz_q.value), min(gz_s.value)), max(max(gz_q.value), max(gz_s.value)), 100)
    
    #  AJF create model linear function for each dataset; could probably fit with more complictaed polynomial, but just want 
    # ... to find average of quasar data and star data best fit lines to find separation region best fit so that stars and quasars can be classified
    def lin(x, m, b):
        line = m*x + b
        return line
    
    # AJF fit quasar data using curve_fit; could also use lmfit, but that's probably overboard for this simple linear application
    (mq, bq), ccq = curve_fit(lin, gz_q, rw1_q, p0 = (1,-1))

    # AJF fit quasar data using curve_fit; could also use lmfit, but that's probably overboard for this simple linear application
    (ms, bs), ccs = curve_fit(lin, gz_s, rw1_s, p0 = (1,-1))
    
    # AJF find average of two fit lines; this will act as 'best fit of separation' or the color cut line
    avg_m = np.mean([mq, ms])
    avg_b = np.mean([bq, bs])
    
    # AJF return original, non-NaN eliminated color indexes
    
    return gz_q1, rw1_q1, gz_s1, rw1_s1, lin, avg_m, avg_b, mq, bq, ms, bs




def classify_qs(gz_q, rw1_q, gz_s, rw1_s, lin, avg_m, avg_b, tab_q, tab_s):
    """ Classify each target in both input tables as either a star or quasar (in the original application) based on the color cut line.
    Write this classification into the final results table

    Parameters:
    ----------  
    gz_q : :class: astropy.units.quantity.Quantity
        G-Z color index for all targets for object type 1 (quasar, in original application)
    rw1_q : :class: astropy.units.quantity.Quantity
        R-W1 color index for all targets for object type 1 (quasar, in original application)
    gz_s : :class: astropy.units.quantity.Quantity
        G-Z color index for all targets for object type 2 (star, in original application)
    rw1_s : :class: astropy.units.quantity.Quantity
        R-W1 color index for all targets for object type 2 (star, in original application)     
    lin : :class: function
        model linear function
    avg_m : :class: numpy.float64
        slope of 'separation/color-cut' line - found by averaging slopes of G-Z and R-W1 color index data
    avg_b : :class: numpy.float64
        y-intercept of 'separation/color-cut' line - found by averaging y-int of G-Z and R-W1 color index data for each object type     
    tab_q : :class: astropy.table.table.QTable
        dust-corrected object type 1 (quasars, in original case) cross-matched table with magnitudes and fluxes
    tab_s : :class: astropy.table.table.QTable
        dust-corrected object type 1 (stars, in original case) cross-matched table with magnitudes and fluxes

    Returns:
    ----------
    tab_q : :class: astropy.table.table.QTable
        dust-corrected object type 1 (quasars, in original case) cross-matched table with magnitudes and fluxes; now includes color-cut classification object type
    tab_s : :class: astropy.table.table.QTable
        dust-corrected object type 1 (stars, in original case) cross-matched table with magnitudes and fluxes; now includes color-cut classification object type

    
    """
    
    # AJF create lists of color index tuples for each object type
    tuple_q = list(zip(gz_q, rw1_q))
    tuple_s = list(zip(gz_s, rw1_s))
    
    # AJF create a list of strings for all targets for object type 1; if value of either g-z or r-w1 index is NaN, then write Cannot Determine...
    # ... if R-W1(G-Z) is larger than the color cut line R-W1 value, then this object is likely a quasar;
    # ... if R-W1(G-z) is less than color cut line, then object is likely a star
    class_q = ['Cannot Determine' if np.isnan(x[0].value) or np.isnan(x[1].value)
    else 'Likely Quasar' if x[1].value > lin(x[0].value, avg_m, avg_b) and x[0].value < 4
    else 'Likely Star' 
    for x in tuple_q]
    
    # AJF do same for object type 2
    class_s = ['Cannot Determine' if np.isnan(x[0].value) or np.isnan(x[1].value) 
    else 'Likely Star' if x[1].value < lin(x[0].value, avg_m, avg_b) or x[0].value > 4
    else 'Likely Quasar' 
    for x in tuple_s]

    # AJF write these lists to a new column in each table, classifying them according to the color cut
    tab_q['Color Cut Class'] = class_q
    tab_s['Color Cut Class'] = class_s

    return tab_q, tab_s



# ------------------------------------------------------------------------------------------------------------------------------------------------------------- #
# ------------------------------------------------------------------------------------------------------------------------------------------------------------- #
# AJF below code copied directly from ChatGPT; finds the duplicate rows in both tables
# AJF Noticed that quasar table had 320 rows, but row numbers only ranged 1-316, indicating some sort of mismatch or duplication...
# AJF ...so just to make simple, plugged into ChatGPT which wrote below code. works well for checking duplicate values/close objects...
# AJF ...and perhaps helps in identifying survey identification mistakes?

def duplicate(tab_q, tab_s):
    """ Chat-GPT created code that checks to find duplicate values in the 'Inout Survey Row Number' column of the input survey data.
    Indicates which input survey targets have more than one object that was cross-matched with the legacy data.
    
    Parameters:
    ----------
    tab_q : :class: astropy.table.table.QTable
        dust-corrected object type 1 (quasars, in original case) cross-matched table 
    tab_s : :class: astropy.table.table.QTable
        dust-corrected object type 1 (stars, in original case) cross-matched table 
        
    Returns:
    ----------
    None - prints out duplicate rows    
    
    """
    
    # Find duplicate entries based on 'Input Survey Row Number' in tab_q
    counts_q = Counter(tab_q["Input Survey Row Number"])
    duplicates_q = [num for num, count in counts_q.items() if count > 1]

    # Find duplicate entries based on 'Input Survey Row Number' in tab_s
    counts_s = Counter(tab_s["Input Survey Row Number"])
    duplicates_s = [num for num, count in counts_s.items() if count > 1]

    print("\n\nDuplicate entries in tab_q:", duplicates_q)
    print("Duplicate entries in tab_s:", duplicates_s)

    # Print the duplicate rows and the ones around them for tab_q and tab_s
    for num in duplicates_q:
        # Find the index of the duplicate row in the table tab_q
        duplicate_idx_q = np.where(tab_q["Input Survey Row Number"] == num)[0]
        
        # Print the rows around the duplicate, with a range of 2 before and 2 after
        for idx in duplicate_idx_q:
            start_q = max(0, idx - 2)  # Avoid going below index 0
            end_q = min(len(tab_q), idx + 3)  # Avoid going beyond the table length
            print(f"\nShowing rows around duplicate with Survey Row Number {num} in tab_q:")
            print(tab_q[start_q:end_q])
            print("\n" + "="*50 + "\n")  # Separator for readability

    for num in duplicates_s:
        # Find the index of the duplicate row in the table tab_s
        duplicate_idx_s = np.where(tab_s["Input Survey Row Number"] == num)[0]
        
        # Print the rows around the duplicate, with a range of 2 before and 2 after
        for idx in duplicate_idx_s:
            start_s = max(0, idx - 2)  # Avoid going below index 0
            end_s = min(len(tab_s), idx + 3)  # Avoid going beyond the table length
            print(f"\nShowing rows around duplicate with Survey Row Number {num} in tab_s:")
            print(tab_s[start_s:end_s])
            print("\n" + "="*50 + "\n")  # Separator for readability

# ------------------------------------------------------------------------------------------------------------------------------------------------------------- #
# ------------------------------------------------------------------------------------------------------------------------------------------------------------- #



def check_and_plot(gz_q, rw1_q, gz_s, rw1_s, lin, avg_m, avg_b, mq, bq, ms, bs, tab_q, tab_s):
    """ Check for false positives/negatives in the classifiction data for each input table (i.e. in orignal application, identifies stars that were mis-classified as quasars and vice versa)
    Plots the original, unclassified data with the color cut overlaid, then plots the classified data (with false pos/neg data highlighted) for comparison
    
    Parameters:
    ----------   
    gz_q : :class: astropy.units.quantity.Quantity
        G-Z color index for all targets for object type 1 (quasar, in original application)
    rw1_q : :class: astropy.units.quantity.Quantity
        R-W1 color index for all targets for object type 1 (quasar, in original application)
    gz_s : :class: astropy.units.quantity.Quantity
        G-Z color index for all targets for object type 2 (star, in original application)
    rw1_s : :class: astropy.units.quantity.Quantity
        R-W1 color index for all targets for object type 2 (star, in original application)     
    lin : :class: function
        model linear function
    avg_m : :class: numpy.float64
        slope of 'separation/color-cut' line - found by averaging slopes of G-Z and R-W1 color index data
    avg_b : :class: numpy.float64
        y-intercept of 'separation/color-cut' line - found by averaging y-int of G-Z and R-W1 color index data for each object type
    mq : :class: numpy.float64
        slope of G-Z (x) vs. R-W1 (y) data for object type 1 (quasar, in original application)
    bq : :class: numpy.float64
        y-intercept of G-Z (x) vs. R-W1 (y) data for object type 1 (quasar, in original application)
    ms : :class: numpy.float64
        slope of G-Z (x) vs. R-W1 (y) data for object type 2 (star, in original application)   
    bs : :class: numpy.float64
        y-intercept of G-Z (x) vs. R-W1 (y) data for object type 1 (quasar, in original application)  
    tab_q : :class: astropy.table.table.QTable
        dust-corrected object type 1 (quasars, in original case) cross-matched table, including magnitudes, fluxes, and color-cut classification
    tab_s : :class: astropy.table.table.QTable
        dust-corrected object type 1 (stars, in original case) cross-matched table, including magnitudes and fluxes, and color-cut classification
        
    Returns:
    ----------
    None - plots the original data (unclassified) alongside the classified data to compare 
    
    
    """
    
    # AJF split up quasar table into color cut classifications for plotting
    q_class_idx = (tab_q['Color Cut Class'] == 'Likely Quasar') 
    q_class_idx_s = (tab_q['Color Cut Class'] == 'Likely Star')
    q_class_idx_cd = (tab_q['Color Cut Class'] == 'Cannot Determine') 
    
    # AJF index quasar table based on classification
    true_q_gz, true_q_rw1 = gz_q[q_class_idx], rw1_q[q_class_idx]
    
    # AJF all objects in this table are ACTUALLY quasars, but may have been mis-identified as stars
    false_neg_q_gz, false_neg_q_rw1 = gz_q[q_class_idx_s], rw1_q[q_class_idx_s]    
    
    # AJF split up stars table into color cut classifications for plotting
    s_class_idx = (tab_s['Color Cut Class'] == 'Likely Star')
    s_class_idx_q = (tab_s['Color Cut Class'] == 'Likely Quasar')
    s_class_idx_cd = (tab_s['Color Cut Class'] == 'Cannot Determine') 
    
    # AJF index quasar table based on classification
    true_s_gz, true_s_rw1 = gz_s[s_class_idx], rw1_s[s_class_idx]
    
    # AJF split up stars table into color cut classifications for plotting
    false_neg_s_gz, false_neg_s_rw1 = gz_s[s_class_idx_q], rw1_s[s_class_idx_q]   

    # AJF combine all color-cut-defined quasar and star g-z data into two arrays for plotting
    class_q_gz = np.concatenate( (true_q_gz, false_neg_s_gz) )
    class_s_gz = np.concatenate( (true_s_gz, false_neg_q_gz) )
    
    # AJF combine all color-cut-defined quasar and star r-w1 data into two arrays for plotting
    class_q_rw1 = np.concatenate( (true_q_rw1, false_neg_s_rw1) )
    class_s_rw1 = np.concatenate( (true_s_rw1, false_neg_q_rw1) )

    # AJF need linspace of x values for fitting function results
    x = np.linspace(min(min(gz_q.value), min(gz_s.value)), max(max(gz_q.value), max(gz_s.value)), 100)
    
    # Create functions for the three fit lines (quasar, star, separator)
    q_linfit = lin(x, mq, bq)
    s_linfit = lin(x, ms, bs)
    avg_linfit = lin(x, avg_m, avg_b)    
    
    # AJF plot initialize
    fig, ax = plt.subplots(2, figsize = (15,20), sharex = True)
    ax1, ax2 = ax    
    fig.subplots_adjust(hspace=0.02)
    
    # AJF plot falsly identified stars first; zorder not working
    ax2.scatter(false_neg_s_gz, false_neg_s_rw1, color = 'black', marker = '*', s = 200, label = 'Stars that are\nFalsely Identified\nas Quasars')

    # AJF plot code-defined quasars and stars (via classify function)
    ax2.scatter(class_q_gz, class_q_rw1, color = 'red', label = 'Quasars', s = 25, alpha = 0.75)
    ax2.scatter(class_s_gz, class_s_rw1, color = 'blue', label = 'Stars', s = 25, alpha = 0.75)

    # AJF plot falsly identified quasars last; zorder not working
    ax2.scatter(false_neg_q_gz, false_neg_q_rw1, color = 'black', marker = '+', s = 600, label = 'Stars that are\nFalsely Identified\nas Quasars')

        
    # AJF plot the color indixes for stars and quasars
    ax1.scatter(gz_q, rw1_q, color='red', label='Quasars')
    ax1.scatter(gz_s, rw1_s, color='blue', label='Stars')

    # AJF plot the best fit lines for quasar and stars alongside the average of these two (the separation best fit proxy)
    ax1.plot(x, q_linfit, 'r--', label='Quasar Fit')
    ax1.plot(x, s_linfit, 'b--', label='Star Fit')
    ax1.plot(x, avg_linfit, 'k--', label='Average Fit')
    
    # AJF create color cut lines after looking at plot
    # AJF make linspace going up to intersection of lines (about g-z = 4)
    x_short = np.linspace(min(min(gz_q.value), min(gz_s.value)), 4, 100)
    ccut = lin(x_short, avg_m, avg_b)
    
    # AJF plot this shorter color cut alongside vertical line at g-z = 4; upper left space represents region where object is likely to be quasar
    for a in ax:
        a.plot(x_short, ccut, color = 'green', label = 'Color Cut', alpha = 0.6, linewidth = 2)
        a.vlines(4, lin(4, avg_m, avg_b), max(max(gz_q.value), max(gz_s.value)) , color = 'green', alpha = 0.6, linewidth = 2)

    # AJF add auto-minor ticks (4 per section), increase major tick frequnecy, add grid, add legends
    for a in ax:
        a.xaxis.set_minor_locator(AutoMinorLocator(5))
        a.yaxis.set_minor_locator(AutoMinorLocator(5))
        a.locator_params(axis='both', nbins=15)
        a.grid(True, alpha = 0.75)
        a.legend(loc = 'upper right', bbox_to_anchor = (0.16,1), fontsize = 10, markerscale = 1)
    
    # AJF make ticks and labels on both top and bottom x
    ax1.xaxis.set_tick_params(labelbottom=False, labeltop=True)
    ax2.xaxis.set_tick_params(labelbottom=True, labeltop=False)
    
    # AJF set title of plot and axis titles
    fig.suptitle('Color Cut', weight = 600, fontsize = 18, y = 0.93)
    ax1.set_title(f'Input Data Actual Object Type', y = 0.9, x = 0.84, weight = 800)
    ax2.set_title('Color Cut Classified Object Type', y = 0.9, x = 0.86, weight = 800)
    ax2.set_xlabel('G - Z', fontsize = 14)
    ax2.set_ylabel('R - W1', fontsize = 14, y = 1)
    
    # AJF save and plot
    plt.savefig('color_cut_gz_rw1.png', format = 'png')
    plt.show()    
    
 
 
 
def log_and_print(tab_q, tab_s, path, path2, radius, ans, col_nams, perc_num):
    """ Keeps record of all the relevant final results; prints these to the terminal and then logs some messages for later user review. 
    
    Parameters:
    ----------     
    tab_q : :class: astropy.table.table.QTable
        dust-corrected object type 1 (quasars, in original case) cross-matched table, including magnitudes, fluxes, and color-cut classification
    tab_s : :class: astropy.table.table.QTable
        dust-corrected object type 1 (stars, in original case) cross-matched table, including magnitudes and fluxes, and color-cut classification
    path : :class: string
        path to directory where object type's fit files are located
    path2 : :class: string
        path to directory where legacy fit files are located
    radius : :class: float
        radius in arcseconds to use in cross-matching input fits files to legacy data
    ans : :class: string
        input argument indicating user would like to just run the color cut code (y) or also run the cross matching code (n)
    col_nams : :class: list
        list of column names that the user desires to pull from the legacy fits files
    perc_num : :class: list
        a list containing the percentages of object types 1 and 2 cross-matched with legacy data as well as the number of objects matched
    
    Returns:
    ----------
    None - plots the original data (unclassified) alongside the classified data to compare 
    
    
    """
    
    # AJF split up quasar table into color cut classifications for finding percent of incorrect classifications
    q_idx = (tab_q['Color Cut Class'] == 'Likely Quasar') 
    qcs_idx = (tab_q['Color Cut Class'] == 'Likely Star')
    qccd_idx = (tab_q['Color Cut Class'] == 'Cannot Determine') 
    
    # AJF split up stars table into color cut classifications for finding percent of incorrect classifications
    s_idx = (tab_s['Color Cut Class'] == 'Likely Star')
    scq_idx = (tab_s['Color Cut Class'] == 'Likely Quasar')
    sccd_idx = (tab_s['Color Cut Class'] == 'Cannot Determine') 
    
    # AJF use np.sum to determine how many True values are in each boolean array (i.e. how many quasars were correctly identified (q_idx = True))
    q, qcs, qccd = np.sum(q_idx), np.sum(qcs_idx), np.sum(qccd_idx)
    s, scq, sccd = np.sum(s_idx), np.sum(scq_idx), np.sum(sccd_idx)

    # AJF find fractions that were correctly identified vs falsely identified vs could not be determined due to NaN values
    frac_q = (q/(q+qcs+qccd))*100
    frac_qcs = (qcs/(q+qcs+qccd))*100
    frac_qccd = (qccd/(q+qcs+qccd))*100

    frac_s = (s/(s+scq+sccd))*100
    frac_scq = (scq/(s+scq+sccd))*100
    frac_sccd = (sccd/(s+scq+sccd))*100
    
    # AJF print and write to log all relevant information, including issued command, resulting data, cross-matching and correctly identifed objects percentages, etc.
    print(f'\n\nThe percentage of quasars correctly identified as such by using the color cut was {frac_q:.3f}%,\nwhile the percentage of quasars that were misidentified as stars was {frac_qcs:.3f}%.')
    print(f'The percentage that contained NaN values, and thus could not be used in the color cut, was {frac_qccd:.3f}%.')
    print(f'\n\nThe percentage of stars correctly identified as such by using the color cut was {frac_s:.3f}%,\nwhile the percentage of stars that were misidentified as quasars was {frac_scq:.3f}%.')
    print(f'The percentage that contained NaN values, and thus could not be used in the color cut, was {frac_sccd:.3f}%.\n\n')
    log = open('log_color_cut_18.txt', 'a')
    log.write('='*150)
    log.write(f'\n{datetime.today().strftime("%Y-%m-%d %H:%M:%S")} MDT')
    log.write(f'\nCommand-line used:\n$ python color_cut_18.py {path} {path2} {radius} {ans}')
    for c in col_nams:
        log.write(f' {c}')
    
    # AJF write this seciton if user chose to NOT perform new cross-matching process
    if ans in ['Yes', 'yes', 'y', 'Y']:
        log.write('\n\nPerformed dust correction with dust_correct()')
        log.write('\nPerformed conversion from flux to magnitude with flux_to_mag()')
        log.write('\nPerformed color cut for g-z vs. r-w1 indexes with color_cut()')
        log.write('\nPerformed classification of data into quasar group and stars group using color cut line with classify_qs()')
        log.write('\nPerformed classification cross-check and plotted color-indexed data with classified data using check_and_plot()')
        log.write(f'\n\nThe percentage of quasars correctly identified as such by using the color cut was {frac_q:.3f}%,\nwhile the percentage of quasars that were misidentified as stars was {frac_qcs:.3f}%.')
        log.write(f'\nThe percentage that contained NaN values, and thus could not be used in the color cut, was {frac_qccd:.3f}%.')
        log.write(f'\n\nThe percentage of stars correctly identified as such by using the color cut was {frac_s:.3f}%,\nwhile the percentage of stars that were misidentified as quasars was {frac_scq:.3f}%.')
        log.write(f'\nThe percentage that contained NaN values, and thus could not be used in the color cut, was {frac_sccd:.3f}%.')    
        log.write(f'\n\nThe number of quasars successfully matched in the crossmatching procedure was {perc_num[2]}, which makes the percentage matched {perc_num[0]:.3f}%')
        log.write(f'\nThe number of stars successfully matched in the crossmatching procedure was {perc_num[3]}, which makes the percentage matched {perc_num[1]:.3f}%')
        log.write('\n'+'='*150+'\n'*5)
        log.close()
    
    # AJF write this seciton if user chose to perform new cross-matching process
    else:
        log.write('\n\nPerformed reading path and table data.')
        log.write('\nPerformed cross-matching list file list creation and actual cross-matching, resulting in the creation of a table of cross-matched quasars and cross-matched stars.')
        log.write('\nPerformed dust correction with dust_correct()')
        log.write('\nPerformed conversion from flux to magnitude with flux_to_mag()')
        log.write('\nPerformed color cut for g-z vs. r-w1 indexes with color_cut()')
        log.write('\nPerformed classification of data into quasar group and stars group using color cut line with classify_qs()')
        log.write('\nPerformed classification cross-check and plotted color-indexed data with classified data using check_and_plot()')
        log.write(f'\n\nThe percentage of quasars correctly identified as such by using the color cut was {frac_q:.3f}%,\nwhile the percentage of quasars that were misidentified as stars was {frac_qcs:.3f}%.')
        log.write(f'\nThe percentage that contained NaN values, and thus could not be used in the color cut, was {frac_qccd:.3f}%.')
        log.write(f'\n\nThe percentage of stars correctly identified as such by using the color cut was {frac_s:.3f}%,\nwhile the percentage of stars that were misidentified as quasars was {frac_scq:.3f}%.')
        log.write(f'\nThe percentage that contained NaN values, and thus could not be used in the color cut, was {frac_sccd:.3f}%.')    
        log.write(f'\n\nThe number of quasars successfully matched in the crossmatching procedure was {perc_num[2]}, which makes the percentage matched {perc_num[0]:.3f}%')
        log.write(f'\nThe number of stars successfully matched in the crossmatching procedure was {perc_num[3]}, which makes the percentage matched {perc_num[1]:.3f}%')
        log.write('\n'+'='*150+'\n'*5)
        log.close()        
 
 

def main():# AJF executes this section first (highest 'shell' of code)
    # AJF add description
    par = argparse.ArgumentParser(description=f'Read in fits files for two different types of objects (say. quasars and stars) and, if necessary. cross-match them with Legacy survey data to extract their FLUX values. Perform a color cut and classify the objects based on this cut.')
    par.add_argument("path", type = str, help = 'path to directory where stars and quasars are located; will read in this data as astropy table; try /d/scratch/ASTR5160/week10')
    par.add_argument("path2", type = str, help = 'path to file where legacy survey is located; will read in this data as astropy table; try /d/scratch/ASTR5160/data/legacysurvey/dr9/south/sweep/9.0')
    par.add_argument("radius", type = float, help = 'radius in arcseconds to attempt to match coordinates; used in leg_query function')
    par.add_argument("ans", type = str, help = 'y will skip the cross-matching step and read in pre-written cross-matched fits files; n goes through legacy matching process (use y if youve already done the cross-matching and want a quick result for debugging, for ex.)')
    par.add_argument("c", type = str, nargs = '+', help = 'list of column names youd like to use from legacy survey sweep file(s), not including RA or Dec')
    
    # AJF parse arguments and rename them
    arg = par.parse_args()    
    path = arg.path
    path2 = arg.path2
    radius = arg.radius
    col_nams = arg.c
    ans = arg.ans
    
    # AJF ignore astropy warnings, make printing prettier
    warnings.simplefilter('ignore', category=AstropyWarning)    
    
    # AJF create list of all relevant legacy flux columns
    flux_cols = ['FLUX_G', 'FLUX_R', 'FLUX_Z', 'FLUX_W1', 'FLUX_W2', 'FLUX_W3', 'FLUX_W4']
    
    # AJF create list of flux columns that are actually being used in code (given in command line argument c)
    # AJF command line argument c can also intake other columns like BRICKID, etc, so need to match command line argument c to full flux list for use in convert_to_mag
    final_fc = [f for f in col_nams if f in flux_cols]
    
    # AJF create list of flux columns that are actually being used in code AFTER DUST CORRECTION (given in command line argument c) 
    final_fc_after_dust = [f'{f}_DC' for f in final_fc]
    
    if ans in ['Yes', 'yes', 'y', 'Y']:
    
        # AJF read the percentages of objects that were matched
        with open('perc_and_num.txt', 'r') as read:
            line = read.readline()
            perc_num = line.split()
            perc_num = [float(f) for f in perc_num]
        # AJF print percentgaes of objects cross-matched successfully
        print(f'\nThe number of quasars successfully matched in the crossmatching procedure was {perc_num[2]}, which makes the percentage matched {perc_num[0]:.3f}%')
        print(f'\nThe number of stars successfully matched in the crossmatching procedure was {perc_num[3]}, which makes the percentage matched {perc_num[1]:.3f}%')
        
        # read in files
        print(f'\nReading in fits files...\n')
        tab_q = QTable.read('tab_q_before_dust.fits')    
        tab_s = QTable.read('tab_s_before_dust.fits')         

        # AJF correct for dust extinction along lightpaths
        dust_correct(tab_q, tab_s)
        
        # AJF convert the nanomaggy fluxes into magnitudes
        convert_to_mag(final_fc_after_dust, tab_q, tab_s)
        
        # AJF do color cut and plot it
        gz_q, rw1_q, gz_s, rw1_s, lin, avg_m, avg_b, mq, bq, ms, bs = color_cut(tab_q, tab_s)
        
        # AJF classify each quasar and star based on color cut
        tab_q, tab_s = classify_qs(gz_q, rw1_q, gz_s, rw1_s, lin, avg_m, avg_b, tab_q, tab_s)
        
        # AJF check to see how well the color cut correctly classified the input data
        check_and_plot(gz_q, rw1_q, gz_s, rw1_s, lin, avg_m, avg_b, mq, bq, ms, bs, tab_q, tab_s)
        
        # AJF write matched tables to fits file for easy reading
        tab_q.write('tab_q.fits', overwrite=True)
        tab_s.write('tab_s.fits', overwrite=True)
        
        print(f'\nThis is quasar matching table:\n\n{tab_q}')
        print(f'\n\nThis is stars matching table:\n\n{tab_s}')
        
        # AJF log all that was done, including time, command-line, and functions run
        log_and_print(tab_q, tab_s, path, path2, radius, ans, col_nams, perc_num)
        
        # AJF CHAT-GPT code below that checks for duplicates within table; i.e. input survey has two matches in legacy sweep files within search radius
        #duplicate(tab_q, tab_s)
        
        
    else:
        print(f'\nGoing through matching process...\n')
        
        # AJF read the path provided
        ra_s, dec_s, z_s, ra_q, dec_q, z_q = read1(path)
        
        # AJF compile list of legacy files that match the ra and dec of the provided input survey data
        uf_q = lql(ra_q, dec_q, path2, len(ra_q))
        uf_s = lql(ra_s, dec_s, path2, len(ra_s))
    
        # AJF perform the cross-matching using cross_match.py from week8
        print(f'\nPerforming Quasar Cross-Matching...')
        tab_leg_q, tab_q, idq, id_leg_q, perc_list_q, num_list_q = lq(uf_q, path2, ra_q, dec_q, len(ra_q), radius, col_nams) 
        print(f'\nPerforming Star Cross-Matching...')
        tab_leg_s, tab_s, ids, id_leg_s, perc_list_s, num_list_s = lq(uf_s, path2, ra_s, dec_s, len(ra_s), radius, col_nams)       

        # AJF write percentage of objects in input survey that were successfully crossmatched to file for later recall if ans = yes
        perc_q = float(np.mean( (np.array(perc_list_q)) ))
        perc_s = float(np.mean( (np.array(perc_list_s)) ))
        perc = [perc_q, perc_s]
        
        # AJF write number of objects matched for each survey to compare to number of rows in final tables
        num_q = sum(num_list_q)
        num_s = sum(num_list_s)
        perc.append(float(num_q))
        perc.append(float(num_s))
        
        # AJF write the above here
        percf = open('perc_and_num.txt', 'w')
        percf.write(f'{perc_q} {perc_s} {num_q} {num_s}')
        percf.close()

        # AJF print percentgaes of objects cross-matched successfully
        print(f'\nThe number of quasars successfully matched in the crossmatching procedure was {num_q}, which makes the percentage matched {perc_q:.3f}%')
        print(f'\nThe number of stars successfully matched in the crossmatching procedure was {num_s}, which makes the percentage matched {perc_s:.3f}%')

        # AJF write matched tables before dust correction for use in ans=yes if-statement to fits file for easy reading
        # AJF this method (table.write) is an astropy.write function - this is different than open('tab_q', 'w') method
        tab_q.write('tab_q_before_dust.fits', overwrite=True)
        tab_s.write('tab_s_before_dust.fits', overwrite=True)
    
        # AJF correct for dust extinction along lightpaths
        dust_correct(tab_q, tab_s)
        
        # AJF convert the nanomaggy fluxes into magnitudes
        convert_to_mag(final_fc_after_dust, tab_q, tab_s)
        
        # AJF do color cut and plot it
        gz_q, rw1_q, gz_s, rw1_s, lin, avg_m, avg_b, mq, bq, ms, bs = color_cut(tab_q, tab_s)
        
        # AJF classify each quasar and star based on color cut
        tab_q, tab_s = classify_qs(gz_q, rw1_q, gz_s, rw1_s, lin, avg_m, avg_b, tab_q, tab_s)
        
        # AJF check to see how well the color cut correctly classified the input data
        check_and_plot(gz_q, rw1_q, gz_s, rw1_s, lin, avg_m, avg_b, mq, bq, ms, bs, tab_q, tab_s)
        
        # AJF write matched tables to fits file for easy reading
        tab_q.write('tab_q.fits', overwrite=True)
        tab_s.write('tab_s.fits', overwrite=True)
        
        print(f'\nThis is quasar matching table:\n\n{tab_q}')
        print(f'\n\nThis is stars matching table:\n\n{tab_s}')
        
        # AJF log all that was done, including time, command-line, and functions run
        log_and_print(tab_q, tab_s, path, path2, radius, ans, col_nams, perc)
        
        # AJF CHAT-GPT code below that checks for duplicates within table; i.e. input survey has two matches in legacy sweep files within search radius
        #duplicate(tab_q, tab_s)
        
        
    
    

if __name__=='__main__':
    main() 
