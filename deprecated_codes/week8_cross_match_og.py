import argparse
import os
from tqdm import tqdm
import time
from datetime import datetime

from astropy.table import QTable
import astropy.coordinates as coord
from astropy.coordinates import SkyCoord
from astropy import units as u

# AJF use chatgpt to ignore unit warnings
import warnings
from astropy.utils.exceptions import AstropyWarning

import numpy as np
import matplotlib.pyplot as plt

# AJF tried to import modules directly as line below, but did not work because of various errors
# from master_scripts.io import decode_sweep_name as dsn

# created by AJF 3/18/25
# last edited by AJF 3/23/25

# comments: could clean this up at some point but have alreayd put way too much time into making this code work (like 8 hours :(  )


def simple_plot(path):
    """ Plot all FIRST data located at given path; just do a simple plot for visualizing
    
    Parameters:
    ----------
    path : :class: string
        linux directory path to FIRST survey data; try /d/scratch/ASTR5160/data/first/first_08jul16.fits

    Returns:
    ----------
    ra : :class: numpy.ndarray
        all ra values of FIRST survey data pulled from file at path (degrees)
    dec : :class: numpy.ndarray
        all dec values of FIRST survey data pulled from file at path (degrees)
    tab_f : :class: 
    
    
    """
    
    # AJF try /d/scratch/ASTR5160/data/first/first_08jul16.fits, read in ra and dec
    tab_f = QTable.read(path) 
    ra = np.array(tab_f['RA'])
    dec = np.array(tab_f['DEC'])
    
    # AJF simple plot
    plt.plot(ra, dec, 'r.')
    #plt.show()
    plt.close()
    
    return ra, dec, tab_f
        



    
def sdss_dr9_query(ra, dec, n):
    """ Query the SDSS online database for cross-matching objects from FIRST survey
    
    Parameters:
    ----------
    ra : :class: numpy.ndarray
        all ra values of FIRST survey data pulled from file at path (degrees)
    dec : :class: numpy.ndarray
        all dec values of FIRST survey data pulled from file at path (degrees)
    n : class: int
        number of rows that are pulled from FIRST survey data
        
    Returns:
    ----------
    None - writes query results to query_results_sdss.txt
        
    """
    
    print(f'\nSDSS query progress below:\n')
    
    with open('query_results_sdss.txt', 'a') as app:
        app.write("\nCommand used in script:\nos.system(f'python ../master_scripts/sdssDR9query.py {r} {d} >> query_results.txt')\n\nResults:\n")  
    
        
    # AJF find out how much time for loop takes
    start = time.time()
    
    # AJF lot to unpack here: tqdm is progress bar, and total = len(ra) helps tqdm determine how many times for loop runs so progress bar is displayed correctly
    # AJF zip function pairs elements of arrays that have same index together into tuples
    # AJF os.system allows command-line interaction 
    # AJF >> ''.txt does same as append mode in .write/.open functions    
    for r, d in tqdm(zip(ra, dec), total = len(ra)):        
        os.system(f'python ../master_scripts/sdssDR9query.py {r} {d} >> query_results_sdss.txt')

    # AJF end stopwatch and print time
    end = time.time()
    tt = end - start
    print(f'Total Time SDSS Online Query Took for {n} queries: {tt:.3f} s\n')

    # AJF make sure next time code is ran, write is on a new line
    with open('query_results_sdss.txt', 'a') as app:
        app.write("\n\n")



def leg_query_list(ra, dec, path, n):
    """ Query the local legacy survey database and build a list of files that have FIRST objects in them, based
    on ra and dec values
    
    Parameters:
    ----------
    ra : :class: numpy.ndarray
        n ra values of FIRST survey data pulled from file at path (degrees)
    dec : :class: numpy.ndarray
        n dec values of FIRST survey data pulled from file at path (degrees)
    path: :class: string
        path2 in arguments of main; path to legacy survey data; try /d/scratch/ASTR5160/data/legacysurvey/dr9/north/sweep/9.0
    n : class: int
        number of rows that are pulled from FIRST survey data
        
    Returns:
    ----------
    unique_files : :class: numpy.ndarray
        an array of all the unique Legacy survey filenames containing cross-matched objects from FIRST
    
    """

    # AJF 'borrow' ADM decode_sweep_name code... tried to import directly but couldn't :(
    # AJF create a list of all sweep files to pull ra/dec boxes from
    filelist = [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f)) and f.startswith('sweep')]
    
    # AJF pull all ra min and max values for each sweep file
    ramin = [float(f[6:9]) for f in filelist]
    ramax = [float(f[14:17]) for f in filelist]
    
    decmin = [float(f[10:13]) if f[9] == 'p' else -1*float(f[10:13]) for f in filelist]
    decmax = [float(f[18:21]) if f[17] == 'p' else -1*float(f[18:21]) for f in filelist]
    
    # AJF zip all ra/dec pairs into list of tuples (i.e. for sweep-170p080-180p085.fits, list index value would be (170, 180, 80, 85) )
    radec = list(zip(ramin, ramax, decmin, decmax))
    
    # AJF is_in_box implies using for loop (or some iterable) to run through full recarray (first 100 objects in fits file, for example) each loop
    # ... and check if each ra is in the indexed radec box - i.e., each iteration of loop has certain tuple of radec box, and each iteration runs through
    # ... full recarray to check that iteration's radec against all objects
    # AJF would it be faster to do the 'opposite'? iterate the for loop over the recarray, so first ra/dec pair of recarray is checked agianst all
    # ... possible radec tuples in first iteraiton of loop; second iter of loop uses second indices of recarray and checks against all radec tuples, etc.?
    
    # AJF make empty 1d array for filling with name of legacy fits file that each object would be in; indexes of this array correspond to index of object in recarray
    full_files = np.empty( (n), dtype = object )
       
    # AJF iterate through all radec tuples; each loop, check all ra/dec values for all objects against given radec tuple; if object fits in this tuple's radec range...
    # AJF then write true to ii; then, find which legacy file this radec tuple represents, and write that file's name to the object's index in full_files    
    for ind, f in enumerate(radec):
        ii = ( (ra >= f[0]) & (ra < f[1]) & (dec >= f[2]) & (dec < f[3]) )        
        filestring = filelist[ind]       
        full_files[ii] = filestring    
   
    # AJF find unique filenames (should have 11 legacy fits files)
    unique_files = np.unique(full_files)

    return unique_files



   

def leg_query(uf, path, ra_f, dec_f, n, radius, col_nams):
    """ Using the filelist generated in leg_query_list, actually find the cross-matched objects in legacy
    and write both the FIRST objects and their legacy cross-matches to a fits file (table)
    
    Parameters:
    ----------
    uf : :class: numpy.ndarray
        an array of all the unique Legacy survey filenames containing cross-matched objects from FIRST
    path: :class: string
        path2 in arguments of main; path to legacy survey data; try /d/scratch/ASTR5160/data/legacysurvey/dr9/north/sweep/9.0        
    ra_f : :class: 
        all ra values of FIRST survey data pulled from file at path (degrees)
    dec_f : :class: 
        all dec values of FIRST survey data pulled from file at path (degrees)
    n : class: int
        number of rows that are pulled from FIRST survey data
    radius : :class: float
        radius in arcseconds to attempt to match coordinates to each other in
    col_nams : :class: list
        
    Returns:
    ----------
    tab_leg : :class: astropy.table.table.QTable
        legacy survey fits sweep file read into QTable
    tab_cross : :class: astropy.table.table.QTable
        table read from cross_table.fits file; is the table created from cross-matching first n FIRST survey results with legacy survey
    idf : :class: numpy.ndarray
        Boolean array indicating the indices of user-inputted survey object's ra and dec rows that cross-match user legacy survey  
    id_leg : :class: numpy.ndarray
        Boolean array indicating the indices of legacy survey rows that cross-match user inputted survey object's ra and dec    
    
       
    """
    # AJF initialize full arrays of size n; fill with NaN so no type errors will show up (since a value of 'None' is not compatible with actual float)
    ra_leg_col, dec_leg_col = np.full( (n),np.nan, dtype = float ), np.full( (n), np.nan, dtype = float )

    # AJF object number for input survey (column 1)
    num = np.arange(1, n+1, 1)    

    # AJF create QTable with input survey ra and dec and legacy ra and dec (empty)
    final_table = QTable([num, ra_f*u.deg, dec_f*u.deg, np.full( (n), np.nan, dtype = float), np.full( (n), np.nan, dtype = float)],
    names = ['Input Survey Data Row Number', 'Input Survey Data RA', 'Input Survey Data Dec','Legacy RA', 'Legacy Dec'])   
    
    # append colnames to table as empty arrays
    for c in col_nams:
        final_table[c] = np.full( (n), np.nan )
    
    # AJF run through unique file list; this way, not loading in each legacy file for all n rows of FIRST survey; finds ra and dec of all...
    # ... objects cross-matched in one unique legacy file and adds these to index of NaN arrays (above) that match index of object in FIRST survey
    print(f'\n\nLocal Legacy cross-matching progress below:\n')
    
    # AJF find out how much time for loop takes
    start = time.time()  
    
    for idx, f in enumerate(tqdm(uf)):
        tab_leg = QTable.read(path + '/' + f)
        ra_leg = tab_leg['RA']
        dec_leg = tab_leg['DEC']
        c_first = SkyCoord(ra = ra_f*u.degree, dec = dec_f*u.degree, frame = 'icrs')
        c_leg = SkyCoord(ra = ra_leg, dec = dec_leg, frame = 'icrs')
        
        # AJF id_leg is index of legacy survey to find cross-matched object, cross-matches to FIRST object indexed by idf
        id_leg, idf, d2, d3 = c_first.search_around_sky(c_leg, radius*u.arcsec)

        # AJF index ra and dec for legacy survey
        ra_leg_col[idf] = ra_leg[id_leg]
        dec_leg_col[idf] = dec_leg[id_leg]
        
        # AJF add Legacy RA and Dec automatically        
        final_table['Legacy RA'] = ra_leg_col
        final_table['Legacy RA'].unit = ra_leg.unit
        final_table['Legacy Dec'] = dec_leg_col
        final_table['Legacy Dec'].unit = dec_leg.unit
                
        for nam in col_nams:
            # AJF fix the dtype of QTable column and, if string, fill with empty string values
            if tab_leg[nam].dtype.kind in ['S', 'U']:
                if final_table[nam].dtype.kind not in ['S', 'U']:
                    final_table[nam] = np.full( (n), '', dtype = 'U20')
                final_table[nam][idf] = tab_leg[nam][id_leg] 
                    
                    
            elif final_table[nam].dtype != 'f':
                final_table[nam] = final_table[nam].astype(float)  
                final_table[nam][idf] = tab_leg[nam][id_leg] 

                    
       ############### need to define custom unit             
            
   
    # AJF end stopwatch and print time
    end = time.time()
    tt = end - start
    print(f'Total Time Legacy Local Query Took for {n} queries: {tt:.3f} s\n')
    
    # AJF write the table to fits file
    final_table.write('final_table.fits', overwrite=True)

    # AJF read the fits file into table to ensure it works
    read_ft = QTable.read('final_table.fits')
    
    print(f'this is final table:\n\n{read_ft}')
    
    return tab_leg, final_table, idf, id_leg




   
    
def main():# AJF executes this section first (highest 'shell' of code)
    # AJF add description
    par = argparse.ArgumentParser(description='')
    par.add_argument("path1", type = str, help = 'path to file where data is located; will read in this data as astropy table; try /d/scratch/ASTR5160/data/first/first_08jul16.fits')
    par.add_argument("path2", type = str, help = 'path to file where legacy survey is located; will read in this data as astropy table; try /d/scratch/ASTR5160/data/legacysurvey/dr9/north/sweep/9.0')
    par.add_argument("n", type = int, help = 'number of initial rows you would like to use from data located at FIRST path; i.e., giving 100 will use first 100 rows of data')
    par.add_argument("radius", type = float, help = 'radius in arcseconds to attempt to match coordinates; used in search_around_sky')
    par.add_argument("-c", type = str, nargs = '+', help = 'list of column names youd like to use from legacy survey sweep file(s)', required = True)
    arg = par.parse_args()
    path1 = arg.path1
    path2 = arg.path2
    n = arg.n
    radius = arg.radius
    col_nams = arg.c
    
    # AJF ignore astropy warnings, make printing prettier
    warnings.simplefilter('ignore', category=AstropyWarning)
    
    # AJF run simple code to plot full fits file ra/dec
    ra, dec, tab_f = simple_plot(path1)
    
    # AJF only use first n rows from FIRST fits file from now on:
    ra_f, dec_f = ra[:n], dec[:n]
    
    # AJF write little intro for each time code is run with info about command used and date/time ran 
    with open('query_results_sdss.txt', 'a') as app:
        app.write('------------------------------')
        app.write(f'\n{datetime.today().strftime("%Y-%m-%d %H:%M:%S")} MDT')
        app.write(f'\nCommand-line: $ python cross_match.py {path1} {n}\n')
    
    # AJF run sdss query function - cross match FIRST survey with sdss 
    #sdss_dr9_query(ra_f, dec_f, n)

    # AJF build local legacy survey query function list of fits files; use only first n rows of recarray
    uf = leg_query_list(ra_f, dec_f, path2, n)

    # AJF run local query and write FIRST survey data with crossmatched legacy data into fits file
    tab_leg, tab_cross, idf, id_leg = leg_query(uf, path2, ra_f, dec_f, n, radius, col_nams)
    
    # AJF print!
    # print(f'\nThis is my legacy cross-matched table:\n\n {tab_cross}\n')    





if __name__=='__main__':
    main() 
