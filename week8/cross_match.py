import argparse
import os
from tqdm import tqdm
import time
from datetime import datetime

from astropy.table import QTable, hstack, vstack
import astropy.coordinates as coord
from astropy.coordinates import SkyCoord
from astropy import units as u
from astropy.units import UnitConversionError

# AJF use chatgpt to ignore unit warnings
import warnings
from astropy.utils.exceptions import AstropyWarning

import numpy as np
import matplotlib.pyplot as plt

# AJF tried to import modules directly as line below, but did not work because of various errors
# from master_scripts.io import decode_sweep_name as dsn

# created by AJF 3/18/25
# last edited by AJF 4/1/25

# comments: could clean this up at some point but have alreayd put way too much time into making this code work (like 40 hours :(  )


def simple_plot(path):
    """ Plot all input survey data located at given path; just do a simple plot for visualizing
    
    Parameters:
    ----------
    path : :class: string
        linux directory path to input survey data; try /d/scratch/ASTR5160/data/first/first_08jul16.fits

    Returns:
    ----------
    ra : :class: numpy.ndarray
        all ra values of input survey data pulled from file at path (degrees)
    dec : :class: numpy.ndarray
        all dec values of input survey data pulled from file at path (degrees)    
    
    """
    
    # AJF try /d/scratch/ASTR5160/data/first/first_08jul16.fits for FIRST data if desired, read in ra and dec
    tab_f = QTable.read(path) 
    ra = np.array(tab_f['RA'])
    dec = np.array(tab_f['DEC'])
    
    # AJF simple plot
    plt.plot(ra, dec, 'r.')
    #plt.show()
    plt.close()
    
    return ra, dec
        



    
def sdss_dr9_query(ra, dec, n, path):
    """ Query the SDSS online database for cross-matching objects from input survey
    
    Parameters:
    ----------
    ra : :class: numpy.ndarray
        all ra values of input survey data pulled from file at path (degrees)
    dec : :class: numpy.ndarray
        all dec values of input survey data pulled from file at path (degrees)
    n : class: int
        number of rows that are pulled from input survey data
    path : :class: string
        linux directory path to input survey data; try /d/scratch/ASTR5160/data/first/first_08jul16.fits for FIRST
    
    Returns:
    ----------
    None - writes query results to query_results_sdss.txt
        
    """
    
    # AJF write little intro for each time code is run with info about command used and date/time ran 
    app = open('query_results_sdss.txt', 'a')
    app.write('------------------------------')
    app.write(f'\n{datetime.today().strftime("%Y-%m-%d %H:%M:%S")} MDT')
    app.write(f'\nCommand-line: $ python cross_match.py {path} {n}\n')    
    print(f'\nSDSS query progress below:\n')
    app.write("\nCommand used in script:\nos.system(f'python ../master_scripts/sdssDR9query.py {r} {d} >> query_results.txt')\n\nResults:\n")
    app.close()  
    
    
    # AJF lot to unpack here: tqdm is progress bar, and total = len(ra) helps tqdm determine how many times for loop runs so progress bar is displayed correctly
    # AJF zip function pairs elements of arrays that have same index together into tuples
    # AJF os.system allows command-line interaction 
    # AJF >> ''.txt does same as append mode in .write/.open functions   
    
    # AJF find out how much time for loop takes
    start = time.time()
 
    for r, d in tqdm(zip(ra, dec), total = len(ra)):        
        os.system(f'python ../master_scripts/sdssDR9query.py {r} {d} >> query_results_sdss.txt')

    # AJF end stopwatch and print time
    end = time.time()
    tt = end - start
    print(f'Total Time SDSS Online Query Took for {n} queries: {tt:.3f} s\n')

    # AJF make sure next time code is ran, write is on a new line, then close
    app = open('query_results_sdss.txt', 'a')
    app.write('\n\n')
    app.close()





def leg_query_list(ra, dec, path, n):
    """ Query the local legacy survey database and build a list of files that have input survey objects in them, based
    on ra and dec values
    
    Parameters:
    ----------
    ra : :class: numpy.ndarray
        n ra values of input survey data pulled from file at path (degrees)
    dec : :class: numpy.ndarray
        n dec values of input survey data pulled from file at path (degrees)
    path: :class: string
        path2 in arguments of main; path to legacy survey data; try /d/scratch/ASTR5160/data/legacysurvey/dr9/north/sweep/9.0
    n : class: int
        number of rows that are pulled from input survey data
        
    Returns:
    ----------
    unique_files : :class: numpy.ndarray
        an array of all the unique Legacy survey filenames containing cross-matched objects from input survey
    
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
   
    # AJF find unique filenames 
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
        all ra values of input survey data pulled from file at path (degrees)
    dec_f : :class: 
        all dec values of input survey data pulled from file at path (degrees)
    n : class: int
        number of rows that are pulled from input survey data
    radius : :class: float
        radius in arcseconds to attempt to match coordinates to each other in
    col_nams : :class: list
        list of all desired column names to include in final table / fits file
        
    Returns:
    ----------
    tab_leg : :class: astropy.table.table.QTable
        legacy survey fits sweep file read into QTable
    final_table : :class: astropy.table.table.QTable
        table read from cross_table.fits file; is the table created from cross-matching first n input survey results with legacy survey
    idf : :class: numpy.ndarray
        array indicating the indices of user-inputted survey object's ra and dec rows that cross-match user legacy survey  
    id_leg : :class: numpy.ndarray
        array indicating the indices of legacy survey rows that cross-match user inputted survey object's ra and dec 
           
    """
    
    
    # AJF run through unique file list; this way, not loading in each legacy file for all n rows of input survey; finds ra and dec of all...
    # ... objects cross-matched in one unique legacy file and adds this cross-matched table to a list, of which each sub-table is joined together after loop
    print(f'\n\nLocal Legacy cross-matching progress below:\n')
    
    # AJF initialize input data table
    input_table = QTable([np.arange(1, n+1, 1), ra_f * u.deg, dec_f * u.deg],
                         names=['Input Survey Row Number', 'Input RA', 'Input Dec'])

    # AJF create empty list to append each cross-matched table to for storage and empty list to append percent of input objects that were matched
    table_list = []
    perc_list = []
    num_list = []

    # AJF define all unique units that will be used
    nanomag = u.def_unit('nanomaggy')

    # AJF add row number, ras and decs to col_nams
    other_nams = ['Input Survey Row Number', 'Input RA', 'Input Dec', 'RA', 'DEC']
    col_nams = other_nams + col_nams

    # AJF create list of flux columns to strip units from
    flux_cols = ['FLUX_G', 'FLUX_R', 'FLUX_Z', 'FLUX_W1', 'FLUX_W2', 'FLUX_W3', 'FLUX_W4']

    # AJF initialize input survey coordinates before for loop 
    c_input = SkyCoord(ra=ra_f*u.deg, dec=dec_f*u.deg, frame='icrs') 

    # AJF go through each unique legacy file only once
    # AJF find out how much time for loop takes
    start = time.time() 
    for f in tqdm(uf):
        
        # AJF read in legacy sweep file, read in ra and dec
        tab_leg = QTable.read(path + '/' + f)
        ra_leg, dec_leg = tab_leg['RA'], tab_leg['DEC']

        # AJF match coordinates in arg.radius circumference
        c_leg = SkyCoord(ra=ra_leg, dec=dec_leg, frame='icrs')
        id_leg, idf, extra1, extra2= c_input.search_around_sky(c_leg, radius*u.arcsec)
   
        # AJF print notice that this file contained no matches if no matching indices 
        if len(id_leg) == 0 or len(idf) == 0:
            tqdm.write(f'No cross-matches in file {f}')
            continue

        # AJF index each table to matching values
        input_idx = input_table[idf] 
        leg_idx = tab_leg[id_leg]   
        
        # AJF find what number percent of objects in input survey had a match
        num_input = len(idf)
        perc = float((num_input/len(ra_leg))*100)
        perc_list.append(perc)
        num_list.append(int(num_input))

        # AJF combine row-indexed input table and row-indexed legacy table
        comb_table = hstack([input_idx, leg_idx], join_type='outer')
        
        # AJF only keep columns that are in list of names col_nams (includes by default input survey row number, ra, dec, and legacy ra/dec, and then any command line -c names)
        comb_table = comb_table[col_nams]
        
        # AJF strips units from flux columns (nanomaggy columns) if these are included in command line argument -c
        for fc in flux_cols:
            if fc in col_nams:
                comb_table[fc] = comb_table[fc].value

        # AJF store the cross-matched table in a list
        # AJF this list stores all cross-matched tables since each for loop iteraiton loads and cross-matches a different legacy sweep file with all rows of input table
        # AJF would it be easier to use something like addition assignment type scheme += or maybe vstack each table each iteration?
        # AJF probably not, as this would create a second new table every iteration
        table_list.append(comb_table)

    # AJF combine all tables in list together for the final table; checks to make sure there was at least one cross-matched ra and dec
    if table_list:
        final_table = vstack(table_list, join_type='outer')
        final_table.sort('Input Survey Row Number') # AJF sort the final table by input row number (descending)
    else:
        print(f'\nNo cross-matches in any legacy sweeps.\n')  
        
    # AJF rename the plain ra and dec columns to show they are legacy ra and legacy dec
    final_table.rename_column('RA', 'Legacy RA')
    final_table.rename_column('DEC', 'Legacy Dec')
    
    # AJF add custom nanomaggy units back onto fluxes from legacy survey, if flux arguments are pulled from command line -c
    for fc in flux_cols:
        if fc in final_table.colnames:
            final_table[fc].unit = nanomag
   
    # AJF end stopwatch and print time needed to completely create final table
    end = time.time()
    tt = end - start
    print(f'Total Time Legacy Local Query and Building Final Table Took for {n} matches: {tt:.3f} s\n')
    

    
    return tab_leg, final_table, idf, id_leg, perc_list, num_list



   
    
def main():# AJF executes this section first (highest 'shell' of code)
    # AJF add description
    par = argparse.ArgumentParser(description='Cross-Match Input Survey Data Files (in fits file format) with Legacy Survey Data (in fits file format)')
    par.add_argument("path1", type = str, help = 'path to file where data is located; will read in this data as astropy table; try /d/scratch/ASTR5160/data/first/first_08jul16.fits')
    par.add_argument("path2", type = str, help = 'path to file where legacy survey is located; will read in this data as astropy table; try /d/scratch/ASTR5160/data/legacysurvey/dr9/north/sweep/9.0')
    par.add_argument("n", type = int, help = 'number of initial rows you would like to use from data located at FIRST path; i.e., giving 100 will use first 100 rows of data')
    par.add_argument("radius", type = float, help = 'radius in arcseconds to attempt to match coordinates; used in search_around_sky')
    par.add_argument("c", type = str, nargs = '+', help = 'list of column names youd like to use from legacy survey sweep file(s) input like: FLUX_G FLUX_R for example')
    arg = par.parse_args()
    path1 = arg.path1
    path2 = arg.path2
    n = arg.n
    radius = arg.radius
    col_nams = arg.c
    
    # AJF ignore astropy warnings, make printing prettier (lots of units in legacy files are not defined astropy units which makes printing to terminal ugly)
    warnings.simplefilter('ignore', category=AstropyWarning)
    
    # AJF run simple code to plot full fits file ra/dec
    ra, dec = simple_plot(path1)
    
    # AJF only use first n rows from input fits file from now on:
    ra_f, dec_f = ra[:n], dec[:n]
    
    # AJF run sdss query function - cross match input survey with sdss 
    sdss_dr9_query(ra_f, dec_f, n, path1)

    # AJF build local legacy survey query function list of fits files; use only first n rows of recarray
    uf = leg_query_list(ra_f, dec_f, path2, n)

    # AJF run local query to build final table containing all cross-matched data
    tab_leg, final_table, idf, id_leg, perc_list, num_list = leg_query(uf, path2, ra_f, dec_f, n, radius, col_nams)

    # AJF write the table to fits file
    final_table.write('final_table.fits', overwrite=True)

    # AJF read the fits file into table to ensure it works
    read_ft = QTable.read('final_table.fits')
    
    # AJF print!
    print(f'\nThis is my legacy cross-matched table:\n\n {read_ft}\n')    





if __name__=='__main__':
    main() 
