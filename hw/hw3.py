import argparse
import time
from tqdm import tqdm
import os

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator

from astropy.table import QTable, hstack, vstack
from astropy.coordinates import SkyCoord
from astropy import units as u

from week8.cross_match import leg_query_list as lql
from master_scripts.sdssDR9query import sdssQuery

# AJF use chatgpt to ignore unit warnings
import warnings
from astropy.utils.exceptions import AstropyWarning

from halo import Halo


# AJF import a chat-gpt co-written code that auto-writes docstrings with variables included
from master_scripts.docstring_wrapper import log_sphinx_io as ds
# AJF note: @docstring is a wrapper that auto-writes docstrings for the function directly below it
# AJF see master_scripts/docstring_wrapper for more details


# AJF created 5/14/25
# AJF last modified 5/19/25
# example command to run:
# python hw3.py /d/scratch/ASTR5160/data/first/first_08jul16.fits /d/scratch/ASTR5160/data/legacysurvey/dr9/north/sweep/9.0 1 163 50 3 y

#@ds
def restrict_input(coord, input_tbl):
    """
    Filters an input astropy table to only keep coordinates within a specified separation from a specified ra dec coordinate
    
    Parameters
    ----------
    coord : :class: list
        a list formatted like [ra dec separation_maximum]
    input_tbl : :class: astropy.table.table.QTable
        the input astropy table to filter based on the input coordinates
        
    Returns
    ----------
    :class: astropy.table.table.QTable
        the filtered astropy table
    :class: astropy.table.table.QTable
        the original, unfiltered astropy table
        
    """
    
    # AJF define skycoord to search around from user-defined argument
    center = SkyCoord(ra = coord[0]*u.deg, dec = coord[1]*u.deg, frame = 'icrs')
     
    # AJF load in input ra and dec to match; make into skycoord
    rai = np.array(input_tbl['RA'])
    deci = np.array(input_tbl['DEC'])
    coordi = SkyCoord(ra = rai*u.deg, dec = deci*u.deg, frame = 'icrs')
    
    # AJF find separation between center coordinate and all coordinates in input survey...
    # AJF... then create a boolean mask to filter input survey ra and dec that are within user-defined region of center coordinate (coord[2])
    seps = coordi.separation(center)   
    boundary = coord[2]*u.deg
    idx = seps < boundary
    
    # AJF filter the input survey ra and dec   
    raif, decif = rai[idx], deci[idx]
    
    # AJF filter the original table
    tab = input_tbl[idx]

    # AJF define row number of original input survey to keep track for future reference if needed; first index of np.where tuple is indices where True
    rows = np.where(idx)[0]
    
    # AJF create new table containing all input survey sources that are within radius of center coordinate
    input_final_tbl = QTable([rows, raif*u.deg, decif*u.deg], names=['Input Survey Original Row Number', 'Input RA', 'Input Dec'])    

    #print(input_final_tbl)

    return input_final_tbl, tab
    





#@ds
def leg_query(uf, path, input_final_tbl, radius, flux_cols):
    """ Using the filelist generated in leg_query_list and a user-specified table (input_final_tbl), 
    finds the cross-matched objects in the path-derived file (usually legacy sweeps)
    and outputs both the input objects and their legacy cross-matches as astropy tables
    
    Parameters
    ----------
    uf : :class: numpy.ndarray
        an array of filename strings - found from leg_query_list function - essentially a list of fits files that could contain objects that match the input table
    path : :class: str
        the path to which the uf file is found - this is a command-line argument
    input_final_tbl : :class: astropy.table.table.QTable
        the input survey table that the user would like to cross-match; usually a quasar table or a first survey fits file
    radius : :class: float
        the radius of cross-matching in arcseconds - usually set at 1 arcsecond - also a command-line argument
    flux_cols : :class: list
        a list of all relevant fluxes that need their units stripped; these are the only non-coordinate/ID columns that are kept in the output tables
        
    Returns
    ----------
    :class: astropy.table.table.QTable
        the cross-matched input table - i.e., if input was a table of quasars located in an ra/dec bin, the resulting table is that input quasar table, but 
        only containing objects that can also be found in the other table
    :class: astropy.table.table.QTable
        the cross-matched path-derived survey table - i.e. same as above, but not the input table
              
    """
    
    
    # AJF run through unique file list; this way, not loading in each legacy file for all n rows of input survey; finds ra and dec of all...
    # ... objects cross-matched in one unique legacy file and adds this cross-matched table to a list, of which each sub-table is joined together after loop
    print(f'\n\nLocal Legacy cross-matching progress below:\n')
    
    # AJF create empty list to append each cross-matched table to for storage and empty list to append percent of input objects that were matched
    table_list = []
    perc_list = []
    num_list = [] 
    
    # AJF create another list that is all neccessary columns for hw; omit any other columns to avoid vstack error of combining tables with 'UnrecognizedUnit' (like 1/deg^2)
    all_cols = ['RA', 'DEC'] + flux_cols

    # AJF pull ra and dec data from that table
    rai, deci = input_final_tbl['Input RA'], input_final_tbl['Input Dec']

    # AJF initialize input survey coordinates before for loop 
    c_input = SkyCoord(ra=rai, dec=deci, frame='icrs') 

    # AJF go through each unique legacy file only once 
    # AJF create empty list to append all tables to 
    all_tabs = []
    
    print(f'Reading in Legacy fits tables from sweep files...\n')
    
    # AJF run through all legacy files and load in all relevant columns from these tables all at omce
    for f in tqdm(uf):
        
        # AJF read in legacy sweep file
        tab = QTable.read(path + '/' + f)
        
        # AJF keep only the neccessary columns (otherwise, will get UnrecognizedUnit error)
        tab = tab[all_cols]
        
        # AJF strips units from flux columns (nanomaggy columns) to prevent nanomaggy unit error (UnrecognizedUnit)
        for fc in flux_cols:
            tab[fc] = tab[fc].value

        # AJF append table to list of tables
        all_tabs.append(tab)
    
    # AJF start spinner
    print('\n')
    spinner = Halo(text='Matching to input survey data with search_around_sky...', spinner = 'pong', color = 'green')    
    spinner.start()
    
    # AJF combine all tables together
    tab_leg = vstack(all_tabs)
    
    # AJF read in ra and dec from legacy for cross-match
    ra_leg, dec_leg = tab_leg['RA'], tab_leg['DEC']

    # AJF match coordinates in arg.radius circumference ONLY ONCE to ALL loaded-in legacy tables
    c_leg = SkyCoord(ra=ra_leg, dec=dec_leg, frame='icrs')
    id_leg, idf, extra1, extra2= c_input.search_around_sky(c_leg, radius*u.arcsec)

    # AJF index each table to matching values
    input_tbl = input_final_tbl[idf] 
    leg_tbl = tab_leg[id_leg]   

    # AJF stop spinner
    spinner.stop()

    # AJF end function here since cross-matching done; now, need to combine tables, but in case user needs to account for weird units (like nanomaggies) or convert thing...
    # ... AJF then end cross-match here and make separate hstack function
    
    return input_tbl, leg_tbl
        








#@ds
def neg_flux(table, band):
    """
    Changes all negative flux values in a certain band-column in a table to NaN values
    
    Parameters
    ----------
    table : :class: astropy.table.table.QTable
        an astropy table where any negative fluxes (which represent no image taken in that band) may cause issues and need to be changed to NaN
    band : :class: str
        the band where negative fluxes may occur
        
    Returns
    ----------
    :class: astropy.table.table.QTable
        the masked table which has no negative fluxes - all those replaced by NaN

    """
    # AJF create mask based on if f is positive or negative to eliminate runtime errors from negative numbers in log (could be done easily with if statement but trying to avoide those :) )
    col = table[band].value
    neg = (col<=0)
    col[neg] = np.nan
    table[band] = col
    
    return table
    






#@ds
def band_mask(table, diff, band1, band2):
    """
    Does a 'greater-than' color cut on the provided table based on the provided bands 
    
    Parameters
    ----------
    table : :class: astropy.table.table.QTable
        input table on which to perform color cut
    diff : :class: float
        the value to compare the difference in band magnitudes (color index) to 
    band1 : :class: str
        string name of first band
    band2 : :class: str
        string name of second band to subtract FROM first band
        
    Returns
    ----------
    :class: astropy.table.table.QTable
        the color-cut input table
    :class: numpy.ndarray
        the color-cut mask (can be applied to another table of same length)


    """   
    
    # AJF make all negative fluxes for each band in table = NaN value
    table = neg_flux(table, band1) 
    table = neg_flux(table, band2)
    
    # AJF convert nanomaggies to magnitudes for band1 and band2   
    mag1 = convert_to_mag(table[band1])
    mag2 = convert_to_mag(table[band2])
    
    # AJF find color difference
    color = np.array(mag1) - np.array(mag2)
    
    # AJF mask based on WISE colors
    iib = (color > float(diff))
    
    # AJF index final tables with band color criteria
    table = table[iib]

    return table, iib
    





#@ds
def mag_mask(table, mag, mag_col):
    """
    Filters a table's magnitudes of specified band to be less than (brighter) than specified magnitude number
    
    Parameters
    ----------
    table : :class: astropy.table.table.QTable
        input table that needs to be filtered based on magnitude in a certain band
    mag : :class: int
        the lowest magnitude that objects should have in the filered table
    mag_col : :class: str
        the name of the band to restrict
        
    Returns
    ----------
    :class: astropy.table.table.QTable
        the magnitude-filtered input table
    :class: numpy.ndarray
        the magnitude mask (can be applied to another table of same length)
    
    
    """
    
    # AJF calculate user-input band magnitudes
    magr = convert_to_mag(table[mag_col])
    
    # AJF mask based on magnitudes
    iir = (np.array(magr) < int(mag))
    
    # AJF index the final tables based on this mask too
    table = table[iir]
    
    return table, iir
           





#@ds
def clean_table(table, flux_cols, nmg):
    """
    Cleans up an input table's appearance and organization

    Parameters
    ----------
    table : :class: astropy.table.table.QTable
        the input (usually cross-matched and combined) table that needs to be made prettier 
    flux_cols : :class: list
        list of flux columns in the input table that should be assigned units
    nmg : :class: astropy.units.core.IrreducibleUnit
        the custom nanomaggy unit
        
    Returns
    ----------
    :class: astropy.table.table.QTable
        the input table, but made prettier / cleaned up
        
    """
    
    # AJF rename the plain ra and dec columns to show they are legacy ra and legacy dec
    table.rename_column('RA', 'Legacy RA')
    table.rename_column('DEC', 'Legacy Dec')
    
    # AJF add custom nanomaggy units back onto fluxes from legacy survey
    for fc in flux_cols:
        table[fc].unit = nmg
    
    # AJF sort by RA
    table.sort('Input RA')
    
    return table
    





#@ds
def duplicate_check(table):
    """
    Finds if and where duplicates in a (cross-matched) table may occur
    
    Parameters
    ----------
    table : :class: astropy.table.table.QTable
        input table that might contain duplicates (especially after a cross-matching process)
        
    Returns
    ----------
    :class: numpy.int64
        a number representing the total number of duplicate rows in the input table
    :class: list
        a list of the first survey ID numbers where the duplicate occurs in 
    :class: astropy.units.quantity.Quantity
        the RA where the duplicates occur

    """
    
    # AJF first find dupliactes of Input Survey:
    # AJF find unique input survey row values and how many of each there are
    vals, counts = np.unique(table['Input Survey Original Row Number'], return_counts = True)
    
    # AJF find the duplicate values (if count>1, there is duplicates)
    dupi = list(vals[counts>1])
    
    # AJF find out how many duplicate rows there are for input survey
    dup_sum_input = np.sum(counts-1)
    
    # AJF use isin function to create Boolean mask where dup = value of input survey row number 
    dup_mask = np.isin(table['Input Survey Original Row Number'], dupi)
    
    # AJF now find the rows that are duplicated
    dup_row_input = table[dup_mask]
    
    # AJF now find duplicates in legacy survey data with the same process, but use RA instead
    # AJF find unique input survey row values and how many of each there are
    vals, counts = np.unique(table['Legacy RA'], return_counts = True)
    
    # AJF find the duplicate values (if count>1, there is duplicates)
    dupl = (vals[counts>1])

    # AJF find out how many duplicate rows there are for legacy survey
    dup_sum_leg = np.sum(counts-1)
    
    # AJF use isin function to create Boolean mask where dup = value of input survey row number 
    dup_mask = np.isin(table['Legacy RA'], dupl)
    
    # AJF now find the rows that are duplicated
    dup_row_leg = table[dup_mask]    
    
    # AJF print all duplcate rows
    #print(f'\nDuplicate Input Survey Objects:\n\n{dup_row_input}\n\nDuplicate Legacy Survey Objects:\n\n{dup_row_leg}\n\n')
    
    # AJF find out total number of duplicate rows
    total_dup = dup_sum_input + dup_sum_leg
    
    return total_dup, dupi, dupl
    







#@ds
def convert_to_mag(nmgy_col):
    """
    Converts the input column of nanomaggy fluxes into magnitudes
    
    Parameters
    ----------
    nmgy_col : :class: astropy.table.column.Column
        an input table column of nanomaggy fluxes
            
    Returns
    ----------
    :class: list
        a list (column) of magnitudes whose indices' values correspond to the fluxes above

    """
    
    # AJF convert nanomaggies to magnitudes; use list comp    
    mag = [float(f'{22.5 - 2.5*np.log10(f):.3f}') for f in nmgy_col]
    
    return mag
    






#@ds
def mag_to_flux(mag_col):
    """
    Converts the input list of magnitudes into (unitless) nanomaggy fluxes
    
    Parameters
    ----------
    mag_col : :class: list
        the input list of magnitudes to convert to fluxes
            
    Returns
    ----------
    :class: list
        the list of input magnitudes, now converted into fluxes

    """
    # AJF convert fluxes back into nanomaggies
    flux = [float(f'{10**( (m-22.5)/(-2.5) )}') for m in mag_col]
    
    return flux
    







#@ds
def sdss_query(rai, deci, final_table):
    """
    Copied in part from ADM code sdssDR9query.py; queries the sdss server for matching objects from input coordinates (within
    0.02 arcminutes of input coords)
    
    Parameters
    ----------
    rai : :class: numpy.ndarray
        an array of ra coordinates, usually from a survey, that you want to find sdss objects for
        
    deci : :class: numpy.ndarray
        an array of dec coordinates, usually from a survey, that you want to find sdss objects for
        
    final_table : :class: astropy.table.table.QTable
        master table to which details of the sdss objects found should be added - like SDSS RA, DEC, and flux values
    
    Returns
    ----------
    :class: astropy.table.table.QTable
        the master table, now updated with SDSS obbject details
        
    :class: int
        the total number of objects that had sdss matches
    
    """
    # AJF initialize all empty lists to add to table
    num_obj, sdss_ra, sdss_dec, sdss_u, sdss_g, sdss_r, sdss_i, sdss_z = [], [], [], [], [], [], [], []

    # AJF for loop to run over list of all input RA and Dec to see if SDSS has a match
    # AJF copy over Adam's code for a majority of the SDSS query
    print(f'\n\nPerforming SDSS Query...\n\n')
    for r, d in tqdm(zip(rai, deci), total = len(rai)): 

        # ADM initialize the query.
        qry = sdssQuery()

        # ADM the query to be executed. You can substitute any query, here!
        query = """SELECT top 1 ra,dec,u,g,r,i,z,GNOE.distance*60 FROM PhotoObj as PT
        JOIN dbo.fGetNearbyObjEq(""" + str(r) + """,""" + str(d) + """,0.02) as GNOE
        on PT.objID = GNOE.objID ORDER BY GNOE.distance"""

        # ADM execute the query.
        qry.query = query
        for line in qry.executeQuery():
            result = line.strip()

        # ADM NEVER remove this line! It won't speed up your code, it will
        # ADM merely overwhelm the SDSS server (a denial-of-service attack)!
        time.sleep(1)

        # ADM and AJF the server returns a byte-type string. Convert it to a string.
        result = result.decode()
        
        # AJF split the resulting string into a list of strings at commas, if string contains commas
        obj = result.split(',')
        
        # AJF create a boolean integer that indicates whether source has been found (True, 1) or...
        # ... if no objects match (False, 0) and convert to list of floats or nan, respectively
        obj_mask = int(len(obj)>1)
        
        # AJF instead of using if statement, use the boolean mask to either have list of floats or list of nan
        obj = obj_mask*obj + (1-obj_mask)*[np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan]
        obj = [float(data) for data in obj]
        
        # AJF append all values to their respective lists to add to table
        num_obj.append(obj_mask)
        sdss_ra.append(obj[0])
        sdss_dec.append(obj[1])
        sdss_u.append(obj[2])
        sdss_g.append(obj[3])
        sdss_r.append(obj[4])
        sdss_i.append(obj[5])
        sdss_z.append(obj[6])
    
    # AJF add all lists as columns in final_table
    final_table['SDSS RA'] = sdss_ra * u.deg
    final_table['SDSS Dec'] = sdss_dec * u.deg
    final_table['SDSS U'] = sdss_u * u.mag
    final_table['SDSS G'] = sdss_g * u.mag
    final_table['SDSS R'] = sdss_r * u.mag
    final_table['SDSS I'] = sdss_i * u.mag
    final_table['SDSS Z'] = sdss_z * u.mag
    
    # AJF arrange the columns correctly
    order = ['Input Survey Original Row Number', 'Input RA', 'Input Dec','Legacy RA', 'Legacy Dec', 'SDSS RA', 'SDSS Dec',
    'Leg _G', 'Leg _R', 'Leg _Z', 'Leg W1', 'Leg W2', 'Leg W3', 'Leg W4','SDSS U', 'SDSS G', 'SDSS R', 'SDSS I', 'SDSS Z']
    final_table = final_table[order]
    
    # AJF rename Legacy _ columns 
    final_table.rename_column('Leg _G', 'Leg G')
    final_table.rename_column('Leg _R', 'Leg R')
    final_table.rename_column('Leg _Z', 'Leg Z')
    
    # AJF calculate number of SDSS objects matched
    sum_sdss = sum(num_obj)
    
    return final_table, sum_sdss
        
    

#@ds
def ubritel_func(init_cols, ugrizw1234, final_table, nmg):
    """
    Extract the brightest object in the u-band from the final master table
    
    Parameters
    ----------
    init_cols : :class: list
        a list of default column names - usually just ras, decs, and ID numbers       
    ugrizw1234 : :class: list
        a list of the 9 flux names       
    final_table : :class: astropy.table.table.QTable
        the master table from which to find the brightest u-band object
    nmg : :class: astropy.units.core.IrreducibleUnit
        custom nanomaggy unit
    
    Returns
    ----------
    :class: astropy.table.table.QTable
        a single-row table that contains the brightest u-band object information
        
    :class: numpy.ndarray
        a list of flux values (ugriz-w1234) extracted from the resulting single-row table (i.e. brightest object's fluxes in each band)
    
    """


    # AJF add this list to the initial (RA/Dec) columns
    ugriz_cols = init_cols + ugrizw1234 

    # AJF find min of u-band, ignoring any nan values
    u_min_idx = np.nanargmin(final_table['SDSS U'].value)

    # AJF index final table so it is just min of u band object; keep only ugrizw1234 and ra/dec columns for ubritel; make ubritel a table, not a row element
    ubritel = final_table[ugriz_cols][u_min_idx:u_min_idx+1]
    
    # AJF convert all these mags to fluxes
    # AJF store flux values in array for final spectrum plot
    fluxes = np.zeros(9)
    for i, mag in enumerate(ugrizw1234):
        # AJF extract each magnitude column's value (without units), then pass it as a list to mag to flux function to convert
        val_list = [ubritel[mag].value[0]]
        flux = mag_to_flux(val_list)  
        ubritel[mag] = flux
        fluxes[i] = flux[0]
        # AJF add custom nanomaggy unit to the new column  
        ubritel[mag].unit = nmg

    return ubritel, fluxes
    






#@ds
def plot(fluxes):
    """
    Create the plots displaying the spectrum of the brightest u-band object (ubritel) - display the observed optical regime
    as well as the full spectrum
    
    Parameters
    ----------
    fluxes : :class: numpy.ndarray
        a list of flux values (ugriz-w1234) extracted from the resulting single-row table (i.e. brightest object's fluxes in each band)
            
    Returns
    ----------
    None - plots the flux as a function of wavelength for ugriz-w1w2w3w4 bands

    
    """
    # AJF plot initialize
    fig, ax = plt.subplots(2, figsize = (20,15))

    # AJF create wavelength array in nm
    wave = np.array([354.3, 477.0, 623.1, 762.5, 913.4, 3400, 4600, 12000, 22000])

    # AJF plot each subplot with relevant data
    ax[0].plot(wave, fluxes, 'b.', markersize = 15)
    ax[1].plot(wave[0:5], fluxes[0:5], 'b.', markersize = 15)
    
    # AJF add auto-minor ticks (4 per section), increase major tick frequnecy, add grid, add legends
    for a in ax:
        a.xaxis.set_minor_locator(AutoMinorLocator(5))
        a.yaxis.set_minor_locator(AutoMinorLocator(5))
        a.locator_params(axis='both', nbins=15)
        a.grid(True, alpha = 0.9)
    
    # AJF set title of plot and axis titles
    fig.suptitle('ubrite1 Spectrum in Observed Frame', weight = 600, fontsize = 16, y = 0.93)
    for a in ax:
        a.set_xlabel(r'Wavelength (nm)', fontsize = 12)
        a.set_ylabel(r'Flux (nanomaggy)', fontsize = 12)
    
    # AJF plot
    #plt.savefig('fig', format = 'pdf')
    plt.show()







def main():# AJF executes this section first (highest 'shell' of code)
    # AJF start timer
    start = time.time()
    
    # AJF add description
    par = argparse.ArgumentParser(description='Cross-matches input survey data with Legacy survey data, extracts SDSS data for these matched objects, and finds the brightest object in the u-band')
    par.add_argument("path1", type = str, help = 'path to file where input data is located; will read in this data as astropy table; try /d/scratch/ASTR5160/data/first/first_08jul16.fits')
    par.add_argument("path2", type = str, help = 'path to file where legacy survey is located; will read in this data as astropy table; try /d/scratch/ASTR5160/data/legacysurvey/dr9/north/sweep/9.0')
    par.add_argument("radius", type = float, help = 'radius in arcseconds to attempt to match coordinates; used in search_around_sky')
    par.add_argument("coord", type = float, nargs = 3, help = 'Region to search around initially; format RA DEC CIRCULAR_REGION_RADIUS in decimal degrees; ex, for RA=163 deg., DEC=50 deg, region to search=3 deg radius: 163 50 3 ')
    arg = par.parse_args()
    
    path1 = arg.path1
    path2 = arg.path2
    radius = arg.radius
    coord = arg.coord
    
    # AJF ignore astropy warnings, make printing prettier (lots of units in legacy files are not defined astropy units which makes printing to terminal ugly)
    warnings.simplefilter('ignore', category=AstropyWarning)

    # AJF define all unique units that will be used
    nmg = u.def_unit('nanomaggy') 

    # AJF create list of flux columns
    leg_flux_cols = ['FLUX_G', 'FLUX_R', 'FLUX_Z', 'FLUX_W1', 'FLUX_W2', 'FLUX_W3', 'FLUX_W4']
    sdss_flux_cols = ['SDSS U', 'SDSS G', 'SDSS R', 'SDSS I', 'SDSS Z']
    flux_cols = leg_flux_cols + sdss_flux_cols

    # AJF read in input survey data as tables
    input_tbl = QTable.read(path1)

    # AJF make a new table that only keeps input data within certain coord separation (argument "coord")
    input_final_tbl, no_use_tab = restrict_input(coord, input_tbl)

    # AJF pull ra and dec data from that table
    rai, deci = input_final_tbl['Input RA'].value, input_final_tbl['Input Dec'].value
   
    # AJF find list of sweep files that contain final input survey coordinates
    uf = lql(rai, deci, path2, len(rai))

    # AJF do cross-match
    input_tbl, leg_tbl = leg_query(uf, path2, input_final_tbl, radius, leg_flux_cols)
    
    # AJF mask based on wise colors
    leg_tbl, ii = band_mask(leg_tbl, 0.5, 'FLUX_W1', 'FLUX_W2')
    input_tbl = input_tbl[ii]
       
    # AJF mask based on r band magnitudes
    leg_tbl, ii = mag_mask(leg_tbl, 22, 'FLUX_R')
    input_tbl = input_tbl[ii]
       
    # AJF recombine row-indexed input table and row-indexed legacy table after masking
    final_table = hstack([input_tbl, leg_tbl], join_type='outer')
    
    # AJF clean up final table 
    final_table = clean_table(final_table, leg_flux_cols, nmg)
    
    # AJF find common columns from full flux_col list and the final table's columns in case there is a difference
    common_cols = list(set(leg_flux_cols) & set(final_table.columns))
    
    # AJF print final_table before conversion to mag
    #print(f'Input Survey / Legacy Fluxes Table\n\n{final_table}\n\n')
      
    # AJF convert all columns from custom nanomaggy unit to magnitude unit
    for band in common_cols:
        final_table = neg_flux(final_table, band)
        final_table[band] = (convert_to_mag(final_table[band].value))*u.mag
        final_table.rename_column(band, 'Leg '+band[-2:])
    
    # AJF look for duplicates
    dup, dupi, dupl = duplicate_check(final_table)
    
    # AJF print final results of cross-match and masking
    # print(f'Final Table of Matched and Masked Objects:\n\n{final_table}\n\n')
    print(f'\nLength of final table after masking for Mag_R<22 and WISE color W1-W2>0.5: {len(final_table)}\n')
    print(f'Number of duplicates in that table is {dup} which occur(s) at input survey row number(s) {dupi} and/or legacy RA(s) {dupl}.\n')
    print(f'Thus, true number of unique objects cross-matched between input survey and LEGACY data, masked with WISE W1-W2 color > 0.5 and R-band Mag < 22 is: {len(final_table)-dup}\n')
    print(f'Unsure what the best way to combine the duplicate rows is...average, just delete one, etc.?\n')   

    # AJF extract input ra and dec from final_table
    rai, deci = final_table['Input RA'].value, final_table['Input Dec'].value

    # AJF perform sdss query
    final_table, sum_sdss = sdss_query(rai, deci, final_table)

    print(f'\nThe total number of SDSS objects that matched the input data is {sum_sdss}.\n')   
    
    # AJF create list of initial number and ra/dec columns
    init_cols = ['Input Survey Original Row Number','Input RA', 'Input Dec','Legacy RA', 'Legacy Dec', 'SDSS RA', 'SDSS Dec']

    # AJF create list for ugriz-w1234 mag to flux conversion
    ugrizw1234 = ['SDSS U', 'Leg G', 'Leg R', 'SDSS I', 'Leg Z', 'Leg W1', 'Leg W2', 'Leg W3', 'Leg W4']

    # AJF find brightest U-band object and return it, along with list of flux values for it
    ubritel, fluxes = ubritel_func(init_cols, ugrizw1234, final_table, nmg)

    # AJF print ubritel after editing
    print(f'\nBrightest Object in SDSS U-Band:\n{ubritel}\n')

    # AJF plot fluxes as a function of center wavelength
    plot(fluxes)
    
    # AJF print the final result table
    print(f'\n\n\nFinal table with input survey, legacy survey, and SDSS:\n{final_table}\n')
    print('-'*150)
    # AJF print comments to screen
    print(f'\nCOMMENTS:\n')
    print(f'This object, identified by SDSS as SDSS J104240.11+483403.4, is identified as a quasar with redshift of about 1.035. This means that, for example,')
    print(f'the observed fluxes in the u and g bands are actually shifted from the far/near UV; the optical is quite bright, and the infrared (WISE bands)')
    print(f'are even brighter, dominating the spectrum (at both observed and rest frame wavelengths). By restricting the r band to less than 22, we expect to see bright')
    print(f'optical sources; by restricting the WISE W1-W2 color cut to greater than 0.5, we also preferentially choose quasars, although not as restrictive as Stern et al. 2012')
    print(f'suggests. By cross-matching LEGACY surveys with FIRST, we expect radio-loud objects with infrared excess; thus, with these criterion, it makes sense that this object')
    print(f'is a quasar. ubrite1 is a radio-loud AGN with UV and optical emission (perhaps emission lines!), suggesting active accretion.\n')

    # AJF calculate the total time taken for the code to run and print it
    end = time.time()
    tt = end - start
    print(f'Total time the code took was {int(tt//60)} min {tt%60:.3f} sec\n')   

if __name__=='__main__':
    main() 
