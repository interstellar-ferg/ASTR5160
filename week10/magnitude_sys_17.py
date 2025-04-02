import numpy as np

import argparse
import os

from astropy.table import QTable
import astropy.coordinates as coord
from astropy.coordinates import SkyCoord
from astropy import units as u 

# AJF use chatgpt to ignore unit warnings
import warnings
from astropy.utils.exceptions import AstropyWarning

from halo import Halo

from week8.cross_match import leg_query_list as lql

# AJF created 3/19
# AJF last edited 3/24

# comments: 3/24: are final legacy values correct?


def ubvri_to_ugriz(v_mag, bv, ri, ra, dec):
    """ Converts ubvri magnitudes to ugriz filter base magnitudes and compares these to sdss
    
    Parameters:
    ----------
    v_mag : :class: float
        v-band magnitude of star
    bv : :class: 
        b-v color index of star
    ri : :class: float
        r-i color index of star
    ra : :class: float
        ra in decimal degrees of star
    dec : :class: float
        dec in decimal degrees of star
        
    Returns:
    ----------
    calc : :class: numpy.ndarray
        an array of all ubvri to ugriz calculated g, r, and z magnitudes; uses 2005 transformations; includes some NaN points for WISE bands
    sdssf : :class: numpy.ndarray
        an array of g, r, z band results from SDSS query; includes some NaN points for WISE bands
    
    """
    # AJF use halo spinner
    print('\n')
    spinner = Halo(text='Querying SDSS...', spinner = 'earth')
    spinner.start()
    
    # AJF use jester et al 2005 transformations; according to https://james.as.arizona.edu/~psmith/61inch/ATLAS/charts/c109.html...
    # ... PG1633+099A has R-I of 0.511 and U-B > 0, so use following equations for this specific catagory    
    g_mag = v_mag + 0.6*bv - 0.12
    r_mag = v_mag - 0.42*bv + 0.11
    rz = 1.72*ri - 0.41
    z_mag = r_mag - rz
    
    # AJF find sdss magnitudes, write them to temporary file
    os.system(f'python ../master_scripts/sdssDR9query.py {ra} {dec} >> mag_sdss_results.txt')
    
    # AJF read last line of file, then delete temp file
    with open('mag_sdss_results.txt', 'r') as read:
        for line in read:
            # AJF split the string by commas
            reader = line.strip().split(',')
    
    # AJF convert each string into floats
    sdss = [float(r) for r in reader]  
    
    # AJF delete file    
    os.system(f'rm mag_sdss_results.txt')
    
    # AJF create calculated mags array and sdss query results array 
    calc = np.array((f'{g_mag:.3f}', f'{r_mag:.3f}', f'{z_mag:.3f}', np.nan, np.nan, np.nan, np.nan))
    sdssf = np.array((f'{sdss[3]:.3f}', f'{sdss[4]:.3f}', f'{sdss[6]:.3f}', np.nan, np.nan, np.nan, np.nan))
    
    # AJF stop spinner b/c function complete
    spinner.stop()
    
    return calc, sdssf




def get_flux(path, ra, dec):
    """ Find the flux values contained in legacy sweep files for your object
    
    Parameters:
    ----------
    path : :class: string
        directory path to legacy sweep files; is used in lql function
    ra : :class: float
        ra in decimal degrees of star
    dec : :class: float
        dec in decimal degrees of star
        
    Returns:
    ----------
    fluxes : :class: numpy.ndarray
        array containing rounded nanomaggy fluxes from legacy survey
        
    """

    # AJF use halo spinner
    spinner = Halo(text='Querying Local Legacy Sweep Files...', spinner = 'moon')
    spinner.start()    
    
    # AJF find object's legacy sweep file using leg_query_list function
    # AJF lql will produce array of filenames; for n=1, will produce array of length 1, so just choose first index [0]
    # AJF only one object so n = 1
    n = 1
    fn = lql(ra, dec, path, n)[0]
    
    # AJF raise error if no sweep file found
    if fn:
        pass
    else:
        raise ValueError('No sweep filename was found for the ra and dec provided. Check path to north or south directory first.')
    fp = path + '/' + fn
    
    # AJF load in legacy stuff
    tab = QTable.read(path + '/' + fn)
    
    # AJF find index of ra, dec cross-match in legacy; make ra/dec into arrays
    ra_a = np.array([ra])
    dec_a = np.array([dec])
    c_obj = SkyCoord(ra = ra_a*u.deg, dec = dec_a*u.deg, frame ='icrs')
    c_leg = SkyCoord(ra = tab['RA'], dec = tab['DEC'], frame = 'icrs')
    id_leg, id_obj, d2, d3 = c_obj.search_around_sky(c_leg, 1*u.arcsec)
    
    # AJF raise error if no object found
    if len(id_obj) == 0:
        raise ValueError(f'\nNo object was found in cross matching procedure, although fits file was found: {fn}\n')
    
    # AJF write nanomaggy fluxes to array for use in table
    fluxes = np.array( (tab['FLUX_G'][id_leg], tab['FLUX_R'][id_leg], tab['FLUX_Z'][id_leg], 
    tab['FLUX_W1'][id_leg], tab['FLUX_W2'][id_leg], tab['FLUX_W3'][id_leg], tab['FLUX_W4'][id_leg])  ).flatten()
    
    # AJF round all to 3 decimals
    fluxes = np.round(fluxes, 3)
    
    # AJF stop spinner b/c function complete
    spinner.stop()

    return fluxes
    
    


def get_mag(fluxes):
    """ Convert the nanomaggy fluxes from legacy survey to magnitudes

    Parameters:
    ----------
    fluxes : :class: numpy.ndarray
        array containing rounded nanomaggy fluxes from legacy survey
        
    Returns:
    ----------
    mag : :class: list
        list containing magnitudes of object, converted directly from legacy survey fluxes (nanomaggies)   
    
    """
    
    # AJF use list comp to convert nanomaggy fluxes to mags
    mag = [f'{22.5 - 2.5*np.log10(f):.3f}' if f>0 else np.nan for f in fluxes]
    
    return mag




def final_results(calc, sdss, leg_flux, leg_mag):
    """ Contain all of the data in a pretty astropy table/fits file and print it

    Parameters:
    ----------
    calc : :class: numpy.ndarray
        an array of all ubvri to ugriz calculated g, r, and z magnitudes; uses 2005 transformations; includes some NaN points for WISE bands
    sdssf : :class: numpy.ndarray
        an array of g, r, z band results from SDSS query; includes some NaN points for WISE bands        
    leg_flux : :class: numpy.ndarray
        array containing rounded nanomaggy fluxes from legacy survey
    leg_mag : :class: list
        list containing magnitudes of object, converted directly from legacy survey fluxes (nanomaggies)
        
    Returns:
    ----------
    mag_tab : :class: astropy.table.table.QTable
        table that dispalys the calculated ugriz magnitudes, the sdss magnitudes, and the legacy magnitudes/fluxes for easy comparison
    
    """


    
    # AJF create band-name column for result table; create results table
    bands = ['g', 'r', 'z', 'W1', 'W2', 'W3', 'W4']

    print(f"bands length: {len(bands)}")
    print(f"calc length: {len(calc)}")
    print(f"sdss length: {len(sdss)}")
    print(f"leg_mag length: {len(leg_mag)}")
    print(f"leg_flux length: {len(leg_flux)}")    

    mag_tab = QTable([bands, calc, sdss, leg_mag, leg_flux], names = ('Bands','UBVRI-ugriz Mags', 'SDSS Mags', 'Legacy Mags', 'Legacy Flux (nanomaggies)'))

    # AJF write the table to fits file
    mag_tab.write('final_mag_table.fits', overwrite=True)

    # AJF read the fits file into table to ensure it works
    tab_check = QTable.read('final_mag_table.fits')
    print(f'This is the final result:\n\n {tab_check}\n')

    return mag_tab   
    




def main():# AJF executes this section first (highest 'shell' of code)
    # AJF add description
    par = argparse.ArgumentParser(description='Converts some user-inputted ubvri magnitudes to ugriz magnitudes; cross-matches input objects ra/dec to legacy survey and converts legacy survey fluxes into mags')
    par.add_argument("v_mag", type = float, help = 'v-band magnitude of star')
    par.add_argument("bv", type = float, help = 'b-v color index of star')
    par.add_argument("ri", type = float, help = 'r-i color index of star')
    par.add_argument("path", type = str, help = 'path to file where legacy survey is located; will read in this data as astropy table; try /d/scratch/ASTR5160/data/legacysurvey/dr9/south/sweep/9.0')
    par.add_argument("ra", type = float, help = 'ra value of object in decimal degrees youd like to convert flux to mag for')
    par.add_argument("dec", type = float, help = 'dec value of object in decimal degrees youd like to convert flux to mag for')
    
    arg = par.parse_args()

    v_mag = arg.v_mag
    bv = arg.bv
    ri = arg.ri
    path = arg.path
    ra = arg.ra
    dec = arg.dec

    # AJF ignore astropy warnings, make printing prettier
    warnings.simplefilter('ignore', category=AstropyWarning)

    # AJF execute conversion function
    calc, sdss_f = ubvri_to_ugriz(v_mag, bv, ri, ra, dec)
    
    # AJF execute flux-finding function
    fluxes = get_flux(path, ra, dec)
    
    # AJF execute convrsion flux to mag function
    mag = get_mag(fluxes)
    
    # AJF create table of results
    final = final_results(calc, sdss_f, fluxes, mag)
    
    # AJF print comment after running code:
    wise_mags = [float(f) for f in final["Legacy Mags"][3:].value]
    print(f'\n4/1/24: Unsure if final magnitudes are correct; seem slightly off, but maybe within tolerance.\nWISE magnitudes (W1 W2 W3 W4) are {wise_mags}')
    print(f'W4 is NaN, which means the nanomaggy flux was less than 0 and thus the object was not detected in the W4 band.\n')

if __name__=='__main__':
    main() 
