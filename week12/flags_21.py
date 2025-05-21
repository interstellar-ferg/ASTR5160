import numpy as np
import argparse
from tqdm import tqdm
from halo import Halo

from astropy.table import QTable, hstack, vstack
from astropy.coordinates import SkyCoord
from astropy import units as u

# AJF use chatgpt to ignore unit warnings
import warnings
from astropy.utils.exceptions import AstropyWarning

from week8.cross_match import leg_query_list as lql
from hw.hw3 import restrict_input as ri
from hw.hw3 import convert_to_mag as f_to_m
from hw.hw3 import neg_flux as nf
from hw.hw4 import max_mag_cut

# AJF import a chat-gpt co-written code that auto-writes docstrings with variables included
from master_scripts.docstring_wrapper import log_sphinx_io as ds
# AJF note: @docstring is a wrapper that auto-writes docstrings for the function directly below it
# AJF see master_scripts/docstring_wrapper for more details


# AJF create 5/18/25
# AJF last modifed 5/20/25
# AJF example command line to run:
# python flags_21.py /d/scratch/ASTR5160/week10/qsos-ra180-dec30-rad3.fits /d/scratch/ASTR5160/data/legacysurvey/dr9/south/sweep/9.0 1 180 30 3



#@ds
def separation_single_coord(path2, uf, rai, deci):
    """
    Find the object in a given table (uf) that has the minimum separation from the given ra, dec coordinates rai deci
    
    Parameters
    ----------
    path2 : :class: str
        path to the file where uf is located
    uf : :class: str
        the filename (fits file) that will be converted to a table to be compared to given coordinates
    rai : :class: float
        the ra coordinate to find an object closest to 
    deci : :class: float
        the dec coordinate to find an object closest to
    
    Returns
    ----------
    :class: astropy.table.table.QTable
        a single-row table containing the object that was found to be closest to given rai deci coordinate    
    :class: astropy.coordinates.angles.Angle
        the distance between the rai deci coordinate and the object found

    """

    # AJF read in sweep file as table
    tab = QTable.read(path2 + '/' + uf)
    
    # AJF find ra and dec of table
    ra = tab['RA']
    dec = tab['DEC']
    
    # AJF create SkyCoords for each
    coordi = SkyCoord(ra = rai*u.deg, dec = deci*u.deg)
    coordtab = SkyCoord(ra = ra, dec = dec)
    
    # AJF find the separation between single coordinate and all coordinates of table
    seps = coordtab.separation(coordi)
    
    # AJF find minimum and index of minimum of seps
    min_ii = np.argmin(seps)
    min_sep = seps[min_ii]

    # AJF apply index to sweeps table to find single object; return as table for easier viewing
    obj = tab[min_ii:min_ii+1]
    
    return obj, min_sep





#@ds
def cross_match_table_to_table(tab1, tab2, radius):
    """
    Coordinate-matches two tables together and returns both matched tables
    
    Parameters
    ----------
    tab1 : :class: astropy.table.table.QTable
        an input astropy table, usually from fits file
    tab2 : :class: astropy.table.table.QTable
        another input astropy table, usually from fits file
    radius : :class: float
        
    Returns
    ----------
    :class: astropy.table.table.QTable
        the output astropy table, but indexed so that only the cross-matched objects with table 2 remain
    :class: astropy.table.table.QTable
        the output astropy table, but indexed so that only the cross-matched objects with table 1 remain

    """
    # AJF start spinner
    spinner = Halo(text='Coordinate matching two tables...', spinner = 'pong', color = 'cyan')    
    spinner.start()
    
    # AJF make SkyCoords for each table
    ra1, dec1 = tab1['RA'], tab1['DEC']
    if ra1.unit == u.deg:
        coord1 = SkyCoord(ra = ra1, dec = dec1)
    else: 
        coord1 = SkyCoord(ra = ra1 * u.deg, dec = dec1 * u.deg)
   
    ra2, dec2 = tab2['RA'], tab2['DEC']
    if ra2.unit == u.deg:
        coord2 = SkyCoord(ra=ra2, dec = dec2)
    else:
        coord2 = SkyCoord(ra=ra2*u.deg, dec = dec2*u.deg)
        
    # AJF match coordinates within 1 arcsecond
    tab1_ii, tab2_ii, extra1, extra2= coord2.search_around_sky(coord1, radius*u.arcsec)

    # AJF apply indexing to table1 and table2
    tab1 = tab1[tab1_ii]
    tab2 = tab2[tab2_ii]
    
    spinner.stop()
    
    return tab1, tab2






def main():
    # AJF add description
    par = argparse.ArgumentParser(description='C')
    par.add_argument("qpath", type = str, help = 'path to file where reference quasars are located; try /d/scratch/ASTR5160/week10/qsos-ra180-dec30-rad3.fits')
    par.add_argument("path2", type = str, help = 'path to file where input sweep/fits file to check number of quasars; will read in this data as astropy table; try /d/scratch/ASTR5160/data/legacysurvey/dr9/south/sweep/9.0')
    par.add_argument("radius", type = float, help = 'radius in arcseconds to attempt to match coordinates; used in search_around_sky; try 1')
    par.add_argument("coord", type = float, nargs = 3, help = 'Region to search around; format RA DEC CIRCULAR_REGION_RADIUS in decimal degrees; ex, for RA=163 deg., DEC=50 deg, region to search=3 deg radius: 3 ')
    
    arg = par.parse_args()
    
    qpath = arg.qpath
    path2 = arg.path2
    radius = arg.radius
    coord = arg.coord

    # AJF ignore astropy warnings, make printing prettier (lots of units in legacy files are not defined astropy units which makes printing to terminal ugly)
    warnings.simplefilter('ignore', category=AstropyWarning)
    
    # AJF initialize coordinates
    rai = 188.53667
    deci = 21.04572

    # AJF make quick note that if single coord is close to edge of sweep file range (i.e. sweep ras go every 10 degrees, sweep decs go every 5 degrees) should include another file
    rrai = round(rai, -1)
    rdeci = 5*round(deci/5)
    ra_diff = np.abs(rrai - rai)
    dec_diff = np.abs(rdeci - deci)

    if ra_diff < 0.5:
        print(f'{rai} is only {ra_diff} from the edge of the sweep file found, so there may be an object closer in another file.')
    if dec_diff < 0.5:
        print(f'{deci} is only {dec_diff} from the edge of the sweep file found, so there may be an object closer in another file.')

    # AJF start spinner
    print(f'\n')
    spinner = Halo(text='Finding corresponding sweep file and finding minimum separation object...', spinner = 'pong', color = 'red')    
    spinner.start()

    # AJF find sweeps so that closest object can be found
    uf = lql(rai, deci, path2, 1)

    # AJF find separation between the single coordinate and all coordinates in sweep file; find the minimum separation and return the object with min sep
    min_sep_obj, min_sep = separation_single_coord(path2, uf[0], rai, deci)
    spinner.stop()
    print(f'\nObject found! Has minimum separation of {min_sep} and type {min_sep_obj["TYPE"][0]}\n\n{min_sep_obj}\n')

    # AJF rename min_sep_obj to obj
    obj = min_sep_obj

    # AJf inspect ALLMASK flags
    allg, allr, allz = obj['ALLMASK_G'], obj['ALLMASK_R'], obj['ALLMASK_Z']
    print(f'g:\n{allg}\nr:\n{allr}\nz:\n{allz}')
    print(f'\nDue to the above, it seems that if bits = 2 total for all bands, then that corresponds to all bands being saturated. Looking at the Legacy image')
    print(f'directly, it appears very bright in the center, then becomes dimmer radially, which is probably why it is labeled as EXP (exponential decay of brightness')
    print(f'So, with that being said, I think it is a galaxy that is perhaps saturated in the center, but not for the full image.\n')

    # AJF find all sweep files that could contain ra/dec within 3 degrees of 180, 30
    ramin = 177
    ramax = 183
    decmin = 27
    decmax = 33
    ra = np.array([ramin, ramax, ramin, ramax])
    dec = np.array([decmin, decmax, decmax, decmin])

    uf = lql(ra, dec, path2, len(ra))

    # AJF set up list of tables to append tables to so that they can be combined eventually
    all_tab = []

    # AJF print note
    print(f'\nRunning through all sweep files and applying separation < 3 degree, r<20, and type = psf cuts...\n')
    for f in tqdm(uf):
        # AJF read in sweep file as table
        tab = QTable.read(path2+'/'+f)

        # AJF restrict the tables to objects within 3 degrees of input coords (180, 30) in my case
        na, tab = ri(coord, tab)

        # AJF mask so that only PSF is left
        tab = tab[(tab['TYPE']=='PSF')]

        # AJF convert r fluxes to mags
        tab = nf(tab, 'FLUX_R')
        tab['FLUX_R'] = (f_to_m(tab['FLUX_R'].value))*u.mag

        # AJF mask r < 20 objects
        tab, iir = max_mag_cut(tab, 'FLUX_R', 20)

        # AJF keep track of column names and units; units list will ensure table after concatenate has proper units
        colnams = tab.colnames
        units = [tab[col].unit for col in colnams]

        # AJF append table to list of all tables
        all_tab.append(tab)

    # AJF combine all tables together
    arrs_tab = np.concatenate(all_tab)

    # AJF back to table
    psfobjs = QTable(arrs_tab)

    # AJF put units back onto all columns
    for col, un in zip(colnams, units):
        psfobjs[col].unit = un

    print(f'\n\npsfobjs table after cuts:\n{psfobjs}\n\nNumber of psfobjs = {len(psfobjs)}\n')

    # AJF read in confirmed-quasar table
    quas = QTable.read(qpath)

    # AJF now perform cross-match with qsos data
    qsos, tab2 = cross_match_table_to_table(psfobjs, quas, radius)    
    print(f'\n\nConfirmed quasars in psfobjs:\n\n{qsos}\n')
    print(f'\nNumber of qsos (confirmed quasars in psfobjs) is {len(qsos)}.\n')
    print(f'Stopping here - completed all red tasks but not full python tasks (i.e. did not complete number 4 in slides).\n\n')
    
    
    
if __name__=='__main__':
    main() 
