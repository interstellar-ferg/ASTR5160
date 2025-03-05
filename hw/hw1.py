from astropy import units as u
from astropy.coordinates import SkyCoord, EarthLocation, AltAz
from astropy.time import Time
from astropy.table import QTable

import numpy as np

from calendar import monthrange as monran
import sys
import argparse

# created Feb 21 2025 by AJF
# last editing done: March 3 2025


def find_quas(mon, year):
    """ Intakes a user-specified month and year and, by using SkyCoords, finds the airmass of each quasar contained within the file located at:
    /d/scratch/ASTR5160/week4/HW1quasarfile.txt for each night at 11 pm MST at Kitt Peak. Finds the lowest (non-negative) airmass for each night
    and writes the date/time, and this quasar's coordinates (hhmmss.ss deg-'-"), ra and dec, and airmass to an astropy table (fits file)

    Parameters:
    ----------
    mon : :class: string
        User-given month of the year; given as two-digits, such as 01, 02, 11, etc. where 01 is January, 02 is February and so on
    year : :class: string
        User-given year to use in finding airmass values

    Returns:
    ----------
    tab1 : :class: 
        Astropy table containing number of data-rows corresponding to number of days in variable mon (31 for 01 (Jan), 30 for 04 (April), etc.).
        Contains date, and lowest-airmass-quasar's coordinates, ra/dec, and airmass
    
    """

    # AJF create if/else statements for input to ensure months are inputted in the correct format
    if int(mon) < 1 or int(mon) > 12:
        print(f'\nError:\nTry again and choose a number between 01 and 12; 01 for Jan, 02 for Feb., etc.\n')
        sys.exit(0)
    elif mon not in ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12']:
        print(f'\nError:\nTry again and ensure to use 01 for 1 (Jan), 02 for 2 (Feb), etc\n')
        sys.exit(0)
    else:
        
        # AJF get Kitt Peak Coord
        kitt = EarthLocation.of_site('kpno')

        # AJF set utc time difference and find 11:00 pm mst --> utc (7 hours)
        utc_diff = -7 * u.hour 
        t = Time(f'{year}-{mon}-01T23:00:00.000', format = 'fits')
        t1 = t-utc_diff

        # AJF create array of one month's worth of 11:00:00pm starting from day 1 of month
        # AJF monran finds the number of days in the month provided
        dday = np.arange(0, monran(2025, int(mon))[1] )*u.day
        t_arr = t1+dday
        
        # AJF create array of 11 pm's for given month that display MST time not UTC time (for display in table)
        t_arr_mst = t+dday
        
        # AJF create frame that runs through 11 pm over course of input month at kitt peak
        frame_zen = AltAz(obstime = t_arr, location = kitt)
        
        # AJF create array of coordinates containing the location of zenith at kitt peak every night in given month at 11 pm
        zens = SkyCoord(alt=np.full(len(t_arr), 90)*u.degree, az = np.full(len(t_arr), 90)*u.degree, frame = frame_zen)
        
        # AJF convert zenith coordinates to ra/dec format to use as a check in final table results    
        zens2 = zens.icrs
        
        # AJF read in astropy table from adam's directory as skycoord object
        quas_tab = QTable.read('/d/scratch/ASTR5160/week4/HW1quasarfile.txt', names = ['Quasars'], format = 'ascii.no_header', data_start=0)
        data = quas_tab['Quasars']
        
        # AJF initialize empty arrays for ra and dec
        ra = np.empty(len(data), dtype = f'U12')
        dec = np.empty(len(data), dtype = f'U12')
        
        # AJF convert hms.ss deg arcmn arcsec to format skycoord can read
        for i in range(len(data)):
            # AJF ra is first 9 characters of each line in quasarfile
            ra_str = (data[i])[0:9]
            # AJF dec is last 9 characters of each line in quasarfile
            dec_str = (data[i])[9:18]
            # AJF split into xxhxxmxx.xxs for ra and (sign)xxdxxmxx.xs for dec
            ra_str = f"{ra_str[:2]}h{ra_str[2:4]}m{ra_str[4:]}s"
            dec_str = f"{dec_str[:3]}d{dec_str[3:5]}m{dec_str[5:]}s" 
            # AJF assign/append into empty array of length of quasarfile (i.e. make new arrays containing SkyCoord-readable ra and dec)
            ra[i] = ra_str
            dec[i] = dec_str

        # AJF create skycoord of all quasars with ra and dec in icrs frame
        quas = SkyCoord(ra = ra, dec = dec, unit = (u.hourangle, u.deg), frame = 'icrs') 
        
        # AJF create empty arrays for recarray table     
        q_air = np.empty( len(quas) )
        airmass = np.empty ( len(t_arr) )
        ra = np.empty( len(t_arr) )
        dec = np.empty( len(t_arr) )
        coords = np.empty( len(t_arr), dtype = f'U24')
        
        # AJF find minimum airmass above 1; step 1, go to first day of month, run through all quasars at the 11pm on day 1 of month...
        # ... and find their airmass. find the minimum and the index of the minumum of these 1001 airmasses for day 1, use the index to ...
        # ... find the ra, dec, and coordinate value associated with that airmass, then do for day 2 (restart for loop for i = 1, etc.)
        for i in range(len(t_arr)):
            for j in range(len(quas)):
                q_aa = quas[j].transform_to(frame_zen[i])
                air = q_aa.secz
                q_air[j] = air
            airmass[i] = min(q_air[q_air>1])
            # AJF use the index of minimum airmass to find index of ra and dec values and original coordinate values
            ind = np.where(q_air == min(q_air[q_air>1]))
            ra[i] = quas[ind[0][0]].ra.value
            dec[i] = quas[ind[0][0]].dec.value
            coords[i] = data[ind[0][0]]
            

        # AJF note: I believe an alternate method of finding the quasar closest to zenith (lowest airmass) could be to use separation method;...
        # ... using zens SkyCoord created above, find the separation between each night's zenith coordinate and all 1001 quasar's location for that night...
        # ... the smallest separation on a given night at 11 pm indicates the quasar closest to zenith; simply find the airmass using altaz frame for that...
        # ... quasar, and do this for each night. attempted to do this way, but seemed to take longer to find, for example,...
        # ... 30 * 1001 separations than to find 30*1001 airmasses, but perhaps was doing incorrectly
        
        # AJF create astropy table
        # AJF include np.round(zens2.ra.value, 5), np.round(zens2.dec.value, 5) in list of columns and 'Zenith RA', 'Zenith Dec' in list of names if you...
        # ... would like to check the final table against the zenith coordinates of kpno at 11 pm for a given month/day
        tab = QTable([t_arr_mst.value, coords, np.round(ra, 5), np.round(dec, 5), np.round(airmass, 4)], names=('Date', 'Quasar Coordinates', 'RA', 'Dec', 'Airmass'), meta={'name': 'tab'})

	# AJF write the table to fits file
        tab.write('air.fits', overwrite=True)

        # AJF read the fits file into table to ensure it works
        tab1 = QTable.read('air.fits')
        print(f'\nHere is a table containing all the lowest-airmass objects observable at Kitt Peak at 11 pm every night of {mon}-{year}:\n\n {tab1}')
        
        return tab1
        


def main(): # AJF executes this section first (highest 'shell' of code)
    parser = argparse.ArgumentParser(description='Finds which quasar, out of a specific list, is at the lowest airmess (closest to zenith) at 11 pm on every day of a given month')
    
    # AJF add user-defined month and year
    parser.add_argument('mon', metavar = 'mon', type = str, help = 'number of month; 01 for January, 02 for Feb., etc.')
    parser.add_argument('year', metavar = 'year', type = str, help = 'year: 2024, 2017, etc')
    arg = parser.parse_args()
    mon = arg.mon
    year = arg.year
    
    # AJF execute all functions
    tab1 = find_quas(mon, year)




if __name__=='__main__':
    main() 
