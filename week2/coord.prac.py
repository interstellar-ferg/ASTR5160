from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.time import Time
from astropy.coordinates import EarthLocation
import numpy as np

def func1():
	""" Converts different coordinates (ra and dec) to degrees, converts today's date/time to mjd and jd, and then displays list of days near today
	
	Parameters:
	----------
	none

	Returns:
	----------
	none

	Notes:
	----------
	Could make this a broader function in the future for user-inputted coordinates/user-inputted jd arrays.
	"""

	# AJF initialize degree-arcmin-arcsec coords astropy (is there an easier, more streamlined way of doing this?)
	c1 = SkyCoord(ra = '23h12m11s', dec = -1 * (40*u.degree + 12*u.arcmin + 13*u.arcsec))
	print(f'\nRA\nUsing astropy: ra = {c1.ra.degree} degrees\nUsing analytical method: ra = {15*(23+12/60+11/3600)} degrees\n')
	print(f'DEC\nUsing astropy: dec = {c1.dec.degree} degrees\nUsing analytical method: dec = {-1*(40+12/60+13/3600)} degrees\n')
	
	# AJF get current time and convert it to jd and mjd, checking the difference to ensure astropy calc correct 
	time = Time.now()
	jd_now = time.jd
	mjd_now = time.mjd
	diff = jd_now - mjd_now
	print(f'This is current jd: {jd_now} and this is current mjd: {mjd_now} and this is difference between the two: {diff}.\n')
	
	# AJF make an array of current mjd +/- 5 days and display it in iso YDM HMS format (i.e. "current" calendar)
	mjd_array = np.arange(mjd_now-5, mjd_now+5, 1)
	mjd_converted = Time(mjd_array, format = 'mjd')
	print(f'These are the five days before and after today converted from mjd format to iso Y-D-M H-M-S format: \n{mjd_converted.iso}')


def func2():
	"""
	Will eventually calculate airmass of object; playing around with airmass calcs in separate file	

	"""
	wiro = EarthLocation(lat = 41*u.degree + 5*u.arcmin + 49*u.arcsec, lon = 105*u.degree + 58*u.arcmin + 33*u.arcsec, height = 2943*u.meter)
	


def main(): # AJF executes this section first (highest 'shell' of code) and references other built functions as needed

	func1()
	func2()


if __name__=='__main__':
    main() 
