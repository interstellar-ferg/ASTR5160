from astropy import units as u
from astropy.coordinates import SkyCoord, EarthLocation, AltAz
from astropy.time import Time
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
	""" Calculates the altitude and airmass of a given target ar a given date and time.

	Parameters:
	----------
	none

	Returns:
	----------	
	none

	Notes:
	----------
	Eventually make this broader function so user can input star coords and time they want to check airmass and altitude. Right now uses date set in code, needs
	to use user-inputted data (like "today" or "2025-2-8 23:00:00", etc.)
	"""
	
	# AJF create wiro location on earth and target coordinates on sky
	# AJF either wiro = line below will work, yield slightly diferent answers based on slightly diff. coords
	# wiro = EarthLocation(lat = 41*u.degree + 5*u.arcmin + 49*u.arcsec, lon = -1*105*u.degree + 58*u.arcmin + 33*u.arcsec, height = 2943*u.m)
	wiro = EarthLocation.of_address('Wyoming Infrared Observatory')
	target = SkyCoord(ra = '12h00m00s', dec = 30*u.degree)

	# AJF create times to check altitude/airmass
	utc_diff = -7 * u.hour 
	tn_11pm = Time("2025-2-5 23:00:00")-utc_diff
	nm_11pm = Time("2025-3-5 23:00:00")-utc_diff

	# AJF perform altitude and airmass calculations using altaz from astropy
	target_alt_tn = target.transform_to(AltAz(obstime = tn_11pm, location = wiro))
	target_alt_nm = target.transform_to(AltAz(obstime = nm_11pm, location = wiro))

	# AJF print out altitude and airmass!
	print(f'\nThis is the altitude and airmass of the star tonight at 11 pm: {target_alt_tn.alt:.6} and {target_alt_tn.secz}')
	print(f'This is the altitude and airmass of the star in one month at 11 pm: {target_alt_nm.alt:.6} and {target_alt_nm.secz}\n')


bear_mountain = EarthLocation(lat=41.3 * u.deg, lon=-74 * u.deg, height=390 * u.m)
utcoffset = -4 * u.hour  # Eastern Daylight Time
time = Time("2012-7-12 23:00:00") - utcoffset

def main(): # AJF executes this section first (highest 'shell' of code) and references other built functions as needed

	func1()
	func2()


if __name__=='__main__':
    main() 
