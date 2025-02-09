import matplotlib.pyplot as plt
import numpy as np
from astropy import units as u
import astropy.coordinates as coord
from astropy.coordinates import SkyCoord, EarthLocation, AltAz
from astropy.time import Time
from matplotlib.ticker import AutoMinorLocator

# created in and around Feb 7 2025 by AJF
# last editing done: Feb. 8 2025


def cart(): 
    """ Changes a certain sky coordinate into the cartesian coordinate representation

    Parameters:
    ----------
    None

    Returns:
    ----------
    c1 : :class: astropy.coordinates.sky_coordinate.SkyCoord
        orignal coordinates in whatever form they were inputted in
    """

    # AJF initialize coordinates using week2 example
    c1 = SkyCoord(ra = '23h12m11s', dec = -1 * (40*u.degree + 12*u.arcmin + 13*u.arcsec))
    
    # calculate x, y, z from formulas using c1 original representation, converting degrees to radians
    x_calc = np.cos(c1.ra.radian)*np.cos(c1.dec.radian)
    y_calc = np.sin(c1.ra.radian)*np.cos(c1.dec.radian)
    z_calc = np.sin(c1.dec.radian)

    # AJF do hard change in coordinates type - .cartesian changes the coordinates from a SkyCoord class ('high class') to a representation class
    # AJF which is used for math/operations, not for use in further coordinate transforms (I think)
    # AJF from what I gather, .cartesian changes the representation of the coordinates in a certain frame, while...
    # AJF representation_type changes the actual frame of reference (i.e. still SkyCoord)
    c1cart = c1.cartesian
    print(f'\nThis is type for c1: {type(c1)}\nand c1 itself:\n{c1}\n\nThis is type for c1.cartesian: {type(c1cart)}\nand c1.cartesian itself:\n{c1.cartesian}')

    # AJF soft change in coordinates - change basis of coordinates to cartesian, but leave as SkyCoord class
    c1.representation_type = coord.CartesianRepresentation
    print(f'\nThis is type for c1 after c1.representation_type change: {type(c1)}\nand the new c1 after c1.rep_type itself:\n{c1}\n')
    print(f'Notice how c1 is still a SkyCoord after using representation_type; this changes the coordinate axes so that it is not ra and dec, but is x, y, and z')
    print(f'c1.ra no longer works here; c1.x is:\n{c1.x}\n')

    # AJF from documentation here: https://github.com/astropy/astropy/issues/9940 seems that representation_type should be used more often than not

    # AJF check if changing rep_type gives correct transformations
    print(f'This is c1.x, c1.y, c1.z after changing rep_type: {c1.x}, {c1.y}, {c1.z}\n This is x, y, z calculated from original ra and dec in radians: {x_calc}, {y_calc}, {z_calc}')
    print(f'They match!\n')
    return c1





def gal_center(): 
    """ Finds the coordinates of the galactic center in standard icrs format and finds location in constellation

    Parameters:
    ----------
    None

    Returns:
    ----------
    cent : :class: astropy.coordinates.sky_coordinate.SkyCoord
        skycoord of galactic center in galactic coordinates

    """

    # AJF b is galactoc latitude, l is gal. longitude; set frame to galactic
    cent = SkyCoord(b = 0*u.hour, l = 0*u.degree, frame = 'galactic')

    # AJF convert galactic frame to icrs frame (J2000) and find ra, dec of gal. center in both degrees and hms
    ra_h, ra_m, ra_s = cent.icrs.ra.hms.h, cent.icrs.ra.hms.m, cent.icrs.ra.hms.s
    ra_deg, dec_deg = cent.icrs.ra.deg, cent.icrs.dec.deg
    print(f'This is ra (H, M, S) and dec (degrees) of galactic center: {ra_h}h {ra_m}m {ra_s}s, {dec_deg}')
    print(f'This is ra and dec of the galactic center in degrees: {ra_deg}, {dec_deg}\n')

    # AJF see below for location og gal. center in Sag.
    print(f'The gal. center is located in the Sagittarius constellation, near the edge. The constellation has stars ranging from about RAs of about 19h to about 18h and Decs')
    print(f'ranging from about -34.5 degrees to -25.5 degrees; thus, since the galactic center has dec right in the middle of the range, and a lower RA than all the stars,')
    print(f'it is located in the central (up-down) and "rightmost"/"western" part of Sagittarius.\n')
    return cent





def movement(): 
    """ Calculates the change in the zenith direction's galactic coordinates in Laramie, WY

    Parameters:
    ----------
    None

    Returns:
    ----------
    zenith_gal : :class: astropy.coordinates.sky_coordinate.SkyCoord
        a compilation of the zenith direction's galactic coordinates across one year using the LST method
    zenith_gal2 : :class: astropy.coordinates.sky_coordinate.SkyCoord
        a compilation of the zenith direction's galactic coordinates across one year using the AltAz method
    dday.value : :class: numpy.ndarray
        array of 365 days (numbers)

    """

    # AJF create laramie location on earth and target coordinates on sky
    laramie = EarthLocation(lat = 41.3114*u.degree, lon = -105.5911*u.degree)

    # AJF set utc time difference and find current day's 11:59:59 pm
    utc_diff = -7 * u.hour 
    ct = Time.now()
    ct1 = ct.to_value('iso', subfmt = 'date')
    t1 = Time(f'{ct1} 23:59:59')-utc_diff

    # AJF create array of one year's worth of 11:59:59pm starting from current night
    dday = np.arange(0, 366, 1)*u.day
    t_arr = t1+dday

    # AJF since RA = LST at HA = 0 (zenith), find LST for each day and use this as the RA for the coordinates in the sky
    lst_arr = t_arr.sidereal_time('apparent', longitude = laramie.lon).degree*u.degree

    # AJF create a latitude (dec) array with laramie's latitude for coordinates
    lat_arr = np.full(np.shape(lst_arr), laramie.lat)*u.degree

    # AJF set up coordinates
    zenith = SkyCoord(ra = lst_arr, dec = lat_arr)

    # AJF convert this "time-array" of coordinates to galactic
    zenith_gal = zenith.galactic

    # AJF confirm altitude of object is close to zenith (90 degrees) at 11:59:59 pm; confirm airmass is close to 1
    zenith_alt_az = zenith.transform_to(AltAz(obstime = t_arr, location = laramie))

    # AJF because calculating lst and alt/az is intensive, this method may be slighlty incorrect
    print(f'\nThis is quick check on values of altitude for doing it this way (mean should be 90 with no std_dev): {np.mean(zenith_alt_az.alt.value):.8} +/- {np.std(zenith_alt_az.alt.value):.4} degrees\n')     

    # --------------------------------------------------------------------------------------------------------------------------------------------
    # AJF could also probably use the fact that zenith literally means altitude is 90 degrees, so just create frame where time runs for a year...
    # and fix the coordinates as altitude = 90, az = 0, then convert these coordinates to galactic and see how it changes

    # AJF create an AltAz frame that runs through a year's worth of zenith's location at 11:59:59 pm in laramie, starting at today's date 
    frame1 = AltAz(obstime = t_arr, location = laramie)

    # AJF create SkyCoordinate for each date, but use the fact that at zenith, altitude must be 90 degrees and azimuthal must be 0
    zenith2 = SkyCoord(alt = np.full(len(dday), 90)*u.degree, az = np.full(len(dday), 0)*u.degree, frame = frame1)

    # AJF convert this coordinate to galactic coordinates
    zenith_gal2 = zenith2.galactic
    return zenith_gal, zenith_gal2, dday.value





def plotting(z1,z2, year):
    """ Plots the results of movement(), showing the change in galactic long. and lat. as a function of time
    in pretty colors!

    Parameters:
    ----------
    z1 : :class: astropy.coordinates.sky_coordinate.SkyCoord
        a compilation of the zenith direction's galactic coordinates across one year using the LST method
    z2 : :class: astropy.coordinates.sky_coordinate.SkyCoord
        a compilation of the zenith direction's galactic coordinates across one year using the AltAz method
    year : :class: numpy.ndarray
        array of 365 days (numbers)

    Returns:
    ----------
    None - plots and saves a figure

    """

    # set up plot with 3x1 grid with same x axis
    fig, ax = plt.subplots(3, figsize = (15,15), sharex=True, layout = 'constrained')

    # set up each scatterplot with colormap info
    p1 = ax[0].scatter(z1.l.degree, z1.b.degree, c = year, cmap='hot', s = 8, label = 'LST-Method')
    p2 = ax[1].scatter(z2.l.degree, z2.b.degree, c = year, cmap = 'cool', s = 8, label = 'AltAz Method')
    p3 = ax[2].scatter(z1.l.degree, z1.b.degree, c = year, cmap='hot', s = 4, label = 'LST-Method')
    p4 = ax[2].scatter(z2.l.degree, z2.b.degree, c = year, cmap = 'cool', s = 2, label = 'AltAz-Method')

    # set up each colorbar, change its location with pad
    bar1 = plt.colorbar(p1, pad = 0.04)
    bar2 = plt.colorbar(p2, pad = 0.04)
    bar3 = plt.colorbar(p3, pad = 0.02)
    bar4 = plt.colorbar(p4, pad = 0.01)

    # set the xlim of the entire x axis
    ax[0].set_xlim(0, 360)

    # set colorbar labels
    bar1.set_label('Number of days after today')
    bar2.set_label('Number of days after today')
    bar3.set_label('Number of days after today')

    # AJF add auto-minor ticks (4 per section), increase major tick frequnecy, add grid, add legends
    for a in ax:
        a.xaxis.set_minor_locator(AutoMinorLocator(5))
        a.yaxis.set_minor_locator(AutoMinorLocator(5))
        a.locator_params(axis='both', nbins=15)
        a.grid(True, alpha = 0.25)
        a.legend(loc = 'best')
    
    # Format x and y axis labels and title
    ax[2].set_xlabel(r'Galactic Longitude, l ($^\circ$)', fontsize = 14)
    ax[1].set_ylabel(r'Galactic Latitude, b ($^\circ$)', fontsize = 14)
    fig.suptitle('Galactic Coordinates of the Zenith Direction in Laramie, WY for the Following Year', fontsize = 16)

    # save the figure to cwd and plot it  
    plt.savefig('gal_zenith.png', format = 'png')
    plt.show()




def main(): # AJF executes this section first (highest 'shell' of code)
    c1 = cart()
    cent = gal_center()
    z1, z2, year = movement()
    plotting(z1, z2, year)




if __name__=='__main__':
    main() 
