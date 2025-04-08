import matplotlib.pyplot as plt
import numpy as np
from numpy.random import random

import argparse

import warnings

# AJF last modified: 4/8/25
# comments: 



def coord_type(radec):
    """ Compile a new list of the user-defined ra/dec box thats in decimal degrees
    
    Parameters:
    ----------
    radec: :class: list
        user-defined ra and dec min/maxes; used as the boundaries of a lat-lon region

    Returns:
    ----------
    rdf: :class: list
        user-defined ra and dec, but converted to decimal degrees if inputted in HH/DD format

    """       
    
    # AJF try list comp below
    # AJF read in ra and dec and make combined radec_final (rdf) list of decimal degree coordinates
    ras = [ convert_to_deci(r, True) if ':' in r else float(r) for r in radec[:2] ]    
    decs = [ convert_to_deci(d, False) if ':' in d else float(d) for d in radec[2:] ]
    
    rdf = ras + decs
    
    # AJF wrote for loop with if statements, then tried to do this with list comprehension above ... 
    # AJF ... is list comp actually more efficient/better in this case?
    
    """
    # AJF split radec into 4 separate string
    for i, coord in enumerate(radec):
        if ':' in coord and i < 2:
            # AJF ra in HH:MM:SS.SS format
            radec[i] = convert_to_deci(coord, i)
            
        elif ':' not in coord and i<2:
            # AJF ra in decimal degrees; change string to float
            radec[i] = float(coord)
            
        elif ':' in coord and i>=2:
            # AJF dec in DD:MM:SS.SS
            radec[i] = convert_to_deci(coord, i)
            
        else:
            # AJF dec in decimal degrees; change string to float 
            radec[i] = float(coord)   
    """
    
    return rdf
    
 
 
 
 
    

def convert_to_deci(coord, ra_flag):
    """ Convert user-inputted ra/dec box coords from HH:MM:SS.SS or DD:MM:SS.SS format to decimal degrees
    
    Parameters:
    ----------
    coord: :class: string
        user-defined ra and dec min/max; used as the boundaries of a lat-lon region; is in HH/DD format
    ra_flag: :class: boolean
        indicates whether the input coordinate is RA (first sub-coord of RA is HH, between 0 and 24) or Dec (which ranges from -90 to 90 and is in degrees already)

    Returns:
    ----------
    final: :class: float
        final coordinate in decimal degree format

    """   

    # AJF handle case when negative DD for dec
    if coord[0] == '-':
        sign = -1
    else:
        sign = 1
    
    # AJF split the coordinate at :    
    splits = coord.split(':')
    
    # AJF do ra; hours*15 = degrees, then arcmin/60 = degrees and arcsec/3600 = degrees
    if ra_flag == True and len(splits) == 3:
        final = float(splits[0])*15 + float(splits[1])/60 + float(splits[2])/3600
        
    # AJF do dec; DD = degrees, then arcmin/60 = degrees and arcsec/3600 = degrees; use sign for dec; HH for RA is never negative ( 0 to 24 )
    elif ra_flag == False and len(splits) == 3:
        final = sign * (float(splits[0]) + float(splits[1])/60 + float(splits[2])/3600)
    
    # AJF raise error if ra is not formatted correctly in HH:MM:SS.SS type format
    elif ra_flag == True:
        raise ValueError(f'RA Coordinate {coord} is not formatted correctly. Need HH:MM:SS.SS or Decimal Degrees')
    
    # AJF raise error if dec is not formatted correctly in DD:MM:SS.SS type format
    else: 
        raise ValueError(f'Dec Coordinate {coord} is not formatted correctly. Need DD:MM:SS.SS or Decimal Degrees')
               
    return final







def area(rdf): 
    """ Find the area of the user-defined lat-lon region using the traditional formula (located in slides)
    
    Parameters:
    ----------
    rdf: :class: list
        user-defined ra and dec, but converted to decimal degrees if inputted in HH/DD format
        
    Returns:
    ----------
    area_ster: :class: float
        area of user-defined lat-lon region in steradians, calculated from PP slides formula
    area_sqd: :class: float
        area of user-defined lat-lon region in square degrees; also calculated from PP slides formula

    """ 
    
    # AJF perform standard area calculation without spherical cap formula    
    # AJF convert all degrees to radians
    rdf = [np.radians(c) for c in rdf]
    ra_min, ra_max, dec_min, dec_max = rdf 
    
    # AJF all are in radians now, so use (ra2 - ra1) * ( sin(dec2) - sin(dec1) ) for steradians
    area_ster = (ra_max - ra_min) * ( np.sin(dec_max) - np.sin(dec_min) )

    # AJF calculate area in square degrees; (180/pi)^2 * (ra2 - ra1) * ( sin(dec2) - sin(dec1) ) if ra, dec in radians
    # AJF use (180/pi) * (ra2 - ra1) * ( sin(dec2) - sin(dec1) ) if ra in degrees
    # AJF use ra in radians formula
    area_sqd = (180/np.pi)**2 * ( ra_max - ra_min ) * ( np.sin(dec_max) - np.sin(dec_min) ) 

    # AJF compare areas
    """
    check_area = area_ster * (180/np.pi)**2
    print(f'area in sq deg: {(area_sqd)}; area in ster converted back to deg for check: {check_area}; area in ster: {area_ster}')
    """
    
    return area_ster, area_sqd







def compare_area(rdf):
    """ Compare the area of the lat-lon region derived formulaically to the area derived from spherical cap formula
    
    Parameters:
    ----------
    rdf: :class: list
        user-defined ra and dec, but converted to decimal degrees if inputted in HH/DD format
        
    Returns:
    ----------
    ll_ster: :class: float
        area of user-defined lat-lon region in steradians, calculated from PP slides formula; calculated using area()
    cap_area: :class: float
        area of user-defined lat-lon region in square degrees; also calculated from PP slides formula; calculated using area()
            
    """
    
    # AJF use the user-defined min declination as the base dec for spherical cap bound by dec and find the area of the cap formulaically
    dec_min = rdf[2]    
    cap_area = 2*np.pi*( 1 - np.sin(np.radians(dec_min)) )
    
    # AJF define a spherical cap as a 'lat-lon rectangle', find area using lat-lon (ll) rectangle area function
    radec_cap = [float(0), float(360), dec_min, float(90)]
    ll_ster, ll_sqd = area(radec_cap)

    # AJF compare formulaic way to lat-lon area function
    print(f'\nFor cap lat-lon region RA: {radec_cap[0]} to {radec_cap[1]}, Dec: {radec_cap[2]} to {radec_cap[3]}, area function returns {ll_ster:.6f} steradians.')
    print(f'Spherical cap area formula ( 2 * pi * sin({radec_cap[2]}) ) returns {cap_area:.6f} steradians.')
    
    # AJF print matching or warning if not matching    
    if np.round(ll_ster, 6) == np.round(cap_area, 6):
            print(f'They match!')
    else:
        warnings.warn('\nSomething went wrong with spherical cap area calculation, but code can proceed.\n')

    # AJF define entire hemisphere and do same calculations as above to show RA 0 - 360 and dec 0 - 90 as requested in HW
    hs_area = 2*np.pi*( 1 - np.sin(np.radians(0)) )
    radec_hs = [0, 360, 0, 90]
    ll_ster, ll_sqd = area(radec_hs)
    print(f'\nFor a hemisphere, the area calculated using the area function is: {ll_ster:.6f} steradians.')
    print(f'The hemispheres area calculated via the spherical cap formula is: {hs_area:.6f} steradians.')

    # AJF print matching or warning if not matching    
    if np.round(ll_ster, 6) == np.round(hs_area, 6):
            print(f'They match!')
    else:
        warnings.warn('\nSomething went wrong with spherical cap area calculation, but code can proceed.\n')    
    
    # AJF return the two areas in case user needs them for something
    return ll_ster, cap_area






    
def plot_aitoff(rdf, path):
    """ Plot several lat-lon rectangles on the aitoff projection
    
    Parameters:
    ----------
    rdf: :class: list
        user-defined ra and dec, but converted to decimal degrees if inputted in HH/DD format
    path: :class: string
        user-defined path that figure will be saved to
     
    Returns:
    ----------
    None - plots lat-lon rectangles in aitoff projection and saves it to user-defined directory (path)
    
    """

    # AJF use user input ra_min and ra_max for plotting    
    # AJF handle cse where input ras wrap projection
    if rdf[1] > float(180) and rdf[0] < float(180):
        # AJF mirror input ra > 180 across x axis, find radians, then make negative
        ra_tup = np.radians(rdf[0]), -1*np.radians( 180 - (rdf[1]-180) )  
    elif rdf[1] > float(180) and rdf[0] > float(180):
        # AJF do same as line above, but if both ra_min and ra_max are larger than 180 degrees
        ra_tup = -1*np.radians(180 - (rdf[0]-180)), -1*np.radians(180 - (rdf[1]-180) )     
    else:
        ra_tup = np.radians(rdf[0]), np.radians(rdf[1])
    
    # AJF create list of dec tuples to plot lat-lon rectangles (in degrees); can easily be changed 
    decs = [(-70, -50), (-45, -30), (-10, 40), (60, 75)]
    
    # AJF convert dec degrees to radians (keep in degrees above for easy user modification)
    decs = [ (np.radians(f[0]), np.radians(f[1])) for f in decs ]
    
    # AJF create vertical line coordinates
    v_coords_min = [(ra_tup[0], ra_tup[0], dec[0], dec[1]) for dec in decs]
    v_coords_max = [(ra_tup[1], ra_tup[1], dec[0], dec[1]) for dec in decs]    
    v_coords = v_coords_min + v_coords_max

    # AJF handle case when input coordinates wrap around projection
    # AJF create horizontal line coordinates that start/stop at boundary wrap
    h_coords_low_r = [(ra_tup[0], np.pi, dec[0], dec[0]) for dec in decs]
    h_coords_upp_r = [(ra_tup[0], np.pi, dec[1], dec[1]) for dec in decs]
    h_coords_low_l = [(-np.pi, ra_tup[1], dec[0], dec[0]) for dec in decs]
    h_coords_upp_l = [(-np.pi, ra_tup[1], dec[1], dec[1]) for dec in decs] 

    # AJF create horizontal line coordinates w/out wrap
    h_coords_min = [(ra_tup[0], ra_tup[1], dec[0], dec[0]) for dec in decs]
    h_coords_max = [(ra_tup[0], ra_tup[1], dec[1], dec[1]) for dec in decs]   
    
    # AJF handle case when input coordinates wrap around projection
    if rdf[1] > float(180) and rdf[0] < float(180):
        h_coords = h_coords_low_r + h_coords_upp_r + h_coords_low_l + h_coords_upp_l
    else:
        h_coords = h_coords_min + h_coords_max            
    
    # AJF set up figure - use plt.figure, not plt.subplots, because need to specify aitoff projection in axes creation
    fig= plt.figure(figsize = (15, 15))
    
    # AJF create subplot axis with aitoff projection
    ax = plt.subplot(111, projection = 'aitoff')

    # AJF set up grid
    ax.grid(True, alpha = 0.5, color = 'b', linewidth = 1)

    # AJF plot the vertical lines for each radec box
    for v in v_coords:
        x = v[:2]
        y = v[2:]
        ax.plot(x, y, color = 'r')
    
    # AJF plot the horizontal lines for each radec box
    for h in h_coords:
        x = h[:2]
        y = h[2:]
        ax.plot(x, y, color = 'r')
    
    # AJF save the plot to the user-defined directory
    plt.savefig(path+'/aitoff_proj.png', format = 'png')
    plt.show()
    
    # AJF dont need to return anything; function used for plotting, and function saves plot to user-defined dir







def populate_and_plot(rdf, area_ster, path):
    """ Plot the user-defined lat-lon region filled in with tons of points and make sure the fractional area of this region is close to the fraction of points in the lat-lon region to total points
    
    Parameters:
    ----------
    rdf: :class: list
        user-defined ra and dec, but converted to decimal degrees if inputted in HH/DD format
    area_ster: :class: float
        area of user-defined lat-lon region in steradians, calculated from PP slides formula (to compare to points fraction) 
    path: :class: string
        user-defined path that figure will be saved to
     
    Returns:
    ----------
    None - plots lat-lon rectangles in aitoff projection and saves it to user-defined directory (path)
    
    """
    
    # AJF part 2 of HW
    # AJF number of points
    N = 1000000

    # AJF create array of ra and dec and plot on sphere surface
    # AJF create N points between 0 and 1, then scale up to 0 to 2, then subtract 1 to get range (-1, 1), then multiply by 180 to get...
    # AJF ... -180 to 180 (evenly spaced/random), then convert to radians (-pi, pi)
    ra = (np.radians(180*((2*random(N))-1)))
    
    # AJF shift random-generated (0,1) to (0,2) range, then subtract from 1 so that range is (1, -1), then take arcsin of this to get values
    # ... ranging from -pi/2 to pi/2
    # ...(depends on sine, so is uniform area across sphere, not cartesian)
    dec = (np.arcsin(1.-random(N)*2))

    # AJF convert all user-defined ra/dec mins and maxes into radians
    ra_min, ra_max, dec_min, dec_max = np.radians(rdf[0]), np.radians(rdf[1]), np.radians(rdf[2]), np.radians(rdf[3])
    
    # AJF handle the wrapping case; for example, if user inputs 45 to 300 as ra_min and ra_max (degrees), ra_max is wrapped around the aitoff projection ...
    # AJF ... since aitoff goes from -pi to pi, but radians(300) is 5pi/3 . need to convert this to a value between -pi and pi... do this by full period (2pi) shift...
    # AJF ... and need to do this for ra_min OR ra_max
    if ra_min > np.pi:
        ra_min = ra_min - 2*np.pi
    if ra_max > np.pi:
        ra_max = ra_max - 2*np.pi

    # AJF create masks for the randomly-generated full-sky ra and dec based on beng between ra_min and ra_max, but take into account wrapping/shifting to aitoff projection space
    if ra_min > ra_max:
        # AJF handle case where ra_min (say, 60 degrees = pi/3, for ex) is larger than ra_max (say. 300 degrees = 5pi/3 = shifted to -pi/3)
        # AJF need to keep everything 'right' of 60 degrees (60 to 180) and everything 'left' of 300 degrees (180 to 300), so need 'OR' logic
        mask = ((ra > ra_min) | (ra < ra_max)) & (dec > dec_min) & (dec < dec_max)
    else:
        # AJF normal mask; no wrapping necessary
        mask = ((ra > ra_min) & (ra < ra_max)) & (dec > dec_min) & (dec < dec_max)

    # AJF filter points based on mask
    raf = ra[mask]
    decf = dec[mask]
    
    # AJF set up figure - use plt.figure, not plt.subplots, because need to specify aitoff projection in axes creation
    fig= plt.figure(figsize = (15, 15))
    
    # AJF create subplot axis with aitoff projection
    ax = plt.subplot(111, projection = 'aitoff')

    # AJF set up grid
    ax.grid(True, alpha = 0.5, color = 'b', linewidth = 1)

    # AJF plot
    ax.scatter(raf, decf, s = 0.1, color = 'g')
    
    # AJF save the plot to the user-defined directory
    plt.savefig(path+'/aitoff_proj.png', format = 'png')
    plt.show()    

    # AJF intake area derived from area() function and divide this by 4*pi (area of spherical surface in steradiabs); then, compare that fraction to...
    # AJF ... (number of points located in lat-lon rectangle)/(total number of plotted points)    
    # AJF print to terminal the final area results
    # AJF print the area fraction of masked space calculated directly using area() function (theoretical/total area)
    print(f'\nHere is the fractional area, directly calculated from the area() function / formulaically (true fraction): {(area_ster/(4*np.pi)):6f}')
    
    # AJF print the ratio of (points kept after masking the randomly-evenly generated ra) to (total number of original points) ; should be very similar to true area fraction
    print(f'Here is the fractional area calculated from the number of masked (kept) coordinates to the total number of randomly-generated coordinates: {(len(raf)/N):.6f}')
    print(f'If everything has run correctly and N (={N}) is large, they should be extremely similar.\n')







def main(): # AJF executes this section first (highest 'shell' of code)
    par = argparse.ArgumentParser(description='Second homework; intakes min/max RA and Dec values in decimal degrees or HH:MM:SS.SS / DD:MM:SS.SS format and finds areas / plots lat-lon rectangles / lat-lon regions in aitoff projection')
    
    ####################################
    # AJF important! use terminal commmand like: python hw2.py . -- ra_min ra_max dec_min dec_max 
    # AJF -- allows argparse to properly intake negative numbers ion string rather than - as optional argument
    ####################################
    
    # create input path argument for saving plots
    par.add_argument("path", type = str, help = 'Directory path where you want to save the Aitoff projections plotted in this module.')
    
    # AJF accept input coordinates like SDSS does
    # AJF example: 12:00:00.00 13:00:00.00 9:30:00.00 12:00:00.00 is a lat-lon rectangle from RA 12 hours - RA 13 hours (180 degrees to 195 degrees) and from Dec 9.5 degrees to 12 degrees
    # AJF example 2: 10:30:00.00 11:00:00.00 30.00 40.00 is a lat lon rectangle from RA 10 hours, 30 minutes to 11 hours (157.5 degrees to 165 degrees) and Dec 30 degrees to 40 degrees
    par.add_argument("radec", type = str, nargs = 4, help = 'Right Ascension and Declination for the users lat-lon rectangle; format like -- ra_min ra_max dec_min dec_max ; need to include -- so that negative numbers are parsed correctly ; can intake ra in decimal degrees or HH:MM:SS and dec in decimal degrees or DD:MM:SS')
        
    # AJF parse and rename arguments
    arg = par.parse_args()
    path = arg.path
    radec = arg.radec

    # AJF parse up coordinates - make into decimal degrees
    rdf = coord_type(radec)
    
    # AJF find the area of the user-defined lat-lon region
    area_ster, area_sqd = area(rdf)
    
    # AJF compare the area() function to the general spherical cap method (a request made in the hw handout)
    ll_ster, cap_area = compare_area(rdf)
    
    # AJF plot lat-lon regions hard-coded (not user defined, >15 degree lat-lon rectangles; requested in hw)
    plot_aitoff(rdf, path)
    
    # AJF populate the user-defined lat-lon region with points and return the fractional area of the masked region (i.e. the area in steradians of the green points vs 4pi, the total area)
    populate_and_plot(rdf, area_ster, path)



    
if __name__=='__main__':
    main() 
