from astropy.table import QTable
from astropy.table import Table
import matplotlib.pyplot as plt
import numpy as np
import argparse

def func1(path):
	""" Reads fits file, converts to table so that data can be plotted.

	Parameters
	----------
	path : :class: 'string'
		filepath to fits file
	
	Returns:
	----------
	ext : :class: 'astropy.table.column.Column'
		first column of extinction data
	ra : :class: 'astropy.table.column.Column'
		column of right ascension data
	dec : :class: 'astropy.table.column.Column'
		column of declination data
	"""
	
	# AJF read in fits file, converts to table and reads appropriate columns 
	objs = Table.read(path)
	ext = objs['EXTINCTION'][:,0]
	dec = objs['DEC']
	ra = objs['RA']
	
	# AJF index extinction data per above 0.22 criteria
	ext_index = (ext > 0.22)

	# AJF index the ra and dec data based on the boolean results of the exteinction criteria
	dec_index = dec[ext_index]
	ra_index = ra[ext_index]

	# AJF plot the full ra and dec data, as well as overplotted indexed ra and dec
	fig = plt.figure(figsize = (12,10))
	plt.title('Dec as a Function of RA')
	plt.xlabel('RA')
	plt.ylabel('Dec')
	plt.plot(ra, dec, 'r.', markersize = 10)
	plt.plot(ra_index, dec_index, 'b+', markersize = 8)
	plt.show()
	return ext, ra, dec
	


def func2(ext, ra, dec):
	""" Create a table and write it to a fits file, then read it to ensure it works properly.

	Parameters
	----------
	ext : :class: 'astropy.table.column.Column'
		first column of extinction data
	ra : :class: 'astropy.table.column.Column'
		column of right ascension data
	dec : :class: 'astropy.table.column.Column'
		column of declination data
	
	Returns:
	----------
	tab1 : :class: 'astropy.table.table.Table'
		qtable created from ra, dec, and random-generated 3-array
	"""
	# AJF generate 3-array (3 columns, 100 rows) of random data uniformly picked from min value of minimum of extinction to max value of max extinction
	rand = np.random.uniform(min(ext), max(ext), (100,3))

	# AJF create the table
	table1 = QTable([ra, dec, rand], names=('ra', 'dec', 'rand'), meta={'name': 'table1'})

	# AJF write the table to fits file
	table1.write('rec_tbl_1.fits', overwrite=True)

	# AJF read the fits file into table to ensure it works
	tab1 = Table.read('rec_tbl_1.fits')
	print(f'This is my recarray:\n\n {tab1}')
	return tab1



def main(): # AJF executes this section first (highest 'shell' of code) and references other built functions as needed
	parser = argparse.ArgumentParser(description='Reads in datafile path and plots Declination as a Function of RA; additonally excludes certain extinction values; generates recarray with random integers')
	parser.add_argument('filepath', metavar = 'filename', type = str, help = 'path to file, something like /d/scratch/ASTR5160/week2/struc.fits')
	arg = parser.parse_args()
	path = arg.filepath

	# AJF read in fits file, plot data, extract final extrinction, ra, and dec 
	ext, ra, dec = func1(path)

	# AJF generate a 3-array (3 columns, 100 data points each) of 300 random data points, then generate a fits file from these arrays
	tab1 = func2(ext, ra, dec)
	


if __name__=='__main__':
    main() 
