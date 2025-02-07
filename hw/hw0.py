from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import numpy as np
import argparse
from matplotlib.ticker import AutoMinorLocator

# created and edited by AJF in and around Feb. 6 2025
# last editing done: Feb. 7 2025 by AJF


def xyyerr(m,b):
	""" Intakes values for m and b, randomly chooses 10 x-values in the domain [0,10), finds 10 y-values from y = mx + b,
	bumps each of these y-values in +/- y direction via a random offset chosen from a (u = each y-value, sigma = 0.5) Gaussian distribution
	with yerr = 0.5 for all 10 y-values

	Parameters
	----------
	m : :class: float
		user-defined slope of a line
	b : :class: float
		user-defined y-intercept of a line

	Returns:
	----------
	x_arr : :class: numpy.ndarray
		ten x-values generated from uniformly from a distribution from 0 to 10
	y_arr : :class: numpy.ndarray
		arrau of y-values generated from the user-defined m and b parameters and x_arr, then perterbed by dy values
	y_err : :class: numpy.ndarray
		array of errors set to value of 0.5
	og_y_arr : :class: numpy.ndarray
		unperturbed y-values
	dy_arr : :class: numpy.ndarray
		perturbation values added to og_y_arr to create y_arr
	"""
	
	# AJF create 10 floats for x-values in [0,10) domain
	x_arr = np.random.uniform(0,10, (10))

	# AJF use y = mx+b to get y values from above x-values
	y_arr = m*x_arr + b
	og_y_arr = y_arr

	"""
	# AJF below is for loop for choosing dy based on current y-value
	dy_arr = np.zeros((10,1))
	for i in range(len(y_arr)):
		u = y_arr[i]
		sig = 0.5
		dy = np.random.normal(u, sig, 1)
		dy = dy*(np.random.choice([1,-1], 1))
		y_arr[i] = y_arr[i]+dy
		dy_arr[i] = dy

	"""

	# AJF offset each y value by +/-dy chosen from Gaussian distribution with u = y-value, sigma = 0.5
	# AJF choose dy based off fixed value (u is fixed, not dependent on y-value) 
	u = 0
	sig = 0.5
	dy_arr = np.random.normal(u, sig, (10))

	"""
	# AJF if mean value of Gaussian is very positive, then no negative dy will be pulled, so use the following for loop to randomize +/-
	for i in range(len(dy)):
		dy_arr[i] = dy_arr[i]*(np.random.choice([1,-1], 1))

	"""
	# AJF add in wiggle to y_arr
	y_arr = y_arr + dy_arr		
			
	# AJF create y_err array
	y_err = np.full(10, 0.5)

	return x_arr, y_arr, y_err, og_y_arr, dy_arr
	



def fitting(x_arr, y_arr, y_err):
	"""Fits the discrete data generated in xyyerr function with a one-dimensional line (i.e. y=mx + b) using scipy curvefit

	Parameters
	----------
	x_arr : :class: numpy.ndarray
		ten x-values generated from uniformly from a distribution from 0 to 10
	y_arr : :class: numpy.ndarray
		arrau of y-values generated from the user-defined m and b parameters and x_arr, then perterbed by dy values
	y_err : :class: numpy.ndarray
		array of errors set to value of 0.5
	
	Returns:
	----------
	m_fit : :class: numpy.float64
		slope parameter of the fitted line
	b_fit : :class: numpy.float64
		y-intercept parameter of the fitted line

	"""

	# AJF create linear fit function
	def f1(x, m, b):
		return m*x + b

	# AJF actually use curvefit to find best m and b values using original x and y data x_arr and y_arr
	vals, cov = curve_fit(f1, x_arr, y_arr, sigma = y_err)
	m_fit = vals[0]
	b_fit = vals[1]

	return m_fit, b_fit





def plotting(x_arr, y_arr, y_err, m, b, m_fit, b_fit):
	""" Create a beautiful plot of the original user-defined line (with m and b), the discrete data generated from the randomly-generated
	x values and perturbed y-values, and the subsequent best fit line to this discrete data.

	Parameters
	----------
	x_arr : :class: numpy.ndarray
		ten x-values generated from uniformly from a distribution from 0 to 10
	y_arr : :class: numpy.ndarray
		arrau of y-values generated from the user-defined m and b parameters and x_arr, then perterbed by dy values
	y_err : :class: numpy.ndarray
		array of errors set to value of 0.5
	m : :class: float
		user-defined slope of a line
	b : :class: float
		user-defined y-intercept of a line
	m_fit : :class: numpy.float64
		slope parameter of the fitted line
	b_fit : :class: numpy.float64
		y-intercept parameter of the fitted line
	
	Returns:
	----------
	None - plots function and saves figure to current working directory

	"""
	# AJF set up lots of x-points for range [0,10]
	x_cont = np.linspace(0,10,100)

	# AJF set up plot
	# AJF use latex for x and y labels and offset title upwards by 15 points
	fig, ax = plt.subplots(figsize = (12,10))
	ax.set_title('Comparing Best-Fit Line to User-Defined Parameters', fontsize = 15, y = 1, pad = 15)
	ax.set_xlabel('$x$', fontsize = 15)
	ax.set_ylabel('$y$', fontsize = 15)

	# AJF plot discrete data
	ax.errorbar(x_arr, y_arr, yerr = y_err, fmt = 'r.', label = 'Discrete data from random sample')

	# AJF plot fitted line
	y_fit = m_fit * x_cont + b_fit
	ax.plot(x_cont, y_fit, 'c-', label = 'Line constructed from best-fit m and b')

	# AJF plot original m and b line
	og_y = m*x_cont + b
	ax.plot(x_cont, og_y, 'k', linestyle = (0, (5,5)), label = 'Line constructed from user-defined m and b')

	# AJF create legend at specific location, add transparent grid
	ax.grid(True, alpha = 0.25)
	ax.legend(bbox_to_anchor = (0.15, 0.7), loc = 'center left', bbox_transform=fig.transFigure, framealpha = 0.35)

	# AJF add auto-minor ticks (4 per section), increase major tick frequnecy, add ticks on upper and right of plot
	ax.xaxis.set_minor_locator(AutoMinorLocator(5))
	ax.yaxis.set_minor_locator(AutoMinorLocator(5))
	ax.locator_params(axis='both', nbins=15)
	ax.tick_params(which = 'both', top=True, right = True, labelsize = 12)

	# AJF save figure as png in cwd
	plt.savefig('hw0_plot.png', format = 'png')

	# AJF show plot
	plt.show()




def main(): # AJF executes this section first (highest 'shell' of code) and references other built functions as needed
	# AJF initialize command-line inputs for slope and intercept using argparse
	parser = argparse.ArgumentParser(description='Intakes m and b values, plots y values as function of randomly-chosen x values in domain [0,10], fits curve to these coordinates and plots.')
	parser.add_argument('m', metavar = 'm', type = float, help = 'desired slope m of y=mx+b to find y values from')
	parser.add_argument('b', metavar = 'b', type = float, help = 'desired y-intercept b of y=mx+b to find y values from')
	arg = parser.parse_args()
	m = arg.m
	b = arg.b

	# AJF execute three functions to finally produce plot 
	x_arr, y_arr, y_err, og_y_arr, dy_arr = xyyerr(m,b)
	m_fit, b_fit = fitting(x_arr, y_arr, y_err)
	plotting(x_arr, y_arr, y_err, m, b, m_fit, b_fit)

	print(f'\nThis was original m and b: {m}, {b}\nThis is fitted line m and b: {m_fit:.6}, {b_fit:.6}\n')


	

if __name__=='__main__':
    main() 
