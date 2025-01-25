import numpy as np
import matplotlib.pyplot as plt
import argparse

def func1(fn):

	"""

	Plot datafile in cartesian space; plot discrete data points (i.e., (1,1), (2,2), etc.)
	
	"""

	data = np.loadtxt(fn, dtype = int, delimiter = ' ') # loads in text
	c1, c2 = data[:,0], data[:,1]
	fig = plt.figure(figsize = (12,10))
	plt.plot(c1,c2, 'r-')
	plt.plot(c1, c2, 'y+', markersize = 20)
	plt.title('Task 1: Discrete Data')
	plt.savefig('fig_c1_1.pdf', format = 'pdf') # saves figure as pdf
	plt.show()
	return



def func2(x_val):

	"""

	Plug in user-defined float into y-function, get result

	"""

	y = x_val**2 + 3*x_val + 8 # uses function to find y value of input x value
	return y



def func3(lims):

	"""

	Plot y-function as a function of a user-defined x-domain. 
	
	"""

	x_arr = np.linspace( (int(lims[0])), (int(lims[1])), 100 ) # creates array based on user input with 100 discrete points
	y = x_arr**2 + 3*x_arr + 8
	fig = plt.figure(figsize = (12,10))
	plt.plot(x_arr,y, 'k-')
	plt.title('Task 3: Plot a Function of an Array of Values')
	plt.savefig('fig_c1_2.pdf', format = 'pdf')
	plt.show() 
	return




def main(): # executes this section and references other built functions as needed
	parser = argparse.ArgumentParser(description='Reads in datafile and plot the second column as a function of the first; reads in x-value to find function value; plots function y = x^2 + 3x + 8')
	parser.add_argument('filename', metavar = 'filename', help = 'Name of file from which to pull data to plot.')
	parser.add_argument('val', metavar = 'val', type = float, help = 'x value to plug into y = x^2 + 3x + 8')
	parser.add_argument('-lim', metavar = '--limits', nargs = '+', help = 'low high x values to plug into y = x^2 + 3x + 8 (format: low high ; i.e., -l 2 10 )') # nargs = + allows for multiple command-line inputs
	arg = parser.parse_args()

	fn = arg.filename   
	x_val = arg.val  
	lims = arg.lim

	func1(fn)
	y = func2(x_val)
	print(f'\nTask 2: This is the y value for x = {x_val}: {y}\n')
	func3(lims)
	
if __name__=='__main__':
    main() 
   
