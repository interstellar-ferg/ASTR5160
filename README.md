# ASTR5160
Code for ASTR-5160 at the University of Wyoming for Spring 2025, Year 1 PhD

The following is a brief overview/summary of topics covered each week:

Week1:
- 1. Python introduction
    - basics on where to code, setting up directories, etc.
- 2. Github intro
    - setting up repos, command-line commands, how to set one up, basics

Week2: 
- 3. Recarrays
    - how to create and read recarrays; their purpose in astro coding
- 4. Review of astro coordinates
    - a brief review of sidereal time, ra/dec, airmass, and how to use astropy library to manipulate these things

Week3:
- 5. Review of astro coordinates; converting coordinates
    - review of precession/equitorial coordinate systems, changing frames (cartesian, galactic, etc)
- 6. Dust maps
    - how to code dust maps in python; calculate extinction values and plot projection of dust density in sky

Week4:
- 7. Plotting projections (like Aitoff, lambert)
    - plot ra and dec values on an aitoff projection; equal density of points on a sphere; find extinction value across projection
- 8. Distance on sphere
    - get comfortable using astropy separation, search_around_sky to calculate points within a certain distance of target, etc.

Week5:
- 9. Area on sphere/HEALPix
    - Calculating areas of regions on a sphere and using HEALPiz binning of equal area pixels to note exactly where targets are on sphere
- 10. Sphereical caps
    - construct caps (basically, regions on sphere) that, when upon intersection, can be used to construct regions/masks on the surface of sphere

Week6: 
- 11. Mangle (masks, creating intersection of caps)
    - explains how to use module Mangle for creating general masks (intersection of caps to create certain shapes/regions on sphere)
- 12. General masking applications
    - continuation of 11; explains general uses of masking, plots masks and shapes bound by ra/dec caps

Week7: 
- 13. GitHib (uploading to repository, new branches)
    - explains commands for setting up new branches, uploading code to github, comments, commits/add
- 14. Python using functions from other scripts
    - how to import code from other scripts and use it

Week8:
- 15. SQL Queries
    - search SDSS surveys and large databases with SQL searches; load in data
- 16. Cross-Matching Surveys
    -  find the common objects between two surveys; usually takes on set of coordinates and finds objects within 1 arcsec of those coordinates

Week9:
- BREAK

Week10:
- 17. Magnitude systems
    - understand and convert magnitudes and fluxes (vega and AB mags, nanomaggies, etc) - understand UBVRI, ugriz
- 18. Classification methods (color cuts)
    - convert fluxes/mags and find color cuts to classify data

Week11:
- BREAK

Week12:
- 21. Flagging bad data
    - use flags and bitmasks to filter out bad data from Legacy sweeps (overexposure, type, other spuruous images)
- 22. Adventures in machine learning (INCOMPLETE)
    - use k-NN mapping techniques to make a machine learning algorithm (INCOMPLETE)

Week13:
- 23. Fitting a line: Chi-Squared
    - fit data with a model fit determined by chi-square method of reduction
- 24. Fitting correlated data: Chi-Squared (INCOMPLETE)
    - fit data where each parameter is not necessarily independent using the chi-squared method (INCOMPLETE)

Week 14:
- 25. Using the likelihood function and bayesian stats to fit data - MCMC and Metro-Hastings (INCOMPLETE)
    - create a metro-hastings style algorithm for fitting a linear model to data (INCOMPLETE)
- 26. The emcee package
    - use the python emcee package to fit a linear model to data (find best fit parameters m and b using emcee)



