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




