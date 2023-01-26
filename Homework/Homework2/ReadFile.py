#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 24 15:07:02 2023

@author: Max

This Script will serve as a stepping stone for later scripts,it serves the 
basic funtion of reading a file that contains particle data and returns the 
time in Myr, total number of particles, and and array that contains the 
particle's type, position, velocity and mass values.

"""

import numpy as np
import astropy.units as u #importing necessary pacakges

def Read(filename): #creating our function that will read files
    file = open(str(filename),"r") #opens file in read mode while & converting into string if not already one
    line1 = file.readline() #reads first line of file for data extraction
    label, value = line1.split() #splits that first line of data into 2 variables, the second is the 
    # time value we want
    time = float(value)*u.Myr #takes that time value and provides it with the appropriate units
    line2 = file.readline() #reads second line in file for data extraction
    label, value2 = line2.split() #splits that second line into 2 parts so we can get the data we need without
    #grabbing the label as well
    total = float(value2) #records the number value of all the particles 
    data = np.genfromtxt(filename,dtype=None,names=True,skip_header=3)
    """
    Line[22] notes: using np.genfromtxt we are able to generate a data array that uses the labels 
    within the original text document in order to arrange all of the data, it skips the first 3
    lines as those pertain to the values that we already recorded in earlier 
    lines or are not relevant to what we need.
    """
    return time,total,data

#a = (Read("MW_000.txt"))
#data = a[2]
#print(data['type'][1])
"""
these last 3 lines[26-28] were leftovers from testing the script, I chose to keep them
in case I need to come back and review the script and test it for more use
"""