#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 24 19:46:17 2023

@author: Max

This script is building on the Readfile script that I wrote before, this one
will use the Read function in order to manipulate the data and calculate the
velocity and distance magnitudes as well as extract the mass of a particle
given its type and number in series
"""

import numpy as np
import astropy.units as u #importing necessary libraries
from ReadFile import * #importing ReadFile script that was written earlier

def ParticleInfo(filename,ptype,number):    #defining our function
    
    file = Read(filename) #using Read func to return data value for use
    data = file[2] #data array is [2] in values returned, defining this array
    #for use later
    
    index = np.where(data['type'] == ptype) #creating a filter that only 
    #contain particles that match the desired type for future use
    
    particle = number-1 #this is a syntax correction so that when we request
    #the 1st particle and input 1 for our number it will return the 0th
    #element in the array which for all intents and purposes is the 1st one
   
    x = (data['x'][index])[particle]
    y = (data['y'][index])[particle]
    z = (data['z'][index])[particle] #using the filter created earlier to grab
    #the different position values for the particle that matches the desired
    #type and position in the array
   
    d = np.sqrt((x**2)+(y**2)+(z**2)) #calculating the 3D distance magnitude
    #using basic magnitude calculation to get 3D magnitude which entails
    #taking the square root of the sum of all the position values squared
    
    d = np.around(d,3)*u.kpc #rounding 3D distance magnitude to 3rd decimal &
    # giving it appropriate units of kpc
   
    vx = (data['vx'][index])[particle]
    vy = (data['vy'][index])[particle]
    vz = (data['vz'][index])[particle] #using filter in similar manner as in 
    #lines 31-33 but instead of grabbing position values we grab velocities
   
    vmag = np.sqrt((vx**2)+(vy**2)+(vz**2)) #taking square root of the sum of
    #velocity values squared in order to get velocity magnitude
    
    vmag = (np.around(vmag,3))*u.km/u.s #rounding velocity magnitude to 3 
    #decimals and giving it units of km/s
   
    mass = ((data['m'][index])[particle]) #using filter to obtain mass of 
    #desired particle
    
    mass = mass * (10**-10)*u.Msun #since the original data is in units of
    # 10e10 Msun and the desired units is just Msun we have to multiply by a
    #factor of 10e-10
    
    return d,vmag,mass #returning distance & velocity magnitude as well as 
#mass of particle

"""
a = ParticleInfo("MW_000.txt",2,100)
dlyr = np.around((a[0]).to(u.lyr),3)
print("The 3D distance is {} or {} \
,the 3D velocity is {} \
and mass is {}"\
      .format(a[0],dlyr,a[1],a[2]))
 """
 #left over testing code to see if the funtion operates as we wanted it to
 #leaving this in the code in case I need to come back and make sure it still
 #works