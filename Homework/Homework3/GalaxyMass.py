#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  1 18:32:36 2023

@author: Max
"""

import numpy as np
import astropy.units as u #importing necessary libraries
from ReadFile import * #importing ReadFile script that was written earlier
from tabulate import tabulate

def ComponentMass(filename,Type):
    """ 
    This function will take in a file and use the Read function created before
    to extract crucial data that will allow us to calculate the total mass
    of a given particle type
    
    Inputs:
        filename:'string'
        This will be the name of the data file being read
        Type:'int'
        This will be the number corresponding to the particle type that we 
        are looking for
        
    Outputs:
        roundmass:'float'
        This will be the total mass of a given particle type in units of 1e12 
        solar mass
    """
    
    file= Read(filename) #reading in data from given file using read funtion
    data = file[2]      #obtaining full data set table from read funtion
    index = np.where(data['type'] == Type) #taking all the data of particles 
    #of the desired type
    mass = ((data['m'][index])) #creating an array of all the masses of the 
    #particles of that given type
    totalmass= np.sum(mass) / 100 #summing all of the masses together and 
    #converting from 1e10 solar mass to 1e12 solar mass
    roundmass = np.round(totalmass,3) #rounding to the third decimal place
    
    return roundmass

# from this line until 63 we are calculating total mass for each particle type
# of 3 different galaxies

m33_1 = ComponentMass('M33_000.txt', 1)
m33_2 = ComponentMass('M33_000.txt', 2)
m33_3 = ComponentMass('M33_000.txt', 3)
m33_total = m33_1 + m33_2 + m33_3 #summing the masses together
m33_fbar = (m33_2 + m33_3) / m33_total #dividing stellar mass by total mass

m31_1 = ComponentMass('M31_000.txt', 1)
m31_2 = ComponentMass('M31_000.txt', 2)
m31_3 =ComponentMass('M31_000.txt', 3)
m31_total = m31_1 + m31_2 + m31_3
m31_fbar = (m31_2 + m31_3) / m31_total


mw_1 = ComponentMass('MW_000.txt', 1)
mw_2 = ComponentMass('MW_000.txt', 2)
mw_3 = ComponentMass('MW_000.txt', 3)
mw_total = mw_1 + mw_2 +mw_3
mw_fbar = (mw_2 + mw_3) / mw_total

#mass calculations end here

#creating a table using the tabulate package to present the data gathered
table = [['Galaxy Name', 'Halo Mass(Msun*1e12)', 'Disk Mass(Msun*1e12)'
         ,'Bulge Mass(Msun*1e12)','Total Mass(Msun*1e12)','f_bar'], 
        ['Milky Way', mw_1, mw_2,mw_3,mw_total,mw_fbar],  
        ['M31', m31_1, m31_2,m31_3,m31_total,m31_fbar], 
        ['M33', m33_1, m33_2,m33_3,m33_total,m33_fbar]]

print(tabulate(table, headers='firstrow', tablefmt='fancy_grid'))

