#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 29 19:12:14 2023

@author: Max
"""

'''The goal if this code is to hopefully check the difference in distance in 
seperation and maybe use that later to simulate the dark matter of the resulting
child galaxy created 
'''

import numpy as np
import astropy.units as u
import astropy.table as tbl

import matplotlib.pyplot as plt
from ReadFile import Read

#from ParticleProperties import ParticleInfo

def fusion(filename,filename2):
    """This function will calculate the difference in distance between 2 
    galaxy snap files for each particle and see if any particles are close 
    enough to it for them to combine
    
    Inputs:
        filename:'str'
            galaxy snap filename
        filename2:'str'
            other galaxy snap filename
    output:
        newmass:'1darray'
            array containing the mass of all the new combined dark mass
            particles
        newpos:'1darray'
            array containing new positions of combined dark mass particles
    """
    data = Read(filename)[2] #getting only the data array from both galaxy files
    data2 = Read(filename)[2]
    
    index = np.where(data['type'] == 2) #only grabbing dark matter halo particles
    
    x1=data['x'][index] * 3.086e19
    y1=data['y'][index] * 3.086e19
    z1=data['z'][index] * 3.086e19
    x2=data['x'][index] * 3.086e19
    y2=data['y'][index] * 3.086e19
    z2=data['z'][index] * 3.086e19
    
    newpos = np.zeros([len(x1),3])
    
    print(len(x1))
    for i in range(0,len(x1),1):
        if np.sqrt((x1[i]-x2[i])**2+(y1[i]-y2[i])**2+(z1[i]-z2[i])**2) <= .1:
            xav = np.average([x1[i],x2[i]])
            yav = np.average([y1[i],y2[i]])
            zav = np.average([z1[i],z2[i]])
            newpos[i,0] , newpos[i,1],newpos[i,2] = xav ,yav , zav
    return newpos

a = fusion('MW_000.txt','M31_000.txt')
b = fusion('MW_200.txt','M31_200.txt')
plt.figure()
plt.xlim(-30,30)
plt.ylim(-30,30)
plt.plot(a[0],a[1],'.',color='red',label='000')
plt.plot(b[0],b[1],'.',color='blue',label='200')
plt.legend()
plt.show

