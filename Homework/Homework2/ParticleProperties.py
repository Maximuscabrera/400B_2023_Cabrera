#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 24 19:46:17 2023

@author: Max
"""

import numpy as np
import astropy.units as u
from ReadFile import *

def ParticleInfo(filename,ptype,number):
    
    file = Read(filename)
    data = file[2]
    index = np.where(data['type'] == ptype)
    particle = number-1
   
    x = (data['x'][index])[particle]
    y = (data['y'][index])[particle]
    z = (data['z'][index])[particle]
   
    d = np.sqrt((x**2)+(y**2)+(z**2))
    d = np.around(d,3)*u.kpc
   
    vx = (data['vx'][index])[particle]
    vy = (data['vy'][index])[particle]
    vz = (data['vz'][index])[particle]
   
    vmag = np.sqrt((vx**2)+(vy**2)+(vz**2))
    vmag = (np.around(vmag,3))*u.km/u.s
   
    mass = ((data['m'][index])[particle])
    mass = mass * (10**-10)*u.Msun
    
    return d,vmag,mass
    
a = ParticleInfo("MW_000.txt",2,100)

dlyr = (a[0]).to(u.lyr)
print("The 3D distance is {} or {} \
,the 3D velocity is {} \
and mass is {}"\
      .format(a[0],dlyr,a[1],a[2]))