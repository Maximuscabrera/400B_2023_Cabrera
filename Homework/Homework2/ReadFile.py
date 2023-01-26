#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 24 15:07:02 2023

@author: Max
"""

import numpy as np
import astropy.units as u #importing necessary pacakges

def Read(filename): #creating our function that will read files
    file = open(str(filename),"r") #opens file in read mode while also converting into string if not already one
    line1 = file.readline() #reads first line of file for data extraction
    label, value = line1.split()
    time = float(value)*u.Myr
    line2 = file.readline()
    label, value2 = line2.split()
    total = float(value2)
    data = np.genfromtxt(filename,dtype=None,names=True,skip_header=3,)
    return time,total,data

#a = (Read("MW_000.txt"))
#data = a[2]
#print(data['type'][1])