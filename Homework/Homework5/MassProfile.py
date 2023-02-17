#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 16 17:03:12 2023

@author: Max
"""
import numpy as np
import astropy.units as u
import astropy.table as tbl
from astropy.constants import G

from ReadFile import *
from CenterOfMass_Solution import *

G = G.to(u.kpc * u.km**2/u.s**2/u.Msun)

class MassProfile:
#class to define a mass profile of a given galaxy and snap number
    
    def __init__(self,galaxy,snap):
        """This class will define our mass profile of a given galaxy and snap

    Inputs:
        galaxy: string
            a string with the galxies name eg. MW,M31
        snap: int
            snapshot number eg. 0,1,etc
            
            """
        #adding snap number to 000 str
        ilbl = '000' + str(snap)
        #removing all but the last 3 digits
        ilbl = ilbl[-3:]
        self.filename = "%s_"%(galaxy) +ilbl + '.txt'
        self.gname = galaxy
        #reading file and seperating out data
        self.time , self.total, self.data = Read(self.filename)
        
        #storing the xyz coordinates with appropriate units
        self.x = self.data['x'] * u.kpc
        self.y = self.data['y'] * u.kpc
        self.z = self.data['z'] * u.kpc
        #storing mass data but not assinging units yt
        self.m = self.data['m']
        
    def MassEnclosed(self,ptype,r):
        """
        This function will calculate enclosed mass of a given type of particle
        at an array of radii in units of Msun
    
    Inputs:
        ptype: int
            number type of particle eg. 1,2,3
        r: 1d array
            array of radii magnitudes
    Outputs:
        mass_enc: 1d array
            array containing mass enclosed in the given radii
        """
        
        #first defining center of mass of galaxy with .1kpc tolerance
        
        COM_P = CenterOfMass(self.filename,ptype).COM_P(0.1) 
        #making index for data filtering
        index = np.where(self.data['type'] == ptype)
        #filtering x y z and getting values relative to COM
        x1 = self.x[index] - COM_P[0]
        y1 = self.y[index] - COM_P[1]
        z1 = self.z[index] - COM_P[2]
        
        #getting magnitude and removing units so it is in radii
        r1 = (np.sqrt(x1**2 + y1**2 + z1**2)) / u.kpc
        m = self.m[index]
        #making zeros array that we will add to
        mass_enc = np.zeros(len(r))
        
        for i in range(len(r)):
            
            
            #creating a filter for all particles within the set radius
            index2 = np.where(r1 < r[i])      
            #filling in the specific slot of the 
            mass_enc[i] = np.sum(m[index2])
       #giving appropriate units for the mass enclosed
        mass_enc = mass_enc * u.Msun * 1e10

        return mass_enc
    
    def MassEnclosedTotal(self,r):
        """
        This function will calculate the total enclosed mass within an array 
        of radii
        
    Inputs:
        r: 1d array
            array of radii magnitudes
    
    Outputs:
        mass_enc_tot: 1d array
            array containing all mass enclosed
        """
        #we will be doing this for all the different types of particles
        type1 = np.where(self.data['type'] == 1)
        type2 = np.where(self.data['type'] == 2)
        COM_P1 = CenterOfMass(self.filename,1).COM_P(0.1)
        COM_P2 = CenterOfMass(self.filename,2).COM_P(0.1)
        x1 = self.x[type1] - COM_P1[0]
        y1 = self.y[type1] - COM_P1[1]
        z1 = self.z[type1] - COM_P1[2]
        r1 = (np.sqrt(x1**2 + y1**2 + z1**2)) / u.kpc
        m1 = self.m[type1]
        #getting x y z and m values for type 1 particles 
        x2 = self.x[type2] - COM_P2[0]
        y2 = self.y[type2] - COM_P2[1]
        z2 = self.z[type2] - COM_P2[2]
        r2 = (np.sqrt(x2**2 + y2**2 + z2**2)) / u.kpc
        m2 = self.m[type2]
        #getting x y z and m values for type 1 particles 
        
        mass_enc1 = np.zeros(len(r))
        mass_enc2 = np.zeros(len(r))
        
        for i in range(len(r)):
            
            index1 = np.where(r1 <r[i]) #creating radii index for type 1 part
            
            mass_enc1[i] = np.sum(m1[index1]) #adding mass values to array
            
            index2 = np.where(r2 <r[i]) #creating radii index for type 2 part
           
            mass_enc2[i] = np.sum(m2[index2])
        if self.gname == 'M33':
            mass_enc_tot = (mass_enc1 + mass_enc2) * u.Msun * 1e10
        else:
            type3 = np.where(self.data['type'] == 3)
            COM_P3 = CenterOfMass(self.filename,3).COM_P(0.1)  
            x3 = self.x[type3] - COM_P3[0]
            y3 = self.y[type3] - COM_P3[1]
            z3 = self.z[type3] - COM_P3[2]
            r3 = (np.sqrt(x3**2 + y3**2 + z3**2)) / u.kpc
            m3 = self.m[type3]          
            mass_enc3 = np.zeros(len(r))
            for i in range(len(r)):
                
                index3 = np.where(r3 <r[i]) #creating radii index for type 3 part
                
                mass_enc3[i] = np.sum(m3[index3])
            mass_enc_tot = (mass_enc1 + mass_enc2 + mass_enc3) * u.Msun * 1e10
        
        return mass_enc_tot

    def hernquist_mass(r,a, m_halo):
    
        """ 
    Function that defines the Hernquist 1990 mass profile 
    Inputs:
        r: astropy quantity
            Galactocentric distance in kpc
        a: astropy quantity
            scale radius of the Hernquist profile in kpc
        m_halo: float
            total halo mass in units of 1e12 Msun 
    Ouputs:
        mass:  astropy quantity
        total mass within the input radius r in Msun 
        """
        mass = m_halo*1e12*r**2/(a+r)**2*u.Msun # Hernquist mass  
        return mass
    def CircularVelocity(self,ptype,r):
        """ This funtion will calculate the circular velocity
        using the enclosed mass at each radius assumig spherical symetry
        
        v_c = (GM/R)**2
    Inputs:
        r: 1d array
            array of different radii
        ptype: int
            particle type eg 1,2,3
    Output:
        v_c: 1d array
            array of velocities at different radii in km/s
            
        
        """
        v_c = np.zeros(len(r))
        
        m = self.MassEnclosed(ptype, r)
        
        rkpc = r * u.kpc
       
        for i in range(len(r)):
            
            a = (m[i] * G / rkpc[i])
            a = np.sqrt(a) * u.s / u.km #need to remove units to add to array

            v_c[i] = a


        v_c = v_c * u.km / u.s
        return v_c
    
    def CircularVelocityTotal(self,r):
         """ This funtion will calculate the  total circular velocity
        using the enclosed mass at each radius assumig spherical symetry
        
        v_c = (GM/R)**2
    Inputs:
        r: 1d array

    Output:
        v_c: 1d array
            array of velocities at different radii in km/s
    """
         v_c = np.zeros(len(r))
    
         m = self.MassEnclosedTotal(r)
         
         rkpc = r * u.kpc
       
         for i in range(len(r)):
            
            a = (m[i] * G / rkpc[i])
            a = np.sqrt(a) * u.s / u.km #need to remove units to add to array

            v_c[i] = a
        
         return v_c
    def HernquistVCirc(r,a,m_halo):
        """ This funtion will calculate the circular velocity from the
        Hernquist mass profile
    Inputs:
        r: astropy quantity
            Galactocentric distance in kpc
        a: astropy quantity
            scale radius of the Hernquist profile in kpc
        m_halo: float
            total halo mass in units of 1e12 Msun 
    Output:
        v_c: astropy quantity
            circular velocity in km / s
        
        """
        
        mass = m_halo*1e12*r**2/(a+r)**2*u.Msun # Hernquist mass  
        
        v_c = np.sqrt(G*mass / r)
        
        return v_c
    
a = MassProfile('MW', 0)
r = np.arange(0.25,30.5,1.5)

b = a.CircularVelocityTotal(r)
print(b)
