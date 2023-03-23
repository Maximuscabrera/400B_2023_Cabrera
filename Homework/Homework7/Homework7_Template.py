
# # Homework 7 Template
# 
# Rixin Li & G . Besla
# 
# Make edits where instructed - look for "****", which indicates where you need to 
# add code. 




# import necessary modules
# numpy provides powerful multi-dimensional arrays to hold and manipulate data
import numpy as np
# matplotlib provides powerful functions for plotting figures
import matplotlib.pyplot as plt
# astropy provides unit system and constants for astronomical calculations
import astropy.units as u
import astropy.constants as const
# import Latex module so we can display the results with symbols
from IPython.display import Latex

# **** import CenterOfMass to determine the COM pos/vel of M33
from CenterOfMass import *

# **** import the GalaxyMass to determine the mass of M31 for each component
from GalaxyMass import *

# # M33AnalyticOrbit




class M33AnalyticOrbit:
    """ Calculate the analytical orbit of M33 around M31 """
    
    def __init__(self,filename): # **** add inputs
        """ **** ADD COMMENTS """

        ### get the gravitational constant (the value is 4.498502151575286e-06)
        self.G = const.G.to(u.kpc**3/u.Msun/u.Gyr**2).value
        
        ### **** store the output file name
        self.filename = filename
        
        ### get the current pos/vel of M33 
        # **** create an instance of the  CenterOfMass class for M33 
        M33_COM = CenterOfMass("M33_000.txt", 2)
        # **** store the position VECTOR of the M33 COM (.value to get rid of units)
        M33x , M33y, M33z = (M33_COM.COM_P(0.1)).value
        # **** store the velocity VECTOR of the M33 COM (.value to get rid of units)
        M33vx, M33vy, M33vz = (M33_COM.COM_V(M33_COM.COM_P(0.1)[0],M33_COM.COM_P(0.1)[1],M33_COM.COM_P(0.1)[2])).value
        
        ### get the current pos/vel of M31 
        # **** create an instance of the  CenterOfMass class for M31 
        M31_COM = CenterOfMass("M31_000.txt", 2)
        # **** store the position VECTOR of the M31 COM (.value to get rid of units)
        M31x , M31y, M31z = (M31_COM.COM_P(0.1)).value
        # **** store the velocity VECTOR of the M31 COM (.value to get rid of units)
        M31vx, M31vy, M31vz = (M31_COM.COM_V(M31_COM.COM_P(0.1)[0],M31_COM.COM_P(0.1)[1],M31_COM.COM_P(0.1)[2])).value
        
        ### store the DIFFERENCE between the vectors posM33 - posM31
        
        r = [M33x-M31x,M33y-M31y,M33z-M31z]
        v = [M33vx-M31vx,M33vy-M31vy,M33vz-M31vz]
        
        # **** create two VECTORs self.r0 and self.v0 and have them be the
        # relative position and velocity VECTORS of M33
        self.r0 = r
        self.v0 = v
        ### get the mass of each component in M31 
        ### disk
        # **** self.rdisk = scale length (no units)
        self.rdisk = 5
        # **** self.Mdisk set with ComponentMass function. Remember to *1e12 to get the right units. Use the right ptype
        self.Mdisk = ComponentMass("M31_000.txt",2) *1e12
        ### bulge
        # **** self.rbulge = set scale length (no units)
        self.rbulge = 1
        # **** self.Mbulge  set with ComponentMass function. Remember to *1e12 to get the right units Use the right ptype
        self.Mbulge = ComponentMass("M31_000.txt",3) *1e12
        # Halo
        # **** self.rhalo = set scale length from HW5 (no units)
        self.rhalo = 62
        # **** self.Mhalo set with ComponentMass function. Remember to *1e12 to get the right units. Use the right ptype
        self.Mhalo =ComponentMass("M31_000.txt",1) *1e12
    
    
    def HernquistAccel(self, M , r_a , r): # it is easiest if you take as an input the position VECTOR 
        """This function will calculate the Hernquist acceleration and return it
        as a unitless vector
        Inputs
        ------
            M:'Float'
                Total halo or bulge mass
            r_a:'int'
                Scale length
            r:'1d array'
                position array containing x,y,z coords
        Outputs
        -------
            Hern:'1da rray'
                The hernquist acceleration stored as a vector array"""
        
        ### **** Store the magnitude of the position vector
        rmag = np.linalg.norm(r) 
        
        ### *** Store the Acceleration
        
        Hern =  -(self.G * M)/(rmag*(r_a+rmag)**2) * r #follow the formula in the HW instructions
        # NOTE: we want an acceleration VECTOR so you need to make sure that in the Hernquist equation you 
        # use  -G*M/(rmag *(ra + rmag)**2) * r --> where the last r is a VECTOR 
        
        return Hern
    
    
    
    def MiyamotoNagaiAccel(self,M,r_d,r):# it is easiest if you take as an input a position VECTOR  r 
        """This function will calculate the disk acceleration using the
        miyamoto nagai profile
        Inputs
        ------
            M:'float'
                Total disk mass
            r_d:'int'
                scale length of disk
            r:'1d array'
                position array
        Outputs
        -------
            Miya:'1darray'
                array containg miyamoto nagai profile acceleration
        """

        
        ### Acceleration **** follow the formula in the HW instructions
        # AGAIN note that we want a VECTOR to be returned  (see Hernquist instructions)
        # this can be tricky given that the z component is different than in the x or y directions. 
        # we can deal with this by multiplying the whole thing by an extra array that accounts for the 
        # differences in the z direction:
        #  multiply the whle thing by :   np.array([1,1,ZSTUFF]) 
        # where ZSTUFF are the terms associated with the z direction
        x,y,z = r[0],r[1],r[2]
        
        R = np.sqrt((x**2)+(y**2))
        z_d = r_d / 5.0
        B = r_d + np.sqrt((z**2)+(z_d**2))
        
        pot = -(self.G * M) / np.sqrt((R**2)+(B**2))
        
        a = pot * r
        a[2] = a[2] * B / (np.sqrt((z**2)+z_d**2)) #correction for z coordinate
        
        Miya = a
        
        return Miya
        # the np.array allows for a different value for the z component of the acceleration
     
    
    def M31Accel(self , r): # input should include the position vector, r
        """ This function will take the sum all of the accelerations together
        of M31
        
        Inputs
        ------
            r:'1d array'
                positional array
        Outputs:
        -------
            atot'1darray'
                total acceleration of all parts of M31 summed together 
        """
        ahalo = self.HernquistAccel(self.Mhalo,self.rhalo,r)
        abulge = self.HernquistAccel(self.Mbulge,self.rbulge,r)
        adisk = self.MiyamotoNagaiAccel(self.Mdisk,self.rdisk,r)
        ### Call the previous functions for the halo, bulge and disk
        # **** these functions will take as inputs variable we defined in the initialization of the class like 
        # self.rdisk etc. 
        c = np.add(ahalo,abulge)
        atot = np.add(c,adisk)
            # return the SUM of the output of the acceleration functions - this will return a VECTOR 
        return atot
    
    
    
    def LeapFrog(self,r,v,dt): # take as input r and v, which are VECTORS. Assume it is ONE vector at a time
        """This funtion will update the position and velocity vectors of the galaxy
        through time
        
        Inputs:
        -------
            r:'1d array'
                positional vector
            v:'1d array'
                velocity vector
            dt:'float'
                time step size
        outputs
        -------
            vnew:'1d array'
                new velocity vector
            rnew:'1d array'
                new positional vector
            
            """
        
        # predict the position at the next half timestep
        rhalf = r + v * (dt/2)
        
        # predict the final velocity at the next timestep using the acceleration field at the rhalf position 
        vnew = v + self.M31Accel(rhalf) * dt
        
        # predict the final position using the average of the current velocity and the final velocity
        # this accounts for the fact that we don't know how the speed changes from the current timestep to the 
        # next, so we approximate it using the average expected speed over the time interval dt. 
        rnew = rhalf + vnew * (dt/2)
        
        return rnew,vnew # **** return the new position and velcoity vectors
    
    
    
    def OrbitIntegration(self, t0, dt, tmax):
        """This function will compute the orbit of a galaxy given a start and end
         time using the leapfrog integrater
         Inputs:
         -------
             t0:'float'
                 initial time in gyr
             dt:'float'
                 time step size
            tmax:'float'
                final time in gyr
         
        """
        # initialize the time to the input starting time
        t=t0
        # initialize an empty array of size :  rows int(tmax/dt)+2  , columns 7
        orbit = np.zeros([int(tmax/dt)+2,7])
        
        # initialize the first row of the orbit
        orbit[0] = t0, *tuple(self.r0), *tuple(self.v0)
        # this above is equivalent to 
        # orbit[0] = t0, self.r0[0], self.r0[1], self.r0[2], self.v0[0], self.v0[1], self.v0[2]
        
        
        # initialize a counter for the orbit.  
        i = 1 # since we already set the 0th values, we start the counter at 1
        
        # start the integration (advancing in time steps and computing LeapFrog at each step)
        while (t < tmax):  # as long as t has not exceeded the maximal time 
            
            # **** advance the time by one timestep, dt
            t += dt
            # **** store the new time in the first column of the ith row
            orbit[i,0] = t
            
            # ***** advance the position and velocity using the LeapFrog scheme
            # remember that LeapFrog returns a position vector and a velocity vector  
            # as an example, if a function returns three vectors you would call the function and store 
            # the variable like:     a,b,c = function(input)
            rold = orbit[i-1, 1:4] #grabbing positional columns of previous iteration
            vold = orbit[i-1, 4:8] #grabbing velocity columns for previous iteration
            
            rnew, vnew = self.LeapFrog(rold, vold, dt)
            # ****  store the new position vector into the columns with indexes 1,2,3 of the ith row of orbit
            # TIP:  if you want columns 5-7 of the Nth row of an array called A, you would write : 
            # A[n, 5:8] 
            orbit[i, 1:4] = rnew
            orbit[i, 4:8] = vnew
            # where the syntax is row n, start at column 5 and end BEFORE column 8
            
            
            # ****  store the new position vector into the columns with indexes 1,2,3 of the ith row of orbit
            
            print(i)
            # **** update counter i , where i is keeping track of the number of rows (i.e. the number of time steps)
            i +=1
            
        
        
        # write the data to a file
        np.savetxt(self.filename, orbit, fmt = "%11.3f"*7, comments='#', 
                   header="{:>10s}{:>11s}{:>11s}{:>11s}{:>11s}{:>11s}{:>11s}"\
                   .format('t', 'x', 'y', 'z', 'vx', 'vy', 'vz'))
        
        # there is no return function

a=M33AnalyticOrbit('test')
a.OrbitIntegration(0, 0.5, 10)
