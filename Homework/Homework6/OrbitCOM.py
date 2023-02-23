

# Homework 6 Template
# G. Besla & R. Li




# import modules
import numpy as np
import astropy.units as u
from astropy.constants import G
import os
# import plotting modules
import matplotlib.pyplot as plt
import matplotlib

# my modules
from ReadFile import Read
# Step 1: modify CenterOfMass so that COM_P now takes a parameter specifying 
# by how much to decrease RMAX instead of a factor of 2
from CenterOfMass2 import CenterOfMass





def OrbitCOM(galaxy,start,end,n=5):
    """function that loops over all the desired snapshots to compute the COM pos and vel as a function of time.
    inputs:
          galaxy: 'string'
              name of galaxy eg 'MW'
         start: 'int'
             the number of the first snapshot to be read in
        end: 'int'
            the number of the last snapshot to be read in
        n: 'int, default of 5'
            the interval at which we will return the COM
    outputs: 
        fileout: 'file'
            a file containing the time, COM postion and velocity vectors of
            galaxy in each snapshot
    """
    
    # compose the filename for output
    fileout = 'orbit_{}.txt'.format(galaxy)
    #  set tolerance and VolDec for calculating COM_P in CenterOfMass
    delta = 0.1
    
    if galaxy == 'M33': #quick check to give appropriate volDec value for M33
        VolDec = 4
    else: #for any other galxies volDec = 2
        VolDec = 2
    # for M33 that is stripped more, use different values for VolDec    
    # generate the snapshot id sequence 
    snap_id = np.arange(start,end+n,n)
    #print(snap_id)
    # it is always a good idea to also check if the input is eligible (not required)

    # initialize the array for orbital info: t, x, y, z, vx, vy, vz of COM
    orbit = (np.zeros([len(snap_id),7]))
    # a for loop 
    
    for i in range(len(snap_id)):
        
        ilbl = '000' +str(snap_id[i])
        
        ilbl = ilbl[-3:]
        
        filename ='%s_'%(galaxy) +ilbl +'.txt'
        
        # compose the data filename (be careful about the folder)
        
        # Initialize an instance of CenterOfMass class, using disk particles
        
        GalaxyCom = CenterOfMass(filename,2)
        
        # Store the COM pos and vel. Remember that now COM_P required VolDec
        orbit[i,0] = (GalaxyCom.time).value / 1e3 #Storing snap time in first column
        
        orbit[i,1] , orbit[i,2] , orbit[i,3] = (GalaxyCom.COM_P(delta,VolDec)).value #getting COM_P
        
        x , y , z = GalaxyCom.COM_P(delta,VolDec) #getting values for COM_P with units for COM_V calc
        
        orbit[i,4] , orbit[i,5] , orbit[i,6] =(GalaxyCom.COM_V(x,y,z)).value #getting COM_V and storing values
        
        
        # print snap_id to see the progress
        
        print(snap_id[i])
        
        #save all data to text file
    np.savetxt(fileout, orbit, fmt = "%11.3f"*7, comments='#',\
               header="{:>10s}{:>11s}{:>11s}{:>11s}{:>11s}{:>11s}{:>11s}".format\
                   ('t', 'x', 'y', 'z', 'vx', 'vy', 'vz'))






# Recover the orbits and generate the COM files for each galaxy
# read in 800 snapshots in intervals of n=5
# Note: This might take a little while - test your code with a smaller number of snapshots first! 

#os.chdir('/Users/Max/400B/400B_2023_Cabrera/Homework/Homework6/MW')
#OrbitCOM('MM',0, 800)

#os.chdir('/Users/Max/400B/400B_2023_Cabrera/Homework/Homework6/M31')
#OrbitCOM('M31',0, 800)

#os.chdir('/Users/Max/400B/400B_2023_Cabrera/Homework/Homework6/M33')
#OrbitCOM('M33',0, 800)


# Read in the data files for the orbits of each galaxy that you just created
# headers:  t, x, y, z, vx, vy, vz
# using np.genfromtxt

########
"""
MWorb = np.genfromtxt('orbit_MW.txt',dtype=None,names=True,skip_header=0)
M31orb = np.genfromtxt('orbit_M31.txt',dtype=None,names=True,skip_header=0)
M33orb = np.genfromtxt('orbit_M33.txt',dtype=None,names=True,skip_header=0)
"""
########

# function to compute the magnitude of the difference between two vectors 
def Vectordiff(orb1,orb2):
    """
    This funtion will calculate the difference in magnitude of two vectors from the
    orbit.txt files generated from the OrbitCom funtion
    
    Inputs:
        orb1: nparray
            an array containing all data generated in the txt file for a given orbit
        orb2
            an array containing all data generated in the txt file for a given orbit
    Outputs:
        vmag: nparray
            an array containing the difference in velocity vectors from all velocity vectors
        dmag: nparray
            an array containing the difference in magnitude from all the seperation vectors
    """
    x1 = orb1['x']
    y1 = orb1['y']
    z1 = orb1['z']
    vx1 = orb1['vx']
    vy1 = orb1['vy']
    vz1 = orb1['vz']
    
    x2 = orb2['x']
    y2 = orb2['y']
    z2 = orb2['z']
    vx2 = orb2['vx']
    vy2 = orb2['vy']
    vz2 = orb2['vz']
    dx = x1 - x2
    dy = y1 - y2
    dz = z1 - z2
    dvx = vx1 - vx2
    dvy = vy1 - vy2
    dvz = vz1 - vz2
    
    dmag = np.sqrt((dx**2) + (dy**2) + (dz**2))
    
    vmag = np.sqrt((dvx**2) + (dvy**2) + (dvz**2))
    
    return vmag,dmag

# You can use this function to return both the relative position and relative velocity for two 

########
"""
MW_M31 = Vectordiff(MWorb,M31orb)

M33_M31 = Vectordiff(M33orb,M31orb)

timescale = MWorb['t']
"""
#######
# galaxies over the entire orbit  




# Determine the magnitude of the relative position and velocities 

#########
"""
plt.figure()
plt.title('MW M31 velocity seperations')
plt.plot(timescale,MW_M31[0],color='red')
plt.xlabel('time Gyr')
plt.ylabel('seperation km / s')
plt.show()

plt.figure()
plt.title('MW M31 orbit seperations')
plt.plot(timescale,MW_M31[1],color='red')
plt.xlabel('time Gyr')
plt.ylabel('seperation kpc')
plt.show()

plt.figure()
plt.title('M33 M31 velocity seperations')
plt.plot(timescale,M33_M31[0],color='red')
plt.xlabel('time Gyr')
plt.ylabel('seperation km / s')
plt.show()

plt.figure()
plt.title('M33 M31 orbit seperations')
plt.plot(timescale,M33_M31[1],color='red')
plt.xlabel('time Gyr')
plt.ylabel('seperation kpc')
plt.show()
"""
#########

# of MW and M31

# of M33 and M31
"""
QUESTIONS:
    1. MW and M31 will have 3 close encounters over the next 12 Gyr
    2. the relative velocity is the time derivative of the relative seperation
    3. MW and M31 merge roughly around 6 Gyr and at the same time M33 and M31
    will enter a point of relative maxima

"""
# Plot the Orbit of the galaxies 
#################################




# Plot the orbital velocities of the galaxies 
#################################

