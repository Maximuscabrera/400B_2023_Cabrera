

# import modules
import numpy as np
import astropy.units as u
import astropy.constants as const

# import plotting modules
import matplotlib.pyplot as plt
import matplotlib

# my modules from previous homeworks
from ReadFile import Read
from CenterOfMass import CenterOfMass
from GalaxyMass import ComponentMass
from matplotlib.colors import LogNorm

"""
Intro:
    The goal of this code was to be able to plot the mass profile of the merger 
    betweenMW and M31 at a snap of 455, the reason why I chose this snap was because 
    of testing I did using the orbit center of mass code in order to find when the 
    seperation between the two centers was at a minimum. From thereI checked the 
    time stamp of when this occured which was listed to be 6.5 gyr and then I had 
    to track down which snap file had that correct time, I also could have just 
    found the snap by realizing that it was listingthe 91st element in the snap 
    reading array and that it was iterating through snaps with a step size of 5 
    so 5 * 91 = 455.


Code:
    This code will produce the mass profile of the two merged galaxies at this
    specific snap and maybe a few moreto show how its profile evolves after the 
    initially collide by concatenating the data together for all the particles, 
    along with a few tweaks to the mass profile code. It will also produce a figure 
    that will help establish the scale length by finding the difference between the 
    hernquist profile and the simulated mass profile and graphing it over the the radius in kpc
    in order to make sure that it is within an acceptable range.
Problems:
    My other components for the galaxies seem to not be showing up in the figures 
    I'm producing as well as the total mass profile line looking more like a step 
    function which doesnt seem right. in any case I'm finding an appropriate scale mass to be
    around 1300

"""
#importing contour code from lab 7
import scipy.optimize as so



import scipy.optimize as so

def find_confidence_interval(x, pdf, confidence_level):
    return pdf[pdf > x].sum() - confidence_level

def density_contour(xdata, ydata, nbins_x, nbins_y, ax=None, **contour_kwargs):
    """ Create a density contour plot.
    Parameters
    ----------
    xdata : numpy.ndarray
    ydata : numpy.ndarray
    nbins_x : int
        Number of bins along x dimension
    nbins_y : int
        Number of bins along y dimension
    ax : matplotlib.Axes (optional)
        If supplied, plot the contour to this axis. Otherwise, open a new figure
    contour_kwargs : dict
        kwargs to be passed to pyplot.contour()
        
    Example Usage
    -------------
     density_contour(x pos, y pos, contour res, contour res, axis, colors for contours)
     e.g.:
     density_contour(xD, yD, 80, 80, ax=ax, 
         colors=['red','orange', 'yellow', 'orange', 'yellow'])

    """

    H, xedges, yedges = np.histogram2d(xdata, ydata, bins=(nbins_x,nbins_y), normed=True)
    x_bin_sizes = (xedges[1:] - xedges[:-1]).reshape((1,nbins_x))
    y_bin_sizes = (yedges[1:] - yedges[:-1]).reshape((nbins_y,1))

    pdf = (H*(x_bin_sizes*y_bin_sizes))
    
    X, Y = 0.5*(xedges[1:]+xedges[:-1]), 0.5*(yedges[1:]+yedges[:-1])
    Z = pdf.T
    fmt = {}
    
    ### Adjust Here #### 
    
    # Contour Levels Definitions
    one_sigma = so.brentq(find_confidence_interval, 0., 1., args=(pdf, 0.68))
    two_sigma = so.brentq(find_confidence_interval, 0., 1., args=(pdf, 0.95))
    three_sigma = so.brentq(find_confidence_interval, 0., 1., args=(pdf, 0.99))
    four_sigma = so.brentq(find_confidence_interval, 0., 1., args=(pdf, 0.80))
    
    # You might need to add a few levels


    # Array of Contour levels. Adjust according to the above
    levels = [one_sigma,four_sigma, two_sigma, three_sigma][::-1]
    
    # contour level labels  Adjust accoding to the above.
    strs = ['0.68','0.80','0.95', '0.99'][::-1]

    
    ###### 
    
    if ax == None:
        contour = plt.contour(X, Y, Z, levels=levels, origin="lower", **contour_kwargs)
        for l, s in zip(contour.levels, strs):
            fmt[l] = s
        plt.clabel(contour, contour.levels, inline=True, fmt=fmt, fontsize=12)

    else:
        contour = ax.contour(X, Y, Z, levels=levels, origin="lower", **contour_kwargs)
        for l, s in zip(contour.levels, strs):
            fmt[l] = s
        ax.clabel(contour, contour.levels, inline=True, fmt=fmt, fontsize=12)
    
    return contour

def RotateFrame(posI,velI):
    """a function that will rotate the position and velocity vectors
    so that the disk angular momentum is aligned with z axis. 
    
    PARAMETERS
    ----------
        posI : `array of floats`
             3D array of positions (x,y,z)
        velI : `array of floats`
             3D array of velocities (vx,vy,vz)
             
    RETURNS
    -------
        pos: `array of floats`
            rotated 3D array of positions (x,y,z) such that disk is in the XY plane
        vel: `array of floats`
            rotated 3D array of velocities (vx,vy,vz) such that disk angular momentum vector
            is in the +z direction 
    """
    
    # compute the angular momentum
    L = np.sum(np.cross(posI,velI), axis=0)
    # normalize the vector
    L_norm = L/np.sqrt(np.sum(L**2))


    # Set up rotation matrix to map L_norm to z unit vector (disk in xy-plane)
    
    # z unit vector
    z_norm = np.array([0, 0, 1])
    
    # cross product between L and z
    vv = np.cross(L_norm, z_norm)
    s = np.sqrt(np.sum(vv**2))
    
    # dot product between L and z 
    c = np.dot(L_norm, z_norm)
    
    # rotation matrix
    I = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    v_x = np.array([[0, -vv[2], vv[1]], [vv[2], 0, -vv[0]], [-vv[1], vv[0], 0]])
    R = I + v_x + np.dot(v_x, v_x)*(1 - c)/s**2

    # Rotate coordinate system
    pos = np.dot(R, posI.T).T
    vel = np.dot(R, velI.T).T
    
    return pos, vel

class MassProfile:
    '''Class that measures and plots mass profiles and rotation curves of
    simulation snapshots'''
    
    def __init__(self, galaxy1, snap, galaxy2):
        ''' This class reads snapshots and plots the mass profiles 
        and rotation curves of galaxies.

        PARAMETERS
        ----------
        galaxy1 and galaxy2 : `str; 'MW', 'M31', or 'M33'`
                Name of the galaxy to read in
        snap1 and snap 2 : `int`
            Number of the snapshot to read in
        '''
        
        # Determine Filename
        # add a string of the filenumber to the value "000"
        ilbl = '000' + str(snap)
        # remove all but the last 3 digits
        ilbl = ilbl[-3:]
        # create filenames
        self.filename1='%s_'%(galaxy1) + ilbl + '.txt'
        self.filename2='%s_'%(galaxy1) + ilbl + '.txt'
        # read the particle data                                                                                             
        self.time, self.total, self.data1 = Read(self.filename1)
        self.time2, self.total2, self.data2 = Read(self.filename2)

        # store the mass, positions, velocities of all particles   
        
        #concatenate for each data field 
        self.type = np.concatenate([self.data1['type'],self.data2['type']])                           
        self.m = np.concatenate([self.data1['m'],self.data2['m']])#*u.Msun
        self.x = np.concatenate([self.data1['x'],self.data2['x']])*u.kpc
        self.y =  np.concatenate([self.data1['y'],self.data2['y']])*u.kpc
        self.z =  np.concatenate([self.data1['z'],self.data2['y']])*u.kpc
    
        # store galaxy name
        self.gname1 = galaxy1 
        self.gname2 = galaxy2
        
        # converting G to units of kpc*km^2/s^2/Msun
        self.G = const.G.to(u.kpc*u.km**2/u.s**2/u.Msun) 
        self.com = CenterOfMass(self.filename1,2)
        self.com2 =  CenterOfMass(self.filename2,2)
        
        #storing average center of mass position for xyz vector between the 2 galaxies for later
        #calculations
        self.com_pos = [np.average([self.com.COM_P(0.1)[0].value,self.com2.COM_P(0.1)[0].value]),\
                   np.average([self.com.COM_P(0.1)[1].value,self.com2.COM_P(0.1)[1].value]),\
                   np.average([self.com.COM_P(0.1)[2].value,self.com2.COM_P(0.1)[2].value])]
        #doing the same but for center of mass velocity 
        vx1 , vy1 , vz1 = self.com.COM_V(self.com_pos[0]*u.kpc,self.com_pos[1]*u.kpc,self.com_pos[2]*u.kpc)
        vx2 , vy2 , vz2 = self.com2.COM_V(self.com_pos[0]*u.kpc,self.com_pos[1]*u.kpc,self.com_pos[2]*u.kpc)
        self.com_vel = [np.average([vx1.value,vx2.value]),np.average([vy1.value,vy2.value])\
                        ,np.average([vz1.value,vz2.value])]
            
    def massEnclosed(self, ptype, radii):
        '''This method computes and returns the mass profile of the galaxy
        based on the specified particle type.

        PARAMETERS
        ----------
        ptype : `int; 1, 2, or 3`
            particle type
        radii : `np.ndarray`
            array of radius bin edges, in kpc

        RETURNS
        -------
        m_enc : `np.ndarray`
            array containing the mass within the radii specified 
            by r, in Msun
        '''
    
        # Determine the COM position using Disk Particles
        # Disk Particles afford the best centroiding.
        # Create a COM object for dark matter particles
        com = CenterOfMass(self.filename1,2)
        com2 =  CenterOfMass(self.filename2,2)
        # Store the COM position of the galaxy
        # Set Delta = whatever you determined to be a 
        #good value in Homework 4.
        
        #take center of mass of both galaxies aand average them
        com_pos = [np.average([com.COM_P(0.05)[0].value,com2.COM_P(0.05)[0].value]),\
                   np.average([com.COM_P(0.05)[1].value,com2.COM_P(0.05)[1].value]),\
                   np.average([com.COM_P(0.05)[2].value,com2.COM_P(0.05)[2].value])]
        #doing .value because numpy arrays dont like holding units quantities
            
            
        # create an array to store indexes of particles of desired Ptype

        #make array of concatenaated data types together                                                
        index = np.where(self.type == ptype) 
        #must use type array created so that we get the correct indicies

        # Store positions of particles of given ptype from the COMP. 
        xG = self.x[index] - com_pos[0]*u.kpc
        yG = self.y[index] - com_pos[1]*u.kpc
        zG = self.z[index] - com_pos[2]*u.kpc
            
        # Compute the mag. of the 3D radius
        rG = np.sqrt(xG**2 + yG**2 + zG**2)
            
        # store mass of particles of a given ptype
        mG = self.m[index]
            
        # Array to store enclosed mass as a function of the 
        #input radius array
        m_enc = np.zeros(np.size(radii))
        # equivalently: 
        # m_enc = np.zeros_like(radii)
    
        # loop through the radii array
        for i in range(np.size(radii)):
            # Only want particles within the given radius
            indexR = np.where(rG <  radii[i]*u.kpc)
            m_enc[i] = np.sum(mG[indexR])         
        
        # return the array of enclosed mass with appropriate units
        return m_enc*u.Msun*1e10
        
    
    def massEnclosedTotal(self, radii):    
        '''This method computes and returns the mass profile of 
        the galaxy based on ALL particles.

        PARAMETERS
        ----------
        radii : `np.ndarray`
            array of radius bin edges, in kpc

        RETURNS
        -------
        m_enc : `np.ndarray`
            array containing the mass within the radii
            specified by r, in Msun
        '''
     
        # Sum up all the mass of each component.
        m_enc = self.massEnclosed(1,radii) + self.massEnclosed(2,radii) + self.massEnclosed(3,radii)
    
        # Recall that M33 only has 2 components!  No bulge
        #if (self.gname1 or self.gname2 == 'M33'):
            #m_enc = self.massEnclosed(1,radii)+ self.massEnclosed(2,radii)  
          
        return m_enc
    
        
        
    def hernquistMass(self, r, a, mhalo):
        ''' This method returns the mass enclosed within a radius based on
        the analytic Hernquist density profile.

        PARAMETERS
        ----------
        r : `float` 
            radius to compute mass within in kpc
        a : `float`
            Hernquist profile scale radius in kpc
        mhalo : `astropy.Quantity`
            total halo mass in Msun

        RETURNS
        -------
        m_enc : `astropy.Quantity'
            mass enclosed by r in Msun
        '''

        # adding units
        r = r*u.kpc
        a = a*u.kpc
        
        # compute numerator and denominator separately
        A = mhalo * r**2
        B = (a + r)**2
        
        return A/B
       
    def faceOn(self):
        """Creating a function that will quickly plot the face on
        histogram of the dark matter particles at a given snap"""
        
        xD1 = self.com.x - self.com_pos[0]
        yD1 = self.com.y - self.com_pos[1]
        zD1 = self.com.z - self.com_pos[2]
        #getting center of mass relative positions for first galaxy
        
        vxD1 = self.com.vx - self.com_vel[0]
        vyD1 = self.com.vy - self.com_vel[1]
        vzD1 = self.com.vz - self.com_vel[2]
        #same as the previous 3 lines but getting relative velocities

        #index = np.where(self.type == 2)
        r1 = np.array([xD1,yD1,zD1]).T 
        v1 = np.array([vxD1,vyD1,vzD1]).T
        #creating 3d array that contains the vectors

        
        xD2 = (self.com2.x - self.com_pos[0])
        yD2 = (self.com2.y - self.com_pos[1])
        zD2 = (self.com2.z - self.com_pos[2])
        
        vxD2 = (self.com2.vx - self.com_vel[0])
        vyD2 = (self.com2.vy - self.com_vel[1])
        vzD2 = (self.com2.vz - self.com_vel[2])
        #same thing as before but for galaxy 2 so getting relative velocity and positions

        
        r2 = np.array([xD2,yD2,zD2]).T 
        v2 = np.array([vxD2,vyD2,vzD2]).T
        #putting that data into a 3d array
        
        rn1 ,vn1 = RotateFrame(r1,v1)
        
        rn2 ,vn2 = RotateFrame(r2,v2)    
       #using rotate frame function developed in class to make the data face on
        fig, ax= plt.subplots(figsize=(8, 8))
        plt.hist2d(rn1[:,0],rn1[:,1],bins = 800 , norm=LogNorm(),cmap='gray')
        plt.hist2d(rn2[:,0],rn2[:,1],bins = 800 , norm=LogNorm(),cmap='gray')
        
        #plotting histogram of data from both galaxies to get a solid image
        #both are set to gray because in testing the particels are ontop of eachother
        #so the particles from different galaxies do not show up
        
        plt.colorbar()
        #density_contour(rn1[:,0], rn1[:,1], 80, 80, ax=ax, colors=['blue','blue','blue','blue'],linewidth=0.5)
       #density_contour(rn2[:,0], rn2[:,1], 80, 80, ax=ax, colors=['red','red','red','red'])
        plt.xlabel(' ', fontsize=10)
        plt.ylabel(' ', fontsize=10)


        plt.ylim(-100,100)
        plt.xlim(-100,100)
        
        plt.title(self.filename1 + ' ' + self.filename2 +' face on')
        label_size = 22
        matplotlib.rcParams['xtick.labelsize'] = label_size 
        matplotlib.rcParams['ytick.labelsize'] = label_size
        #plt.savefig(self.filename1 + ' ' + self.filename2 +' face on'+'.png')
def mEncPlot(galaxy1,snap1,galaxy2,plot_name,a):
    '''
    Plots the total and component-wise mass profile of a galaxy, 
    along with the analytic expectation from the Hernquist profile.

    PARAMETERS
    ----------
    galaxy1 and glaxy 2 : `str; 'MW', 'M31', or 'M33'`
        Name of the galaxy to read in
    snap1 : `int`
        Number of the snapshot to read in
    plot_name : 'str'
        Filename to save the plot under
    a : `float`
        Hernquist scale radius for analytic plot
    '''

    # read in galaxy information
    mProf = MassProfile(galaxy1, snap1,galaxy2) # mass profile of both galaxies

    # finding filename 
    ilbl = '000' + str(snap1) # pad snapshot number
    ilbl = ilbl[-3:] # cut off leading digits so we end up 
      #with a three-digit snap number
    filename1='%s_'%(galaxy1) + ilbl + '.txt'
    filename2='%s_'%(galaxy2) + ilbl + '.txt'   
    #grabbing correct file name
    M_halo_tot = (ComponentMass(filename1, 1) + ComponentMass(filename2,1))* 1e12 * u.Msun 
        # halo mass in Msun

    # radius array in kpc
    r_arr = np.linspace(0.1, 30, 100)

    # calculate mass profiles
    m_halo = mProf.massEnclosed(1, r_arr)
    #m_disk = mProf.massEnclosed(2, r_arr) 
    #m_bulge = mProf.massEnclosed(3, r_arr)
    #not worried about disk or bulge particles so we exclude them
    m_tot = mProf.massEnclosedTotal(r_arr)

    # make plot
    plt.figure()
    plt.plot(r_arr, mProf.hernquistMass(r_arr, a, M_halo_tot), 
            c='cyan', label='Analytic Halo, a={} kpc'.format(a))
    plt.plot(r_arr, m_halo, c='b', linestyle=':', label='Halo')
    
    #plt.plot(r_arr, m_disk, c='r', linestyle='-.', label='Disk')
    #plt.plot(r_arr, m_bulge, c='g', linestyle='--', label='Bulge')
    #again not focused on these particels so they are not reflected in graph
    
    plt.plot(r_arr, m_tot, c='k', linewidth=3, label='Total')
    # other formatting 
    plt.xlabel('r [kpc]') 
    plt.ylabel('$M_{enc}$ $[M_\\odot]$') 
    plt.yscale('log')
    plt.ylim(1e8, 5e11) 
    plt.title(galaxy1+galaxy2+' Mass Profile '+ str(snap1))
    plt.legend()
    # save as image
    plt.savefig(plot_name)
    
    plt.figure()
    #plotting the residual, M_hern-m_halo/m_halo
    plt.plot(r_arr,abs(mProf.hernquistMass(r_arr, a, M_halo_tot) - m_halo)/m_halo,label='residual')
    plt.yscale('log')
    plt.ylabel('(M_hern - M_tot) / M_tot')
    plt.xlabel('r [kpc')
    plt.title('Residuals')
    plt.legend()
    plt.savefig('Residuals_'+plot_name)





# plot mass profiles
mEncPlot('MW', '0', 'M31', 'MW_M31_0',60)
mEncPlot('MW', 455,'M31', 'MW_M31_455_mass.png', 135)
mEncPlot('MW', 500,'M31', 'MW_M31_500_mass.png', 140)
mEncPlot('MW', 600,'M31', 'MW_M31_600_mass.png', 90)
mEncPlot('MW', 700,'M31', 'MW_M31_700_mass.png', 130)
mEncPlot('MW', 800,'M31', 'MW_M31_800_mass.png', 140)

#a = MassProfile('MW', 455, 'M31')
#a1 = MassProfile('MW', 500, 'M31')
#a2 = MassProfile('MW', 600, 'M31')
#a3 = MassProfile('MW', 700, 'M31')
#a4 = MassProfile('MW', 800, 'M31')


'code to plot face on histogram figures '
#a.faceOn()
#a1.faceOn()
#a2.faceOn()
#a3.faceOn()
#a4.faceOn()

#plotting face on contour


