import xarray as xr
import numpy as np
from math import pi

## LWA Calculation
def eqlat(Z500, area, lat, hemisphere):
    R = 6.378*1.e6
    
    nlat = len(lat)
    fi_t = np.zeros(nlat)    # fi_t is an array storing latitudes which correspond to different qgpv levels 
    
    # MODIFIED_yjh - ## Create the qgpv levels evenly. nlat levels in total
    # maxlevel = Z500[:,:].max()
    # minlevel = Z500[:,:].min()
    # levs = np.linspace(minlevel, maxlevel, nlat)   

    maxlevel = float(Z500.max())
    minlevel = float(Z500.min())
    levs = np.linspace(minlevel, maxlevel, nlat)
    
    #calculate the equivalent latitude
    if hemisphere==1:            
        for k in np.arange(nlat):
            A = sum(area[Z500[:,:] <= levs[k]])       #Note, for Z500, the bounded area is where Z500 < contour, becasue Z500 decreases with latitudes in NH and increases with latitudes in SH
            fi = np.arcsin(1 - A/(2*pi*(R**2)))       #This is the formula to calculate the equivlant latitude which corresponds to the kth qgpv level
            fi = fi * (180/pi)            
            fi_t[nlat-1-k] = fi
        
        q_part = np.interp(lat, fi_t[:], levs[::-1])  # A simple interploration that give equivalent qgpv values to each latitude from our data.
    
    elif hemisphere==2:
        for k in np.arange(nlat):
            A = sum(area[Z500[:,:] <= levs[k]])      
            fi = np.arcsin(A/(2*pi*(R**2)) - 1)
            fi = fi * (180/pi)            
            fi_t[k] = fi
        
        q_part = np.interp(lat[:], fi_t[:], levs[:]) 
        
    return q_part

### This function is to do the line integration and output the final LWA_Z, LWA_Z_A, LWA_Z_C ###
def lwa(Z500, q_part, nlat, nlon, laa, lat, dphi, hemisphere):
    import numpy as np
    
    if hemisphere ==1:           
        LWA = np.zeros((nlat,nlon))
        LWA_A = np.zeros((nlat,nlon))
        LWA_C = np.zeros((nlat,nlon))
        for k in np.arange(nlat):
            QB = np.zeros((nlat,nlon))
            QB_A = np.zeros((nlat,nlon))
            QB_C = np.zeros((nlat,nlon))
            
            q = Z500[:,:] - q_part[k]                  #Difference between Z500 and reference Z500 for the kth latitude
            
            QB[(laa>=lat[k]) & (q>=0)] = 1             #Z500 decreases with latitude, so we need areas where q>=0 and latitude >= lat[k]
            QB[(laa<=lat[k]) & (q<=0)] = -1
            LWA[k,:] = np.sum(QB * q * dphi, axis=0)  
 
            QB_A[(laa>=lat[k]) & (q>=0)] = 1           #Z500 decreases with latitude, so we need areas where q>=0 and latitude >= lat[k]
            LWA_A[k,:] = np.sum(QB_A * q * dphi, axis=0) 

            QB_C[(laa<=lat[k]) & (q<=0)] = -1
            LWA_C[k,:] = np.sum(QB_C * q * dphi, axis=0) 
                       
    elif hemisphere == 2:
        LWA = np.zeros((nlat,nlon))
        LWA_A = np.zeros((nlat,nlon))
        LWA_C = np.zeros((nlat,nlon))
        for k in np.arange(nlat):
            QB = np.zeros((nlat,nlon))
            QB_A = np.zeros((nlat,nlon))
            QB_C = np.zeros((nlat,nlon))
            
            q = Z500[:,:] - q_part[k] 
                            
            QB[(laa<=lat[k]) & (q>=0)] = 1              #Z500 increases with latitude, so we need areas where q>=0 but latitude <= lat[k]
            QB[(laa>=lat[k]) & (q<=0)] = -1
            LWA[k,:] = np.sum(QB * q * dphi, axis=0) 

            QB_A[(laa<=lat[k]) & (q>=0)] = 1            #Z500 increases with latitude, so we need areas where q>=0 but latitude <= lat[k]
            LWA_A[k,:] = np.sum(QB_A * q * dphi, axis=0) 

            QB_C[(laa>=lat[k]) & (q<=0)] = -1
            LWA_C[k,:] = np.sum(QB_C * q * dphi, axis=0)       
            
    return LWA, LWA_A, LWA_C

def Cal(ds, lat_name, lon_name, time_name):
    from math import pi
    import numpy as np
    import math
       
    # Latitude has to go from 90 to 0 to -90   
    ds = ds.sortby(lat_name, ascending = False)
    lat, lon, z = ds[lat_name].values, ds[lon_name].values, ds.values
    
    nlat, nlon , ndays = len(lat), len(lon), ds[time_name].shape[0]
   
    Eq1 = int(nlat/2)-1 # Latitude for Equator # [89]
    Eq2 = nlat - Eq1 -1 # [91] !MODIFIED_yjh: for nlat = 181 instaed of 180 (Eq2 = nlat - Eq1)

    dlat = (lat[0] - lat[1]) * pi /180
    dlon = (lon[1] - lon[0]) * pi /180
    
    R = 6.378*1.e6
    slat0 = np.sin(lat*pi/180)
    clat0 = np.cos(lat*pi/180)
    clat = np.abs(clat0[:, np.newaxis] * np.ones((nlat,nlon)))   
    dphi = R * dlat * clat      
    
    area = np.zeros((nlat,nlon))

    for la in np.arange(nlat):
        if lat[la] == 90:
            area[la,:] = (R**2) * (1-np.sin(pi/2-dlat/2)) * dlon
        elif lat[la] == -90:
            area[la,:] = (R**2) * (1-np.sin(pi/2-dlat/2)) * dlon
        else:
            area[la,:] = (R**2) * (np.sin(lat[la]*pi/180 + dlat/2)-np.sin(lat[la]*pi/180 - dlat/2)) * dlon
    
    LWA_td = np.zeros((ndays,nlat,nlon))   
    LWA_td_A = np.zeros((ndays,nlat,nlon))
    LWA_td_C = np.zeros((ndays,nlat,nlon))
    
    ti = -1
    ###--------The core code-----------------
    for t in np.arange(ndays):
        ti+=1
                   
        Z500 = ds.isel(**{time_name: t})
                  
        ###Core part for calculating LWA       
        LWA_z = np.zeros((nlat,nlon))
        LWA_z_A = np.zeros((nlat,nlon))
        LWA_z_C = np.zeros((nlat,nlon))
        
        ##------------------------Northern Hemisphere------------------------------
        lat1 = lat[Eq1::-1]
        nlat1 = len(lat1)
        dphi1 = dphi[Eq1::-1]
        loo1,laa1 = np.meshgrid(lon,lat1)
               
        q_part1 = eqlat(Z500[Eq1::-1,:], area[Eq1::-1], lat[Eq1::-1], 1)  #1 is for NH
        LWA_z1, LWA_z1_A, LWA_z1_C = lwa(Z500[Eq1::-1,:], q_part1, nlat1, nlon, laa1, lat1, dphi1,1)
        
        ##-------------------------Southern Hemisphere----------------------------- 
        lat2 = lat[-1:-Eq2:-1]
        nlat2 = len(lat2)
        dphi2 = dphi[-1:-Eq2:-1]
        loo2,laa2 = np.meshgrid(lon,lat2)
           
        q_part2 = eqlat(Z500[-1:-Eq2:-1,:], area[-1:-Eq2:-1], lat[-1:-Eq2:-1], 2)   #2 is for SH     
        LWA_z2, LWA_z2_A, LWA_z2_C = lwa(Z500[-1:-Eq2:-1,:], q_part2, nlat2, nlon, laa2, lat2, dphi2,2)    
                             
        LWA_z[0:Eq1+1,:] = LWA_z1[::-1,:]
        LWA_z[Eq2:nlat,:] = LWA_z2[::-1,:]
        LWA_td[ti,:,:] = LWA_z

        LWA_z_A[0:Eq1+1,:] = LWA_z1_A[::-1,:]
        LWA_z_A[Eq2:nlat,:] = LWA_z2_A[::-1,:]
        LWA_td_A[ti,:,:] = LWA_z_A
        
        LWA_z_C[0:Eq1+1,:] = LWA_z1_C[::-1,:]
        LWA_z_C[Eq2:nlat,:] = LWA_z2_C[::-1,:]
        LWA_td_C[ti,:,:] = LWA_z_C
        
        print(f'Day{ti+1} LWA_Cal finished ---', flush = True) 
    
    # #  LWA Calulation Part II
    cos = np.zeros(nlat)
    
    for x in np.arange(nlat):
        cos[x] = np.cos(math.radians(lat[x]))
    
    for x in np.arange(nlat):
        LWA_td[:,x,:]=LWA_td[:,x,:]/cos[x]


    return LWA_td, LWA_td_A, LWA_td_C, lat, lon
    
