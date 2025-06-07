
###### Code for separate 3 types of blocks (ridge, trough, dipole) ######
#### Method: Focus on the peaking date, calculate the total LWA_AC and LWA_C of the block region ####
Blocking_ridge_date = [];  Blocking_ridge_lon = []; Blocking_ridge_lat=[];  Blocking_ridge_peaking_date = [];   Blocking_ridge_peaking_lon = []; Blocking_ridge_peaking_lat=[];  Blocking_ridge_duration = [];  Blocking_ridge_velocity = [];    Blocking_ridge_area = [];   Blocking_ridge_peaking_LWA = [];   Blocking_ridge_A = []; Blocking_ridge_C = [];   Blocking_ridge_label =[]
Blocking_trough_date = []; Blocking_trough_lon =[]; Blocking_trough_lat=[]; Blocking_trough_peaking_date = [];  Blocking_trough_peaking_lon =[]; Blocking_trough_peaking_lat=[]; Blocking_trough_duration = []; Blocking_trough_velocity = [];   Blocking_trough_area = [];  Blocking_trough_peaking_LWA = [];  Blocking_trough_A = []; Blocking_trough_C = []; Blocking_trough_label =[]
Blocking_dipole_date = []; Blocking_dipole_lon =[]; Blocking_dipole_lat=[]; Blocking_dipole_peaking_date = [];  Blocking_dipole_peaking_lon =[]; Blocking_dipole_peaking_lat=[]; Blocking_dipole_duration= []; Blocking_dipole_velocity =[];    Blocking_dipole_area = [];   Blocking_dipole_peaking_LWA = [];  Blocking_dipole_A = []; Blocking_dipole_C = []; Blocking_dipole_label = []
lat_range=int(int((90-np.max(Blocking_peaking_lat))/dlat)*2+1)
lon_range=int(30/dlon)+1


for n in np.arange(len(Blocking_lon)):

    LWA_AC_sum = 0
    LWA_C_sum = 0
    Blocking_A = []
    Blocking_C = []
        
    ### peaking date information ###
    peaking_date_index = Date.index(Blocking_peaking_date[n])
    peaking_lon_index = np.squeeze(np.array(np.where( lon[:]==Blocking_peaking_lon[n])))
    peaking_lat_index = np.squeeze(np.array(np.where( lat[:]==Blocking_peaking_lat[n]))) 
    
    t = np.squeeze(np.where(np.array(Blocking_date[n]) == np.array(Blocking_peaking_date[n] )))
    
    file_LWA = Dataset(path_LWA[peaking_date_index],'r')
    LWA_max  = file_LWA.variables['LWA_Z500'][0,0,peaking_lat_index,peaking_lon_index]
    file_LWA.close() 
    

    ### date LWA_AC ###
    file_LWA_AC = Dataset(path_LWA_AC[peaking_date_index],'r')
    LWA_AC  = file_LWA_AC.variables['LWA_Z500'][0,0,180:,:]
    file_LWA_AC.close()
    
    ### date LWA_C ###
    file_LWA_C = Dataset(path_LWA_C[peaking_date_index],'r')
    LWA_C  = file_LWA_C.variables['LWA_Z500'][0,0,180:,:]
    file_LWA_C.close()
    
    ### shift the field to make the block center location at the domain center ###
    LWA_AC = np.roll(LWA_AC, int(nlon/2)-peaking_lon_index, axis=1)
    LWA_C = np.roll(LWA_C,   int(nlon/2)-peaking_lon_index, axis=1)
    WE = np.roll(Blocking_label[n][t], int(nlon/2)-peaking_lon_index, axis=1)
    lon_roll = np.roll(lon,   int(nlon/2)-peaking_lon_index)

    LWA_AC = LWA_AC[  :, int(nlon/2)-int(lon_range/2):int(nlon/2)+int(lon_range/2)+1]
    LWA_C = LWA_C[    :, int(nlon/2)-int(lon_range/2):int(nlon/2)+int(lon_range/2)+1]
    WE = WE[   :, int(nlon/2)-int(lon_range/2):int(nlon/2)+int(lon_range/2)+1]       ### Note that we need to confine the blocking region to +-15 degrees
    
    LWA_AC_d = np.zeros((nlat_NH, lon_range))
    LWA_C_d = np.zeros((nlat_NH, lon_range))
    LWA_AC_d[WE == True]  = LWA_AC[WE== True]
    LWA_C_d[WE == True]  =  LWA_C[WE == True]
    
    
    LWA_AC_sum += LWA_AC_d.sum()
    LWA_C_sum += LWA_C_d.sum()
    Blocking_A.append(LWA_AC_d.sum())
    Blocking_C.append(LWA_C_d.sum())

### if the anticyclonic LWA is much stronger than cytclonic LWA, then it is defined as ridge ###
### if the anticyclonic LWA is comparable with cyclonic LWA, then it is defined as dipole ###
### if the anticyclonic LWA is weaker than cyclonic LWA, then it is defined as trough events ###
    if LWA_AC_sum > 10 * LWA_C_sum :
        Blocking_ridge_date.append(Blocking_date[n]);                 Blocking_ridge_lon.append(Blocking_lon[n]);                  Blocking_ridge_lat.append(Blocking_lat[n])
        Blocking_ridge_peaking_date.append(Blocking_peaking_date[n]); Blocking_ridge_peaking_lon.append(Blocking_peaking_lon[n]);  Blocking_ridge_peaking_lat.append(Blocking_peaking_lat[n]); Blocking_ridge_peaking_LWA.append(LWA_max)
        Blocking_ridge_duration.append(len(Blocking_date[n]));        Blocking_ridge_velocity.append(Blocking_velocity[n]);        Blocking_ridge_area.append(Blocking_peaking_area[n]); Blocking_ridge_label.append(Blocking_label[n])
        Blocking_ridge_A.append(Blocking_A);                          Blocking_ridge_C.append(Blocking_C)
    elif LWA_C_sum > 2 * LWA_AC_sum:
        Blocking_trough_date.append(Blocking_date[n]);                 Blocking_trough_lon.append(Blocking_lon[n]);                 Blocking_trough_lat.append(Blocking_lat[n])
        Blocking_trough_peaking_date.append(Blocking_peaking_date[n]); Blocking_trough_peaking_lon.append(Blocking_peaking_lon[n]); Blocking_trough_peaking_lat.append(Blocking_peaking_lat[n]); Blocking_trough_peaking_LWA.append(LWA_max)
        Blocking_trough_duration.append(len(Blocking_date[n]));        Blocking_trough_velocity.append(Blocking_velocity[n]);       Blocking_trough_area.append(Blocking_peaking_area[n]); Blocking_trough_label.append(Blocking_label[n])
        Blocking_trough_A.append(Blocking_A);                          Blocking_trough_C.append(Blocking_C)
    else:
        Blocking_dipole_date.append(Blocking_date[n]);                 Blocking_dipole_lon.append(Blocking_lon[n]);                 Blocking_dipole_lat.append(Blocking_lat[n])
        Blocking_dipole_peaking_date.append(Blocking_peaking_date[n]); Blocking_dipole_peaking_lon.append(Blocking_peaking_lon[n]); Blocking_dipole_peaking_lat.append(Blocking_peaking_lat[n]); Blocking_dipole_peaking_LWA.append(LWA_max)
        Blocking_dipole_duration.append(len(Blocking_date[n]));        Blocking_dipole_velocity.append(Blocking_velocity[n]);       Blocking_dipole_area.append(Blocking_peaking_area[n]); Blocking_dipole_label.append(Blocking_label[n])
        Blocking_dipole_A.append(Blocking_A);                          Blocking_dipole_C.append(Blocking_C)

    print(n)
    
Blocking_diversity_date= [];   Blocking_diversity_lon= []; Blocking_diversity_lat= []; Blocking_diversity_date= []; Blocking_diversity_peaking_date= []; Blocking_diversity_peaking_lon= [];  Blocking_diversity_peaking_lat=[]; Blocking_diversity_peaking_LWA=[]; Blocking_diversity_duration=[]; Blocking_diversity_area=[]; Blocking_diversity_velocity=[]; Blocking_diversity_A = []; Blocking_diversity_C = []; Blocking_diversity_label = []
Blocking_diversity_date.append(Blocking_ridge_date);   Blocking_diversity_lon.append(Blocking_ridge_lon); Blocking_diversity_lat.append(Blocking_ridge_lat); Blocking_diversity_peaking_date.append(Blocking_ridge_peaking_date); Blocking_diversity_peaking_lat.append(Blocking_ridge_peaking_lat); Blocking_diversity_peaking_lon.append(Blocking_ridge_peaking_lon); Blocking_diversity_peaking_LWA.append(Blocking_ridge_peaking_LWA); Blocking_diversity_velocity.append(Blocking_ridge_velocity); Blocking_diversity_duration.append(Blocking_ridge_duration); Blocking_diversity_area.append(Blocking_ridge_area);  Blocking_diversity_A.append(Blocking_ridge_A) ;         Blocking_diversity_C.append(Blocking_ridge_C)   ;  Blocking_diversity_label.append(Blocking_ridge_label)
Blocking_diversity_date.append(Blocking_trough_date);  Blocking_diversity_lon.append(Blocking_trough_lon); Blocking_diversity_lat.append(Blocking_trough_lat); Blocking_diversity_peaking_date.append(Blocking_trough_peaking_date); Blocking_diversity_peaking_lat.append(Blocking_trough_peaking_lat); Blocking_diversity_peaking_lon.append(Blocking_trough_peaking_lon);Blocking_diversity_peaking_LWA.append(Blocking_trough_peaking_LWA); Blocking_diversity_velocity.append(Blocking_trough_velocity); Blocking_diversity_duration.append(Blocking_trough_duration); Blocking_diversity_area.append(Blocking_trough_area);  Blocking_diversity_A.append(Blocking_trough_A) ; Blocking_diversity_C.append(Blocking_trough_C) ;  Blocking_diversity_label.append(Blocking_trough_label)    
Blocking_diversity_date.append(Blocking_dipole_date);  Blocking_diversity_lon.append(Blocking_dipole_lon); Blocking_diversity_lat.append(Blocking_dipole_lat); Blocking_diversity_peaking_date.append(Blocking_dipole_peaking_date); Blocking_diversity_peaking_lat.append(Blocking_dipole_peaking_lat); Blocking_diversity_peaking_lon.append(Blocking_dipole_peaking_lon); Blocking_diversity_peaking_LWA.append(Blocking_dipole_peaking_LWA); Blocking_diversity_velocity.append(Blocking_dipole_velocity); Blocking_diversity_duration.append(Blocking_dipole_duration); Blocking_diversity_area.append(Blocking_dipole_area); Blocking_diversity_A.append(Blocking_dipole_A) ; Blocking_diversity_C.append(Blocking_dipole_C) ;  Blocking_diversity_label.append(Blocking_dipole_label)
