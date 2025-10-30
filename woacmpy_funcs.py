import numpy as np
from netCDF4 import Dataset
from datetime import datetime,timedelta
import woacmpy_v1.woacmpy_global as wg
import woacmpy_v1.woacmpy_classes as wc
import woacmpy_v1.woacmpy_observations as wo
import woacmpy_v1.woacmpy_track as wt
from woacmpy_v1.pyhycom import thickness2depths
from woacmpy_v1.woacmpy_utilities import indx
from wrf import getvar,interplevel
import gsw
from scipy.interpolate import griddata
from matplotlib.dates import date2num


# ================== Analysis functions used in WOACMPY ===============================


def initialize_fields():

    print('Initializing fields... \n')

    set_spray_inputs()    # Set list of input fields for offline spray calculations
    set_slice_inputs()    # Set field for WRF slice fields
    set_RMWfilt_inputs()    # Set fields for WRF fields filtered by RMW limits

    # 1. Add fields that won't be covered by big loop in step 2 =======================
    # Import lat/lon for all domains and WRF landmask since they're regularly used
    for r in wc.Run.AllRuns:
        for d in [1,2,3]:    # Atmospheric model
            if wg.active_doms[d]:
                add_field(r,1,d,'Nat')
                add_field(r,2,d,'Nat')
                add_field(r,3,d,'Nat')
        for d in [4,5,6]:    # Wave model
            if wg.active_doms[d]:
                add_field(r,13,d,'Nat')
                add_field(r,14,d,'Nat')
        for d in [7,8,9]:    # Ocean model
            if wg.active_doms[d]:
                add_field(r,16,d,'Nat')
                add_field(r,17,d,'Nat')
    # If Map with curly vectors used, import required fields
    for fi in wc.Fig.AllFigs:
        for fr in fi.myframes:
            #if (fr.type == 'Map') and (fr.typeparams['crlyvect'][0] == 'SurfCurr'):
            #    for r in uc.Run.AllRuns:
            #        add_field(r,18,5,'Nat')    # u_surf
            #        add_field(r,19,5,'Nat')    # v_surf
            if (fr.type == 'Map') and (fr.typeparams['crlyvect'][0] == 'wspd10'):
                for r in wc.Run.AllRuns:
                    add_field(r,4,1,'Nat')    # u10
                    add_field(r,5,1,'Nat')    # v10
    # If plotting by storm quadrants, import quadrant mask
    if wg.useSRinfo:
        for fi in wc.Fig.AllFigs:
            for fr in fi.myframes:
                if ((fr.type == 'ScatStat') and (fr.typeparams[1] == True)) or \
                        ((fr.type == 'Scatter') and (fr.typeparams[0] == True)):
                    for r in wc.Run.AllRuns:
                        for d in [1,2,3]:
                            if wg.active_doms[d]:
                                add_field(r,71,d,'Der')    # Quadrant mask
                                add_field(r,10,d,'Der')    # X-distance to storm center
                                add_field(r,11,d,'Der')    # Y-distance to storm center
    """
    # If plotting map of model - OISST, import Hycom SST
    for fi in uc.Fig.AllFigs:
        for fr in fi.myframes:
            if (fr.type == 'Map') and (fr.typeparams['2Dfield'][0] == 'OISSTdiff'):
                for r in uc.Run.AllRuns:
                    add_field(r,31,5,'Nat')    # Hycom sst
    # If using 3D Hycom fields, import depth
    for fi in uc.Fig.AllFigs:
        for fr in fi.myframes:
            if fr.type in ['HycVert']:
                for r in uc.Run.AllRuns:
                    add_field(r,28,5,'Nat')    # Hycom depth
    # If plotting MLT on Hycom vertical profiles, import MLT
    for fi in uc.Fig.AllFigs:
        for fr in fi.myframes:
            if fr.type == 'HycVert':
                if fr.typeparams[7]:
                    for r in uc.Run.AllRuns:
                        add_field(r, 28,5,'Nat')    # Hycom 3d depth
                        add_field(r, 29,5,'Nat')    # Hycom 3d temperature
                        add_field(r,474,5,'Nat')    # Hycom 3d salinity
                        add_field(r,475,5,'Der')    # Hycom 3d sigma
                        add_field(r,476,5,'Der')    # Mixed layer depth
    """
    # If plotting bulk TC properties, import Rstorm, RbyRMW, dz, and z
    if wg.useSRinfo:
        for fi in wc.Fig.AllFigs:
            for fr in fi.myframes:
                if fr.type == 'BulkTCProps':
                    for r in wc.Run.AllRuns:
                        for d in [1,2,3]:
                            if wg.active_doms[d]:
                                add_field(r, 10,d,'Der')    # X-distance to storm center
                                add_field(r, 11,d,'Der')    # Y-distance to storm center
                                add_field(r, 12,d,'Der')    # Radial distance to storm center
                                add_field(r,375,d,'Der')    # R/RMW
                                add_field(r,450,d,'Nat')    # Z-staggered perturbation geopotential
                                add_field(r,451,d,'Nat')    # Z-staggered base state geopotential
                                add_field(r,722,d,'Der')    # WRF model layer thickness
                                add_field(r,459,d,'Der')    # WRF half level heights
    # If plotting time series using a 3D field, import dz
    for fi in wc.Fig.AllFigs:
        for fr in fi.myframes:
            if fr.type == 'TimeSeries':
                for ser in fr.typeparams[0]:
                    if ser[0] in ['VolInt','VolMean']:
                        for r in wc.Run.AllRuns:
                            for d in [1,2,3]:
                                if wg.active_doms[d]:
                                    add_field(r,450,d,'Nat')    # Z-staggered perturbation geopotential
                                    add_field(r,451,d,'Nat')    # Z-staggered base state geopotential
                                    add_field(r,722,d,'Der')    # WRF model layer thickness
    # If using RMW calculated from d03 10-m windspeed, import required fields
    if wg.useSRinfo:
        calcRMWfromd03 = False
        for fi in wc.Fig.AllFigs:
            for fr in fi.myframes:
                if fr.type == 'ScatStat':
                    if fr.typeparams[4][0]:
                        calcRMWfromd03 = True
                elif fr.type == 'WrfVert':
                    if fr.typeparams['RMWsurf'][0]:
                        calcRMWfromd03 = True
                elif fr.type == 'HovRstorm':
                    if fr.typeparams[8][0]:
                        calcRMWfromd03 = True
        if calcRMWfromd03:
            for r in wc.Run.AllRuns:
                if wg.active_doms[3]:
                    add_field(r,4,3,'Nat')    # u10
                    add_field(r,5,3,'Nat')    # v10
                    add_field(r,6,3,'Der')    # wspd10
                    add_field(r,10,3,'Der')    # X-distance to storm center
                    add_field(r,11,3,'Der')    # Y-distance to storm center
                    add_field(r,12,3,'Der')    # Radial distance to storm center
    # If using motion-relative coordinates for WRF vertical cross sections or hovmollers, import motion-relative X,Y
    if wg.useSRinfo:
        motrelXY = False
        for fi in wc.Fig.AllFigs:
            for fr in fi.myframes:
                if fr.type == 'WrfVert':
                    if fr.typeparams['general'][1]:
                        motrelXY = True
                elif fr.type == 'HovRstorm':
                    if fr.typeparams[0]:
                        motrelXY = True
        if motrelXY:
            for r in wc.Run.AllRuns:
                for d in [1,2,3]:
                    if wg.active_doms[d]:
                        add_field(r, 10,d,'Der')    # Earth-relative X-distance to storm center
                        add_field(r, 11,d,'Der')    # Earth-relative Y-distance to storm center
                        add_field(r,503,d,'Der')    # Motion-relative X-distance to storm center
                        add_field(r,504,d,'Der')    # Motion-relative Y-distance to storm center

    # 2. Initialize fields for all figures ==============================================
    for fi in wc.Fig.AllFigs:    # Loop over figures
        for fr in fi.myframes:    # Loop over frames within a figure
            framefiltindx = []    # List holding indices for filtered fields
            for fld in fr.fldindx:    # Loop over fields for each frame
                if wg.field_info[fld][2] == 'DER':
                    for i in range(len(fr.runs)):    # Loop over runs
                        add_field(fr.runs[i],fld,fr.doms[i],'Der')                        
                    look_deeper(fld,fr)
                else:
                    for i in range(len(fr.runs)):    # Loop over runs
                        add_field(fr.runs[i],fld,fr.doms[i],'Nat')
                # Add fields required for filters
                fieldfiltindx = []    # List holding indices for a single filtered field
                if fr.filter in ['strmsea','sea','eyewallsea','strmsea_noeyewall','strmsea_quadFL',\
                                 'strmsea_quadFR','strmsea_quadRL','strmsea_quadRR','strm','eyewall']:
                    for i in range(len(fr.runs)):
                        if fr.filter not in fr.runs[i].myfields[fld][fr.doms[i]].filters:
                            fr.runs[i].myfields[fld][fr.doms[i]].filters.append(fr.filter)
                            # Import other required variables
                            if fr.filter in ['strmsea','sea','eyewallsea','strmsea_noeyewall',\
                                             'strmsea_quadFL','strmsea_quadFR','strmsea_quadRL','strmsea_quadRR']:
                                add_field(fr.runs[i], 3,fr.doms[i],'Nat')    # mask_WRF
                            if fr.filter in ['strmsea','eyewallsea','strmsea_noeyewall',\
                                             'strmsea_quadFL','strmsea_quadFR','strmsea_quadRL','strmsea_quadRR','strm','eyewall']:
                                add_field(fr.runs[i],12,fr.doms[i],'Der')    # Rstorm
                                add_field(fr.runs[i],10,fr.doms[i],'Der')    # x_WRF
                                add_field(fr.runs[i],11,fr.doms[i],'Der')    # y_WRF
                            if fr.filter in ['strmsea_quadFL','strmsea_quadFR','strmsea_quadRL','strmsea_quadRR']:
                                add_field(fr.runs[i],71,fr.doms[i],'Der')    # Quadrant mask
                        fieldfiltindx.append(fr.runs[i].myfields[fld][fr.doms[i]].filters.index(fr.filter))
                framefiltindx.append(fieldfiltindx)
            fr.filtindx = framefiltindx

    # 3. Add any remaining required fields ============================================
    for r in wc.Run.AllRuns:
        if (r.field_impswitches[375,1] == -1) or (r.field_impswitches[375,2] == -1) or \
           (r.field_impswitches[375,3] == -1) or (r.field_impswitches[472,1] == -1) or \
           (r.field_impswitches[472,2] == -1) or (r.field_impswitches[472,3] == -1) or \
           (r.field_impswitches[473,1] == -1) or (r.field_impswitches[473,2] == -1) or \
           (r.field_impswitches[473,3] == -1):    # Get fields to calculate RMW
            add_field(r,4,3,'Nat')    # u10
            add_field(r,5,3,'Nat')    # v10
            add_field(r,6,3,'Der')    # wspd10
            add_field(r,10,3,'Der')    # X-distance to storm center
            add_field(r,11,3,'Der')    # Y-distance to storm center
            add_field(r,12,3,'Der')    # Radial distance to storm center
        for d in [1,2,3]:    # If either uZon or vMer is requested, make sure the other is too
            if r.field_impswitches[448,d] == 1:
                add_field(r,449,d,'Nat')
            if r.field_impswitches[449,d] == 1:
                add_field(r,448,d,'Nat')


def set_spray_inputs():

    # Set input fields for offline spray calculations
    spr_indx = [101,131,161,191,602,650]
    for i in range(len(wc.SprayData.whichspray)):
        if wc.SprayData.whichspray[i] == 'spr4_uwincm':
            wg.field_info[spr_indx[i]][4] = wg.inps_spr4_uwincm
        elif wc.SprayData.whichspray[i] == 'spr4_umwmglb':
            wg.field_info[spr_indx[i]][4] = wg.inps_spr4_umwmglb
        elif wc.SprayData.whichspray[i] == 'csp1_uwincm':
            wg.field_info[spr_indx[i]][4] = wg.inps_csp1_uwincm


def set_slice_inputs():

    # Set input fields for WRF horizontal slices
    indxA = np.nan
    for f in wg.field_info:
        if f[1] == wg.WRFsliceAfield:
            indxA = [f[0]]
    wg.field_info[505][4] = indxA
    wg.field_info[506][4] = indxA
    wg.field_info[507][4] = indxA
    wg.field_info[508][4] = indxA
    wg.field_info[509][4] = indxA


def set_RMWfilt_inputs():

    # Set input fields for WRF fields filtered using RMW limits
    flddic = {550:wg.WRFfield_RMWfilt_A_defs[0],
              551:wg.WRFfield_RMWfilt_B_defs[0],
              552:wg.WRFfield_RMWfilt_C_defs[0],
              553:wg.WRFfield_RMWfilt_D_defs[0],
              554:wg.WRFfield_RMWfilt_E_defs[0],
              555:wg.WRFfield_RMWfilt_F_defs[0],
              556:wg.WRFfield_RMWfilt_G_defs[0],
              557:wg.WRFfield_RMWfilt_H_defs[0],
              558:wg.WRFfield_RMWfilt_I_defs[0],
              559:wg.WRFfield_RMWfilt_J_defs[0],
              560:wg.WRFfield_RMWfilt_K_defs[0],
              561:wg.WRFfield_RMWfilt_L_defs[0],
              562:wg.WRFfield_RMWfilt_M_defs[0],
              563:wg.WRFfield_RMWfilt_N_defs[0],
              564:wg.WRFfield_RMWfilt_O_defs[0],
              565:wg.WRFfield_RMWfilt_P_defs[0],
              566:wg.WRFfield_RMWfilt_Q_defs[0],
              567:wg.WRFfield_RMWfilt_R_defs[0],
              568:wg.WRFfield_RMWfilt_S_defs[0],
              569:wg.WRFfield_RMWfilt_T_defs[0]}
    for fld in [550,551,552,553,554,555,556,557,558,559,\
                560,561,562,563,564,565,566,567,568,569]:
        indxfld = np.nan
        for f in wg.field_info:
            if f[1] == flddic[fld]:
                indxfld = f[0]
        wg.field_info[fld][4] = [375,indxfld,459]


def add_field(r,fld,dom,fldtype):

    # Check if a field exists and add it if it doesn't
    if r.field_impswitches[fld,dom] == 0:
        r.myfields[fld][dom] = wc.Field()
        if fldtype == 'Nat':
            if (wg.field_info[fld][2] in ['WRF','WRTIN'] and dom in [1,2,3]) or \
               (wg.field_info[fld][2] in ['UMWM'] and dom in [4,5,6]) or \
               (wg.field_info[fld][2] in ['HYC']  and dom in [7,8,9]):    # Native field requested on its own model component's grid
                r.field_impswitches[fld,dom] = 1
                wc.Field.AllNativeFields.append(r.myfields[fld][dom])
            else:    # Native field must be remapped from another model component
                r.field_impswitches[fld,dom] = -2
                if wg.field_info[fld][2] in ['WRF']:    # Remap from designated ATM model domain
                    remap_from_dom = wg.remap_from_ATM
                elif wg.field_info[fld][2] in ['UMWM']:    # Remap from designated WAV model domain
                    remap_from_dom = wg.remap_from_WAV
                elif wg.field_info[fld][2] in ['HYC']:    # Remap from designated OCN model domain
                    remap_from_dom = wg.remap_from_OCN
                r.myfields[fld][remap_from_dom] = wc.Field()
                r.field_impswitches[fld,remap_from_dom] = 1
                wc.Field.AllNativeFields.append(r.myfields[fld][remap_from_dom])
        elif fldtype == 'Der':
            r.field_impswitches[fld,dom] = -1


def look_deeper(fld,fr):

    # Initialize fields needed for a derived field
    for needs_fld in wg.field_info[fld][4]:
        if wg.field_info[needs_fld][2] == 'DER':
            for i in range(len(fr.runs)):    # Loop over runs
                add_field(fr.runs[i],needs_fld,fr.doms[i],'Der')
            look_deeper(needs_fld,fr)
        else:
            for i in range(len(fr.runs)):    # Loop over runs
                add_field(fr.runs[i],needs_fld,fr.doms[i],'Nat')


def import_fields():

    print('Importing fields... \n')

    # 1. Import fields needed for all figures ===================================
    for r in wc.Run.AllRuns:    # Loop over runs
        currentTime = r.startTime
        while True:    # Loop over simulation timesteps
            r.time.append(currentTime)
            if wg.useSRinfo:    # Find storm center, direction, and mslp
                lon0,lat0 = wt.getStormCenter(currentTime,r.track)
                r.lon0.append(lon0)
                r.lat0.append(lat0)
                r.strmdir.append(wt.getStormDirection(currentTime,r.track))    # [rad CCW from East]
                r.mslp.append(r.track.mslp[np.argmin(np.abs(date2num(currentTime) - date2num(r.track.time)))])
                r.strmspeed.append(wt.getStormSpeed(currentTime,r.track))
                if (currentTime.hour in [0,6,12,18]) and (currentTime.minute == 0):    # Add 6-hourly values
                    r.lon0_6hrly.append(lon0)
                    r.lat0_6hrly.append(lat0)
                    r.time_6hrly.append(currentTime)
                else:
                    r.lon0_6hrly.append(np.nan)
                    r.lat0_6hrly.append(np.nan)
                    r.time_6hrly.append(np.nan)
            # Loop over all model domains 
            for j in [1,2,3,4,5,6,7,8,9]:
                if wg.active_doms[j]:
                    if 1 in r.field_impswitches[:,j]:
                        for i in range(1,len(r.myfields)):    # Loop over available fields
                            if r.field_impswitches[i,j] == 1:    # Field is requested, so import it
                                if j in [1,2,3]:    # ATM domains
                                    if wg.field_info[i][2] == 'SPRWRT':
                                        ncfile = r.sprwrt_path+'/sprayHFs_d02_'+currentTime.isoformat().replace('T','_')+'.nc'
                                    elif wg.field_info[i][2] == 'WRTIN':
                                        ncfile = wg.write_dir+'/writeflds_'+r.strmtag+'_'+currentTime.isoformat().replace('T','_')+'.nc'
                                    elif wg.wrfout_NUWRF:
                                        ncfile = r.run_path+'/wrfout_d0'+str(j)+'_'+currentTime.strftime('%Y-%m-%d_%H_%M_%S')
                                    else:
                                        ncfile = r.run_path+'/wrfout_d0'+str(j)+'_'+currentTime.isoformat().replace('T','_')
                                elif j in [4]:    # WAV domain, only single UMWM domain used right now
                                    if wg.field_info[i][2] == 'SPRWRT':    # Written output, only occurs for global runs
                                        ncfile = r.sprwrt_path+'/sprayHFs_glb_'+currentTime.isoformat().replace('T','_')+'.nc'
                                    else:
                                        ncfile = r.run_path+'/output/umwmout_'+currentTime.strftime('%Y-%m-%d_%H:%M:%S')+'.nc'
                                elif j in [7]:    # OCN domain, only single HYCOM domain used right now
                                    ncfile = r.run_path+'/archv.'+currentTime.strftime('%Y_%j_%H')+'.nc'
                                nc = Dataset(ncfile,'r')
                                if i == 9:    # WRF SLP, obtained using wrf python function
                                    r.myfields[i][j].grdata.append(getvar(nc,'slp',meta=False))
                                elif i in [34,35]:    # WRF fields taken at Z-staggered model level 1
                                    r.myfields[i][j].grdata.append(nc.variables[wg.field_info[i][3]][0,0,:,:])
                                elif i in [36,37]:    # WRF fields taken at Z-staggered model level 2
                                    r.myfields[i][j].grdata.append(nc.variables[wg.field_info[i][3]][0,1,:,:])
                                elif i in [38,39]:    # WRF fields taken at Z-staggered model level 3
                                    r.myfields[i][j].grdata.append(nc.variables[wg.field_info[i][3]][0,2,:,:])
                                elif i in [42,43,44,48,50,51]:    # WRF fields taken at lowest model mass level
                                    r.myfields[i][j].grdata.append(nc.variables[wg.field_info[i][3]][0,0,:,:])
                                elif i in [16,17]:    # Hycom lat and lon, doesn't change through time
                                    r.myfields[i][j].grdata.append(nc.variables[wg.field_info[i][3]][:,:])
                                elif i in [18,19,31,477]:    # Hycom fields taken at model level 1 (surface)
                                    r.myfields[i][j].grdata.append(nc.variables[wg.field_info[i][3]][0,0,:,:].filled(np.nan))
                                elif i in [20]:    # Hycom 2D fields
                                    r.myfields[i][j].grdata.append(nc.variables[wg.field_info[i][3]][0,:,:].filled(np.nan))
                                elif i in [29,474]:    # Hycom 3D fields
                                    r.myfields[i][j].grdata.append(nc.variables[wg.field_info[i][3]][0,:,:,:].filled(np.nan))
                                elif i == 28:    # Hycom depth
                                    thknssPa = nc.variables['thknss'][0,:,:,:].filled(np.nan)    # Layer thickness [Pa]
                                    thknssM = thknssPa/9806    # Layer thickness [m]
                                    (z_bottom,z_center,z_top) = thickness2depths(thknssM)    # Depth [m]
                                    r.myfields[i][j].grdata.append(z_center)
                                elif i in [57,58]:    # Global UMWM inputs
                                    ncfile2 = r.run_path+'/input/umwmin_'+currentTime.strftime('%Y-%m-%d_%H:%M:%S')+'.nc'
                                    nc2 = Dataset(ncfile2,'r')
                                    r.myfields[i][j].grdata.append(nc2.variables[wg.field_info[i][3]][:,:])
                                elif i in [94,96,97,98,99]:    # Thermodynamics for global spray analysis
                                    ncfile2 = r.thermo_path+'/ERA5GlobalInp_'+currentTime.strftime('%Y-%m-%d_%H:%M:%S')+'.nc'
                                    nc2 = Dataset(ncfile2,'r')
                                    r.myfields[i][j].grdata.append(nc2.variables[wg.field_info[i][3]][0,:,:])
                                elif i in [448,449]:    # WRF 3D winds in Earth coordinates, uses wrf python
                                    if i == 448:
                                        pass
                                    if i == 449:
                                        uv = getvar(nc,'uvmet',meta=False)
                                        r.myfields[448][j].grdata.append(uv[0,:,:,:])
                                        r.myfields[449][j].grdata.append(uv[1,:,:,:])
                                elif i in [450,451,452,453,454,455,456,570,571]:    # WRF 3D fields
                                    r.myfields[i][j].grdata.append(nc.variables[wg.field_info[i][3]][0,:,:,:])
                                elif i == 574:    # WRF radar reflectivity, uses wrf python
                                    r.myfields[i][j].grdata.append(getvar(nc,'dbz',meta=False))
                                elif i == 575:    # WRF equivalent potential temperature, uses wrf python
                                    r.myfields[i][j].grdata.append(getvar(nc,'theta_e',meta=False))
                                elif i in [indx('OM_TMP_top'),indx('OM_S_top'),indx('OM_U_top'),indx('OM_V_top')]:    # WRF ocean model fields - topmost layer
                                    r.myfields[i][j].grdata.append(nc.variables[wg.field_info[i][3]][0,0,:,:])
                                elif i in [indx('OM_TMP'),indx('OM_S'),indx('OM_U'),indx('OM_V'),indx('OM_DEPTH'),\
                                           indx('OM_Q2'),indx('OM_Q2L'),indx('OM_L'),indx('OM_KM'),indx('OM_KH'),\
                                           indx('OM_KQ'),indx('OM_KMB'),indx('OM_KHB')]: # WRF 3D ocean model fields
                                    r.myfields[i][j].grdata.append(nc.variables[wg.field_info[i][3]][0,:,:,:])
                                elif i in [117,147]:    # Spray droplet spectra fields -- only occurs with WRTIN
                                    r.myfields[i][j].grdata.append(nc.variables[wg.field_info[i][3]][0,:,:,:])
                                else:    # For all other variables, import normally as 2D fields
                                    r.myfields[i][j].grdata.append(nc.variables[wg.field_info[i][3]][0,:,:])
            if r.timedel[0] == 'hours':
                currentTime += timedelta(hours=r.timedel[1])
            elif r.timedel[0] == 'minutes':
                currentTime += timedelta(minutes=r.timedel[1])
            if currentTime > r.endTime:break
        # Convert to numpy arrays
        for j in [1,2,3,4,5,6,7,8,9]:
            if wg.active_doms[j]:
                if 1 in r.field_impswitches[:,j]:
                    for i in range(1,len(r.myfields)):
                        if r.field_impswitches[i,j] == 1:
                            if i in [117,147]:    # These WRTIN spray droplet spectra should remain a list in the time dimension
                                pass
                            else:
                                r.myfields[i][j].grdata = np.array(r.myfields[i][j].grdata)
    ## Convert to numpy arrays
    #for f in wc.Field.AllNativeFields:
    #    f.grdata = np.array(f.grdata)

    # 2. Remap fields if needed, only works for 2D fields ===========================================
    for r in wc.Run.AllRuns:    # Loop over runs
        for j in [1,2,3,4,5,6,7,8,9]:
            if wg.active_doms[j]:
                if -2 in r.field_impswitches[:,j]:
                    for i in range(1,len(r.myfields)):
                        if r.field_impswitches[i,j] == -2:    # Field needs to be remapped from native domain
                            # Get source info
                            if wg.field_info[i][2] in ['WRF']:
                                lonSRC = r.myfields[ 1][wg.remap_from_ATM].grdata
                                latSRC = r.myfields[ 2][wg.remap_from_ATM].grdata
                                fldSRC = r.myfields[ i][wg.remap_from_ATM].grdata
                            elif wg.field_info[i][2] in ['UMWM']:
                                lonSRC = r.myfields[13][wg.remap_from_WAV].grdata
                                latSRC = r.myfields[14][wg.remap_from_WAV].grdata
                                fldSRC = r.myfields[ i][wg.remap_from_WAV].grdata
                            elif wg.field_info[i][2] in ['HYC']:
                                lonSRC = r.myfields[16][wg.remap_from_OCN].grdata
                                latSRC = r.myfields[17][wg.remap_from_OCN].grdata
                                fldSRC = r.myfields[ i][wg.remap_from_OCN].grdata
                            # Get target info
                            if j in [1,2,3]:
                                lonTGT = r.myfields[ 1][j].grdata
                                latTGT = r.myfields[ 2][j].grdata
                            elif j in [4,5,6]:
                                lonTGT = r.myfields[13][j].grdata
                                latTGT = r.myfields[14][j].grdata
                            elif j in [7,8,9]:
                                lonTGT = r.myfields[16][j].grdata
                                latTGT = r.myfields[17][j].grdata
                            # Remap
                            fldTGT = []
                            for k in range(np.shape(lonSRC)[0]):
                                latlonSRC_forInterp = np.transpose(np.array([lonSRC[k,:,:].flatten(),latSRC[k,:,:].flatten()]))
                                latlonTGT_forInterp = (lonTGT[k,:,:],latTGT[k,:,:])
                                fldSRC_forInterp = fldSRC[k,:,:].flatten()
                                fldTGT.append(griddata(latlonSRC_forInterp,fldSRC_forInterp,latlonTGT_forInterp,method='linear'))
                            r.myfields[i][j].grdata = np.array(fldTGT)


def apply_all_filters():

    print('Applying filters...\n')

    # Apply requested filters to model fields
    for r in wc.Run.AllRuns:    # Loop over runs
        for d in [1,2,3,4,5,6,7,8,9]:    # Loop over domains
            for fld in range(1,len(r.myfields)):
                if r.myfields[fld][d] == [] or fld in [113,117,122,123,124,125,126,127,128,129,130,143,147,152,153,154,\
                        155,156,157,158,159,160,173,177,182,183,184,185,186,187,188,189,190,203,207,212,213,214,215,216,217,218,\
                        219,220,225,226,227,228,229,230,231,232,233,234,235,236,\
                        614,618,623,624,625,626,627,628,629,630,631,662,666,671,672,673,674,675,676,677,678,679,702,703,704,705,\
                        737,738,739,740,741,742]:
                    pass
                else:
                    for filt in r.myfields[fld][d].filters:
                        apply_filter(filt,r,fld,d)


def apply_filter(filt,r,fld,dom):

    # Apply a selected filter to a selected field
    if filt in ['strmsea','strmsea_quadFL','strmsea_quadFR','strmsea_quadRL','strmsea_quadRR']:
        keep = np.logical_and(r.myfields[12][dom].grdata <= wg.filter_strmsea_radius, \
                              r.myfields[ 3][dom].grdata > 1.5)    # Within storm radius, over sea
    elif filt in ['eyewallsea']:
        keep = np.logical_and(r.myfields[12][dom].grdata <= wg.filter_eyewallsea_radius, \
                              r.myfields[ 3][dom].grdata > 1.5)    # Within eyewall radius, over sea
    elif filt in ['strmsea_noeyewall']:
        keep = np.logical_and(r.myfields[ 3][dom].grdata > 1.5,\
               np.logical_and(r.myfields[12][dom].grdata >  wg.filter_eyewallsea_radius,\
                              r.myfields[12][dom].grdata <= wg.filter_strmsea_radius))    # Within storm but outside eyewall, over sea
    elif filt == 'sea':
        keep = (r.myfields[3][dom].grdata > 1.5)    # Over sea
    elif filt in ['strm']:
        keep = r.myfields[12][dom].grdata <= wg.filter_strmsea_radius    # Within storm radius
    elif filt in ['eyewall']:
        keep = r.myfields[12][dom].grdata <= wg.filter_eyewallsea_radius    # Within eyewall radius
    if filt in ['strmsea_quadFL','strmsea_quadFR','strmsea_quadRL','strmsea_quadRR']:
        quadindx = {'FR':1,'FL':2,'RL':3,'RR':4}    # Dictionary matching filter name to quadrant mask
        keep = np.logical_and(keep,r.myfields[71][dom].grdata == quadindx[filt[-2:]])
    filtered_data = np.where(keep,r.myfields[fld][dom].grdata,np.nan)
    r.myfields[fld][dom].grdata_filt.append(filtered_data)


