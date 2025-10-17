import numpy as np
from datetime import datetime,timedelta
from netCDF4 import Dataset
import os
import woacmpy_v1.woacmpy_classes as wc
import woacmpy_v1.woacmpy_global as wg


# ================= Class and functions to work with observations in WOACMPY ===============


class StormObs:

    StormNamesUsed = []
    ObsObjectsUsed = []

    OISST_limits = {'Harvey'   : [],
                    'Michael_REMSS': [datetime(2018,10,7),datetime(2018,10,11),900,1200,1100,1400],
                    'Michael_NOAA': [datetime(2018,10,7),datetime(2018,10,11),1030,1160,400,520],
                    'Fanapi'   : [],
                    'Florence' : [],
                    'Dorian'   : [],
                    'Irene'    : []}

    def __init__(self,strmname,startTime,endTime):
        self.strmname = strmname    # Name of storm
        self.startTime = startTime    # Start time
        self.endTime = endTime    # End time
        StormObs.StormNamesUsed.append(strmname)
        StormObs.ObsObjectsUsed.append(self)
        # Best Track Data (NHC database)
        self.BT_datetime_full = []
        self.BT_datetime = []
        self.BT_datetime_6hrly = []
        self.BT_datetime_6hrly_full = []
        self.BT_lon0_full = []
        self.BT_lon0 = []
        self.BT_lon0_6hrly = []
        self.BT_lon0_6hrly_full = []
        self.BT_lat0_full = []
        self.BT_lat0 = []
        self.BT_lat0_6hrly = []
        self.BT_lat0_6hrly_full = []
        self.BT_wspdmax_full = []
        self.BT_wspdmax = []
        self.BT_slpmin_full = []
        self.BT_slpmin = []
        # Wind center track (HRD)
        self.windtrack_datetime_full = []
        self.windtrack_lon0_full = []
        self.windtrack_lat0_full = []        
        # OISST Data (RSS)
        self.OISST_datetime = []
        self.OISST_lon = []
        self.OISST_lat = []
        self.OISST_sst = []    # [C]


def get_BT(strm):

    # Import data, clip before and after simulation time, and extract 6hrly subset
    BT_filename = '/home/orca/bwbarr/observations/NHC_besttrack/besttrack_'+strm.strmname+'.txt'
    BT_file = open(BT_filename,'r')
    BT_format = None
    if strm.strmname in ['Harvey','Florence','Michael','Irene']:    # NHC format
        BT_format = 'NHC'
    elif strm.strmname in ['Fanapi','Dorian']:    # JTWC format
        BT_format = 'JTWC'
    if strm.strmname in ['Harvey','Florence','Michael','Dorian','Irene']:    # Longitude given as West
        EastWestCorrection = -1.0
    elif strm.strmname in ['Fanapi']:    # Longitude given as East
        EastWestCorrection = 1.0
    if BT_format == 'NHC':
        BT_data_full_wHeader = BT_file.readlines()
        BT_data_full = BT_data_full_wHeader[1:]    # Remove header
    elif BT_format == 'JTWC':
        BT_data_full = BT_file.readlines()
    for line in BT_data_full:
        linesplit = line.split(',')
        if BT_format == 'NHC':
            strm.BT_datetime_full.append(datetime.strptime(linesplit[0].strip()+linesplit[1].strip(),'%Y%m%d%H%M'))
            strm.BT_lon0_full.append(float(linesplit[5][:-1])*EastWestCorrection)    # [deg E]
            strm.BT_lat0_full.append(float(linesplit[4][:-1]))    # [deg N]
            strm.BT_wspdmax_full.append(float(linesplit[6])/1.94384)    # [m s-1]
            strm.BT_slpmin_full.append(float(linesplit[7]))    # [mb]
        elif BT_format == 'JTWC':
            strm.BT_datetime_full.append(datetime.strptime(linesplit[2].strip(),'%Y%m%d%H'))
            strm.BT_lon0_full.append(float(linesplit[7][:-1])*EastWestCorrection/10.)    # [deg E]
            strm.BT_lat0_full.append(float(linesplit[6][:-1])/10.)    # [deg N]
            strm.BT_wspdmax_full.append(float(linesplit[8])/1.94384)    # [m s-1]
            strm.BT_slpmin_full.append(float(linesplit[9]))    # [mb]
    for i in np.arange(np.size(strm.BT_datetime_full)):
        if (strm.BT_datetime_full[i] >= strm.startTime) and (strm.BT_datetime_full[i] <= strm.endTime):
            strm.BT_datetime.append(strm.BT_datetime_full[i])
            strm.BT_lon0.append(strm.BT_lon0_full[i])
            strm.BT_lat0.append(strm.BT_lat0_full[i])
            strm.BT_wspdmax.append(strm.BT_wspdmax_full[i])
            strm.BT_slpmin.append(strm.BT_slpmin_full[i])
    for i in np.arange(np.size(strm.BT_datetime)):
        if strm.BT_datetime[i].hour in [0,6,12,18]:
            strm.BT_datetime_6hrly.append(strm.BT_datetime[i])
            strm.BT_lon0_6hrly.append(strm.BT_lon0[i])
            strm.BT_lat0_6hrly.append(strm.BT_lat0[i])
    for i in np.arange(np.size(strm.BT_datetime_full)):
        if strm.BT_datetime_full[i].hour in [0,6,12,18]:
            strm.BT_datetime_6hrly_full.append(strm.BT_datetime_full[i])
            strm.BT_lon0_6hrly_full.append(strm.BT_lon0_full[i])
            strm.BT_lat0_6hrly_full.append(strm.BT_lat0_full[i])


def get_windtrack(strm):

    windtrack_filename = '/home/orca/bwbarr/observations/HRD_windtrack/windtrack_'+strm.strmname+'.txt'
    windtrack_file = open(windtrack_filename,'r')
    windtrack_data_full_wHeader = windtrack_file.readlines()
    windtrack_data_full = windtrack_data_full_wHeader[3:]    # Remove header
    for line in windtrack_data_full:
        linesplit = line.split()
        if linesplit[3] != 'N':
            break
        strm.windtrack_datetime_full.append(datetime.strptime(linesplit[0].strip()+linesplit[1].strip(),\
                '%m/%d/%Y%H:%M:%S'))
        strm.windtrack_lon0_full.append(float(linesplit[4])*-1)    # [deg E]
        strm.windtrack_lat0_full.append(float(linesplit[2]))    # [deg N]


def get_OISST(strm):

    print('Needs update.')
    """
    whichOISST = 'REMSS'    # REMSS or NOAA
    startTime   = StormObs.OISST_limits[strm.strmname+'_'+whichOISST][0]
    endTime     = StormObs.OISST_limits[strm.strmname+'_'+whichOISST][1]
    lonmin_indx = StormObs.OISST_limits[strm.strmname+'_'+whichOISST][2]
    lonmax_indx = StormObs.OISST_limits[strm.strmname+'_'+whichOISST][3]
    latmin_indx = StormObs.OISST_limits[strm.strmname+'_'+whichOISST][4]
    latmax_indx = StormObs.OISST_limits[strm.strmname+'_'+whichOISST][5]
    currentTime = startTime
    while True:
        strm.OISST_datetime.append(currentTime)
        if whichOISST == 'REMSS':
            OISST_filename = '/home/orca/data/satellite/sst/remss_sst_ir_mw/'+currentTime.strftime('%Y')+'/'+\
                    currentTime.strftime('%Y%m%d')+'120000-REMSS-L4_GHRSST-SSTfnd-MW_IR_OI-GLOB-v02.0-fv05.0.nc'
        elif whichOISST == 'NOAA':
            OISST_filename = '/home/orca/data/satellite/sst/oisst/avhrr/'+currentTime.strftime('%Y%m')+\
                    '/oisst-avhrr-v02r01.'+currentTime.strftime('%Y%m%d')+'.nc'
        nc = Dataset(OISST_filename,'r')
        lon_vect = nc.variables['lon'][lonmin_indx:lonmax_indx]
        lat_vect = nc.variables['lat'][latmin_indx:latmax_indx]
        if whichOISST == 'REMSS':
            lon_vect[lon_vect > 180.] = lon_vect[lon_vect > 180.] - 360.
        lat,lon = np.meshgrid(lat_vect,lon_vect,indexing='ij')
        strm.OISST_lon.append(lon)
        strm.OISST_lat.append(lat)
        if whichOISST == 'REMSS':
            sst = nc.variables['analysed_sst'][0,latmin_indx:latmax_indx,lonmin_indx:lonmax_indx] - 273.15    # [C]
        elif whichOISST == 'NOAA':
            sst = nc.variables['sst'][0,0,latmin_indx:latmax_indx,lonmin_indx:lonmax_indx]    # [C]
        sst[sst <= 0.0] = np.nan    # Set land points to Nan
        strm.OISST_sst.append(sst)
        currentTime += timedelta(days=1)
        if currentTime > endTime:break
    """


