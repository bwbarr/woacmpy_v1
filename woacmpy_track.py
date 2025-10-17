import numpy as np
from datetime import datetime,timedelta
import pandas as pd
from matplotlib.dates import date2num


# ============== Track-related functions for WOACMPY, adapted from Milan's pygplib.atcf functions =======


def argminDatetime(time0,time):

    # Return the index of datetime array time for which the datetime time0 is nearest
    return np.argmin(np.abs(date2num(time0)-date2num(time)))


class atcf_csv():    # Track data imported from .csv file

    def __init__(self,path):
        self.time = []
        self.lat  = []
        self.lon  = []
        self.wspd = []
        self.mslp = []
        df = pd.read_csv(path,parse_dates=['date_and_time'])
        self.time = np.array([t.to_pydatetime() for t in df['date_and_time']])
        self.lat  = df['lat'].to_numpy()    # [deg N]
        self.lon  = df['lon'].to_numpy()    # [deg E]
        self.wspd = df['max_wind_kts'].to_numpy()*0.514444    # [m s-1]
        self.mslp = df['min_slp_mb'].to_numpy()    # [mb]


def getStormCenter(time,track):

    # Return the (lon,lat) storm center given datetime and atcf track object
    ind = np.argmin(np.abs(date2num(time)-date2num(track.time)))
    return track.lon[ind],track.lat[ind]


def getStormDirection(time,track):

    # Return storm propagation direction given datetime and atcf track objects
    nup = argminDatetime(time+timedelta(hours=3),track.time)
    ndn = argminDatetime(time-timedelta(hours=3),track.time)
    dlon = track.lon[nup]-track.lon[ndn]
    dlat = track.lat[nup]-track.lat[ndn]
    return np.arctan2(dlat,dlon)


def getStormSpeed(time,track):

    # Return storm propagation speed given datetime and atcf track objects
    nup  = argminDatetime(time+timedelta(hours=3),track.time)
    ndn  = argminDatetime(time-timedelta(hours=3),track.time)
    nnow = argminDatetime(time,track.time)
    Re = 6.371e6    # Radius of earth [m]
    dx = (track.lon[nup]-track.lon[ndn])*np.pi/180.*Re*np.cos(track.lat[nnow]*np.pi/180.)    # Zonal displacement [m]
    dy = (track.lat[nup]-track.lat[ndn])*np.pi/180.*Re    # Meridional displacement [m]
    ds = np.sqrt(dx**2 + dy**2)    # Total displacement [m]
    dt = (track.time[nup]-track.time[ndn]).total_seconds()    # Time interval [s]
    return ds/dt


def latlon2xyStormRelative(lon,lat,lon0,lat0,dir=0.5*np.pi):

    # Given (lon,lat) fields, returns (x,y) distance in meters from (lon0,lat0). 
    # If dir [radians] is specified, rotates (x,y) so that positive y-axis is pointing in direction dir.
    Re   = 6.371E6
    d2km = np.pi*Re/180.*1E-3
    d2r  = np.pi/180.
    Y = (lat-lat0)*d2km
    X = (lon-lon0)*d2km*np.cos(lat*d2r)
    x = X*np.sin(dir)-Y*np.cos(dir)
    y = X*np.cos(dir)+Y*np.sin(dir)
    return x,y


