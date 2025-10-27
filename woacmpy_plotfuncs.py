import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.ticker as mticker
import matplotlib.colors as mcols
from matplotlib.patches import FancyArrowPatch
import woacmpy_v1.woacmpy_classes as wc
import woacmpy_v1.woacmpy_global as wg
import woacmpy_v1.woacmpy_observations as wo
import woacmpy_v1.woacmpy_funcs as wf
import woacmpy_v1.woacmpy_allframetypes as wa
from woacmpy_v1.curlyvectors_BK import curly_vectors
from datetime import datetime,timedelta
import cartopy.crs as ccrs
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER,LATITUDE_FORMATTER
import cartopy.feature as cfeature
import matplotlib.patheffects as pe
import matplotlib.dates as mdates
from bisect import bisect
import os
from netCDF4 import Dataset
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.gridspec import GridSpec
from scipy.interpolate import griddata,interp1d
if os.path.islink('/home/orca/bwbarr/python/woacmpy_v1/woacmpy_HACKS.py'):
    from woacmpy_v1.woacmpy_HACKS import *


# ===================== Plotting functions used by WOACMPY ========================


def make_all_figures(datetag):

    print('Making all figures... \n')

    # If observations are used, import and process
    if wg.global_UMWM == False and wg.useSRinfo:
        OISST_storms = []    # OISST (2D field)
        BT_storms = []    # Best track position, MSLP, MWS (Along-track)
        for fi in wc.Fig.AllFigs:
            for fr in fi.myframes:
                # Add to BT_storms
                if fr.type in ['TimeSeries','Map','HovRstorm']:
                    if fr.runs[0].myobs not in BT_storms:
                        BT_storms.append(fr.runs[0].myobs)
                # Add to OISST_storms
                if (fr.type == 'Map') and (fr.typeparams['2Dfield'][0] in ['OISST','OISSTdiff']):
                    if fr.runs[0].myobs not in OISST_storms:
                        OISST_storms.append(fr.runs[0].myobs)
        for strm in OISST_storms:
            wo.get_OISST(strm)
        for strm in BT_storms:
            wo.get_BT(strm)
    # Make all figures for an analysis
    for fi in wc.Fig.AllFigs:
        if fi.type == 'EachStep':
            startTime   = fi.myframes[0].runs[0].startTime
            timedel     = fi.myframes[0].runs[0].timedel
            strmtag     = fi.myframes[0].runs[0].strmtag
            details = [startTime,None,timedel,strmtag,datetag]
            n_timesteps = fi.myframes[0].runs[0].n_timesteps
            for t in range(n_timesteps):
                steplist = [t]
                make_figure(fi,steplist,details)
        elif fi.type == 'AllStepsSameTimes':
            startTime   = fi.myframes[0].runs[0].startTime
            endTime     = fi.myframes[0].runs[0].endTime
            timedel     = fi.myframes[0].runs[0].timedel
            strmtag     = fi.myframes[0].runs[0].strmtag
            details = [startTime,endTime,timedel,strmtag,datetag]
            steplist = None
            make_figure(fi,steplist,details)
        elif fi.type == 'AllStepsDiffTimes':
            details = [None,None,None,None,datetag]
            steplist = None
            make_figure(fi,steplist,details)


def make_figure(fi,steplist,details):

    # Make a single figure
    startTime = details[0]
    endTime = details[1]
    timedel = details[2]
    strmtag = details[3]
    datetag = details[4]
    fi.figobj = plt.figure(figsize=fi.size)
    fi.gs = GridSpec(fi.grid[0],fi.grid[1],figure=fi.figobj)
    for fr in fi.myframes:
        make_frame(fi,fr,steplist)
    if fi.type == 'EachStep':
        if timedel[0] == 'hours':
            currentTime = startTime + timedelta(hours=timedel[1]*steplist[0])
        elif timedel[0] == 'minutes':
            currentTime = startTime + timedelta(minutes=timedel[1]*steplist[0])
        if fi.title[0] is not None:
            plt.suptitle(fi.title[0]+'\n'+currentTime.strftime('%H:%M:%S UTC %d %b %Y'),weight='bold',fontsize=fi.title[1])
    elif fi.type == 'AllStepsSameTimes':
        if fi.title[0] is not None:
            plt.suptitle(fi.title[0]+'\n'  +startTime.strftime('%H:%M:%S UTC %d %b %Y')+' through ' \
                                             +endTime.strftime('%H:%M:%S UTC %d %b %Y'),weight='bold',fontsize=fi.title[1])
    elif fi.type == 'AllStepsDiffTimes':
        strmsused = []
        strmtimes_title = ''
        strmtimes_filename = ''
        for fr in fi.myframes:
            for r in fr.runs:
                if r in strmsused:
                    pass
                else:
                    strmsused.append(r)
        for r in strmsused:
            strmtimes_title    = strmtimes_title+r.strmname+': ' +r.startTime.strftime('%y%m%d%H')+\
                                                             ' to '+r.endTime.strftime('%y%m%d%H')+', '
            strmtimes_filename = strmtimes_filename+'_'+r.strmtag+r.startTime.strftime('%y%m%d%H')+\
                                                               'to'+r.endTime.strftime('%y%m%d%H')
        strmtimes_title = strmtimes_title[:-2]
        if fi.title[0] is not None:
            plt.suptitle(fi.title[0]+'\n'+strmtimes_title,weight='bold',fontsize=fi.title[1])
        if len(strmtimes_filename) > 80:
            strmtimes_filename = '_cmprstrms'
    uses_cartopy = False
    for fr in fi.myframes:
        if fr.type in ['Map']:
            uses_cartopy = True
    if uses_cartopy == False:
        plt.tight_layout()
    plt.subplots_adjust(left=fi.subadj[0],bottom=fi.subadj[1],right=fi.subadj[2],top=fi.subadj[3])
    for mark in fi.marks:
        plt.text(mark[2][0],mark[2][1],mark[0],fontsize=mark[1],transform=plt.gcf().transFigure,weight='bold')
    if fi.type == 'EachStep':
        if timedel[0] == 'hours':
            figname=fi.figtag+'_'+datetag+'_'+strmtag+currentTime.strftime('%y%m%d%H')+'.png'
        elif timedel[0] == 'minutes':
            figname=fi.figtag+'_'+datetag+'_'+strmtag+currentTime.strftime('%y%m%d%H%M')+'.png'
    elif fi.type == 'AllStepsSameTimes':
        if timedel[0] == 'hours':
            figname=fi.figtag+'_'+datetag+'_'+strmtag  +startTime.strftime('%y%m%d%H')+'to' \
                                                         +endTime.strftime('%y%m%d%H')+'.png'
        elif timedel[0] == 'minutes':
            figname=fi.figtag+'_'+datetag+'_'+strmtag  +startTime.strftime('%y%m%d%H%M')+'to' \
                                                         +endTime.strftime('%y%m%d%H%M')+'.png'
    elif fi.type == 'AllStepsDiffTimes':
        figname=fi.figtag+'_'+datetag+strmtimes_filename+'.png'
    # ----------------- Hack Functions -----------------------
    addBEA26Hacks(fi)
    # --------------------------------------------------------
    plt.savefig(figname,dpi=wg.global_dpi)
    print('Saving '+figname+'\n')
    plt.close(fi.figobj)


def make_frame(fi,fr,steplist):

    # Make a single frame in a given figure
    for r in wc.Run.AllRuns:    # Reset 'thing' variables in runs
        r.thing0 = []
        r.thing1 = []
        r.thing2 = []
        r.thing3 = []
        r.thing4 = []
        r.thing5 = []
        r.thing6 = []
        r.thing7 = []
        r.thing8 = []
        r.thing9 = []
    if fr.type in ['Map']:    # Create axis
        fr.axobj = fi.figobj.add_subplot(eval('fi.gs['+fr.gsindx[0]+','+fr.gsindx[1]+']'), \
                                         projection=ccrs.PlateCarree())
    else:
        fr.axobj = fi.figobj.add_subplot(eval('fi.gs['+fr.gsindx[0]+','+fr.gsindx[1]+']'), \
                                         xlim=(fr.limits[0],fr.limits[1]),ylim=(fr.limits[2],fr.limits[3]), \
                                         xscale=fr.scales[0],yscale=fr.scales[1])
    plot_fields(fr,steplist)    # Plot fields
    if fr.title is None:    # Add title
        pass
    else:
        plt.title(fr.title,fontsize=fr.fontsize)
    if fr.type in ['Map_SR','MapDiff2M1_SR','MapCntr_SR']:    # Special formatting for storm-relative map plots
        rng = fr.limits[1]
        fr.axobj.set_aspect('equal')
        fr.axobj.set_xticks(np.arange(-rng+50,rng,50))
        fr.axobj.set_yticks(np.arange(-rng+50,rng,50))
        # Add rings
        for radius in np.arange(50,rng+50,50):
            plt.plot(radius*np.cos(np.linspace(0,2*np.pi,100)),\
                     radius*np.sin(np.linspace(0,2*np.pi,100)),'k-',lw=1)
        # Add storm direction
        if fr.type in ['Map_SR','MapDiff2M1_SR'] and fr.fldname[0] == 'xRot_WRF':    # Plotting motion-relative
            strmdir = np.pi/2
        elif fr.type == 'MapDiff2M1_SR' and fr.fldname[0] == 'x_WRF':    # Don't plot arrow
            strmdir = np.nan
        else:
            strmdir = fr.runs[0].strmdir[steplist[0]]
        plt.plot([-rng*np.cos(strmdir),          rng*np.cos(strmdir)],\
                 [-rng*np.sin(strmdir),          rng*np.sin(strmdir)],'k-',lw=1)
        plt.plot([-rng*np.cos(strmdir+0.5*np.pi),rng*np.cos(strmdir+0.5*np.pi)],\
                 [-rng*np.sin(strmdir+0.5*np.pi),rng*np.sin(strmdir+0.5*np.pi)],'k-',lw=1)
        plt.arrow(0,0,0.5*rng*np.cos(strmdir),0.5*rng*np.sin(strmdir),facecolor='black',edgecolor='black',\
                  shape='full',lw=2,length_includes_head=True,head_width=12,zorder=10)
    if fr.type not in ['Map_SR','MapDiff2M1_SR','MapCntr_SR']:    # Add legend
        if fr.legloc is None:
            pass
        else:
            plt.legend(loc=fr.legloc,framealpha=1.0,ncol=fr.legncol)
    if fr.type not in ['Map_SR','MapDiff2M1_SR','MapCntr_SR','Map']:    # Add gridlines
        plt.grid()
        #plt.grid(b=True)
    if fr.type not in ['Map']:    # Labels and other formatting
        plt.xlabel(fr.labels[0],fontsize=fr.fontsize)
        plt.ylabel(fr.labels[1],fontsize=fr.fontsize)
        fr.axobj.tick_params(axis='both',which='major',labelsize=fr.fontsize)
        if fr.scinot[0] == 'sci':
            plt.ticklabel_format(axis='x',style='sci',scilimits=(0,0))
        if fr.scinot[1] == 'sci':
            plt.ticklabel_format(axis='y',style='sci',scilimits=(0,0))
    if fr.type in ['Map']:    # Cartopy-specific formatting
        fr.axobj.coastlines(resolution='50m')
        if fr.type == 'Map':
            if fr.typeparams['layout'][0] == 'lnd_stock':    # Stock land image
                fr.axobj.stock_img()
            if fr.typeparams['layout'][0] == 'lnd_grey':    # Grey land
                fr.axobj.add_feature(cfeature.NaturalEarthFeature('physical','land', '50m',facecolor='#C0C0C0'),zorder=2)
            if fr.typeparams['layout'][1] == 'ocn_grey':    # Grey ocean
                fr.axobj.add_feature(cfeature.NaturalEarthFeature('physical','ocean','50m',facecolor='grey'),zorder=-1)
        gl = fr.axobj.gridlines(crs=ccrs.PlateCarree(),draw_labels=True,zorder=3)
        gl.top_labels = False
        gl.right_labels = False
        gl.xformatter = LONGITUDE_FORMATTER
        gl.yformatter = LATITUDE_FORMATTER
        gl.xlabel_style = {'size':fr.fontsize}
        gl.ylabel_style = {'size':fr.fontsize}
        fr.axobj.set_extent(fr.limits,crs=ccrs.PlateCarree())
        if fr.type == 'Map':
            gl.xlocator = mticker.FixedLocator(range(fr.limits[0],fr.limits[1]+1+fr.typeparams['layout'][3],fr.typeparams['layout'][3]))
            gl.ylocator = mticker.FixedLocator(range(fr.limits[2],fr.limits[3]+1+fr.typeparams['layout'][3],fr.typeparams['layout'][3]))
    if fr.type in ['TimeSeries']:    # Specify xticks for time series
        fr.axobj.xaxis.set_major_locator(mdates.HourLocator(interval = fr.typeparams[2]))
        fr.axobj.xaxis.set_major_formatter(mdates.DateFormatter('%d %HZ'))
        fr.axobj.tick_params(axis='x',which='major',labelsize=fr.typeparams[4])
        if fr.typeparams[3] == False:
            fr.axobj.tick_params(axis='x',which='major',labelbottom=False)
    if fr.type in ['HovRstorm']:    # Format date labels
        fr.axobj.yaxis.set_major_formatter(mdates.DateFormatter('%m-%d %HZ'))
        fr.axobj.tick_params(axis='y',which='major',labelsize=10)
    if fr.type in ['HycVert','WRFOcnVert']:    # Set y-axis as positive downward
        fr.axobj.invert_yaxis()


def plot_fields(fr,steplist):

    # Call functions from woacmpy_allframetypes to plot desired frame type
    if fr.type == 'Scatter':    # Scatter Plot
        wa.plot_scatter(fr,steplist)
    elif fr.type == '3DScat':    # 3-D Scatter Plot
        wa.plot_3Dscat(fr,steplist)
    elif fr.type == 'ScatStat':    # Scatter statistics
        wa.plot_scatstat(fr,steplist)
    elif fr.type == 'SpecProf':    # Droplet Spectra or Near-surface Vertical Profile Plot
        wa.plot_specprof(fr,steplist)
    elif fr.type == 'TimeSeries':    # Time Series Plot
        wa.plot_timeseries(fr,steplist)
    elif fr.type == 'Map_SR':    # Map - Storm-relative
        wa.plot_mapSR(fr,steplist)
    elif fr.type == 'MapDiff2M1_SR':    # Difference Map - Storm-relative
        wa.plot_mapdiffSR(fr,steplist)
    elif fr.type == 'Map':    # Map
        wa.plot_map(fr,steplist)
    elif fr.type == 'HovRstorm':    # Plot Hovmoller with Rstorm as x-axis
        wa.plot_hovRstorm(fr,steplist)
    elif fr.type == 'HycVert':    # Plot vertical Hycom ocean cross section
        wa.plot_hycvert(fr,steplist)
    elif fr.type == 'WRFOcnVert':    # Plot vertical cross section from WRF ocean model
        wa.plot_wrfocnvert(fr,steplist)
    elif fr.type == 'BulkTCProps':    # Plot of "bulk" TC properties
        wa.plot_bulkTCprops(fr,steplist)
    elif fr.type == 'WrfVert':    # Plot WRF vertical mean cross sections
        wa.plot_wrfvert(fr,steplist)



