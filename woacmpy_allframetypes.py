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
import woacmpy_v1.woacmpy_derivedflds as wd
from woacmpy_v1.curlyvectors_BK import curly_vectors
from woacmpy_v1.woacmpy_utilities import indx
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


# ===================== Definitions for all frame types in WOACMPY ========================


def plot_scatter(fr,steplist_0):

    for j in range(len(fr.runs)):
        if steplist_0 is None:
            steplist = fr.runs[j].steplist
        else:
            steplist = steplist_0
        for n in range(1,len(fr.fldindx)):
            if fr.filter is None:
                xfield = np.array([fr.runs[j].myfields[fr.fldindx[0]][fr.doms[j]].grdata[t,:,:] for t in steplist])
                yfield = np.array([fr.runs[j].myfields[fr.fldindx[n]][fr.doms[j]].grdata[t,:,:] for t in steplist])
            else:
                xfield = np.array([fr.runs[j].myfields[fr.fldindx[0]][fr.doms[j]].grdata_filt[fr.filtindx[0][j]][t,:,:] for t in steplist])
                yfield = np.array([fr.runs[j].myfields[fr.fldindx[n]][fr.doms[j]].grdata_filt[fr.filtindx[n][j]][t,:,:] for t in steplist])
            if fr.typeparams[0] == False:    # No quadrant plotting
                if fr.legtext is None:
                    label = fr.fldname[n]+' - '+fr.runs[j].tag
                elif fr.legtext[j][n-1] is None:
                    label = fr.fldname[n]+' - '+fr.runs[j].tag
                elif fr.legtext[j][n-1] is not None:
                    label = fr.legtext[j][n-1]
                plt.plot(xfield.flatten(),yfield.flatten(),\
                        '.',markersize=2,mec=fr.colors[j][n-1],mfc=fr.colors[j][n-1])
                plt.plot([np.nan],[np.nan],'.',markersize=10,mec=fr.colors[j][n-1],mfc=fr.colors[j][n-1],label=label)
            elif fr.typeparams[0] == True:    # Plot by quadrants
                if (j == 0) and (n == 1):
                    quad = np.array([fr.runs[j].myfields[71][fr.doms[j]].grdata[t,:,:] for t in steplist])
                    labels = ['Front Right','Front Left','Rear Left','Rear Right']
                    colors = ['b','r','g','k']
                    for i in range(4):
                        thisquad = (quad == float(i+1))
                        plt.plot(xfield[thisquad].flatten(),yfield[thisquad].flatten(),\
                                '.',markersize=2,mec=colors[i],mfc=colors[i])
                        plt.plot([np.nan],[np.nan],'.',markersize=10,mec=colors[i],mfc=colors[i],label=labels[i])


#================================================================================


def plot_3Dscat(fr,steplist):

    """
    -------------- Scatter Plot with Colored Markers and Density Contours ----------------
    Make a scatter plot with markers colored by the value of a third field.  Can also
    plot scatter density contours.  Only works with one run.

            Type:       '3DScat'
            Type parameters:
                        [
                         0: Colormap
                         1: [min,max] values for color scale
                         2: List of colorbar tick marks
                         3: Which density contours to plot:
                             'CDF' for fraction of total dataset contributed by bins with equal or greater density,
                             None for no contours
                         4: List of contour intervals
                         5: Contour colors
                         6: Number of bins for density contours
                        ]
            Fields:     [x-field, y-field, color field]
            Runs:       [only one run allowed]
            Colors:     None, not used
            Leg params: [None,None,1], not used
    --------------------------------------------------------------------------------------
    """

    if steplist is None:
        steplist = fr.runs[0].steplist
    # Get data
    xfield = []
    yfield = []
    zfield = []
    for t in steplist:
        if fr.filter is None:
            xfield.append(fr.runs[0].myfields[fr.fldindx[0]][fr.doms[0]].grdata[t,:,:].flatten())
            yfield.append(fr.runs[0].myfields[fr.fldindx[1]][fr.doms[0]].grdata[t,:,:].flatten())
            zfield.append(fr.runs[0].myfields[fr.fldindx[2]][fr.doms[0]].grdata[t,:,:].flatten())
        else:
            xfield.append(fr.runs[0].myfields[fr.fldindx[0]][fr.doms[0]].grdata_filt[fr.filtindx[0][0]][t,:,:].flatten())
            yfield.append(fr.runs[0].myfields[fr.fldindx[1]][fr.doms[0]].grdata_filt[fr.filtindx[1][0]][t,:,:].flatten())
            zfield.append(fr.runs[0].myfields[fr.fldindx[2]][fr.doms[0]].grdata_filt[fr.filtindx[2][0]][t,:,:].flatten())
    xfield = np.array(xfield).flatten()
    yfield = np.array(yfield).flatten()
    zfield = np.array(zfield).flatten()
    # Plot scatter
    plt.scatter(xfield,yfield,s=5,c=zfield,cmap=fr.typeparams[0],vmin=fr.typeparams[1][0],vmax=fr.typeparams[1][1])
    colorbar = plt.colorbar(shrink=0.8,extend='both',ticks=fr.typeparams[2])
    cbytick_obj = plt.getp(colorbar.ax.axes,'yticklabels')
    plt.setp(cbytick_obj,fontsize=12)
    # Plot density contours if selected
    if fr.typeparams[3] is not None:
        # Bin points to get 2D histogram
        numbins = fr.typeparams[6]    # Number of bins along each axis
        binwid_x = (fr.limits[1] - fr.limits[0])/numbins
        binwid_y = (fr.limits[3] - fr.limits[2])/numbins
        bintol_x = binwid_x/2
        bintol_y = binwid_y/2
        xbin = np.linspace(fr.limits[0]+bintol_x,fr.limits[1]-bintol_x,numbins)    # Bin centers, x-axis
        ybin = np.linspace(fr.limits[2]+bintol_y,fr.limits[3]-bintol_y,numbins)    # Bin centers, y-axis
        xindx = np.round((xfield-fr.limits[0])/(fr.limits[1]-fr.limits[0])*numbins-0.5)    # Bin indices on x-axis
        xindx[xindx < 0] = np.nan
        xindx[xindx > (numbins-1)] = np.nan
        yindx = np.round((yfield-fr.limits[2])/(fr.limits[3]-fr.limits[2])*numbins-0.5)    # Bin indices on y-axis
        yindx[yindx < 0] = np.nan
        yindx[yindx > (numbins-1)] = np.nan
        xyindx = xindx + numbins*yindx    # Each point has a unique index in an unraveled x-y array
        xycount = np.zeros((numbins,numbins))    # Matrix to count binned x,y values, initialized with zeros
        for x in range(numbins):
            for y in range(numbins):
                thisbin = (xyindx == x + numbins*y)
                if np.sum(thisbin) > 0:
                    xycount[y,x] = np.sum(thisbin)
        numtot = np.sum(xycount)    # Total number of points in plotted area
        # Plot contours depending on selected method
        if fr.typeparams[3] == 'CDF':    # Plot fraction of total dataset contributed by bins with equal or greater density
            nummax = np.max(xycount)    # Largest value in xycount
            CDFcount = np.full_like(xycount,np.nan)
            for i in range(int(nummax)+1):    # Count up from zero to maximum bin count
                CDFcount[xycount == i] = np.sum(xycount[xycount >= i])    # Number of points contributed by bins with equal or greater number of points
            CDFfrac = CDFcount/numtot    # Fraction of total points contributed
            cntr = plt.contour(xbin,ybin,CDFfrac,fr.typeparams[4],colors=fr.typeparams[5],linewidths=1)
            plt.clabel(cntr,fmt='%.1f',inline=True,colors=fr.typeparams[5])


#================================================================================


def plot_scatstat(fr,steplist_0):

    quadlabels = ['Front Right','Front Left','Rear Left','Rear Right']    # Labels for quadrant plotting
    quadcolscat = ['b','r','g','k']    # Colors for quadrant plotting (scatter)
    quadcolstat = ['b','r','g','k']    # Colors for quadrant plotting (statistics)
    numbins = fr.typeparams[8]    # Number of bins along each axis (usually use 100)
    min_count = fr.typeparams[6]    # Minimum number of elements required in bin to plot (9 is current "best practice")
    if fr.scales[0] == 'linear':    # Use linear x-axis
        binwid_x = (fr.limits[1] - fr.limits[0])/numbins
        bintol_x = binwid_x/2
        xbin = np.linspace(fr.limits[0]+bintol_x,fr.limits[1]-bintol_x,numbins)    # Bin centers, x-axis
    elif fr.scales[0] == 'log':    # Use logarithmic x-axis where powers of bins are linearly spaced
        binwid_x = (np.log10(fr.limits[1]) - np.log10(fr.limits[0]))/numbins    # Here the bin width is the increment in power
        bintol_x = binwid_x/2
        xbin = 10.**np.linspace(np.log10(fr.limits[0])+bintol_x,np.log10(fr.limits[1])-bintol_x,numbins)    # Bin centers, x-axis
    for k in range(len(fr.runs)):
        for n in range(1,len(fr.fldindx)):
            fr.runs[k].thing1.append([np.full(numbins,np.nan) for i in range(4)])    # Mean
            fr.runs[k].thing2.append([np.full(numbins,np.nan) for i in range(4)])    # Standard deviation
            fr.runs[k].thing3.append([np.full(numbins,np.nan) for i in range(4)])    # Median
            fr.runs[k].thing4.append([np.full(numbins,np.nan) for i in range(4)])    # Upper quartile
            fr.runs[k].thing5.append([np.full(numbins,np.nan) for i in range(4)])    # Lower quartile
    for k in range(len(fr.runs)):
        if steplist_0 is None:
            steplist = fr.runs[k].steplist
        else:
            steplist = steplist_0
        for n in range(1,len(fr.fldindx)):
            if fr.filter is None:
                xfield = np.array([fr.runs[k].myfields[fr.fldindx[0]][fr.doms[k]].grdata[t,:,:] for t in steplist])
                yfield = np.array([fr.runs[k].myfields[fr.fldindx[n]][fr.doms[k]].grdata[t,:,:] for t in steplist])
            else:
                xfield = np.array([fr.runs[k].myfields[fr.fldindx[0]][fr.doms[k]].grdata_filt[fr.filtindx[0][k]][t,:,:] for t in steplist])
                yfield = np.array([fr.runs[k].myfields[fr.fldindx[n]][fr.doms[k]].grdata_filt[fr.filtindx[n][k]][t,:,:] for t in steplist])
            if fr.typeparams[1]:
                quad = np.array([fr.runs[k].myfields[71][fr.doms[k]].grdata[t,:,:] for t in steplist])
            # Determine bin indices for each field gridpoint
            if fr.scales[0] == 'linear':    # Using linear x-axis
                xindx = np.round((xfield-fr.limits[0])/(fr.limits[1]-fr.limits[0])*numbins-0.5)
            elif fr.scales[0] == 'log':    # Using logarithmic x-axis
                xindx = np.round((np.log10(xfield)-np.log10(fr.limits[0]))/(np.log10(fr.limits[1])-np.log10(fr.limits[0]))*numbins-0.5)
            xindx[xindx <  0] = np.nan
            xindx[xindx > (numbins-1)] = np.nan
            # Determine statistics
            for x in range(numbins):
                thisbin = np.logical_and(xindx > x-0.01,xindx < x+0.01)
                if fr.typeparams[1] == False:    # Don't plot by storm quadrants
                    if np.nansum(thisbin) >= min_count:
                        fr.runs[k].thing1[n-1][0][x] =     np.nanmean(yfield[thisbin])    # Mean
                        fr.runs[k].thing2[n-1][0][x] =      np.nanstd(yfield[thisbin])    # Std Dev
                        fr.runs[k].thing3[n-1][0][x] = np.nanquantile(yfield[thisbin],0.50)    # Median
                        fr.runs[k].thing4[n-1][0][x] = np.nanquantile(yfield[thisbin],0.75)    # Upp Quart
                        fr.runs[k].thing5[n-1][0][x] = np.nanquantile(yfield[thisbin],0.25)    # Low Quart
                elif fr.typeparams[1] == True:    # Plot by storm quadrants
                    for i in range(4):
                        thisquad = (quad == float(i+1))
                        thisbinquad = np.logical_and(thisbin,thisquad)
                        if np.nansum(thisbinquad) >= min_count:
                            fr.runs[k].thing1[n-1][i][x] =     np.nanmean(yfield[thisbinquad])    # Mean
                            fr.runs[k].thing2[n-1][i][x] =      np.nanstd(yfield[thisbinquad])    # Std Dev
                            fr.runs[k].thing3[n-1][i][x] = np.nanquantile(yfield[thisbinquad],0.50)    # Median
                            fr.runs[k].thing4[n-1][i][x] = np.nanquantile(yfield[thisbinquad],0.75)    # Upp Quart
                            fr.runs[k].thing5[n-1][i][x] = np.nanquantile(yfield[thisbinquad],0.25)    # Low Quart
            # Plot scatter behind statistics if requested (doesn't work for difference)
            if fr.typeparams[2] and fr.typeparams[7] == False:
                if fr.typeparams[1] == False:    # Don't plot by storm quadrants
                    plt.plot(xfield.flatten(),yfield.flatten(),\
                            '.',markersize=2,mec=fr.colors[k][n-1],mfc=fr.colors[k][n-1])
                elif fr.typeparams[1] == True:    # Plot by storm quadrants
                    if (k == 0) and (n == 1):
                        for i in range(4):
                            thisquad = (quad == float(i+1))
                            plt.plot(xfield[thisquad].flatten(),yfield[thisquad].flatten(),\
                                    '.',markersize=2,mec=quadcolscat[i],mfc=quadcolscat[i])
    # Plotting (no storm quadrants)
    if fr.typeparams[1] == False and fr.typeparams[7] == False:    # Don't plot differences
        for k in range(len(fr.runs)):
            for n in range(1,len(fr.fldindx)):
                if fr.legtext is None:
                    label = fr.fldname[n]+' - '+fr.runs[k].tag
                elif fr.legtext[k][n-1] is None:
                    label = fr.fldname[n]+' - '+fr.runs[k].tag
                elif fr.legtext[k][n-1] is not None:
                    label = fr.legtext[k][n-1]
                if fr.typeparams[0] == 'mean_1stdev':    # Plot +/- 1 stdev band
                    plt.fill_between(xbin,fr.runs[k].thing1[n-1][0] - fr.runs[k].thing2[n-1][0],\
                                          fr.runs[k].thing1[n-1][0] + fr.runs[k].thing2[n-1][0],\
                                          facecolor=fr.colors[k][n-1],alpha=0.5)
                    if fr.typeparams[2]:    # Plot white edge of band
                        plt.plot(xbin,fr.runs[k].thing1[n-1][0] - fr.runs[k].thing2[n-1][0],'w-',linewidth=1,label='_nolegend_')
                        plt.plot(xbin,fr.runs[k].thing1[n-1][0] + fr.runs[k].thing2[n-1][0],'w-',linewidth=1,label='_nolegend_')
                elif fr.typeparams[0] == 'med_quart':    # Plot interquartile band
                    plt.fill_between(xbin,fr.runs[k].thing5[n-1][0],  fr.runs[k].thing4[n-1][0],\
                                          facecolor=fr.colors[k][n-1],alpha=0.5)
                if fr.typeparams[0] in ['mean','mean_1stdev']:
                    if fr.typeparams[0] == 'mean' and fr.typeparams[2] == False:    # Don't outline in white
                        if fr.typeparams[5] == True and fr.fldname[0] == 'Rstorm':    # Integrate azimuthally
                            yvals = fr.runs[k].thing1[n-1][0]*xbin*1000*np.pi*2*1e-6    # Units change by factor of [m], value reduced by factor of 1e-6
                            plt.plot(xbin,yvals,                    fr.colors[k][n-1],linewidth=2,label=label,linestyle=fr.typeparams[3][k])
                        else:
                            plt.plot(xbin,fr.runs[k].thing1[n-1][0],fr.colors[k][n-1],linewidth=2,label=label,linestyle=fr.typeparams[3][k])
                    else:    # Outline in white
                        plt.plot(xbin,fr.runs[k].thing1[n-1][0],fr.colors[k][n-1],linewidth=2,label=label,linestyle=fr.typeparams[3][k],\
                                path_effects=[pe.Stroke(linewidth=4,foreground='w'),pe.Normal()])
                elif fr.typeparams[0] in ['med','med_quart']:
                    plt.plot(xbin,fr.runs[k].thing3[n-1][0],fr.colors[k][n-1],linewidth=2,label=label,linestyle=fr.typeparams[3][k])
    elif fr.typeparams[1] == False and fr.typeparams[7]:    # Plot differences for consecutive pairs of runs, only works for 'mean'
        for k in range(len(fr.runs)):
            if (k % 2) != 0:    # Calculate difference and plot by counting the even indices
                pass
            elif (k % 2) == 0:
                for n in range(1,len(fr.fldindx)):
                    label = fr.legtext[k][n-1]
                    if fr.typeparams[0] == 'mean':    # Only use this with 'mean'
                        plt.plot(xbin,fr.runs[k+1].thing1[n-1][0] - fr.runs[k].thing1[n-1][0],fr.colors[k][n-1],\
                                linewidth=2,label=label,linestyle=fr.typeparams[3][k])
    # Plotting (storm quadrants) - only allows one run and one yfield, doesn't work for differences
    elif fr.typeparams[1] == True and fr.typeparams[7] == False:
        for i in range(4):
            if fr.typeparams[0] == 'mean_1stdev':    # Plot +/- 1 stdev band
                plt.fill_between(xbin,fr.runs[0].thing1[0][i] - fr.runs[0].thing2[0][i],\
                                      fr.runs[0].thing1[0][i] + fr.runs[0].thing2[0][i],\
                                      facecolor=quadcolstat[i],alpha=0.5)
            elif fr.typeparams[0] == 'med_quart':    # Plot interquartile band
                plt.fill_between(xbin,fr.runs[0].thing5[0][i],  fr.runs[0].thing4[0][i],\
                                      facecolor=quadcolstat[i],alpha=0.5)
            if fr.typeparams[0] in ['mean','mean_1stdev']:
                if fr.typeparams[2] == False:
                    plt.plot(xbin,fr.runs[0].thing1[0][i],quadcolstat[i],linewidth=2,label=quadlabels[i])
                else:
                    plt.plot(xbin,fr.runs[0].thing1[0][i],quadcolstat[i],linewidth=2,label=quadlabels[i],\
                            path_effects=[pe.Stroke(linewidth=4,foreground='w'),pe.Normal()])
            elif fr.typeparams[0] in ['med','med_quart']:
                if fr.typeparams[2] == False:
                    plt.plot(xbin,fr.runs[0].thing3[0][i],quadcolstat[i],linewidth=2,label=quadlabels[i])
                else:
                    plt.plot(xbin,fr.runs[0].thing3[0][i],quadcolstat[i],linewidth=2,label=quadlabels[i],\
                            path_effects=[pe.Stroke(linewidth=4,foreground='w'),pe.Normal()])
    # Mark surface RMW if requested when Rstorm is on x-axis
    if fr.typeparams[4][0]:    # RMW requested
        if fr.fldname[0] == 'Rstorm':    # Check if Rstorm is on x-axis
            for k in range(len(fr.runs)):    # Loop over runs
                if fr.runs[k].RMW == []:
                    wd.calcRMW(fr.runs[k])    # Calculate RMW
                if steplist_0 is None:
                    steplist = fr.runs[k].steplist
                else:
                    steplist = steplist_0
                RMW_vect = np.array([fr.runs[k].RMW[t] for t in steplist])
                RMW_mean = np.nanmean(RMW_vect)
                plt.plot([RMW_mean,RMW_mean],[fr.limits[2],fr.limits[3]],fr.typeparams[4][1][k],linewidth=1,\
                        linestyle=fr.typeparams[4][2][k],label=fr.typeparams[4][3][k],zorder=0)


#================================================================================


def plot_specprof(fr,steplist_0):

    tol = fr.typeparams[0]    # Tolerance around testfield values [same units as testfield]
    testfield_units = fr.typeparams[1]    # String stating test-field units
    testfield_picked = fr.typeparams[2]    # Specific values of testfield to use [same units as testfield]
    specORprof = fr.typeparams[3]    # 'Spec' for droplet spectrum plot, 'Prof' for vertical profiles
    profType = fr.typeparams[4]    # If 'Prof', 't' for temp, 'q' for spec hum, 'sMy0' for s minus y0
    for k in range(len(fr.runs)):
        if steplist_0 is None:
            steplist = fr.runs[k].steplist
        else:
            steplist = steplist_0
        for t in steplist:
            if fr.filter is None:
                testfield = fr.runs[k].myfields[fr.fldindx[0]][fr.doms[k]].grdata[t,:,:]
            else:
                testfield = fr.runs[k].myfields[fr.fldindx[0]][fr.doms[k]].grdata_filt[fr.filtindx[0][k]][t,:,:]
            for n in range(1,len(fr.fldindx)):
                if specORprof == 'Spec':
                    if   fr.fldindx[n] in [113,117,122,123,124,126,127,128,129,130,225,229,233,737]:    # A
                        r0 = wc.SprayData.r0[0]*1e6    # [um]
                    elif fr.fldindx[n] in [143,147,152,153,154,156,157,158,159,160,226,230,234,738]:    # B
                        r0 = wc.SprayData.r0[1]*1e6    # [um]
                    elif fr.fldindx[n] in [173,177,182,183,184,186,187,188,189,190,227,231,235,739]:    # C
                        r0 = wc.SprayData.r0[2]*1e6    # [um]
                    elif fr.fldindx[n] in [203,207,212,213,214,216,217,218,219,220,228,232,236,740]:    # D
                        r0 = wc.SprayData.r0[3]*1e6    # [um]
                    elif fr.fldindx[n] in [614,618,623,624,625,627,628,629,630,631,702,704,741]:    # E
                        r0 = wc.SprayData.r0[4]*1e6    # [um]
                    elif fr.fldindx[n] in [662,666,671,672,673,675,676,677,678,679,703,705,742]:    # F
                        r0 = wc.SprayData.r0[5]*1e6    # [um]
                    yfield = fr.runs[k].myfields[fr.fldindx[n]][fr.doms[k]].grdata[t]
                    for i in range(testfield.shape[0]):
                        for j in range(testfield.shape[1]):
                            if ~np.isnan(testfield[i,j]):
                                for p in range(len(testfield_picked)):
                                    if testfield[i,j] > testfield_picked[p] - tol and testfield[i,j] < testfield_picked[p] + tol:
                                        plt.plot(r0,yfield[:,i,j],'-',color=fr.colors[k][n-1][p],linewidth=0.5)
                elif specORprof == 'Prof':
                    profiles = fr.runs[k].myfields[fr.fldindx[n]][fr.doms[k]].grdata[t]
                    if profType == 't':
                        z = profiles[0]
                        xfield = profiles[2]    # [C]
                    elif profType == 'q':
                        z = profiles[1]
                        xfield = profiles[3]    # [g kg-1]
                    elif profType == 'sMy0':
                        z = profiles[1]
                        xfield = profiles[4]
                    for i in range(testfield.shape[0]):
                        for j in range(testfield.shape[1]):
                            if ~np.isnan(testfield[i,j]):
                                for p in range(len(testfield_picked)):
                                    if testfield[i,j] > testfield_picked[p] - tol and testfield[i,j] < testfield_picked[p] + tol:
                                        plt.plot(xfield[:,i,j],z[:,i,j],'-',color=fr.colors[k][n-1][p],linewidth=0.5)
    for k in range(len(fr.runs)):
        for n in range(1,len(fr.fldindx)):
            for p in range(len(testfield_picked)):
                if fr.legtext is None:
                    label = '***user must define***'
                elif fr.legtext[k][n-1] is None:
                    label = '***user must define***'
                else:
                    label = fr.legtext[k][n-1] + str(testfield_picked[p]) + ' ' + testfield_units
                plt.plot([np.nan],[np.nan],'-',color=fr.colors[k][n-1][p],linewidth=1,label=label)


#================================================================================


def plot_timeseries(fr,steplist):

    # Make calculations on run data
    for k in range(len(fr.runs)):
        fr.runs[k].thing0 = [[] for n in range(len(fr.fldindx))]
        fr.runs[k].thing1 = [[] for n in range(len(fr.fldindx))]
        fr.runs[k].thing2 = [[] for n in range(len(fr.fldindx))]
        fr.runs[k].thing3 = [[] for n in range(len(fr.fldindx))]
        fr.runs[k].thing4 = [[] for n in range(len(fr.fldindx))]
        fr.runs[k].thing5 = [[] for n in range(len(fr.fldindx))]
        fr.runs[k].thing6 = [[] for n in range(len(fr.fldindx))]
        fr.runs[k].thing7 = [[] for n in range(len(fr.fldindx))]
        fr.runs[k].thing8 = [[] for n in range(len(fr.fldindx))]
        fr.runs[k].thing9 = [[] for n in range(len(fr.fldindx))]
    for k in range(len(fr.runs)):
        steplist_all = fr.runs[k].steplist
        for t in steplist_all:
            for n in range(len(fr.fldindx)):
                if fr.filter is None:
                    yfield = fr.runs[k].myfields[fr.fldindx[n]][fr.doms[k]].grdata[t,...]    # Can use 2D or 3D fields
                else:
                    yfield = fr.runs[k].myfields[fr.fldindx[n]][fr.doms[k]].grdata_filt[fr.filtindx[n][k]][t,:,:]    # Cannot filter 3D fields
                if yfield.ndim == 2:    # Using a 2D field
                    fr.runs[k].thing0[n] = 2    # Record if 2D or 3D field
                    fr.runs[k].thing1[n].append(np.nanmean(yfield))    # Mean
                    fr.runs[k].thing2[n].append(np.nanmax(yfield))    # Maximum
                    fr.runs[k].thing3[n].append(np.nanmin(yfield))    # Min
                    fr.runs[k].thing4[n].append(yfield)    # Raw scatter points
                    fr.runs[k].thing5[n].append(np.nanquantile(yfield,0.25))    # 25th percentile
                    fr.runs[k].thing6[n].append(np.nanquantile(yfield,0.50))    # 50th percentile
                    fr.runs[k].thing7[n].append(np.nanquantile(yfield,0.75))    # 75th percentile
                    fr.runs[k].thing8[n].append(np.nanquantile(yfield,0.95))    # 95th percentile
                    if fr.doms[k] == 1:
                        dA = (12*1000)**2    # Area of grid cell [m2]
                    elif fr.doms[k] == 2:
                        dA = (4*1000)**2    # [m2]
                    elif fr.doms[k] == 3:
                        dA = (1.33333*1000)**2    # [m2]
                    fr.runs[k].thing9[n].append(np.nansum(yfield*dA))    # Integral over surface area, units change by factor of [m2]
                elif yfield.ndim == 3:    # Using a 3D field
                    fr.runs[k].thing0[n] = 3    # Record if 2D or 3D field
                    dz = fr.runs[k].myfields[722][fr.doms[k]].grdata[t,:,:,:]    # WRF model layer thickness [m]
                    if fr.doms[k] == 1:
                        dA = (12*1000)**2    # [m2]
                    elif fr.doms[k] == 2:
                        dA = (4*1000)**2    # [m2]
                    elif fr.doms[k] == 3:
                        dA = (1.33333*1000)**2    # [m2]
                    dV = dz*dA    # Volume of each grid cell [m3]
                    dV[np.isnan(yfield)] = np.nan
                    fr.runs[k].thing1[n].append(np.nansum(yfield*dV))    # Volume-integrated total, units change by factor of [m2]
                    fr.runs[k].thing2[n].append(np.nansum(yfield*dV)/np.nansum(dV))    # Volume average
        # Calculate time derivatives
        for n in range(len(fr.fldindx)):
            if fr.runs[k].thing0[n] == 3:    # Currently only calculating time derivative for volumetric mean
                vmean = np.array(fr.runs[k].thing2[n])    # Volume average for full time series
                ddt_vmean = np.full_like(vmean,np.nan)    # Initialize d/dt series.  Only set up for hourly output, units change by factor of [hr-1]
                if np.size(vmean) >= 3:
                    ddt_vmean[0] = (vmean[1]-vmean[0])/1    # First order forward difference for first point
                    ddt_vmean[-1] = (vmean[-1]-vmean[-2])/1    # First order backward difference for last point
                    ddt_vmean[1:-1] = (vmean[2:]-vmean[:-2])/(2*1)    # Second order central difference for interior points
                fr.runs[k].thing3[n] = ddt_vmean
    # Plot observations
    if fr.typeparams[1][0] == False:
        pass
    elif fr.typeparams[1][0] == 'BT_wspdmax':
        plt.plot(fr.runs[0].myobs.BT_datetime_full,fr.runs[0].myobs.BT_wspdmax_full,fr.typeparams[1][1],color=fr.typeparams[1][2],label='Best Track')
    elif fr.typeparams[1][0] == 'BT_slpmin':
        plt.plot(fr.runs[0].myobs.BT_datetime_full,fr.runs[0].myobs.BT_slpmin_full, fr.typeparams[1][1],color=fr.typeparams[1][2],label='Best Track')
    # Plot model results
    if fr.typeparams[6] == False:    # Plot separate curves for each run
        for k in range(len(fr.runs)):
            if fr.typeparams[9] == False:    # Plot each n field separately
                for n in range(len(fr.fldindx)):
                    for ser in fr.typeparams[0]:
                        # Options for 2D fields
                        if ser[0] == 'Mean':
                            yser = fr.runs[k].thing1[n]
                            lab_tag = ''
                        elif ser[0] == 'Max':
                            yser = fr.runs[k].thing2[n]
                            lab_tag = ''
                        elif ser[0] == 'Min':
                            yser = fr.runs[k].thing3[n]
                            lab_tag = ''
                        elif ser[0] == '25pct':
                            yser = fr.runs[k].thing5[n]
                            lab_tag = '25%'
                        elif ser[0] == '50pct':
                            yser = fr.runs[k].thing6[n]
                            lab_tag = '50%'
                        elif ser[0] == '75pct':
                            yser = fr.runs[k].thing7[n]
                            lab_tag = '75%'
                        elif ser[0] == '95pct':
                            yser = fr.runs[k].thing8[n]
                            lab_tag = '95%'
                        elif ser[0] == 'AreaInt':
                            yser = fr.runs[k].thing9[n]
                            lab_tag = ''
                        # Options for 3D fields
                        if ser[0] == 'VolInt':
                            yser = fr.runs[k].thing1[n]
                            lab_tag = ''
                        elif ser[0] == 'VolMean':
                            yser = fr.runs[k].thing2[n]
                            lab_tag = ''
                        elif ser[0] == 'VolMeanDDT':
                            yser = fr.runs[k].thing3[n]
                            lab_tag = ''
                        # Plot series
                        label = fr.legtext[k][n]+lab_tag
                        if fr.typeparams[7] == False:    # Don't plot scatter behind
                            plt.plot(fr.runs[k].time,yser,ser[1][k],color=fr.colors[k][n],label=label,linewidth=ser[2][k])
                        elif fr.typeparams[7]:    # Plot scatter behind, doesn't work for 3D fields
                            for t in fr.runs[k].steplist:
                                tvect = np.array([fr.runs[k].time[t] for el in range(np.size(fr.runs[k].thing4[n][t]))]).flatten()
                                plt.plot(tvect,fr.runs[k].thing4[n][t].flatten(),'.',markersize=2,mec=fr.colors[k][n],mfc=fr.colors[k][n])
                            plt.plot(fr.runs[k].time,yser,ser[1][k],color=fr.colors[k][n],label=label,linewidth=ser[2][k],\
                                    path_effects=[pe.Stroke(linewidth=4,foreground='w'),pe.Normal()])
                            plt.plot([np.nan],[np.nan],'.',markersize=2,mec=fr.colors[k][n],mfc=fr.colors[k][n],label=label+'Scatter')
                        if steplist is not None:
                            if len(steplist) == 1:    # Show current position
                                plt.plot(fr.runs[k].time[steplist[0]],yser[steplist[0]],'o',mec=fr.colors[k][n],mfc='w',mew=2,markersize=10)
            elif fr.typeparams[9] == True:    # Make a calculation involving two or more n fields
                for ser in fr.typeparams[0]:
                    if ser[0] == 'VolInt1_D_AreaInt0':    # 'VolInt' of field 1 divided by 'AreaInt' of field 0
                        yser = np.array(fr.runs[k].thing1[1])/np.array(fr.runs[k].thing9[0])
                    elif ser[0] == 'VolMean1_D_Mean0':    # 'VolMean' of field 1 divided by 'Mean' of field 0
                        yser = np.array(fr.runs[k].thing2[1])/np.array(fr.runs[k].thing1[0])
                    elif ser[0] == 'VolMeanDDT1_D_AreaInt0':    # 'VolMeanDDT' of field 1 divided by 'AreaInt' of field 0
                        yser = np.array(fr.runs[k].thing3[1])/np.array(fr.runs[k].thing9[0])
                    elif ser[0] == 'VolMeanDDT1_D_Mean0':    # 'VolMeanDDT' of field 1 divided by 'Mean' of field 0
                        yser = np.array(fr.runs[k].thing3[1])/np.array(fr.runs[k].thing1[0])
                    if fr.typeparams[10] is not None:    # Smooth yser
                        for m in range(fr.typeparams[10]):    # Loop and smooth
                            yser_sm = np.full_like(yser,np.nan)    # Initialize smoothed series
                            if np.size(yser) >= 3:
                                yser_sm[0] = (yser[0]+yser[1])/2    # First point
                                yser_sm[-1] = (yser[-2]+yser[-1])/2    # Last point
                                yser_sm[1:-1] = (yser[:-2]+yser[1:-1]+yser[2:])/3    # Interior points
                            yser = np.copy(yser_sm)
                    label = fr.legtext[k][0]
                    plt.plot(fr.runs[k].time,yser,ser[1][k],color=fr.colors[k][0],label=label,linewidth=ser[2][k])
                    if steplist is not None:
                        if len(steplist) == 1:    # Show current position
                            plt.plot(fr.runs[k].time[steplist[0]],yser[steplist[0]],'o',mec=fr.colors[k][0],mfc='w',mew=2,markersize=10)
    elif fr.typeparams[6] == True:    # Plot run2 minus run1 difference
        for n in range(len(fr.fldindx)):
            for ser in fr.typeparams[0]:
                if ser[0] == 'Mean':
                    yser = np.array(fr.runs[1].thing1[n]) - np.array(fr.runs[0].thing1[n])
                elif ser[0] == 'Max':
                    yser = np.array(fr.runs[1].thing2[n]) - np.array(fr.runs[0].thing2[n])
                elif ser[0] == 'Min':
                    yser = np.array(fr.runs[1].thing3[n]) - np.array(fr.runs[0].thing3[n])
                label = fr.legtext[0][n]
                plt.plot(fr.runs[0].time,yser,ser[1][0],color=fr.colors[0][n],label=label,linewidth=ser[2][0])
                if steplist is not None:
                    if len(steplist) == 1:    # Show current position
                        plt.plot(fr.runs[0].time[steplist[0]],yser[steplist[0]],'o',mec=fr.colors[0][n],mfc='w',mew=2,markersize=10)
    # Plot lines
    for line in fr.typeparams[5]:
        if line[4] is None:
            plt.plot(line[0],line[1],'-',color=line[2],linewidth=line[3],solid_capstyle='butt',zorder=-1)
        else:
            plt.plot(line[0],line[1],'-',color=line[2],linewidth=line[3],solid_capstyle='butt',zorder=-1,label=line[4])
    # Plot marks (text)
    for mark in fr.typeparams[8]:
        plt.text(mark[2][0],mark[2][1],mark[0],fontsize=mark[1],color=mark[3],weight='bold')


#================================================================================


def plot_mapSR(fr,steplist):

    """
    ------------------------------ Storm-Relative Map --------------------------------
    Plot a storm-relative map with X and Y coordinates.  Can use earth-relative or
    motion-relative axis directions.  Can plot one field with colors and another using
    contours.  Only works with EachStep option and one run.

            Type:       'Map_SR'
            Type parameters:
                        [
                         0: Colormap
                         1: List of contourf levels
                         2: List of colorbar tick marks
                         3: True/False - Plot zero contour?
                         4: 'lin' or 'log' for color scale (levels and ticks are exponents of 10 for 'log')
                         5: [[list of levels to draw contours],[list of (x,y) tuples for contour labels]], or None for nothing
                        ]
            Fields:     ['x_WRF' (ER axes) or 'xRot_WRF' (MR axes), 'y_WRF' or 'yRot_WRF', z1 field (colors), z2 field (contours, if desired)]
            Runs:       [only one run allowed]
            Colors:     None, not used
            Leg params: [None,None,1], not used
    ----------------------------------------------------------------------------------
    """

    # Get fields
    x = fr.runs[0].myfields[fr.fldindx[0]][fr.doms[0]].grdata[steplist[0],:,:]
    y = fr.runs[0].myfields[fr.fldindx[1]][fr.doms[0]].grdata[steplist[0],:,:]
    if fr.filter is None:
        z1 = fr.runs[0].myfields[fr.fldindx[2]][fr.doms[0]].grdata[steplist[0],:,:]
        if len(fr.fldindx) == 4:
            z2 = fr.runs[0].myfields[fr.fldindx[3]][fr.doms[0]].grdata[steplist[0],:,:]
    else:
        z1 = fr.runs[0].myfields[fr.fldindx[2]][fr.doms[0]].grdata_filt[fr.filtindx[2][0]][steplist[0],:,:]
        if len(fr.fldindx) == 4:
            z2 = fr.runs[0].myfields[fr.fldindx[3]][fr.doms[0]].grdata_filt[fr.filtindx[3][0]][steplist[0],:,:]
    
    # Plot
    if fr.typeparams[4] == 'lin':
        plt.contourf(x,y,z1,fr.typeparams[1],cmap=fr.typeparams[0],extend='both')
        colorbar = plt.colorbar(shrink=0.8,extend='both',ticks=fr.typeparams[2])
    elif fr.typeparams[4] == 'log':
        plt.contourf(x,y,z1,np.power(10.,fr.typeparams[1]),cmap=fr.typeparams[0],extend='both',norm=mcols.LogNorm())
        colorbar = plt.colorbar(shrink=0.8,extend='both',ticks=np.power(10.,fr.typeparams[2]))
    cbytick_obj = plt.getp(colorbar.ax.axes,'yticklabels')
    plt.setp(cbytick_obj,fontsize=13)
    if fr.typeparams[3]:    # Plot zero contour for z1 in grey
        plt.contour(x,y,z1,[0.0],colors='grey')
    plt.contour(x,y,fr.runs[0].myfields[3][fr.doms[0]].grdata[steplist[0],:,:],[1.5],colors='k')    # Plot coastline in black
    if fr.typeparams[5][0] is not None:    # Plot labeled contours
        cntr = plt.contour(x,y,z2,fr.typeparams[5][0],colors='k',linewidths=0.5)
        plt.clabel(cntr,fmt='%d',inline=True,colors='k',manual=fr.typeparams[5][1])


#================================================================================


def plot_mapdiffSR(fr,steplist):

    # Get fields
    if fr.filter is None:
        x1 = fr.runs[0].myfields[fr.fldindx[0]][fr.doms[0]].grdata[steplist[0],:,:]
        y1 = fr.runs[0].myfields[fr.fldindx[1]][fr.doms[0]].grdata[steplist[0],:,:]
        z1 = fr.runs[0].myfields[fr.fldindx[2]][fr.doms[0]].grdata[steplist[0],:,:]
        if fr.typeparams[4] == '2runs':
            x2 = fr.runs[1].myfields[fr.fldindx[0]][fr.doms[1]].grdata[steplist[0],:,:]
            y2 = fr.runs[1].myfields[fr.fldindx[1]][fr.doms[1]].grdata[steplist[0],:,:]
            z2 = fr.runs[1].myfields[fr.fldindx[2]][fr.doms[1]].grdata[steplist[0],:,:]
        elif fr.typeparams[4] == '2flds':
            z2 = fr.runs[0].myfields[fr.fldindx[3]][fr.doms[0]].grdata[steplist[0],:,:]
    else:
        x1 = fr.runs[0].myfields[fr.fldindx[0]][fr.doms[0]].grdata_filt[fr.filtindx[0][0]][steplist[0],:,:]
        y1 = fr.runs[0].myfields[fr.fldindx[1]][fr.doms[0]].grdata_filt[fr.filtindx[1][0]][steplist[0],:,:]
        z1 = fr.runs[0].myfields[fr.fldindx[2]][fr.doms[0]].grdata_filt[fr.filtindx[2][0]][steplist[0],:,:]
        if fr.typeparams[4] == '2runs':
            x2 = fr.runs[1].myfields[fr.fldindx[0]][fr.doms[1]].grdata_filt[fr.filtindx[0][1]][steplist[0],:,:]
            y2 = fr.runs[1].myfields[fr.fldindx[1]][fr.doms[1]].grdata_filt[fr.filtindx[1][1]][steplist[0],:,:]
            z2 = fr.runs[1].myfields[fr.fldindx[2]][fr.doms[1]].grdata_filt[fr.filtindx[2][1]][steplist[0],:,:]
        elif fr.typeparams[4] == '2flds':
            z2 = fr.runs[0].myfields[fr.fldindx[3]][fr.doms[0]].grdata_filt[fr.filtindx[3][0]][steplist[0],:,:]
    
    # Calculate difference
    if fr.typeparams[4] == '2runs':    # Remap Run 2 onto Run 1 grid and take difference
        xy2_forInterp = np.transpose(np.array([x2[~np.isnan(z2)].flatten(),y2[~np.isnan(z2)].flatten()]))    # x2 and y2 reformatted for interpolation
        z2_forInterp = z2[~np.isnan(z2)].flatten()    # z2 reformatted for interpolation
        z2_remapped = griddata(xy2_forInterp,z2_forInterp,(x1,y1),method='linear')    # z2 remapped onto z1 grid
        zdiff = z2_remapped - z1
    elif fr.typeparams[4] == '2flds':    # No remap necessary
        zdiff = z2 - z1

    # Plot difference
    plt.contourf(x1,y1,zdiff,fr.typeparams[2],cmap=fr.typeparams[1],extend='both')
    colorbar = plt.colorbar(shrink=0.8,extend='both',ticks=fr.typeparams[3])
    cbytick_obj = plt.getp(colorbar.ax.axes,'yticklabels')
    plt.setp(cbytick_obj,fontsize=13)
    if fr.typeparams[0][0] is None:
        label = '***User must add label***'
    else:
        label = fr.typeparams[0][0]
    plt.text(fr.limits[0] + 0.02*(fr.limits[1] - fr.limits[0]),\
             fr.limits[3] - 0.05*(fr.limits[3] - fr.limits[2]),label,weight='bold')
    if fr.typeparams[5] is not None:    # Plot a line
        plt.plot(np.linspace(fr.typeparams[5][0][0][0],fr.typeparams[5][0][1][0],num=100),\
                 np.linspace(fr.typeparams[5][0][0][1],fr.typeparams[5][0][1][1],num=100),fr.typeparams[5][1],linewidth=2)


#================================================================================


def plot_map(fr,steplist):

    """
    ----------------------------- Earth-Relative Map -------------------------------
    Plot an earth-relative map on a lon/lat grid, with options to plot 2D fields, 
    1D curves, curly vectors of velocity fields, and storm tracks.  For 2D fields and 
    curly vectors, plots fields for first timestep when using an 'AllSteps' option.

        Type:       'Map'
        Type parameters (dictionary):
                    {
                     'BT_trk':
                         [
                          0: True/False - Plot best track?
                          1: True/False - Plot date labels at 00Z?
                          2: True/False - Plot track outside of run0 model time as thin dashed line?
                         ]
                     'model_trk':
                         [
                          0: True/False - Plot model tracks?
                          1: True/False - Plot date labels at 00Z?
                          2: Track curve formatting, right now only 'default' is available
                         ]
                     '2Dfield':
                         [
                          0: 'Model' for run0, 'ModelDiff' for run1-run0, 'ModelPctDiff' for (run1-run0)/run0*100, or None for nothing
                          1: List of contourf levels
                          2: Colormap
                          3: True/False - Plot colorbar?
                          4: List of colorbar tick marks
                          5: 'lin' or 'log' for color scale (levels and ticks are exponents of 10 for 'log')
                          6: 'All3WRFDoms' to overlay all 3 WRF domains (must manually add fields in woacmpy.py script), or None for nothing
                          7: [[list of levels to draw contours],[list of (x,y) tuples for contour labels] or None for no labels], or [None] for nothing
                         ]
                     '1Dfield':
                         [
                          0: None (not currently used)
                         ]
                     'crlyvect':
                         [
                          0: 'wspd10' for 10m d01 winds, or None for nothing
                          1: Scale factor between field magnitude and curly vector length
                          2: Skip interval
                          3: Arrowhead size with respect to data coordinate
                         ]
                     'layout':
                         [0: 'lnd_stock' for stock cartopy land colors, 'lnd_grey' for grey land, or None for nothing
                          1: 'ocn_grey' for grey ocean, or None for nothing
                          2: List of [lon1,lat1,lon2,lat2] lists for positioning storm track labels, use None for each date that you don't want to show
                          3: Increment in lat/lon gridlines
                         ]
                    }
        Fields:     [field for '2Dfield']
        Runs:       All runs to plot tracks for
        Colors:     One color for each run
        Leg labels: One label for each run
    --------------------------------------------------------------------------------
    """

    if steplist is None:
        steplist = fr.runs[0].steplist
    All3WRFDoms_lon = []
    All3WRFDoms_lat = []
    All3WRFDoms_field2D = []

    # 1. Plot 2D field ----------------------------------------------------------
    if fr.typeparams['2Dfield'][0] is not None:
        # Get 2D data to plot
        if fr.typeparams['2Dfield'][0] in ['Model','ModelDiff','ModelPctDiff']:    # Plot model results
            if fr.typeparams['2Dfield'][6] == 'All3WRFDoms':    # Overlay fields from all 3 WRF domains
                plotdoms = [1,2,3]
            else:
                plotdoms = [fr.doms[0]]
            for d in plotdoms:
                if d in [1,2,3]:    # ATM domains
                    lon = fr.runs[0].myfields[ 1][d].grdata[steplist[0],:,:]
                    lat = fr.runs[0].myfields[ 2][d].grdata[steplist[0],:,:]
                    if fr.typeparams['2Dfield'][6] == 'All3WRFDoms':
                        All3WRFDoms_lon.append(lon)
                        All3WRFDoms_lat.append(lat)
                elif d in [4,5,6]:    # WAV domains
                    lon = fr.runs[0].myfields[13][d].grdata[steplist[0],:,:]
                    lat = fr.runs[0].myfields[14][d].grdata[steplist[0],:,:]
                elif d in [7,8,9]:    # OCN domains
                    lon = fr.runs[0].myfields[16][d].grdata[steplist[0],:,:]
                    lat = fr.runs[0].myfields[17][d].grdata[steplist[0],:,:]
                if fr.fldindx[0] in [591,592]:    # Plot running max value, uses its own grid
                    lon = wg.lon_MaxField
                    lat = wg.lat_MaxField
                if fr.filter is None:
                    z0 = fr.runs[0].myfields[fr.fldindx[0]][d].grdata[steplist[0],:,:]
                    if fr.typeparams['2Dfield'][0] in ['ModelDiff','ModelPctDiff']:
                        z1 = fr.runs[1].myfields[fr.fldindx[0]][d].grdata[steplist[0],:,:]
                else:
                    z0 = fr.runs[0].myfields[fr.fldindx[0]][d].grdata_filt[fr.filtindx[0][0]][steplist[0],:,:]
                    if fr.typeparams['2Dfield'][0] in ['ModelDiff','ModelPctDiff']:
                        z1 = fr.runs[1].myfields[fr.fldindx[0]][d].grdata_filt[fr.filtindx[0][1]][steplist[0],:,:]
                if fr.typeparams['2Dfield'][0] == 'Model':
                    field2D = z0
                elif fr.typeparams['2Dfield'][0] == 'ModelDiff':    # Don't interpolate, assumes same grid
                    field2D = z1 - z0
                elif fr.typeparams['2Dfield'][0] == 'ModelPctDiff':    # Don't interpolate, assumes same grid
                    field2D = (z1 - z0)/z0*100
                if fr.typeparams['2Dfield'][6] == 'All3WRFDoms':
                    All3WRFDoms_field2D.append(field2D)
        # Plot fields
        if fr.typeparams['2Dfield'][6] == 'All3WRFDoms':    # Plot d03 over d02 over d01
            if fr.typeparams['2Dfield'][5] == 'lin':
                plt.contourf(All3WRFDoms_lon[0],All3WRFDoms_lat[0],All3WRFDoms_field2D[0],fr.typeparams['2Dfield'][1],              cmap=fr.typeparams['2Dfield'][2],extend='both',zorder=0)
                plt.contourf(All3WRFDoms_lon[1],All3WRFDoms_lat[1],All3WRFDoms_field2D[1],fr.typeparams['2Dfield'][1],              cmap=fr.typeparams['2Dfield'][2],extend='both',zorder=0)
                plt.contourf(All3WRFDoms_lon[2],All3WRFDoms_lat[2],All3WRFDoms_field2D[2],fr.typeparams['2Dfield'][1],              cmap=fr.typeparams['2Dfield'][2],extend='both',zorder=0)
            elif fr.typeparams['2Dfield'][5] == 'log':
                plt.contourf(All3WRFDoms_lon[0],All3WRFDoms_lat[0],All3WRFDoms_field2D[0],np.power(10.,fr.typeparams['2Dfield'][1]),cmap=fr.typeparams['2Dfield'][2],extend='both',zorder=0,norm=mcols.LogNorm())
                plt.contourf(All3WRFDoms_lon[1],All3WRFDoms_lat[1],All3WRFDoms_field2D[1],np.power(10.,fr.typeparams['2Dfield'][1]),cmap=fr.typeparams['2Dfield'][2],extend='both',zorder=0,norm=mcols.LogNorm())
                plt.contourf(All3WRFDoms_lon[2],All3WRFDoms_lat[2],All3WRFDoms_field2D[2],np.power(10.,fr.typeparams['2Dfield'][1]),cmap=fr.typeparams['2Dfield'][2],extend='both',zorder=0,norm=mcols.LogNorm())
        else:    # Plot the field for the one requested domain
            if fr.typeparams['2Dfield'][5] == 'lin':
                plt.contourf(lon,lat,field2D,fr.typeparams['2Dfield'][1],              cmap=fr.typeparams['2Dfield'][2],extend='both',zorder=0)
            elif fr.typeparams['2Dfield'][5] == 'log':
                plt.contourf(lon,lat,field2D,np.power(10.,fr.typeparams['2Dfield'][1]),cmap=fr.typeparams['2Dfield'][2],extend='both',zorder=0,norm=mcols.LogNorm())
        if fr.typeparams['2Dfield'][3]:    # Add colorbar
            if fr.typeparams['2Dfield'][5] == 'lin':
                colorbar = plt.colorbar(shrink=0.8,extend='both',ticks=fr.typeparams['2Dfield'][4])
            elif fr.typeparams['2Dfield'][5] == 'log':
                colorbar = plt.colorbar(shrink=0.8,extend='both',ticks=np.power(10.,fr.typeparams['2Dfield'][4]))
            cbytick_obj = plt.getp(colorbar.ax.axes,'yticklabels')
            plt.setp(cbytick_obj,fontsize=fr.fontsize)
        if fr.typeparams['2Dfield'][6] is None and fr.typeparams['2Dfield'][7][0] is not None:    # Plot contours on top of colored field
            cntr = plt.contour(lon,lat,field2D,fr.typeparams['2Dfield'][7][0],colors='k',linewidths=0.5)
            if fr.typeparams['2Dfield'][7][1] is not None:    # Label contours
                plt.clabel(cntr,fmt='%d',inline=False,colors='k',manual=fr.typeparams['2Dfield'][7][1])

    # 2. Plot 1D field -----------------------------------------------------
    if fr.typeparams['1Dfield'][0] is not None:    # Not currently used
        pass

    # 3. Plot curly vectors -----------------------------------------------
    if fr.typeparams['crlyvect'][0] is not None:
        lonlat_buffer = 2    # Buffer for cropping fields [deg]
        lon1 = fr.limits[0] - lonlat_buffer
        lon2 = fr.limits[1] + lonlat_buffer
        lat1 = fr.limits[2] - lonlat_buffer
        lat2 = fr.limits[3] + lonlat_buffer
        """
        if fr.typeparams['crlyvect'][0] == 'SurfCurr':
            lon_full = fr.runs[0].myfields[16][5].grdata[steplist[0],:,:]
            lat_full = fr.runs[0].myfields[17][5].grdata[steplist[0],:,:]
            indx_lon1 = np.searchsorted(lon_full[0,:],lon1)
            indx_lon2 = np.searchsorted(lon_full[0,:],lon2)
            indx_lat1 = np.searchsorted(lat_full[:,0],lat1)
            indx_lat2 = np.searchsorted(lat_full[:,0],lat2)
            lon = lon_full[indx_lat1:indx_lat2+1,indx_lon1:indx_lon2+1]
            lat = lat_full[indx_lat1:indx_lat2+1,indx_lon1:indx_lon2+1]
            usurf = fr.runs[0].myfields[18][5].grdata[steplist[0],:,:][indx_lat1:indx_lat2+1,indx_lon1:indx_lon2+1]
            vsurf = fr.runs[0].myfields[19][5].grdata[steplist[0],:,:][indx_lat1:indx_lat2+1,indx_lon1:indx_lon2+1]
            curly_vectors(lon,lat,usurf,vsurf,scale_factor=fr.typeparams['crlyvect'][1],\
                    skip=fr.typeparams['crlyvect'][2],linewidth=0.5,color='k',\
                    arrowhead_min_length=fr.typeparams['crlyvect'][3],\
                    arrowhead_max_length=fr.typeparams['crlyvect'][3],\
                    verbose=False,interp='nn',zorder=1)
            if fr.typeparams['crlyvect'][4] is None:
                label = label + 'Arrows: Surface Current Speed\n'
        """
        if fr.typeparams['crlyvect'][0] == 'wspd10':    # WRF 10m winds from d01
            lon_full = fr.runs[0].myfields[1][1].grdata[steplist[0],:,:]
            lat_full = fr.runs[0].myfields[2][1].grdata[steplist[0],:,:]
            inbounds = np.logical_and(np.logical_and(lon_full > lon1,lon_full < lon2),\
                                      np.logical_and(lat_full > lat1,lat_full < lat2))
            whichlons = np.nonzero(np.sum(inbounds,axis=0))[0]
            whichlats = np.nonzero(np.sum(inbounds,axis=1))[0]
            indx_lon1 = whichlons[0]
            indx_lon2 = whichlons[-1]
            indx_lat1 = whichlats[0]
            indx_lat2 = whichlats[-1]
            lon = lon_full[indx_lat1:indx_lat2+1,indx_lon1:indx_lon2+1]
            lat = lat_full[indx_lat1:indx_lat2+1,indx_lon1:indx_lon2+1]
            u = fr.runs[0].myfields[4][1].grdata[steplist[0],:,:][indx_lat1:indx_lat2+1,indx_lon1:indx_lon2+1]
            v = fr.runs[0].myfields[5][1].grdata[steplist[0],:,:][indx_lat1:indx_lat2+1,indx_lon1:indx_lon2+1]
            curly_vectors(lon,lat,u,v,scale_factor=fr.typeparams['crlyvect'][1],\
                    skip=fr.typeparams['crlyvect'][2],linewidth=0.5,color='k',\
                    arrowhead_min_length=fr.typeparams['crlyvect'][3],\
                    arrowhead_max_length=fr.typeparams['crlyvect'][3],\
                    verbose=False,interp='nn',zorder=4)

    # 4. Plot track datetime labels ---------------------------------------
    if fr.typeparams['BT_trk'][1] or fr.typeparams['model_trk'][1]:
        strm = fr.runs[0].myobs
        track_times = []
        j = -1
        for i in np.arange(np.size(strm.BT_datetime_6hrly_full)):
            if strm.BT_datetime_6hrly_full[i].hour in [0]:
                j += 1
                if fr.typeparams['layout'][2][j] is None:
                    track_times.append(0)
                else:
                    track_times.append(strm.BT_datetime_6hrly_full[i])
                    track_label = '%02d'%strm.BT_datetime_6hrly_full[i].hour + 'Z'\
                            + ' %02d '%strm.BT_datetime_6hrly_full[i].day\
                            + strm.BT_datetime_6hrly_full[i].strftime('%b')
                    plt.text(fr.typeparams['layout'][2][j][0],fr.typeparams['layout'][2][j][1],\
                            track_label,fontsize=10,zorder=10)

    # 5. Plot best track position -----------------------------------------
    if fr.typeparams['BT_trk'][0]:
        strm = fr.runs[0].myobs
        BT_datetime = strm.BT_datetime_full
        BT_lon0 = strm.BT_lon0_full
        BT_lat0 = strm.BT_lat0_full
        BT_datetime_6hrly = strm.BT_datetime_6hrly_full
        BT_lon0_6hrly = strm.BT_lon0_6hrly_full
        BT_lat0_6hrly = strm.BT_lat0_6hrly_full
        if fr.typeparams['BT_trk'][2]:    # Plot track outside of run0 model time as dashed line
            cutbfr       = bisect([dt.timestamp() for dt in BT_datetime],      fr.runs[0].startTime.timestamp())
            cutaft       = bisect([dt.timestamp() for dt in BT_datetime],      fr.runs[0].endTime.timestamp())
            cutbfr_6hrly = bisect([dt.timestamp() for dt in BT_datetime_6hrly],fr.runs[0].startTime.timestamp())
            cutaft_6hrly = bisect([dt.timestamp() for dt in BT_datetime_6hrly],fr.runs[0].endTime.timestamp())
            plt.plot(BT_lon0[:cutbfr],        BT_lat0[:cutbfr],        '--',color='k',zorder=5,linewidth=1)
            plt.plot(BT_lon0[cutbfr-1:cutaft],BT_lat0[cutbfr-1:cutaft],'-', color='k',zorder=5)
            plt.plot(BT_lon0[cutaft-1:],      BT_lat0[cutaft-1:],      '--',color='k',zorder=5,linewidth=1)
            plt.plot(BT_lon0_6hrly[:cutbfr_6hrly],              BT_lat0_6hrly[:cutbfr_6hrly],              'o',color='k',zorder=5,markersize=3)
            plt.plot(BT_lon0_6hrly[cutbfr_6hrly-1:cutaft_6hrly],BT_lat0_6hrly[cutbfr_6hrly-1:cutaft_6hrly],'o',color='k',zorder=5)
            plt.plot(BT_lon0_6hrly[cutaft_6hrly-1:],            BT_lat0_6hrly[cutaft_6hrly-1:],            'o',color='k',zorder=5,markersize=3)
        else:    # Use solid black line for entire best track
            plt.plot(BT_lon0,BT_lat0,'-',color='k',zorder=5)
            plt.plot(BT_lon0_6hrly,BT_lat0_6hrly,'o',color='k',zorder=5)
        plt.plot([np.nan],[np.nan],linestyle='-',color='k',marker='o',mec='k',mfc='k',label='Best Track')
        if fr.typeparams['BT_trk'][1]:
            for i in range(len(BT_datetime_6hrly)):
                if BT_datetime_6hrly[i] in track_times:
                    plt.plot([BT_lon0_6hrly[i],fr.typeparams['layout'][2][track_times.index(BT_datetime_6hrly[i])][2]],\
                             [BT_lat0_6hrly[i],fr.typeparams['layout'][2][track_times.index(BT_datetime_6hrly[i])][3]],\
                             'k',linewidth=0.5,zorder=10)

    # 6. Plot model track position ----------------------------------------
    if fr.typeparams['model_trk'][0]:
        for k in range(len(fr.runs)):
            if fr.legtext is None:
                serlabel = '***user must specify***'
            elif fr.legtext[k][0] is None:
                serlabel = '***user must specify***'
            elif fr.legtext[k][0] is not None:
                serlabel = fr.legtext[k][0]
            if len(steplist) == 1:
                model_lon0 = fr.runs[k].lon0[:steplist[0]+1]
                model_lat0 = fr.runs[k].lat0[:steplist[0]+1]
                model_lon0_6hrly = fr.runs[k].lon0_6hrly[:steplist[0]+1]
                model_lat0_6hrly = fr.runs[k].lat0_6hrly[:steplist[0]+1]
            else:
                model_lon0 = fr.runs[k].lon0
                model_lat0 = fr.runs[k].lat0
                model_lon0_6hrly = fr.runs[k].lon0_6hrly
                model_lat0_6hrly = fr.runs[k].lat0_6hrly
            if fr.typeparams['model_trk'][2] == 'default':
                plt.plot(model_lon0,model_lat0,'-',color=fr.colors[k][0],zorder=6)
                plt.plot(model_lon0_6hrly,model_lat0_6hrly,'o',color=fr.colors[k][0],zorder=6)
                plt.plot([np.nan],[np.nan],linestyle='-',color=fr.colors[k][0],marker='o',label=serlabel)
            if len(steplist) == 1:
                plt.plot(model_lon0[-1],model_lat0[-1],'o',mec=fr.colors[k][0],mfc='#00FF00',mew=2,markersize=9,zorder=6)
            if fr.typeparams['model_trk'][1]:
                for i in range(len(fr.runs[k].time)):
                    if len(steplist) == 1 and fr.runs[k].time[i] > fr.runs[k].time[steplist[0]]:
                        pass
                    else:
                        if fr.runs[k].time[i] in track_times:
                            plt.plot([fr.runs[k].lon0[i],fr.typeparams['layout'][2][track_times.index(fr.runs[k].time[i])][2]],\
                                     [fr.runs[k].lat0[i],fr.typeparams['layout'][2][track_times.index(fr.runs[k].time[i])][3]],\
                                     'k',linewidth=0.5,zorder=10)


#===============================================================================


def plot_hovRstorm(fr,steplist_0):

    steplist = steplist_0
    if steplist_0 is None:
        steplist = fr.runs[0].steplist    # If using AllStepsCmprStrms, user must ensure that runs have same steplist
    if fr.typeparams[0]:    # Use motion-relative coordinates
        Xindx = 503
        Yindx = 504
    else:
        Xindx = 10
        Yindx = 11
    # Hovmoller of model fields: values along a ray or azimuthal average/total of 2D horizontal field.
    # Can do colors for one field and contours for another, with same settings.
    if fr.typeparams[12] is None:    # Plot contourf of fr.fldindx[1] only
        loopflds = [1]
    else:    # Plot contourf of fr.fldindx[1] and contours of fr.fldindx[2]
        loopflds = [1,2]
    for i in loopflds:    # Loop over selected fields and plot
        # 1. Obtain required fields
        findx = fr.fldindx[i]
        if fr.typeparams[1] in ['Diff1M0','PctD1V0']:    # Use run1 minus run0 difference or % diff of run1 versus run0
            x0     = np.array([fr.runs[0].myfields[Xindx][fr.doms[0]].grdata[t,:,:] for t in steplist])    # [km]
            x1     = np.array([fr.runs[1].myfields[Xindx][fr.doms[1]].grdata[t,:,:] for t in steplist])    # [km]
            y0     = np.array([fr.runs[0].myfields[Yindx][fr.doms[0]].grdata[t,:,:] for t in steplist])    # [km]
            y1     = np.array([fr.runs[1].myfields[Yindx][fr.doms[1]].grdata[t,:,:] for t in steplist])    # [km]
            Rstorm = np.array([fr.runs[0].myfields[   12][fr.doms[0]].grdata[t,:,:] for t in steplist])    # [km]
            if fr.filter is None:
                field0 = np.array([fr.runs[0].myfields[findx][fr.doms[0]].grdata[t,:,:] for t in steplist])
                field1 = np.array([fr.runs[1].myfields[findx][fr.doms[1]].grdata[t,:,:] for t in steplist])
            else:
                field0 = np.array([fr.runs[0].myfields[findx][fr.doms[0]].grdata_filt[fr.filtindx[i][0]][t,:,:] for t in steplist])
                field1 = np.array([fr.runs[1].myfields[findx][fr.doms[1]].grdata_filt[fr.filtindx[i][1]][t,:,:] for t in steplist])
            X = x0    # [km]
            Y = y0    # [km]
            field1_remapped = []
            for t in range(len(steplist)):
                xy1_forInterp = np.transpose(np.array([x1[t,:,:].flatten(),y1[t,:,:].flatten()]))    # x1 and y1 reformatted for interpolation
                xy0_grid = (x0[t,:,:],y0[t,:,:])    # field0 grid to interpolate onto
                field1_forInterp = field1[t,:,:].flatten()    # field1 reformatted for interpolation
                field1_remapped.append(griddata(xy1_forInterp,field1_forInterp,xy0_grid,method='linear'))    # Remap field1 onto field0 grid
            field1_remapped = np.array(field1_remapped)
            if fr.typeparams[1] == 'Diff1M0':
                field2D = field1_remapped - field0    # Calculate field1 minus field0 difference (field is 2D + time)
            elif fr.typeparams[1] == 'PctD1V0':
                field2D = (field1_remapped - field0)/field0*100    # Calculate percent difference
        else:    # Use fields for the provided run (only one should be provided)
            X       = np.array([fr.runs[0].myfields[Xindx][fr.doms[0]].grdata[t,:,:] for t in steplist])    # [km]
            Y       = np.array([fr.runs[0].myfields[Yindx][fr.doms[0]].grdata[t,:,:] for t in steplist])    # [km]
            Rstorm  = np.array([fr.runs[0].myfields[   12][fr.doms[0]].grdata[t,:,:] for t in steplist])    # [km]
            if fr.filter is None:
                field2D = np.array([fr.runs[0].myfields[findx][fr.doms[0]].grdata[t,:,:] for t in steplist])
            else:
                field2D = np.array([fr.runs[0].myfields[findx][fr.doms[0]].grdata_filt[fr.filtindx[i][0]][t,:,:] for t in steplist])
        # 2. Calculate azimuthal mean/total or 1D profile
        fieldHov = []
        if fr.typeparams[2] in ['Azimean','Azitot']:    # Calculate azimuthal mean or total
            if fr.typeparams[3] in ['FR','FL','RL','RR','Left','Right']:
                if fr.typeparams[0] == False:    # Only allow quadrant averaging if using motion-relative coordinates
                    Rstorm[:,:,:] = np.nan
                else:
                    if fr.typeparams[3] == 'FR':
                        Rstorm[~np.logical_and(X >= 0,Y >= 0)] = np.nan
                    elif fr.typeparams[3] == 'FL':
                        Rstorm[~np.logical_and(X <  0,Y >= 0)] = np.nan
                    elif fr.typeparams[3] == 'RL':
                        Rstorm[~np.logical_and(X <  0,Y <  0)] = np.nan
                    elif fr.typeparams[3] == 'RR':
                        Rstorm[~np.logical_and(X >= 0,Y <  0)] = np.nan
                    elif fr.typeparams[3] == 'Left':
                        Rstorm[X >= 0] = np.nan
                    elif fr.typeparams[3] == 'Right':
                        Rstorm[X <  0] = np.nan
            for t in range(len(steplist)):
                if fr.doms[0] == 1:
                    numbinsR = 20
                elif fr.doms[0] in [2,3]:
                    numbinsR = 100
                field1D,rbin = plot_hovRstorm_calcmeans(fr,Rstorm[t,:,:],field2D[t,:,:],numbinsR)    # Azimuthal mean at each timestep, and radius vector
                if fr.typeparams[2] == 'Azitot':    # Integrate azimuthally, not set up yet for Left/Right
                    if fr.typeparams[3] in ['FR','FL','RL','RR']:    # Integrate over arc length of pi/2
                        field1D = rbin*1000*np.pi/2*field1D*1e-6    # Units change by factor of [m], value reduced by factor of 1e-6
                    elif fr.typeparams[3] == 'All':    # Integrate over arc length of 2*pi
                        field1D = rbin*1000*np.pi*2*field1D*1e-6    # Units change by factor of [m], value reduced by factor of 1e-6
                fieldHov.append(field1D)
            R1D = rbin    # [km]
        elif fr.typeparams[2] == 'Line':    # Calculate profile along a set line
            line = np.transpose(np.array([np.linspace(fr.typeparams[3][0][0],fr.typeparams[3][1][0],num=100),\
                                          np.linspace(fr.typeparams[3][0][1],fr.typeparams[3][1][1],num=100)]))
            for t in range(len(steplist)):
                XY_forInterp = np.transpose(np.array([X[t,:,:].flatten(),Y[t,:,:].flatten()]))
                field2D_forInterp = field2D[t,:,:].flatten()
                fieldHov.append(np.array(griddata(XY_forInterp,field2D_forInterp,line,method='linear')))
            R1D = np.sqrt(line[:,0]**2 + line[:,1]**2)    # [km]
        fieldHov = np.array(fieldHov)
        # 3. Plot hovmoller fields
        dt = np.array([fr.runs[0].time[t] for t in steplist])
        if i == 1:    # Plot contourf of fr.fldindx[1]
            if fr.typeparams[9] == 'lin':
                plt.contourf(R1D,dt,fieldHov,fr.typeparams[4],cmap=fr.typeparams[5],extend='both')
                if fr.typeparams[6]:    # Plot colorbar
                    colorbar = plt.colorbar(shrink=0.8,extend='both',ticks=fr.typeparams[7])
            elif fr.typeparams[9] == 'log':
                plt.contourf(R1D,dt,fieldHov,np.power(10.,fr.typeparams[4]),cmap=fr.typeparams[5],extend='both',norm=mcols.LogNorm())
                if fr.typeparams[6]:    # Plot colorbar
                    colorbar = plt.colorbar(shrink=0.8,extend='both',ticks=np.power(10.,fr.typeparams[7]))
            if fr.typeparams[6]:
                cbytick_obj = plt.getp(colorbar.ax.axes,'yticklabels')
                plt.setp(cbytick_obj,fontsize=12)
            if fr.typeparams[11] and fr.typeparams[9] == 'lin':    # Plot zero contour
                plt.contour(R1D,dt,fieldHov,[0.0],colors='grey')
        elif i == 2:    # Plot contours of fr.fldindx[2]
            cntr = plt.contour(R1D,dt,fieldHov,fr.typeparams[12][1],colors=fr.typeparams[12][0],linewidths=0.5)
            plt.clabel(cntr,fmt='%d',inline=True,colors=fr.typeparams[12][0],manual=fr.typeparams[12][2])
            plt.plot([np.nan],[np.nan],'-',c=fr.typeparams[12][0],lw=0.5,label=fr.typeparams[12][3])
    # Plot RMW, if requested
    if fr.typeparams[8][0]:
        for k in range(len(fr.runs)):    # Loop over runs
            if fr.runs[k].RMW == []:
                wd.calcRMW(fr.runs[k])    # Calculate RMW
            RMW = np.array([fr.runs[k].RMW[t] for t in steplist])
            plt.plot(RMW,dt,fr.typeparams[8][1][k],linestyle=fr.typeparams[8][2][k],label=fr.typeparams[8][3][k])
    # Plot lines
    for line in fr.typeparams[10]:
        if line[4] is None:
            plt.plot(line[0],line[1],'-',color=line[2],linewidth=line[3],solid_capstyle='butt')
        else:
            plt.plot(line[0],line[1],'-',color=line[2],linewidth=line[3],solid_capstyle='butt',label=line[4])


def plot_hovRstorm_calcmeans(fr,Rstorm,field,numbinsR):
    return plot_wrfvert_calcmeans_1D(fr,Rstorm,field,numbinsR=numbinsR)


#================================================================================


def plot_hycvert(fr,steplist):

    lon     = fr.runs[0].myfields[16][5].grdata[steplist[0],:,:][0,:]    # Longitude vector [deg]
    lat     = fr.runs[0].myfields[17][5].grdata[steplist[0],:,:][:,0]    # Latitude vector [deg]
    depth3D = fr.runs[0].myfields[28][5].grdata[steplist[0],:,:,:]    # 3d depth [m]
    if fr.typeparams[7]:    # Import MLT
        mlt2Drun1 = fr.runs[0].myfields[476][5].grdata[steplist[0],:,:]    # MLT [m]
    if fr.typeparams[6] == False:    # Plot run1
        field3D = fr.runs[0].myfields[fr.fldindx[0]][fr.doms[0]].grdata[steplist[0],:,:,:]
    elif fr.typeparams[6]:    # Plot run2 - run1
        field3D = fr.runs[1].myfields[fr.fldindx[0]][fr.doms[1]].grdata[steplist[0],:,:,:] - \
                  fr.runs[0].myfields[fr.fldindx[0]][fr.doms[0]].grdata[steplist[0],:,:,:]
        if fr.typeparams[7]:    # Import MLT
            mlt2Drun2 = fr.runs[1].myfields[476][5].grdata[steplist[0],:,:]    # MLT [m]
    if fr.typeparams[0] == 'SameLat':
        x = np.array([lon for d in range(np.shape(depth3D)[0])])    # Horizontal coordinate of section
        indx = np.argmin(np.abs(lat - fr.typeparams[1]))    # Index of section
        depth = depth3D[:,indx,:]    # Depth coordinate of section
        field = field3D[:,indx,:]    # Field values of section
        if fr.typeparams[7]:
            xvect = lon    # Longitude vector
            mltrun1 = mlt2Drun1[indx,:]
            if fr.typeparams[6]:
                mltrun2 = mlt2Drun2[indx,:]
    elif fr.typeparams[0] == 'SameLon':
        x = np.array([lat for d in range(np.shape(depth3D)[0])])    # Horizontal coordinate of section
        indx = np.argmin(np.abs(lon - fr.typeparams[1]))    # Index of section
        depth = depth3D[:,:,indx]    # Depth coordinate of section
        field = field3D[:,:,indx]    # Field values of section
        if fr.typeparams[7]:
            xvect = lat    # Latitude vector
            mltrun1 = mlt2Drun1[:,indx]
            if fr.typeparams[6]:
                mltrun2 = mlt2Drun2[:,indx]
    plt.contourf(x,depth,field,fr.typeparams[3],cmap=fr.typeparams[2],extend='both')
    if fr.typeparams[5]:
        colorbar = plt.colorbar(shrink=0.8,extend='both',ticks=fr.typeparams[4])
        cbytick_obj = plt.getp(colorbar.ax.axes,'yticklabels')
        plt.setp(cbytick_obj,fontsize=12)
    if fr.typeparams[7]:    # Plot MLT
        plt.plot(xvect,mltrun1,'k',linestyle='-',linewidth=2,label=fr.legtext[0][0])
        if fr.typeparams[6]:
            plt.plot(xvect,mltrun2,'k',linestyle='--',linewidth=2,label=fr.legtext[1][0])


#================================================================================


def plot_wrfocnvert(fr,steplist):

    """
    ---------------------- Vertical Cross Section of WRF Ocean -----------------------
    Plot a vertical cross section of the WRF ocean.  Sections can only be taken
    along WRF X and Y grid axes (not set up to interpolate onto a user-specified
    line).  Can choose section as 'SameLat' or 'SameLon', where user-specified 
    latitude or longitude corresponds approximately to the center of the WRF 
    domain (map projections may cause X and Y axes to not be coincident with
    south-north and west-east Earth axes).  Can also plot a mixed layer thickness
    from WRF ocean model output.  Can plot for single run or difference between
    two runs with the same vert/horiz grids.  Only works with EachStep option.

            Type:       'WRFOcnVert'
            Type parameters:
                        [
                         0: 'SameLat' to plot a west-east section, or 'SameLon' to plot a north-south section
                         1: float [deg], the latitude or longitude (-180 to 180) at which to take section
                         2: Colormap
                         3: List of contourf levels
                         4: List of colorbar tick marks
                         5: True/False: Plot colorbar?
                         6: True/False: False to plot run0, True to plot run1 - run0 difference
                         7: True/False: Plot MLT?
                        ]
            Fields:     ['OM_DEPTH' (must have this),3D ocean field for contourf, 2D MLT plotted as a curve if requested]
            Runs:       [run0, run1 if taking difference as run1 - run0]
            Colors:     Not used
            Filters:    Not currently set up to allow any filters
            Leg params: Labels are used for the MLT curves
    --------------------------------------------------------------------------------
    """

    lon     = fr.runs[0].myfields[indx( 'lon_WRF')][fr.doms[0]].grdata[steplist[0],:,:]    # WRF longitude [deg], assume run0 grid for all runs
    lat     = fr.runs[0].myfields[indx( 'lat_WRF')][fr.doms[0]].grdata[steplist[0],:,:]    # WRF latitude [deg]
    depth3D = fr.runs[0].myfields[indx('OM_DEPTH')][fr.doms[0]].grdata[steplist[0],:,:,:]    # 3D WRF ocean depth [m]
    if fr.typeparams[7]:    # Import MLT for run0
        MLT2D_run0 = fr.runs[0].myfields[fr.fldindx[2]][fr.doms[0]].grdata[steplist[0],:,:]    # 2D MLT [m]
    if fr.typeparams[6] == False:    # Import run0 3D field
        field3D = fr.runs[0].myfields[fr.fldindx[1]][fr.doms[0]].grdata[steplist[0],:,:,:]
    elif fr.typeparams[6]:    # Import run1 - run0 3D field difference
        field3D = fr.runs[1].myfields[fr.fldindx[1]][fr.doms[1]].grdata[steplist[0],:,:,:] - \
                  fr.runs[0].myfields[fr.fldindx[1]][fr.doms[0]].grdata[steplist[0],:,:,:]
        if fr.typeparams[7]:    # Import MLT for run1
            MLT2D_run1 = fr.runs[1].myfields[fr.fldindx[2]][fr.doms[1]].grdata[steplist[0],:,:]    # 2D MLT [m]
    if fr.typeparams[0] == 'SameLat':
        indxCS = np.argmin(np.abs(lat[:,np.shape(lat)[1]//2] - fr.typeparams[1]))
        x = np.array([lon[indxCS,:] for d in range(np.shape(depth3D)[0])])    # Horizontal (longitude) coordinates of section
        depth = depth3D[:,indxCS,:]    # Vertical (depth) coordinates of section
        field = field3D[:,indxCS,:]    # Field we are looking at
        if fr.typeparams[7]:
            xvect = lon[indxCS,:]    # Longitude vector for MLT curve
            MLT_run0 = MLT2D_run0[indxCS,:]
            if fr.typeparams[6]:
                MLT_run1 = MLT2D_run1[indxCS,:]
    elif fr.typeparams[0] == 'SameLon':
        indxCS = np.argmin(np.abs(lon[np.shape(lon)[0]//2,:] - fr.typeparams[1]))
        x = np.array([lat[:,indxCS] for d in range(np.shape(depth3D)[0])])    # Horizontal (latitude) coordinates of section
        depth = depth3D[:,:,indxCS]    # Vertical (depth) coordinates of section
        field = field3D[:,:,indxCS]    # Field we are looking at
        if fr.typeparams[7]:
            xvect = lat[:,indxCS]    # Latitude vector for MLT curve
            MLT_run0 = MLT2D_run0[:,indxCS]
            if fr.typeparams[6]:
                MLT_run1 = MLT2D_run1[:,indxCS]
    plt.contourf(x,depth,field,fr.typeparams[3],cmap=fr.typeparams[2],extend='both')
    if fr.typeparams[5]:
        colorbar = plt.colorbar(shrink=0.8,extend='both',ticks=fr.typeparams[4])
        cbytick_obj = plt.getp(colorbar.ax.axes,'yticklabels')
        plt.setp(cbytick_obj,fontsize=12)
    if fr.typeparams[7]:    # Plot MLT
        plt.plot(xvect,MLT_run0,'k',linestyle='-',linewidth=2,label=fr.legtext[0][0])
        if fr.typeparams[6]:
            plt.plot(xvect,MLT_run1,'k',linestyle='--',linewidth=2,label=fr.legtext[1][0])


#================================================================================


def plot_bulkTCprops(fr,steplist_0):

    for j in range(len(fr.runs)):
        if steplist_0 is None:
            steplist = fr.runs[j].steplist
        else:
            steplist = steplist_0
        blkprops = np.full((len(fr.fldindx),len(steplist)),np.nan,dtype=float)
        Rstorm = np.array([fr.runs[j].myfields[ 12][fr.doms[j]].grdata[t,:,:] for t in steplist])
        RbyRMW = np.array([fr.runs[j].myfields[375][fr.doms[j]].grdata[t,:,:] for t in steplist])
        for n in range(len(fr.fldindx)):
            if fr.filter is None:
                field = np.array([fr.runs[j].myfields[fr.fldindx[n]][fr.doms[j]].grdata[t,...] for t in steplist])
            else:
                field = np.array([fr.runs[j].myfields[fr.fldindx[n]][fr.doms[j]].grdata_filt[fr.filtindx[n][j]][t,...] for t in steplist])
            if fr.typeparams[0][n][0] in ['AziMnExt','AziMnLoc']:    # Calculate azimuthal-mean profile
                for t in range(len(steplist)):
                    if fr.typeparams[0][n][1] == 'Rstorm':
                        Rcoord = Rstorm[t,:,:]
                    elif fr.typeparams[0][n][1] == 'RbyRMW':
                        Rcoord = RbyRMW[t,:,:]
                    numbins = 100
                    binwid = (fr.typeparams[0][n][2][1] - fr.typeparams[0][n][2][0])/numbins
                    bintol = binwid/2
                    Rbin = np.linspace(fr.typeparams[0][n][2][0]+bintol,fr.typeparams[0][n][2][1]-bintol,numbins)    # Rcoord bin centers
                    Rindx = np.round((Rcoord-fr.typeparams[0][n][2][0])/(fr.typeparams[0][n][2][1]-fr.typeparams[0][n][2][0])*numbins-0.5)
                    Rindx[Rindx <  0] = np.nan
                    Rindx[Rindx > 99] = np.nan
                    fieldmean = np.full_like(Rbin,np.nan)
                    for b in range(numbins):
                        thisbin = np.logical_and(Rindx > b-0.01,Rindx < b+0.01)
                        fieldmean[b] = np.nanmean(field[t,:,:][thisbin])
                    if fr.typeparams[0][n][0] == 'AziMnExt':
                        if fr.typeparams[0][n][3] == 'Max':
                            blkprops[n,t] = np.nanmax(fieldmean)
                        elif fr.typeparams[0][n][3] == 'Min':
                            blkprops[n,t] = np.nanmin(fieldmean)
                    elif fr.typeparams[0][n][0] == 'AziMnLoc':
                        if np.nanmax(fieldmean) >= fr.typeparams[0][n][3] and np.nanmin(fieldmean) <= fr.typeparams[0][n][3]:
                            if fr.typeparams[0][n][4] == 'OutsideMax':
                                fieldmean[Rbin < Rbin[np.nanargmax(fieldmean)]] = np.nan
                            elif fr.typeparams[0][n][4] == 'OutsideMin':
                                fieldmean[Rbin < Rbin[np.nanargmin(fieldmean)]] = np.nan
                            blkprops[n,t] = Rbin[np.nanargmin(np.abs(fieldmean - fr.typeparams[0][n][3]))]
            elif fr.typeparams[0][n][0] in ['AnnExt','AnnMn','AnnInt']:    # Calculate based on all values in an annulus
                for t in range(len(steplist)):
                    if fr.typeparams[0][n][1] == 'Rstorm':
                        Rcoord = Rstorm[t,:,:]
                    elif fr.typeparams[0][n][1] == 'RbyRMW':
                        Rcoord = RbyRMW[t,:,:]
                    field_ann = np.where(np.logical_or(Rcoord < fr.typeparams[0][n][2][0],Rcoord > fr.typeparams[0][n][2][1]),np.nan,field[t,:,:])
                    if fr.typeparams[0][n][0] == 'AnnExt':
                        if fr.typeparams[0][n][3] == 'Max':
                            blkprops[n,t] = np.nanmax(field_ann)
                        elif fr.typeparams[0][n][3] == 'Min':
                            blkprops[n,t] = np.nanmin(field_ann)
                    elif fr.typeparams[0][n][0] == 'AnnMn':
                        blkprops[n,t] = np.nanmean(field_ann)
                    elif fr.typeparams[0][n][0] == 'AnnInt':
                        if fr.doms[j] == 1:
                            dA = (12*1000)**2    # Area of grid cell [m2]
                        elif fr.doms[j] == 2:
                            dA = (4*1000)**2    # [m2]
                        elif fr.doms[j] == 3:
                            dA = (1.33333*1000)**2    # [m2]
                        blkprops[n,t] = np.nansum(field_ann*dA)
                #print(blkprops[n,:])
            elif fr.typeparams[0][n][0] in ['ColInt','ColMn']:    # Integrate over the vortex column to a specified height, or take column mean
                dz_all = np.array([fr.runs[j].myfields[722][fr.doms[j]].grdata[t,:,:,:] for t in steplist])    # Layer thickness [m]
                z_all  = np.array([fr.runs[j].myfields[459][fr.doms[j]].grdata[t,:,:,:] for t in steplist])    # Half layer height [m]
                for t in range(len(steplist)):
                    if fr.typeparams[0][n][1] == 'Rstorm':
                        Rcoord = Rstorm[t,:,:]
                    elif fr.typeparams[0][n][1] == 'RbyRMW':
                        Rcoord = RbyRMW[t,:,:]
                    elim = np.logical_or(Rcoord < fr.typeparams[0][n][2][0],Rcoord > fr.typeparams[0][n][2][1])    # Points to turn into nans
                    field_col = np.array([np.where(elim,np.nan,field[t,k,:,:]) for k in range(np.shape(field)[1])])
                    z = z_all[t,:,:,:]/1000    # Convert to km
                    field_col[z > fr.typeparams[0][n][3]] = np.nan    # Cut off at specified height
                    dz = dz_all[t,:,:,:]
                    if fr.doms[j] == 1:
                        dA = (12*1000)**2    # Area of grid cell [m2]
                    elif fr.doms[j] == 2:
                        dA = (4*1000)**2    # [m2]
                    elif fr.doms[j] == 3:
                        dA = (1.33333*1000)**2    # [m2]
                    dV = dz*dA    # Volume of each grid cell [m3]
                    dV[np.isnan(field_col)] = np.nan
                    if fr.typeparams[0][n][0] == 'ColInt':
                        blkprops[n,t] = np.nansum(field_col*dV)
                    elif fr.typeparams[0][n][0] == 'ColMn':
                        blkprops[n,t] = np.nansum(field_col*dV)/np.nansum(dV)
                #print(blkprops[n,:])
            elif fr.typeparams[0][n][0] in ['NonGridded']:    # Plot a non-gridded property
                if fr.typeparams[0][n][1] == 'mslp':
                    blkprops[n,:] = np.array([fr.runs[j].mslp[t] for t in steplist])    # [mb]
                elif fr.typeparams[0][n][1] == 'strmspeed':
                    blkprops[n,:] = np.array([fr.runs[j].strmspeed[t] for t in steplist])    # [m s-1]
                elif fr.typeparams[0][n][1] == 'U10maxAzimAvg':
                    blkprops[n,:] = np.array([fr.runs[j].U10maxAzimAvg[t] for t in steplist])    # [m s-1]
                elif fr.typeparams[0][n][1] == 'RMW':
                    blkprops[n,:] = np.array([fr.runs[j].RMW[t] for t in steplist])    # [km]
        for n in range(1,len(fr.fldindx)):
            if fr.legtext is None:
                label = '***User must define***'
            elif fr.legtext[j][n-1] is None:
                label = '***User must define***'
            elif fr.legtext[j][n-1] is not None:
                label = fr.legtext[j][n-1]
            if fr.typeparams[1] == 'lines':
                if fr.typeparams[2][j] <= 1:
                    plt.plot(blkprops[0,:],blkprops[n,:],fr.colors[j][n-1],linewidth=fr.typeparams[2][j],linestyle=fr.typeparams[3][j],label=label)
                else:    # Outline in white
                    plt.plot(blkprops[0,:],blkprops[n,:],fr.colors[j][n-1],linewidth=fr.typeparams[2][j],linestyle=fr.typeparams[3][j],label=label,path_effects=[pe.Stroke(linewidth=fr.typeparams[2][j]+3,foreground='w'),pe.Normal()])
            elif fr.typeparams[1] == 'markers':
                mfc = fr.colors[j][n-1] if fr.typeparams[4][j] else 'None'
                plt.plot(blkprops[0,:],blkprops[n,:],mec=fr.colors[j][n-1],mfc=mfc,ls='',ms=fr.typeparams[3][j],marker=fr.typeparams[2][j],label=label)
                if fr.typeparams[5][0]:    # Plot means, this only works for one pair of x,y fields because only one color can be given
                    Mfmt = fr.typeparams[5][1][j]    # Formatting items for this run
                    mfc = Mfmt[3] if Mfmt[2] else 'None'
                    mec = 'k'     if Mfmt[2] else Mfmt[3]
                    plt.plot([np.mean(blkprops[0,:])],[np.mean(blkprops[n,:])],mec=mec,mfc=mfc,ls='',ms=Mfmt[1],marker=Mfmt[0],label=Mfmt[4],zorder=10,mew=2)
                    print('\n')
                    print(np.mean(blkprops[0,:]))
                    print(np.mean(blkprops[n,:]))


#================================================================================


def plot_wrfvert(fr,steplist_0):

    steplist = steplist_0
    if fr.typeparams['general'][0] == 'Azimean':    # AllStepsCmprStrms only available for 'Azimean'
        if steplist_0 is None:
            steplist = fr.runs[0].steplist    # If using AllStepsCmprStrms, user must ensure that runs have same steplist
    if fr.typeparams['general'][1]:    # Use motion-relative coordinates
        Xindx = 503
        Yindx = 504
    else:
        Xindx = 10
        Yindx = 11
    # 2D vertical sections or averages of 3D fields (y-axis is height): loop over types, make calculations, and plot
    for fldtype in ['2Dfield','cntr1','cntr2','cntr3','cntr4','cntr5','cntr6','crlyvect']:    # If using crlyvect, x and y components must be passed in the frame's field list
        if fldtype in fr.typeparams:
            if fr.typeparams[fldtype][0]:    # Field is selected, so proceed
                # 1. Obtain required fields
                if fldtype != 'crlyvect':
                    findx = fr.fldindx[fr.typeparams[fldtype][2]]    # Index of field in input to plot
                if fr.typeparams[fldtype][1] == 'Diff1M0':    # Use run1 minus run0 difference, doesn't work for crlyvect
                    x0     = np.array([fr.runs[0].myfields[Xindx][fr.doms[0]].grdata[t,:,:] for t in steplist])
                    x1     = np.array([fr.runs[1].myfields[Xindx][fr.doms[1]].grdata[t,:,:] for t in steplist])
                    y0     = np.array([fr.runs[0].myfields[Yindx][fr.doms[0]].grdata[t,:,:] for t in steplist])
                    y1     = np.array([fr.runs[1].myfields[Yindx][fr.doms[1]].grdata[t,:,:] for t in steplist])
                    field0 = np.array([fr.runs[0].myfields[findx][fr.doms[0]].grdata[t,:,:,:] for t in steplist])
                    field1 = np.array([fr.runs[1].myfields[findx][fr.doms[1]].grdata[t,:,:,:] for t in steplist])
                    field1_remapped = []
                    for t in range(len(steplist)):
                        field1_remapped_t = []
                        xy1_forInterp = np.transpose(np.array([x1[t,:,:].flatten(),y1[t,:,:].flatten()]))    # x1 and y1 reformatted for interpolation
                        xy0_grid = (x0[t,:,:],y0[t,:,:])    # field0 grid to interpolate onto
                        for k in range(np.shape(field1)[1]):
                            field1_forInterp = field1[t,k,:,:].flatten()    # field1 reformatted for interpolation
                            field1_remapped_t.append(griddata(xy1_forInterp,field1_forInterp,xy0_grid,method='linear'))    # Remap field1 onto field0 grid
                        field1_remapped.append(np.array(field1_remapped_t))
                    field3D = np.array(field1_remapped) - field0    # Calculate field1 minus field0 difference (field is 3D + time)
                    X = x0    # [km]
                    Y = y0    # [km]
                    Z3D    = np.array([fr.runs[0].myfields[459][fr.doms[0]].grdata[t,:,:,:] for t in steplist])/1000    # [km], use field0 heights
                    Rstorm = np.array([fr.runs[0].myfields[ 12][fr.doms[0]].grdata[t,:,:] for t in steplist])    # [km]
                else:    # Use fields for a single selected run
                    whichRun = fr.typeparams[fldtype][1]
                    X       = np.array([fr.runs[whichRun].myfields[Xindx][fr.doms[whichRun]].grdata[t,:,:] for t in steplist])    # [km]
                    Y       = np.array([fr.runs[whichRun].myfields[Yindx][fr.doms[whichRun]].grdata[t,:,:] for t in steplist])    # [km]
                    Z3D     = np.array([fr.runs[whichRun].myfields[  459][fr.doms[whichRun]].grdata[t,:,:,:] for t in steplist])/1000    # [km]
                    Rstorm  = np.array([fr.runs[whichRun].myfields[   12][fr.doms[whichRun]].grdata[t,:,:] for t in steplist])    # [km]
                    if fldtype != 'crlyvect':    # Import a single 3D field
                        field3D      = np.array([fr.runs[whichRun].myfields[findx][fr.doms[whichRun]].grdata[t,:,:,:] for t in steplist])
                    else:
                        if fr.typeparams[fldtype][2] == 'u2ndSR':    # Import vertical velocity and SR radial winds to plot secondary circulation
                            field3Dx = np.array([fr.runs[whichRun].myfields[  496][fr.doms[whichRun]].grdata[t,:,:,:] for t in steplist])    # Radial windspeed
                            field3Dy = np.array([fr.runs[whichRun].myfields[  463][fr.doms[whichRun]].grdata[t,:,:,:] for t in steplist])    # Vertical velocity
                # 2. Calculate azimuthal mean or vertical cross section
                if fr.typeparams['general'][0] == 'Azimean':    # Calculate azimuthal mean
                    if fr.typeparams['general'][2] in ['FR','FL','RL','RR']:
                        if fr.typeparams['general'][1] == False:    # Only allow quadrant averaging if using motion-relative coordinates
                            Rstorm[:,:,:] = np.nan
                        else:
                            if fr.typeparams['general'][2] == 'FR':
                                Rstorm[~np.logical_and(X >= 0,Y >= 0)] = np.nan
                            elif fr.typeparams['general'][2] == 'FL':
                                Rstorm[~np.logical_and(X <  0,Y >= 0)] = np.nan
                            elif fr.typeparams['general'][2] == 'RL':
                                Rstorm[~np.logical_and(X <  0,Y <  0)] = np.nan
                            elif fr.typeparams['general'][2] == 'RR':
                                Rstorm[~np.logical_and(X >= 0,Y <  0)] = np.nan
                    if fldtype != 'crlyvect':    # Average a single 3D field
                        field2D,rbin  = plot_wrfvert_calcmeans(fr,Rstorm,field3D)    # Azimuthal mean of selected field, and radius vector
                    else:    # Average x and y vector components separately
                        field2Dx,rbin = plot_wrfvert_calcmeans(fr,Rstorm,field3Dx)
                        field2Dy,rbin = plot_wrfvert_calcmeans(fr,Rstorm,field3Dy)
                    Z2D,rbin          = plot_wrfvert_calcmeans(fr,Rstorm,Z3D)    # Azimuthal mean of height [km], and radius vector
                    R2D = np.array([rbin for k in range(np.shape(field2D)[0])])    # 2D version of rbin [km]
                    Z1D = np.array([np.nanmean(Z3D[:,k,:,:]) for k in range(np.shape(Z3D)[1])])    # Mean height at each model level [km]
                    R1D = rbin
                elif fr.typeparams['general'][0] == 'CrossSec':    # Calculate cross section, doesn't work for crlyvect
                    # 'CrossSec' should only be used with 'EachStep', but take first time in steplist to be sure
                    field3D_CS = field3D[0,:,:,:]
                    X_CS = X[0,:,:]
                    Y_CS = Y[0,:,:]
                    Z3D_CS = Z3D[0,:,:,:]
                    Rstorm_CS = Rstorm[0,:,:]
                    # Define line and interpolate
                    field2D = []
                    Z2D = []
                    line = np.transpose(np.array([np.linspace(fr.typeparams['general'][2][0][0],fr.typeparams['general'][2][1][0],num=100),\
                                                  np.linspace(fr.typeparams['general'][2][0][1],fr.typeparams['general'][2][1][1],num=100)]))
                    XY_CS_forInterp = np.transpose(np.array([X_CS.flatten(),Y_CS.flatten()]))
                    for k in range(np.shape(field3D_CS)[0]):
                        field3D_CS_forInterp = field3D_CS[k,:,:].flatten()
                        Z3D_CS_forInterp = Z3D_CS[k,:,:].flatten()
                        field2D.append(griddata(XY_CS_forInterp,field3D_CS_forInterp,line,method='linear'))
                        Z2D.append(griddata(XY_CS_forInterp,Z3D_CS_forInterp,line,method='linear'))
                    field2D = np.array(field2D)
                    Z2D = np.array(Z2D)    # [km]
                    R2D = np.array([np.sqrt(line[:,0]**2 + line[:,1]**2) for k in range(np.shape(field2D)[0])])    # [km]
                    Z1D = np.array([np.nanmean(Z3D_CS[k,:,:]) for k in range(np.shape(Z3D_CS)[0])])    # Mean height at each model level [km]
                    R1D = R2D[0,:]    # [km]
                # 3. Plot
                if fldtype == '2Dfield':    # Plot tricontourf or contourf
                    if fr.typeparams['general'][3] == 'Z2D':
                        plt.tricontourf(R2D[~np.isnan(Z2D)].flatten(),Z2D[~np.isnan(Z2D)].flatten(),field2D[~np.isnan(Z2D)].flatten(),\
                                fr.typeparams['2Dfield'][4],cmap=fr.typeparams['2Dfield'][3],extend='both')    # Triangulation needs Nans removed
                    elif fr.typeparams['general'][3] == 'Z1D':
                        plt.contourf(R1D,Z1D,field2D,\
                                fr.typeparams['2Dfield'][4],cmap=fr.typeparams['2Dfield'][3],extend='both')
                    if fr.typeparams['2Dfield'][6]:
                        colorbar = plt.colorbar(shrink=0.8,extend='both',ticks=fr.typeparams['2Dfield'][5])
                        cbytick_obj = plt.getp(colorbar.ax.axes,'yticklabels')
                        plt.setp(cbytick_obj,fontsize=12)
                elif fldtype in ['cntr1','cntr2','cntr3','cntr4','cntr5','cntr6']:    # Plot tricontour or contour
                    # Plot fields
                    if fr.typeparams['general'][3] == 'Z2D':
                        cntr = plt.tricontour(R2D[~np.isnan(Z2D)].flatten(),Z2D[~np.isnan(Z2D)].flatten(),field2D[~np.isnan(Z2D)].flatten(),\
                                fr.typeparams[fldtype][3],colors=fr.typeparams[fldtype][6])
                    elif fr.typeparams['general'][3] == 'Z1D':
                        if fr.typeparams[fldtype][9] is None:
                            cntr = plt.contour(R1D,Z1D,field2D,fr.typeparams[fldtype][3],colors=fr.typeparams[fldtype][6],linewidths=fr.typeparams[fldtype][7])
                        else:
                            cntr = plt.contour(R1D,Z1D,field2D,fr.typeparams[fldtype][3],colors=fr.typeparams[fldtype][6],linewidths=fr.typeparams[fldtype][7],linestyles=fr.typeparams[fldtype][9])
                    # Label contours
                    if fr.typeparams[fldtype][8] is None:    # Auto-label contours
                        fr.axobj.clabel(cntr,inline=True,fontsize=12,fmt=fr.typeparams[fldtype][4])
                    else:    # Manually label contours
                        fr.axobj.clabel(cntr,inline=True,fontsize=12,fmt=fr.typeparams[fldtype][4],manual=fr.typeparams[fldtype][8])
                    # Add dummy curve for legend
                    if fr.typeparams[fldtype][9] is None:
                        plt.plot([np.nan,np.nan],[np.nan,np.nan],fr.typeparams[fldtype][6],label=fr.typeparams[fldtype][5],linewidth=fr.typeparams[fldtype][7])
                    else:
                        plt.plot([np.nan,np.nan],[np.nan,np.nan],fr.typeparams[fldtype][6],label=fr.typeparams[fldtype][5],linewidth=fr.typeparams[fldtype][7],linestyle=fr.typeparams[fldtype][9])
                elif fldtype in ['crlyvect']:    # Plot curly vectors
                    whichWay = 'streamplot'    # Plot using Brandon's 'crlyvect' function or matplotlib's 'streamplot'
                    if whichWay == 'crlyvect':
                        Rcrly = R2D
                        Zcrly = np.array([[z for i in range(100)] for z in Z1D])
                        curly_vectors(Rcrly,Zcrly,field2Dx,field2Dy,scale_factor=fr.typeparams[fldtype][3],skip=fr.typeparams[fldtype][4],\
                                linewidth=0.5,color='k',verbose=False,interp='linear',arrowhead_min_length=0.0,arrowhead_max_length=0.0,\
                                arrowhead_length_factor=0.0)
                    elif whichWay == 'streamplot':
                        Rstream = R1D
                        Zstream = np.linspace(Z1D[0],fr.limits[3],100)    # Have to remap Z to a regular grid
                        finterpx = interp1d(Z1D,field2Dx,axis=0)
                        finterpy = interp1d(Z1D,field2Dy,axis=0)
                        field2Dxstream = finterpx(Zstream)
                        field2Dystream = finterpy(Zstream)
                        plt.streamplot(Rstream,Zstream,field2Dxstream,field2Dystream,linewidth=0.5,color='k',arrowstyle='->',\
                                density=fr.typeparams[fldtype][3])
                        if fr.typeparams[fldtype][4] is not None:
                            plt.plot([np.nan],[np.nan],c='k',marker='$>$',linewidth=0.5,mew=0.5,label=fr.typeparams[fldtype][4])
    # 1D profiles or azimuthal averages of 2D horizontal fields (y-axis is value along profile): loop over types, make calculations, and plot
    for fldtype in ['prof1','prof2','prof3','prof4','prof5','prof6']:
        if fldtype in fr.typeparams:
            if fr.typeparams[fldtype][0]:    # Field is selected, so proceed
                # 1. Obtain required fields
                findx = fr.fldindx[fr.typeparams[fldtype][2]]    # Index of field in input to plot
                if fr.typeparams[fldtype][1] == 'Diff1M0':    # Use run1 minus run0 difference
                    x0     = np.array([fr.runs[0].myfields[Xindx][fr.doms[0]].grdata[t,:,:] for t in steplist])
                    x1     = np.array([fr.runs[1].myfields[Xindx][fr.doms[1]].grdata[t,:,:] for t in steplist])
                    y0     = np.array([fr.runs[0].myfields[Yindx][fr.doms[0]].grdata[t,:,:] for t in steplist])
                    y1     = np.array([fr.runs[1].myfields[Yindx][fr.doms[1]].grdata[t,:,:] for t in steplist])
                    field0 = np.array([fr.runs[0].myfields[findx][fr.doms[0]].grdata[t,:,:] for t in steplist])
                    field1 = np.array([fr.runs[1].myfields[findx][fr.doms[1]].grdata[t,:,:] for t in steplist])
                    field1_remapped = []
                    for t in range(len(steplist)):
                        xy1_forInterp = np.transpose(np.array([x1[t,:,:].flatten(),y1[t,:,:].flatten()]))    # x1 and y1 reformatted for interpolation
                        xy0_grid = (x0[t,:,:],y0[t,:,:])    # field0 grid to interpolate onto
                        field1_forInterp = field1[t,:,:].flatten()    # field1 reformatted for interpolation
                        field1_remapped.append(griddata(xy1_forInterp,field1_forInterp,xy0_grid,method='linear'))    # Remap field1 onto field0 grid
                    field2D = np.array(field1_remapped) - field0    # Calculate field1 minus field0 difference (field is 2D + time)
                    X = x0    # [km]
                    Y = y0    # [km]
                    Rstorm = np.array([fr.runs[0].myfields[ 12][fr.doms[0]].grdata[t,:,:] for t in steplist])    # [km]
                else:    # Use fields for a single selected run
                    whichRun = fr.typeparams[fldtype][1]
                    field2D = np.array([fr.runs[whichRun].myfields[findx][fr.doms[whichRun]].grdata[t,:,:] for t in steplist])
                    X       = np.array([fr.runs[whichRun].myfields[Xindx][fr.doms[whichRun]].grdata[t,:,:] for t in steplist])    # [km]
                    Y       = np.array([fr.runs[whichRun].myfields[Yindx][fr.doms[whichRun]].grdata[t,:,:] for t in steplist])    # [km]
                    Rstorm  = np.array([fr.runs[whichRun].myfields[   12][fr.doms[whichRun]].grdata[t,:,:] for t in steplist])    # [km]
                # 2. Calculate azimuthal mean or 1D profile
                if fr.typeparams['general'][0] == 'Azimean':    # Calculate azimuthal mean
                    if fr.typeparams['general'][2] in ['FR','FL','RL','RR']:
                        if fr.typeparams['general'][1] == False:    # Only allow quadrant averaging if using motion-relative coordinates
                            Rstorm[:,:,:] = np.nan
                        else:
                            if fr.typeparams['general'][2] == 'FR':
                                Rstorm[~np.logical_and(X >= 0,Y >= 0)] = np.nan
                            elif fr.typeparams['general'][2] == 'FL':
                                Rstorm[~np.logical_and(X <  0,Y >= 0)] = np.nan
                            elif fr.typeparams['general'][2] == 'RL':
                                Rstorm[~np.logical_and(X <  0,Y <  0)] = np.nan
                            elif fr.typeparams['general'][2] == 'RR':
                                Rstorm[~np.logical_and(X >= 0,Y <  0)] = np.nan
                    field1D,rbin = plot_wrfvert_calcmeans_1D(fr,Rstorm,field2D)    # Azimuthal mean of selected field, and radius vector
                    R1D = rbin    # [km]
                elif fr.typeparams['general'][0] == 'CrossSec':    # Calculate profile along a set line
                    # 'CrossSec' should only be used with 'EachStep', but take first time in steplist to be sure
                    field2D_CS = field2D[0,:,:]
                    X_CS = X[0,:,:]
                    Y_CS = Y[0,:,:]
                    Rstorm_CS = Rstorm[0,:,:]
                    # Define line and interpolate
                    line = np.transpose(np.array([np.linspace(fr.typeparams['general'][2][0][0],fr.typeparams['general'][2][1][0],num=100),\
                                                  np.linspace(fr.typeparams['general'][2][0][1],fr.typeparams['general'][2][1][1],num=100)]))
                    XY_CS_forInterp = np.transpose(np.array([X_CS.flatten(),Y_CS.flatten()]))
                    field2D_CS_forInterp = field2D_CS.flatten()
                    field1D = np.array(griddata(XY_CS_forInterp,field2D_CS_forInterp,line,method='linear'))
                    R1D = np.sqrt(line[:,0]**2 + line[:,1]**2)    # [km]
                # 3. Plot
                plt.plot(R1D,field1D,fr.typeparams[fldtype][3],linestyle=fr.typeparams[fldtype][4],label=fr.typeparams[fldtype][5])
    # Plot surface RMW, if requested
    if fr.typeparams['RMWsurf'][0]:
        for k in range(len(fr.runs)):    # Loop over runs
            if fr.runs[k].RMW == []:
                wd.calcRMW(fr.runs[k])    # Calculate RMW
            RMW_vect = np.array([fr.runs[k].RMW[t] for t in steplist])
            RMW_mean = np.nanmean(RMW_vect)
            plt.plot([RMW_mean,RMW_mean],[fr.limits[2],0.2*fr.limits[3]],fr.typeparams['RMWsurf'][1][k],linewidth=2,\
                    linestyle=fr.typeparams['RMWsurf'][2][k],label=fr.typeparams['RMWsurf'][3][k])


def plot_wrfvert_calcmeans(fr,Rstorm,field):
    nZ = np.shape(field)[1]
    numbinsR = 100
    binwid_r = (fr.limits[1] - fr.limits[0])/numbinsR
    bintol_r = binwid_r/2
    rbin = np.linspace(fr.limits[0]+bintol_r,fr.limits[1]-bintol_r,numbinsR)    # Bin centers, r-axis
    meanfield = np.full((nZ,numbinsR),np.nan)    # Create array for averaged field
    rindx = np.round((Rstorm-fr.limits[0])/(fr.limits[1]-fr.limits[0])*numbinsR-0.5)
    rindx[rindx <  0] = np.nan
    rindx[rindx > 99] = np.nan
    for i in range(numbinsR):
        thisbinR = (rindx == i)
        for j in range(nZ):
            meanfield[j,i] = np.nanmean(field[:,j,:,:][thisbinR])
    return meanfield,rbin


def plot_wrfvert_calcmeans_1D(fr,Rstorm,field,numbinsR=100):
    binwid_r = (fr.limits[1] - fr.limits[0])/numbinsR
    bintol_r = binwid_r/2
    rbin = np.linspace(fr.limits[0]+bintol_r,fr.limits[1]-bintol_r,numbinsR)    # Bin centers, r-axis
    meanfield = np.full(numbinsR,np.nan)    # Create array for averaged field
    rindx = np.round((Rstorm-fr.limits[0])/(fr.limits[1]-fr.limits[0])*numbinsR-0.5)
    rindx[rindx <  0] = np.nan
    rindx[rindx > (numbinsR-1)] = np.nan
    for i in range(numbinsR):
        thisbinR = (rindx == i)
        meanfield[i] = np.nanmean(field[thisbinR])
    return meanfield,rbin


#================================================================================


