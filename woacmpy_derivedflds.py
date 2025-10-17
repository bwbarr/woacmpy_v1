import numpy as np
from netCDF4 import Dataset
from datetime import datetime,timedelta
import woacmpy_v1.woacmpy_global as wg
import woacmpy_v1.woacmpy_classes as wc
import woacmpy_v1.woacmpy_observations as wo
import woacmpy_v1.woacmpy_track as wt
from woacmpy_v1.pyhycom import thickness2depths
from wrf import getvar,interplevel
import gsw
from spraylibs_v4.util import qsat0,satratio,stabIntM,stabIntH
from spraylibs_v4.heatflux import spraymedHF
from spraylibs_v4.ssgf import whitecapActive_DLM17,whitecap_CF210504
from coare_spray_v1.heatflux import sprayHFs
from scipy.interpolate import griddata
from matplotlib.dates import date2num


# ================== Derived field calculations for WOACMPY ===============================


def calculate_derived_fields():

    print('Calculating derived fields... \n')

    # Calculate fields that are not taken directly from model output
    for r in wc.Run.AllRuns:    # Loop over runs
        for d in [3,1,2,4,5,6,7,8,9]:    # Loop over domains, dom 3 is done first to get RMW calcs if needed
            if wg.active_doms[d]:
                for f in wg.derived_fld_indx:    # Loop over derived fields
                    if r.field_impswitches[f,d] == -1:

                        # Field 6: 10-m Windspeed
                        if f == 6:
                            r.myfields[6][d].grdata = (r.myfields[4][d].grdata**2 \
                                                     + r.myfields[5][d].grdata**2)**0.5    # [m s-1]

                        # Fields 10 and 11: X and Y Storm-Centered Coordinates
                        elif f == 10:
                            n_timesteps = r.myfields[1][d].grdata.shape[0]
                            xy = [wt.latlon2xyStormRelative( \
                                    r.myfields[1][d].grdata[k,:,:],r.myfields[2][d].grdata[k,:,:], \
                                    r.lon0[k],r.lat0[k],0.5*np.pi) for k in np.arange(n_timesteps)]
                            r.myfields[10][d].grdata = np.array([xy[k][0] for k in np.arange(n_timesteps)])    # [km]
                            r.myfields[11][d].grdata = np.array([xy[k][1] for k in np.arange(n_timesteps)])    # [km]

                        # Field 12: Distance to Storm Center
                        elif f == 12:
                            r.myfields[12][d].grdata = (r.myfields[10][d].grdata**2 \
                                                      + r.myfields[11][d].grdata**2)**0.5    # [km]

                        # Field 27: Wave energy dissipation flux [kg s-3]
                        elif f == 27:
                            r.myfields[27][d].grdata = (r.myfields[25][d].grdata**2 \
                                                      + r.myfields[26][d].grdata**2)**0.5    # [kg s-3]

                        # Field 32: Sea surface temperature [C]
                        elif f == 32:
                            r.myfields[32][d].grdata = r.myfields[15][d].grdata - 273.15    # [C]

                        # Field 33: Surface current speed [m s-1]
                        elif f == 33:
                            r.myfields[33][d].grdata = (r.myfields[18][d].grdata**2 + r.myfields[19][d].grdata**2)**0.5

                        # Field 40: Height of lowest model level [m]
                        elif f == 40:
                            r.myfields[40][d].grdata = 0.5*(r.myfields[34][d].grdata + r.myfields[35][d].grdata \
                                                          + r.myfields[36][d].grdata + r.myfields[37][d].grdata)/9.81    # [m]

                        # Field 41: Height of second lowest model level [m]
                        elif f == 41:
                            r.myfields[41][d].grdata = 0.5*(r.myfields[36][d].grdata + r.myfields[37][d].grdata \
                                                          + r.myfields[38][d].grdata + r.myfields[39][d].grdata)/9.81    # [m]
                        # Field 45: Pressure at lowest model level [Pa]
                        elif f == 45:
                            r.myfields[45][d].grdata = r.myfields[43][d].grdata + r.myfields[44][d].grdata    # [Pa]

                        # Field 46: Potential temperature at lowest model level [K]
                        elif f == 46:
                            r.myfields[46][d].grdata = r.myfields[42][d].grdata + 300    # [K]

                        # Field 47: Temperature at lowest model level [K]
                        elif f == 47:
                            r.myfields[47][d].grdata = r.myfields[46][d].grdata* \
                                    (r.myfields[45][d].grdata/1e5)**0.286    # [K]

                        # Field 49: Specific humidity at lowest model level [kg kg-1]
                        elif f == 49:
                            r.myfields[49][d].grdata = r.myfields[48][d].grdata / \
                                    (1.0 + r.myfields[48][d].grdata)    # [kg kg-1]

                        # Field 52: X-wind component at lowest model level [m s-1]
                        elif f == 52:
                            r.myfields[52][d].grdata = 0.5*(r.myfields[50][d].grdata[:,:,:-1] \
                                    + r.myfields[50][d].grdata[:,:,1:])    # [m s-1]

                        # Field 53: Y-wind component at lowest model level [m s-1]
                        elif f == 53:
                            r.myfields[53][d].grdata = 0.5*(r.myfields[51][d].grdata[:,:-1,:] \
                                    + r.myfields[51][d].grdata[:,1:,:])    # [m s-1]

                        # Field 54: Windspeed at lowest model level [m s-1]
                        elif f == 54:
                            r.myfields[54][d].grdata = (r.myfields[52][d].grdata**2 + r.myfields[53][d].grdata**2)**0.5    # [m s-1]

                        # Field 56: Surface enthalpy flux [W m-2]
                        elif f == 56:
                            r.myfields[56][d].grdata = r.myfields[7][d].grdata + r.myfields[8][d].grdata

                        # Field 59: 10-m Windspeed (UMWM global input)
                        elif f == 59:
                            r.myfields[59][d].grdata = (r.myfields[57][d].grdata**2 \
                                                      + r.myfields[58][d].grdata**2)**0.5    # [m s-1]

                        # Field 60: Wave age (dcp/u*) [-]
                        elif f == 60:
                            r.myfields[60][d].grdata = r.myfields[24][d].grdata/r.myfields[21][d].grdata

                        # Field 63: Relative humidity at lowest model level [%]
                        elif f == 63:
                            r.myfields[63][d].grdata = satratio(r.myfields[47][d].grdata,r.myfields[45][d].grdata,r.myfields[49][d].grdata,2)*100    # [%]

                        # Field 68: Total rainrate [mm/hr]
                        elif f == 68:
                            totaccum = r.myfields[66][d].grdata + r.myfields[67][d].grdata    # RAINC + RAINNC [mm]
                            intaccum = np.full_like(totaccum,0)    # Rain accumulated during interval [mm]
                            if np.shape(totaccum)[0] > 1:
                                intaccum[1:,:,:] = totaccum[1:,:,:] - totaccum[:-1,:,:]
                            if r.timedel[0] == 'hours':
                                hrlyaccum = intaccum/r.timedel[1]    # Hourly equivalent accumulated rainrate [mm hr-1]
                            elif r.timedel[0] == 'minutes':
                                hrlyaccum = intaccum*60/r.timedel[1]    # Hourly equivalent accumulated rainrate [mm hr-1]
                            r.myfields[68][d].grdata = hrlyaccum    # [mm hr-1]

                        # Field 69: Minimum sea level pressure [mb]
                        elif f == 69:
                            if d in [1,2,3]:
                                mslp_grid = np.full_like(r.myfields[ 1][d].grdata[0,:,:],0.0)
                            r.myfields[69][d].grdata = np.array([mslp_grid + r.mslp[t] for t in range(len(r.mslp))])

                        # Field 71: Storm quadrant [1=FR,2=FL,3=RL,4=RR]
                        elif f == 71:
                            strmdir = r.strmdir
                            x_WRF = r.myfields[10][d].grdata
                            y_WRF = r.myfields[11][d].grdata
                            theta = np.array([(np.arctan2(y_WRF[t,:,:],x_WRF[t,:,:]) - strmdir[t])*180./np.pi for t in range(len(strmdir))])
                            theta[theta >=  180.] -= 360.
                            theta[theta <  -180.] += 360.
                            quad = np.full_like(x_WRF,np.nan)
                            quad[np.logical_and(theta >= -180.,theta < -90.)] = 4.    # Rear Right
                            quad[np.logical_and(theta >=  -90.,theta <   0.)] = 1.    # Front Right
                            quad[np.logical_and(theta >=    0.,theta <  90.)] = 2.    # Front Left
                            quad[np.logical_and(theta >=   90.,theta < 180.)] = 3.    # Rear Left
                            r.myfields[71][d].grdata = quad

                        # Field 80: Size of gail and hurricane-force windspeeds [km2]
                        elif f == 80:
                            wspdgail = 17    # Gail-force windspeed [m s-1]
                            wspdhurr = 34    # Hurricane-force windspeed [m s-1]
                            if d == 1:
                                gridarea = 144    # [km2]
                            elif d == 2:
                                gridarea = 16    # [km2]
                            elif d == 3:
                                gridarea = 1.3**2    # [km2]
                            wspd10 = r.myfields[6][d].grdata
                            area = np.full_like(wspd10,np.nan)
                            area[:,0,0] = 0
                            area[:,0,1] = 0
                            for i in range(np.shape(wspd10)[0]):
                                area[i,0,0] = np.nansum(wspd10[i,:,:] >= wspdgail)*gridarea    # Gail-force area [km2]
                                area[i,0,1] = np.nansum(wspd10[i,:,:] >= wspdhurr)*gridarea    # Hurr-force area [km2]
                            r.myfields[80][d].grdata = area

                        # Fields 95: Windspeed at reference height for global spray analysis [m s-1]
                        elif f == 95:
                            kappa = 0.41    # von Karman constant
                            ustar = r.myfields[21][d].grdata    # Friction velocity [m s-1]
                            U10   = r.myfields[59][d].grdata    # 10m windspeed [m s-1]
                            z1    = r.myfields[94][d].grdata    # Reference height [m]
                            r.myfields[95][d].grdata = U10 + ustar/kappa*np.log(z1/10)    # [m s-1]

                        # Field 100: Nondimensional depth (dwn*depth) [-]
                        elif f == 100:
                            r.myfields[100][d].grdata = 2*np.pi/r.myfields[55][d].grdata*r.myfields[30][d].grdata

                        # Offline spray calculation fields
                        elif f in [101,131,161,191,602,650]:
                            fldindx = [0,1,2,3,4,5,6,7,8,9,10,11,13,14,16,18,19,20,21,31,32,33,34,35,36,37,38,39,40,41,42,43,44,46,47,48,49]    # Indices of 2D fields in spray calc return
                            specindx = [12,17,22,23,24,25,26,27,28,29,30]    # Indices of 3D spectra/profiles in spray calc return
                            nfld = len(fldindx)
                            nspec = len(specindx)
                            if f == 101:
                                dataindex = 0
                                fldlist = [101,102,103,104,105,106,107,108,109,110,111,112,114,115,116,118,119,120,121,237,238,239,240,241,273,274,323,324,325,326,327,328,329,363,367,421,517]    # Indices in woacmpy_global list
                                speclist = [113,117,122,123,124,125,126,127,128,129,130]
                            elif f == 131:
                                dataindex = 1
                                fldlist = [131,132,133,134,135,136,137,138,139,140,141,142,144,145,146,148,149,150,151,242,243,244,245,246,275,276,330,331,332,333,334,335,336,364,368,422,518]
                                speclist = [143,147,152,153,154,155,156,157,158,159,160]
                            elif f == 161:
                                dataindex = 2
                                fldlist = [161,162,163,164,165,166,167,168,169,170,171,172,174,175,176,178,179,180,181,247,248,249,250,251,277,278,337,338,339,340,341,342,343,365,369,423,519]
                                speclist = [173,177,182,183,184,185,186,187,188,189,190]
                            elif f == 191:
                                dataindex = 3
                                fldlist = [191,192,193,194,195,196,197,198,199,200,201,202,204,205,206,208,209,210,211,252,253,254,255,256,279,280,344,345,346,347,348,349,350,366,370,424,520]
                                speclist = [203,207,212,213,214,215,216,217,218,219,220]
                            elif f == 602:
                                dataindex = 4
                                fldlist = [602,603,604,605,606,607,608,609,610,611,612,613,615,616,617,619,620,621,622,632,633,634,635,636,637,638,639,640,641,642,643,644,645,646,647,648,649]
                                speclist = [614,618,623,624,625,626,627,628,629,630,631]
                            elif f == 650:
                                dataindex = 5
                                fldlist = [650,651,652,653,654,655,656,657,658,659,660,661,663,664,665,667,668,669,670,680,681,682,683,684,685,686,687,688,689,690,691,692,693,694,695,696,697]
                                speclist = [662,666,671,672,673,674,675,676,677,678,679]
                            # Identify which spray libraries to use
                            whichspray = wc.SprayData.whichspray[dataindex]
                            # Get spray data used by all libraries
                            fs         = wc.SprayData.sourcestrength[dataindex]
                            r0         = wc.SprayData.r0[dataindex]    # [m]
                            delta_r0   = wc.SprayData.delta_r0[dataindex]    # [m]
                            SSGFname   = wc.SprayData.SSGFname[dataindex]
                            feedback   = wc.SprayData.feedback[dataindex]
                            profiles   = wc.SprayData.profiles[dataindex]
                            zRvaries   = wc.SprayData.zRvaries[dataindex]
                            sprayLB    = wc.SprayData.sprayLB[dataindex]
                            fdbkfsolve = wc.SprayData.fdbkfsolve[dataindex]
                            scaleSSGF  = wc.SprayData.scaleSSGF[dataindex]
                            chi1       = wc.SprayData.chi1[dataindex]
                            chi2       = wc.SprayData.chi2[dataindex]
                            which_z0tq = wc.SprayData.which_z0tq[dataindex]
                            # Get spray data needed by chosen library
                            if whichspray in ['spr4_uwincm','spr4_umwmglb']:
                                stability       = wc.SprayData.stability[dataindex]
                                fdbkcrzyOPT     = wc.SprayData.fdbkcrzyOPT[dataindex]
                                showfdbkcrzy    = wc.SprayData.showfdbkcrzy[dataindex]
                            elif whichspray in ['csp1_uwincm']:
                                param_delspr_Wi = wc.SprayData.param_delspr_Wi[dataindex]
                                which_stress    = wc.SprayData.which_stress[dataindex]
                                use_gf          = wc.SprayData.use_gf[dataindex]
                                z_ref           = wc.SprayData.z_ref[dataindex]
                            # Run chosen spray libraries
                            if whichspray == 'spr4_uwincm':
                                spray_inp_fields = wg.inps_spr4_uwincm
                            elif whichspray == 'spr4_umwmglb':
                                spray_inp_fields = wg.inps_spr4_umwmglb
                            elif whichspray == 'csp1_uwincm':
                                spray_inp_fields = wg.inps_csp1_uwincm
                            for t in np.arange(r.myfields[spray_inp_fields[0]][d].grdata.shape[0]):
                                if whichspray in ['spr4_uwincm','spr4_umwmglb']:
                                    z_1   = r.myfields[spray_inp_fields[ 0]][d].grdata[t,:,:]    # [m]
                                    U_1   = r.myfields[spray_inp_fields[ 1]][d].grdata[t,:,:]    # [m s-1]
                                    th_1  = r.myfields[spray_inp_fields[ 2]][d].grdata[t,:,:]    # [K]
                                    q_1   = r.myfields[spray_inp_fields[ 3]][d].grdata[t,:,:]    # [kg kg-1]
                                    p_0   = r.myfields[spray_inp_fields[ 4]][d].grdata[t,:,:]    # [Pa]
                                    eps   = r.myfields[spray_inp_fields[ 5]][d].grdata[t,:,:]    # [kg s-3]
                                    dcp   = r.myfields[spray_inp_fields[ 6]][d].grdata[t,:,:]    # [m s-1]
                                    swh   = r.myfields[spray_inp_fields[ 7]][d].grdata[t,:,:]    # [m]
                                    mss   = r.myfields[spray_inp_fields[ 8]][d].grdata[t,:,:]    # [-]
                                    ustar = r.myfields[spray_inp_fields[ 9]][d].grdata[t,:,:]    # [m s-1]
                                    t_0   = r.myfields[spray_inp_fields[10]][d].grdata[t,:,:]    # [K]
                                    AllResults = spraymedHF(z_1,U_1,th_1,q_1,p_0,fs,eps,dcp,swh,mss,ustar,t_0,\
                                            r0=r0,delta_r0=delta_r0,SSGFname=SSGFname,feedback=feedback,getprofiles=profiles,\
                                            zRvaries=zRvaries,stability=stability,sprayLB=sprayLB,fdbkfsolve=fdbkfsolve,\
                                            fdbkcrzyOPT=fdbkcrzyOPT,showfdbkcrzy=showfdbkcrzy,scaleSSGF=scaleSSGF,chi1=chi1,chi2=chi2,\
                                            which_z0tq=which_z0tq)
                                elif whichspray in ['csp1_uwincm']:
                                    z_1   = r.myfields[spray_inp_fields[ 0]][d].grdata[t,:,:]    # [m]
                                    U_1   = r.myfields[spray_inp_fields[ 1]][d].grdata[t,:,:]    # [m s-1]
                                    t_1   = r.myfields[spray_inp_fields[ 2]][d].grdata[t,:,:]    # [K]
                                    q_1   = r.myfields[spray_inp_fields[ 3]][d].grdata[t,:,:]    # [kg kg-1]
                                    p_0   = r.myfields[spray_inp_fields[ 4]][d].grdata[t,:,:]    # [Pa]
                                    eps   = r.myfields[spray_inp_fields[ 5]][d].grdata[t,:,:]    # [kg s-3]
                                    dcp   = r.myfields[spray_inp_fields[ 6]][d].grdata[t,:,:]    # [m s-1]
                                    swh   = r.myfields[spray_inp_fields[ 7]][d].grdata[t,:,:]    # [m]
                                    mss   = r.myfields[spray_inp_fields[ 8]][d].grdata[t,:,:]    # [-]
                                    ustar = r.myfields[spray_inp_fields[ 9]][d].grdata[t,:,:]    # [m s-1]
                                    t_0   = r.myfields[spray_inp_fields[10]][d].grdata[t,:,:]    # [K]
                                    AllResults = sprayHFs(z_1,t_1,q_1,z_1,U_1,p_0,t_0,eps,dcp,swh,mss,fs,\
                                            r0=r0,delta_r0=delta_r0,SSGFname=SSGFname,param_delspr_Wi=param_delspr_Wi,feedback=feedback,\
                                            getprofiles=profiles,zRvaries=zRvaries,sprayLB=sprayLB,fdbksolve=fdbkfsolve,\
                                            scaleSSGF=scaleSSGF,chi1=chi1,chi2=chi2,which_stress=which_stress,ustar_bulk_in=ustar,\
                                            use_gf=use_gf,which_z0tq=which_z0tq,z_ref=z_ref)
                                for i in range(nfld):
                                    if r.field_impswitches[fldlist[i],d] == -1:
                                        r.myfields[fldlist[i]][d].grdata.append(AllResults[fldindx[i]])
                                for i in range(nspec):
                                    if r.field_impswitches[speclist[i],d] == -1:
                                        r.myfields[speclist[i]][d].grdata.append(AllResults[specindx[i]])
                                if wc.SprayData.r0[dataindex] is None:
                                    wc.SprayData.r0[dataindex] = AllResults[15]
                                    wc.SprayData.delta_r0[dataindex] = AllResults[45]
                            for fld in fldlist:
                                if r.field_impswitches[fld,d] == -1:
                                    r.myfields[fld][d].grdata = np.array(r.myfields[fld][d].grdata)

                        # Fields 221 - 224, 700, 701: Net spray sensible heat flux [W m-2]
                        elif f in [221,222,223,224,700,701]:
                            indx = {221:[106,107],
                                    222:[136,137],
                                    223:[166,167],
                                    224:[196,197],
                                    700:[607,608],
                                    701:[655,656]}
                            r.myfields[f][d].grdata = r.myfields[indx[f][0]][d].grdata - r.myfields[indx[f][1]][d].grdata

                        # Fields 225 - 232, 702 - 705: Drop-specific spray heat fluxes due to temp and size change [W m-2 um-1]
                        elif f in [225,226,227,228,229,230,231,232,702,703,704,705]:
                            indx = {225:[118,122,117],    # a, E, dm/dr0
                                    226:[148,152,147],
                                    227:[178,182,177],
                                    228:[208,212,207],
                                    229:[119,123,117],
                                    230:[149,153,147],
                                    231:[179,183,177],
                                    232:[209,213,207],
                                    702:[619,623,618],
                                    703:[667,671,666],
                                    704:[620,624,618],
                                    705:[668,672,666]}
                            for t in range(len(r.myfields[indx[f][1]][d].grdata)):
                                r.myfields[f][d].grdata.append(np.array([r.myfields[indx[f][0]][d].grdata[t,:,:]\
                                        *r.myfields[indx[f][1]][d].grdata[t][r0,:,:]*r.myfields[indx[f][2]][d].grdata[t][r0,:,:]\
                                        for r0 in range(np.shape(r.myfields[indx[f][1]][d].grdata[t])[0])]))

                        # Fields 233 - 236: t_0 - t_zT [K]
                        elif f in [233,234,235,236]:
                            indx = {233:113,
                                    234:143,
                                    235:173,
                                    236:203}
                            t_0 = r.myfields[15][d].grdata
                            t_zT = r.myfields[indx[f]][d].grdata
                            for i in range(len(t_zT)):
                                r.myfields[f][d].grdata.append(np.array([t_0[i,:,:] - t_zT[i][j,:,:] for j in range(np.shape(t_zT[i])[0])]))

                        # Fields 257 - 260: t_0 - t_10Npr [K]
                        elif f in [257,258,259,260]:
                            indx = {257:238,
                                    258:243,
                                    259:248,
                                    260:253}
                            r.myfields[f][d].grdata = r.myfields[15][d].grdata - r.myfields[indx[f]][d].grdata

                        # Fields 261 - 268: Absolute value of HRspr/HTspr and HLspr/HSspr [-]
                        elif f in [261,262,263,264,265,266,267,268]:
                            indx = {261:[105,107],
                                    262:[106,108],
                                    263:[135,137],
                                    264:[136,138],
                                    265:[165,167],
                                    266:[166,168],
                                    267:[195,197],
                                    268:[196,198]}
                            r.myfields[f][d].grdata = np.abs(r.myfields[indx[f][1]][d].grdata/r.myfields[indx[f][0]][d].grdata)

                        # Fields 269 - 272: abs(s10Npr - y0 - 1) [-]
                        elif f in [269,270,271,272]:
                            indx = {269:241,
                                    270:246,
                                    271:251,
                                    272:256}
                            r.myfields[f][d].grdata = np.abs(r.myfields[indx[f]][d].grdata - 1)

                        # Fields 281 - 288, 779 - 780: Total enthalpy flux with and without spray [W m-2]
                        elif f in [281,282,283,284,285,286,287,288,779,780]:
                            indx = {281:[103,104],
                                    282:[273,274],
                                    283:[133,134],
                                    284:[275,276],
                                    285:[163,164],
                                    286:[277,278],
                                    287:[193,194],
                                    288:[279,280],
                                    779:[604,605],
                                    780:[652,653]}
                            r.myfields[f][d].grdata = r.myfields[indx[f][0]][d].grdata + r.myfields[indx[f][1]][d].grdata

                        # Fields 289 - 292: Absolute value of HRspr/Mspr [J g-1]
                        elif f in [289,290,291,292]:
                            indx = {289:[107,116],
                                    290:[137,146],
                                    291:[167,176],
                                    292:[197,206]}
                            r.myfields[f][d].grdata = np.abs(r.myfields[indx[f][0]][d].grdata/r.myfields[indx[f][1]][d].grdata)

                        # Fields 293 - 300: Bowen Ratio with and without spray [-]
                        elif f in [293,294,295,296,297,298,299,300]:
                            indx = {293:[103,104],
                                    294:[273,274],
                                    295:[133,134],
                                    296:[275,276],
                                    297:[163,164],
                                    298:[277,278],
                                    299:[193,194],
                                    300:[279,280]}
                            r.myfields[f][d].grdata = r.myfields[indx[f][0]][d].grdata/r.myfields[indx[f][1]][d].grdata

                        # Fields 301 - 306: Percent difference in spray modifications to SHF and LHF [%]
                        elif f in [301,302,303,304,305,306]:
                            indx = {301:[307,310],
                                    302:[308,311],
                                    303:[307,313],
                                    304:[308,314],
                                    305:[307,316],
                                    306:[308,317]}
                            r.myfields[f][d].grdata = (r.myfields[indx[f][1]][d].grdata - r.myfields[indx[f][0]][d].grdata)\
                                    /np.abs(r.myfields[indx[f][0]][d].grdata)*100

                        # Fields 307 - 318: Spray modifications to SHF, LHF, and enthalpy flux [W m-2]
                        elif f in [307,308,309,310,311,312,313,314,315,316,317,318]:
                            indx = {307:[103,273],
                                    308:[104,274],
                                    309:[281,282],
                                    310:[133,275],
                                    311:[134,276],
                                    312:[283,284],
                                    313:[163,277],
                                    314:[164,278],
                                    315:[285,286],
                                    316:[193,279],
                                    317:[194,280],
                                    318:[287,288]}
                            r.myfields[f][d].grdata = r.myfields[indx[f][0]][d].grdata - r.myfields[indx[f][1]][d].grdata

                        # Fields 319 - 322: Spray HF due to cooling to wetbulb temp [W m-2]
                        elif f in [319,320,321,322]:
                            indx = {319:[105,106],
                                    320:[135,136],
                                    321:[165,166],
                                    322:[195,196]}
                            r.myfields[f][d].grdata = r.myfields[indx[f][0]][d].grdata - r.myfields[indx[f][1]][d].grdata

                        # Fields 351 - 354: Interfacial enthalpy flux [W m-2]
                        elif f in [351,352,353,354]:
                            indx = {351:[101,102],
                                    352:[131,132],
                                    353:[161,162],
                                    354:[191,192]}
                            r.myfields[f][d].grdata = r.myfields[indx[f][0]][d].grdata + r.myfields[indx[f][1]][d].grdata

                        # Fields 355 - 358, 698, 699: Peak radius in dm/dr0 [um]
                        elif f in [355,356,357,358,698,699]:
                            indx = {355:[117,0],
                                    356:[147,1],
                                    357:[177,2],
                                    358:[207,3],
                                    698:[618,4],
                                    699:[666,5]}
                            for t in range(len(r.myfields[indx[f][0]][d].grdata)):
                                dmdr0 = r.myfields[indx[f][0]][d].grdata[t]
                                peakr0 = np.full_like(dmdr0[0,:,:],np.nan)
                                for i in range(peakr0.shape[0]):
                                    for j in range(peakr0.shape[1]):
                                        dmdr0_max = np.nanmax(dmdr0[:,i,j])
                                        if np.isnan(dmdr0_max) or dmdr0_max == 0.0:
                                            peakr0[i,j] = np.nan
                                        else:
                                            peakr0[i,j] = wc.SprayData.r0[indx[f][1]][np.where(dmdr0[:,i,j] == dmdr0_max)[0][0]]*1e6    # [um]
                                r.myfields[f][d].grdata.append(peakr0)
                            r.myfields[f][d].grdata = np.array(r.myfields[f][d].grdata)

                        # Fields 359 - 362: Specific available energy for heat transfer due to temperature change; no wetbulb contribution [J kg-1]
                        elif f in [359,360,361,362]:
                            indx = {359:238,
                                    360:243,
                                    361:248,
                                    362:253}
                            cp_sw = 4200.    # Specific heat capacity of seawater [J kg-1 K-1]
                            r.myfields[f][d].grdata = cp_sw*(r.myfields[15][d].grdata - r.myfields[indx[f]][d].grdata)

                        # Fields 371 - 374: Ratio of storm radius to RMW (based on offline spray calcs, using windfield from selected domain)
                        elif f in [371,372,373,374]:
                            indx = {371:367,
                                    372:368,
                                    373:369,
                                    374:370}
                            U10 = r.myfields[indx[f]][d].grdata    # 10m windspeed [m s-1]
                            Rstorm = r.myfields[12][d].grdata    # Distance to storm center [km]
                            numbins = 100
                            Rmax = 200.    # Max R for taking average [km]
                            binwid = (Rmax-0)/numbins
                            bintol = binwid/2
                            Rbin = np.linspace(0+bintol,Rmax-bintol,numbins)    # Bin centers
                            RbyRMW = []
                            for t in range(np.shape(U10)[0]):
                                Rindx = np.round((Rstorm[t,:,:]-0)/(Rmax-0)*numbins-0.5)
                                Rindx[Rindx <  0] = np.nan
                                Rindx[Rindx > 99] = np.nan
                                U10mean = np.full((numbins,),np.nan)
                                for b in range(numbins):
                                    thisbin = np.logical_and(Rindx > b-0.01,Rindx < b+0.01)
                                    U10mean[b] = np.nanmean(U10[t,:,:][thisbin])    # Azim mean wind profile [m s-1]
                                RMW = Rbin[np.nanargmax(U10mean)]    # Radius of max wind [km]
                                RbyRMW.append(Rstorm[t,:,:]/RMW)    # Ratio of storm radius to RMW
                            r.myfields[f][d].grdata = np.array(RbyRMW)

                        # Field 375: Ratio of storm radius to RMW (based on UWIN-CM d03 wspd10)
                        elif f == 375:
                            if r.RMW == []:
                                calcRMW(r)
                            RMW = r.RMW    # Radius of maximum azimuthally-averaged windspeed [km]
                            Rstorm = r.myfields[12][d].grdata    # Distance to storm center [km]
                            r.myfields[f][d].grdata = np.array([Rstorm[t,:,:]/RMW[t] for t in range(len(RMW))])

                        # Fields 404 - 406: Percent change in heat fluxes due to spray (SPRWRT) [%]
                        elif f in [404,405,406]:
                            indx = {404:[391,394],
                                    405:[392,395],
                                    406:[393,396]}
                            H0pr = r.myfields[indx[f][0]][d].grdata
                            H1   = r.myfields[indx[f][1]][d].grdata
                            dH1byH0pr = (H1 - H0pr)/H0pr*100
                            dH1byH0pr[np.logical_or(np.abs(H0pr) < 1,np.abs(dH1byH0pr) > 500)] = np.nan
                            r.myfields[f][d].grdata = dH1byH0pr

                        # Field 407: ETbar (SPRWRT), with large values filtered out [-]. CURRENTLY JUST A PASS-THROUGH
                        elif f == 407:
                            ETbar = r.myfields[386][d].grdata
                            r.myfields[f][d].grdata = ETbar

                        # Field 408: ERbar (SPRWRT), with large values filtered out [-]. CURRENTLY JUST A PASS-THROUGH
                        elif f == 408:
                            ERbar = r.myfields[387][d].grdata
                            r.myfields[f][d].grdata = ERbar

                        # Fields 409 - 420: Percent change in heat fluxes due to spray (OFF) [%]
                        elif f in [409,410,411,412,413,414,415,416,417,418,419,420]:
                            indx = {409:[273,103],
                                    410:[274,104],
                                    411:[282,281],
                                    412:[275,133],
                                    413:[276,134],
                                    414:[284,283],
                                    415:[277,163],
                                    416:[278,164],
                                    417:[286,285],
                                    418:[279,193],
                                    419:[280,194],
                                    420:[288,287]}
                            H0pr = r.myfields[indx[f][0]][d].grdata
                            H1   = r.myfields[indx[f][1]][d].grdata
                            dH1byH0pr = (H1 - H0pr)/H0pr*100
                            dH1byH0pr[np.logical_or(np.abs(H0pr) < 1,np.abs(dH1byH0pr) > 500)] = np.nan
                            r.myfields[f][d].grdata = dH1byH0pr

                        # Fields 425 - 436: Feedback coefficients alphaS, betaS, and betaL with large values filtered out
                        elif f in [425,426,427,428,429,430,431,432,433,434,435,436]:
                            indx = {425:[109,106],
                                    426:[110,107],
                                    427:[111,108],
                                    428:[139,136],
                                    429:[140,137],
                                    430:[141,138],
                                    431:[169,166],
                                    432:[170,167],
                                    433:[171,168],
                                    434:[199,196],
                                    435:[200,197],
                                    436:[201,198]}
                            coeff  = r.myfields[indx[f][0]][d].grdata    # Feedback coefficient
                            Hspr   = r.myfields[indx[f][1]][d].grdata    # Spray heat flux with feedback
                            wspd10 = r.myfields[6][d].grdata    # 10-m windspeed [m s-1]
                            Hsprpr = Hspr/coeff    # Spray heat flux without feedback
                            coeff_filt = np.copy(coeff)    # Filtered coefficients
                            coeff_filt[np.abs(coeff) > 10] = np.nan    # Filter out very large values outright
                            coeff_filt[np.logical_and(wspd10 >  30,np.abs(Hsprpr) < 10)] = np.nan    # Filter out small denom cases
                            #coeff_filt[np.logical_and(wspd10 <= 30,np.abs(Hsprpr) <  2)] = np.nan    # Filter out small denom cases
                            r.myfields[f][d].grdata = coeff_filt

                        # Fields 437 - 441, 728, 729: Ch10N and Cq10N with spray, with large values filtered out
                        elif f in [437,438,439,440,441,442,443,444,445,446,728,729]:
                            indx = {437:[324,103,273],
                                    438:[331,133,275],
                                    439:[338,163,277],
                                    440:[345,193,279],
                                    441:[397,394,391],
                                    442:[325,104,274],
                                    443:[332,134,276],
                                    444:[339,164,278],
                                    445:[346,194,280],
                                    446:[398,395,392],
                                    728:[716,  7,465],
                                    729:[718,  8,466]}
                            C10N = r.myfields[indx[f][0]][d].grdata    # Unfiltered C10N
                            H1   = r.myfields[indx[f][1]][d].grdata    # HF with spray
                            H0pr = r.myfields[indx[f][2]][d].grdata    # HF without spray
                            C10Nfilt = np.copy(C10N)    # Filtered C10N
                            C10Nfilt[np.abs(H1/H0pr) > 5] = np.nan    # Filter out low H0pr cases
                            r.myfields[f][d].grdata = C10Nfilt

                        # Field 447: Spray modification to surface enthalpy flux [W m-2]
                        elif f == 447:
                            r.myfields[447][d].grdata = r.myfields[81][d].grdata + r.myfields[82][d].grdata

                        # Fields 457, 458, 495, and 496: Tangential and radial windspeeds, Earth- and Storm-Relative [m s-1]
                        elif f in [457,458,495,496]:
                            dx  = r.myfields[ 10][d].grdata    # X position of gridpoint [km]
                            dy  = r.myfields[ 11][d].grdata    # Y position of gridpoint [km]
                            uER = r.myfields[448][d].grdata    # Zonal windspeed, Earth-relative [m s-1]
                            vER = r.myfields[449][d].grdata    # Meridional windspeed, Earth-relative [m s-1]
                            th2D = np.arctan2(dy,dx)    # Angle of rotation, + CCW from +Long [rad]
                            th = np.array([np.array([th2D[t,:,:] for i in range(np.shape(uER)[1])]) for t in range(np.shape(th2D)[0])])
                            if f == 457:    # Tangential windspeed, Earth-relative
                                r.myfields[457][d].grdata = -uER*np.sin(th) + vER*np.cos(th)
                            elif f == 458:    # Radial windspeed, Earth-relative
                                r.myfields[458][d].grdata =  uER*np.cos(th) + vER*np.sin(th)
                            if f in [495,496]:
                                strmdir = np.array(r.strmdir)    # Storm direction [rad CCW from East]
                                strmspd = np.array(r.strmspeed)    # Storm translational speed [m s-1]
                                ustrm = strmspd*np.cos(strmdir)    # Storm motion vector, X-component [m s-1]
                                vstrm = strmspd*np.sin(strmdir)    # Storm motion vector, Y-component [m s-1]
                                uSR = np.array([uER[t,:,:,:] - ustrm[t] for t in range(np.shape(uER)[0])])    # Zonal wspd, storm-rel [m s-1]
                                vSR = np.array([vER[t,:,:,:] - vstrm[t] for t in range(np.shape(vER)[0])])    # Merid wspd, storm-rel [m s-1]
                                if f == 495:    # Tangential windspeed, storm-relative
                                    r.myfields[495][d].grdata = -uSR*np.sin(th) + vSR*np.cos(th)
                                elif f == 496:    # Radial windspeed, storm-relative
                                    r.myfields[496][d].grdata =  uSR*np.cos(th) + vSR*np.sin(th)

                        # Field 459: Height [m]
                        elif f == 459:
                            phi_SZ = r.myfields[450][d].grdata + r.myfields[451][d].grdata    # Z-staggered geopotential [m2 s-2]
                            r.myfields[459][d].grdata = 0.5*(phi_SZ[:,:-1,:,:] + phi_SZ[:,1:,:,:])/9.81    # [m]

                        # Field 460: Potential temperature [K]
                        elif f == 460:
                            r.myfields[460][d].grdata = r.myfields[452][d].grdata + 300    # [K]

                        # Field 461: Pressure [Pa]
                        elif f == 461:
                            r.myfields[461][d].grdata = r.myfields[453][d].grdata + r.myfields[454][d].grdata    # [Pa]

                        # Field 462: Specific humidity [kg kg-1]
                        elif f == 462:
                            r.myfields[462][d].grdata = r.myfields[455][d].grdata / \
                                    (1.0 + r.myfields[455][d].grdata)    # [kg kg-1]

                        # Field 463: Vertical velocity [m s-1]
                        elif f == 463:
                            r.myfields[463][d].grdata = 0.5*(r.myfields[456][d].grdata[:,:-1,:,:] + \
                                                             r.myfields[456][d].grdata[:,1:,:,:])

                        # Field 464: Specific enthalpy [kJ kg-1]
                        elif f == 464:
                            cpd = 1004    # Specific heat capacity of dry air [J kg-1 K-1]
                            cpv = 1952    # Specific heat capacity of water vapor [J kg-1 K-1]
                            Lv = 2.5e6    # Latent heat of vaporization of water [J kg-1]
                            theta = r.myfields[460][d].grdata    # Potential temperature [K]
                            q = r.myfields[462][d].grdata    # Specific humidity [kg kg-1]
                            r.myfields[464][d].grdata = (((1-q)*cpd + q*cpv)*theta + Lv*q)/1000    # [kJ kg-1]

                        # Field 465: Bulk SHF (spray effect removed) [W m-2]
                        elif f == 465:
                            r.myfields[465][d].grdata = r.myfields[7][d].grdata - r.myfields[81][d].grdata

                        # Field 466: Bulk LHF (spray effect removed) [W m-2]
                        elif f == 466:
                            r.myfields[466][d].grdata = r.myfields[8][d].grdata - r.myfields[82][d].grdata

                        # Field 467: Net spray SHF [W m-2]
                        elif f == 467:
                            r.myfields[467][d].grdata = r.myfields[84][d].grdata - r.myfields[85][d].grdata

                        # Field 468: Bulk enthalpy flux (spray effect removed) [W m-2]
                        elif f == 468:
                            r.myfields[468][d].grdata = r.myfields[465][d].grdata + r.myfields[466][d].grdata

                        # Field 469 - 471: Interfacial heat fluxes [W m-2]
                        elif f == 469:    # SHF
                            sprayHF = r.myfields[467][d].grdata
                            sprayHF_noNans = np.where(np.isnan(sprayHF),0.0,sprayHF)
                            r.myfields[469][d].grdata = r.myfields[7][d].grdata - sprayHF_noNans
                        elif f == 470:    # LHF
                            sprayHF = r.myfields[86][d].grdata
                            sprayHF_noNans = np.where(np.isnan(sprayHF),0.0,sprayHF)
                            r.myfields[470][d].grdata = r.myfields[8][d].grdata - sprayHF_noNans
                        elif f == 471:    # Enthalpy flux
                            sprayHF = r.myfields[83][d].grdata
                            sprayHF_noNans = np.where(np.isnan(sprayHF),0.0,sprayHF)
                            r.myfields[471][d].grdata = r.myfields[56][d].grdata - sprayHF_noNans

                        # Field 472 - 473: Radius of max azimean 10-m windspeed and max azimean 10-m windspeed (based on UWIN-CM d03 wspd10)
                        elif f in [472,473]:
                            if r.RMW == []:
                                calcRMW(r)
                            if f == 472:
                                qoi = r.RMW    # Radius of max azimean 10-m windspeed [km]
                            elif f == 473:
                                qoi = r.U10maxAzimAvg    # Max azimean 10-m windspeed [m s-1]
                            dumarr = r.myfields[1][d].grdata[0,:,:]*0    # Dummy array of zeros
                            r.myfields[f][d].grdata = np.array([dumarr+qoi[t] for t in range(len(qoi))])

                        # Field 475: Potential density anomaly [kg m-3]
                        elif f == 475:
                            temp = r.myfields[ 29][d].grdata    # 3d temperature [C]
                            sal  = r.myfields[474][d].grdata    # 3d salinity [g kg-1]
                            sigma = np.array([gsw.sigma0(sal[t,:,:,:],temp[t,:,:,:]) for t in range(np.shape(temp)[0])])    # Potential density anomaly [kg m-3]
                            r.myfields[475][d].grdata = sigma
                        
                        # Field 476: Mixed layer thickness, using Yakelyn's routine based on density
                        elif f == 476:
                            zref = 10    # Reference depth [m]
                            dT = 0.2    # Temperature difference defining MLT [K]
                            depth = r.myfields[ 28][d].grdata    # 3d depth [m]
                            temp  = r.myfields[ 29][d].grdata    # 3d temperature [C]
                            sal   = r.myfields[474][d].grdata    # 3d salinity [g kg-1]
                            sigma = r.myfields[475][d].grdata    # 3d potential density anomaly [kg m-3]
                            mlt_all = []    # Mixed layer thickness [m]
                            for t in range(np.shape(depth)[0]):
                                alpha = gsw.alpha(sal[t,:,:,:],temp[t,:,:,:],depth[t,:,:,:])    # Coeff of therm exp [K-1]
                                rho_ref   = interplevel(sigma[t,:,:,:],depth[t,:,:,:],zref,missing=np.nan) + 1000    # Density at zref [kg m-3]
                                alpha_ref = interplevel(alpha,depth[t,:,:,:],zref,missing=np.nan)    # Coeff of therm exp at zref [K-1]
                                sigma_mlt = rho_ref + alpha_ref*dT*rho_ref - 1000    # Pot density anomaly at MLT [kg m-3]
                                mlt = interplevel(depth[t,:,:,:],sigma[t,:,:,:],sigma_mlt,missing=np.nan)    # Mixed layer thickness [m]
                                mlt_all.append(mlt)
                            r.myfields[476][d].grdata = np.array(mlt_all)

                        # Field 478: Inflow angle, Earth-relative [deg]
                        elif f == 478:
                            uTan =  r.myfields[457][d].grdata    # Tangential windspeed [m s-1]
                            uRad = -r.myfields[458][d].grdata    # Radial windspeed, inflow is positive [m s-1]
                            r.myfields[478][d].grdata = np.arctan2(uRad,uTan)*180/np.pi    # Inflow angle [deg]

                        # Field 479: Inflow angle at LML, Earth-relative [deg]
                        elif f == 479:
                            infang = r.myfields[478][d].grdata    # 3d inflow angle [deg]
                            r.myfields[479][d].grdata = np.array([infang[t,0,:,:] for t in range(np.shape(infang)[0])])

                        # Field 480: Stability parameter at LML
                        elif f == 480:
                            z_LML  = r.myfields[40][d].grdata    # Height of LML [m]
                            ustar  = r.myfields[78][d].grdata    # Friction velocity (WRF output) [m s-1]
                            th_LML = r.myfields[46][d].grdata    # Pot temp at LML [K]
                            HS1    = r.myfields[ 7][d].grdata    # Surface SHF [W m-2]
                            HL1    = r.myfields[ 8][d].grdata    # Surface LHF [W m-2]
                            p_LML  = r.myfields[45][d].grdata    # Pressure at LML [Pa]
                            t_LML  = r.myfields[47][d].grdata    # Temperature at LML [K]
                            q_LML  = r.myfields[49][d].grdata    # Spec hum at LML [kg kg-1]
                            kappa = 0.41    # von Karman constant [-]
                            g = 9.81    # Acceleration due to gravity [m s-2]
                            cp_a = 1004.67    # Specific heat capacity of air [J kg-1 K-1]
                            Lv = 2.43e6    # Latent heat of vap for water at 30C [J kg-1]
                            Rdry = 287.    # Dry air gas constant [J kg-1 K-1]
                            rho_LML = p_LML/(Rdry*t_LML*(1.+0.61*q_LML))    # Air density at LML [kg m-3]
                            denom = kappa*g/th_LML*(-HS1/(rho_LML*cp_a*ustar) - 0.61*th_LML*HL1/(rho_LML*Lv*ustar))    # Denominator of L [m s-2]
                            L = ustar**2/denom    # Monin-Obukhov length [m]
                            r.myfields[480][d].grdata = z_LML/L    # Stability parameter at LML [-]

                        # Field 481: Temperature
                        elif f == 481:
                            th = r.myfields[460][d].grdata    # Potential temperature [K]
                            p  = r.myfields[461][d].grdata    # Pressure [Pa]
                            r.myfields[481][d].grdata = th*(p/1e5)**0.286    # Temperature [K]

                        # Field 482: Virtual potential temperature
                        elif f == 482:
                            w  = r.myfields[455][d].grdata    # Water vapor mixing ratio [kg kg-1]
                            th = r.myfields[460][d].grdata    # Potential temperature [K]
                            eps = 0.622    # Ratio of gas constants for dry air and water vapor
                            r.myfields[482][d].grdata = th*(w + eps)/eps/(1 + w)    # Virtual pot temperature [K]

                        # Field 483: Thermodynamic BL height
                        elif f == 483:
                            dthv = 0.5    # Virt pot temp increase defining tpbl height [K]
                            z_all   = r.myfields[459][d].grdata    # Height [m]
                            thv_all = r.myfields[482][d].grdata    # Virtual potential temperature [K]
                            tpblh = []    # Thermodynamic BL height [m]
                            for t in range(np.shape(z_all)[0]):
                                z = z_all[t,:31,:,:]    # np.choose requires no more than 31 choices
                                thv = thv_all[t,:31,:,:]    # np.choose requires no more than 31 choices
                                thv_top = np.array([thv[0,:,:] + dthv for k in range(np.shape(z)[0])])    # thv at tpbl top [K]
                                indxabv = np.nanargmax(thv > thv_top,axis=0)    # Vertical index of point just above tpbl top
                                indxblw = indxabv - 1    # Vertical index of point just below tpbl top
                                thvseq = [thv[k,:,:] for k in range(np.shape(z)[0])]    # Convert to a list
                                zseq   =   [z[k,:,:] for k in range(np.shape(z)[0])]    # Convert to a list
                                thv_abv = np.choose(indxabv,thvseq)    # thv just above tpbl top [K]
                                thv_blw = np.choose(indxblw,thvseq)    # thv just below tpbl top [K]
                                z_abv   = np.choose(indxabv,zseq)    # z just above tpbl top [m]
                                z_blw   = np.choose(indxblw,zseq)    # z just below tpbl top [m]
                                z_top = z_blw + (z_abv-z_blw)*(thv_top[0,:,:]-thv_blw)/(thv_abv-thv_blw)    # Height of tpbl [m]
                                tpblh.append(z_top)
                            r.myfields[483][d].grdata = np.array(tpblh)

                        # Fields 484 and 499: Dynamic BL height (Earth- and storm-relative)
                        elif f in [484,499]:
                            z_all = r.myfields[459][d].grdata    # Height [m]
                            if f == 484:
                                uRad_all = r.myfields[458][d].grdata    # Radial windspeed, Earth-relative [m s-1]
                            elif f == 499:
                                uRad_all = r.myfields[496][d].grdata    # Radial windspeed, storm-relative [m s-1]
                            dpblh = []    # Dynamic BL height [m]
                            for t in range(np.shape(z_all)[0]):
                                # Lower section
                                z_LWR = z_all[t,:31,:,:]    # np.choose requires no more than 31 choices
                                uRad_LWR = uRad_all[t,:31,:,:]    # np.choose requires no more than 31 choices
                                indxabv_LWR = np.nanargmax(uRad_LWR > 0.,axis=0)    # Vertical index of point just above dpbl top
                                indxblw_LWR = indxabv_LWR - 1    # Vertical index of point just below dpbl top
                                indxblw_LWR[indxblw_LWR == -1] = 0
                                uRadseq_LWR = [uRad_LWR[k,:,:] for k in range(np.shape(z_LWR)[0])]    # Convert to a list
                                zseq_LWR    =    [z_LWR[k,:,:] for k in range(np.shape(z_LWR)[0])]    # Convert to a list
                                uRad_abv_LWR = np.choose(indxabv_LWR,uRadseq_LWR)    # uRad just above dpbl top [m s-1]
                                uRad_blw_LWR = np.choose(indxblw_LWR,uRadseq_LWR)    # uRad just below dpbl top [m s-1]
                                z_abv_LWR    = np.choose(indxabv_LWR,zseq_LWR)    # z just above dpbl top [m]
                                z_blw_LWR    = np.choose(indxblw_LWR,zseq_LWR)    # z just below dpbl top [m]
                                z_top_LWR = z_blw_LWR + (z_abv_LWR-z_blw_LWR)*(0.-uRad_blw_LWR)/(uRad_abv_LWR-uRad_blw_LWR)    # Height of dpbl [m]
                                z_top_LWR[indxabv_LWR == 0] = 0.0    # Points with outflow at lowest level --> dpblh = 0
                                # Upper section
                                z_UPR = z_all[t,25:,:,:]    # Overlap a few levels with the lower section
                                uRad_UPR = uRad_all[t,25:,:,:]    # Overlap a few levels with the lower section
                                indxabv_UPR = np.nanargmax(uRad_UPR > 0.,axis=0)    # Vertical index of point just above dpbl top
                                indxblw_UPR = indxabv_UPR - 1    # Vertical index of point just below dpbl top
                                indxblw_UPR[indxblw_UPR == -1] = 0
                                uRadseq_UPR = [uRad_UPR[k,:,:] for k in range(np.shape(z_UPR)[0])]    # Convert to a list
                                zseq_UPR    =    [z_UPR[k,:,:] for k in range(np.shape(z_UPR)[0])]    # Convert to a list
                                uRad_abv_UPR = np.choose(indxabv_UPR,uRadseq_UPR)    # uRad just above dpbl top [m s-1]
                                uRad_blw_UPR = np.choose(indxblw_UPR,uRadseq_UPR)    # uRad just below dpbl top [m s-1]
                                z_abv_UPR    = np.choose(indxabv_UPR,zseq_UPR)    # z just above dpbl top [m]
                                z_blw_UPR    = np.choose(indxblw_UPR,zseq_UPR)    # z just below dpbl top [m]
                                z_top_UPR = z_blw_UPR + (z_abv_UPR-z_blw_UPR)*(0.-uRad_blw_UPR)/(uRad_abv_UPR-uRad_blw_UPR)    # Height of dpbl [m]
                                z_top_UPR[np.nanmax(uRad_UPR > 0.,axis=0) == 0] = np.nan    # Points with no outflow at any height --> dpblh undefined
                                # Select appropriate height
                                z_top = np.copy(z_top_LWR)    # Start by assuming lower section value is correct
                                useUPR = np.nanmax(uRad_LWR > 0.,axis=0) == 0    # Lower section is all inflow, so use upper section
                                z_top[useUPR] = z_top_UPR[useUPR]
                                dpblh.append(z_top)
                            r.myfields[f][d].grdata = np.array(dpblh) 

                        # Field 485: Air density
                        elif f == 485:
                            p = r.myfields[461][d].grdata    # Pressure [Pa]
                            t = r.myfields[481][d].grdata    # Temperature [K]
                            w = r.myfields[455][d].grdata    # Water vapor mixing ratio [kg kg-1]
                            Rdry = 287.    # Dry air gas constant [J kg-1 K-1]
                            eps = 0.622    # Ratio of gas constants for dry air and water vapor
                            tv = t*(w + eps)/eps/(1 + w)    # Virtual temperature [K]
                            r.myfields[485][d].grdata = p/Rdry/tv    # Air density [kg m-3]

                        # Field 486: Specific angular momentum times 1e-6 (storm-relative)
                        elif f == 486:
                            Rstorm_all = r.myfields[ 12][d].grdata*1000    # Distance to storm center (2D+time) [m]
                            uTan_all   = r.myfields[495][d].grdata    # Storm-relative tangential windspeed (3D+time) [m s-1]
                            lat_all    = r.myfields[  2][d].grdata    # Latitude (2D+time) [deg N]
                            Omega = 7.292e-5    # Angular rotation rate of earth [rad s-1]
                            f_all = 2*Omega*np.sin(np.pi/180*lat_all)    # Coriolis parameter [s-1]
                            M_all = []    # Specific angular momentum [m2 s-1]
                            for t in range(np.shape(uTan_all)[0]):    # Loop over timesteps
                                Rstorm = np.array([Rstorm_all[t,:,:] for k in range(np.shape(uTan_all)[1])])
                                f      = np.array([f_all[t,:,:]      for k in range(np.shape(uTan_all)[1])])
                                uTan = uTan_all[t,:,:,:]
                                M_all.append(Rstorm*uTan + 0.5*f*Rstorm**2)
                            r.myfields[486][d].grdata = np.array(M_all)/1e6    # Divide by 1e6

                        # Fields 487 and 488: Vertical flux of k and M through top of thermodynamic BL
                        elif f in [487,488]:
                            pblh_all  = r.myfields[483][d].grdata    # Thermodynamic BL height (2D+time) [m]
                            z_all     = r.myfields[459][d].grdata    # Height (3D+time) [m]
                            wvert_all = r.myfields[463][d].grdata    # Vertical velocity (3D+time) [m s-1]
                            rhoa_all  = r.myfields[485][d].grdata    # Air density (3D+time) [kg m-3]
                            if f == 487:
                                qoi_all = r.myfields[464][d].grdata*1000    # Specific enthalpy (3D+time) [J kg-1]
                            elif f == 488:
                                qoi_all = r.myfields[486][d].grdata*1e6    # Specific ang mom (3D+time) [m2 s-1]
                            vertflux_all = rhoa_all*wvert_all*qoi_all    # Vertical advective flux at all gridpoints, [W m-2] or [kg s-2]
                            BLtopflux_all = []
                            for t in range(np.shape(z_all)[0]):    # Loop over timesteps
                                pblh     =     pblh_all[t,:,:]    # 2D [m]
                                z        =        z_all[t,:,:,:]    # 3D [m]
                                vertflux = vertflux_all[t,:,:,:]    # 3D, [W m-2] or [kg s-2]
                                BLtopflux_all.append(interplevel(vertflux,z,pblh,missing=np.nan))
                            r.myfields[f][d].grdata = np.array(BLtopflux_all)/1000    # [kW m-2] or [Mg s-2]

                        # Field 489: Vertical velocity at top of WRF BL
                        elif f == 489:
                            pblh_all  = r.myfields[ 62][d].grdata    # WRF BL height (2D+time) [m]
                            z_all     = r.myfields[459][d].grdata    # Height (3D+time) [m]
                            wvert_all = r.myfields[463][d].grdata    # Vertical velocity (3D+time) [m s-1]
                            wvert_BLtop = np.array([interplevel(wvert_all[t,:,:,:],z_all[t,:,:,:],pblh_all[t,:,:],missing=np.nan) for t in range(np.shape(z_all)[0])])
                            r.myfields[489][d].grdata = wvert_BLtop

                        # Fields 490, 491, 537, 538, 539, and 540: Vertically-integrated radial flux of k and M within BL (WRF, thermodynamic, or dynamic; using all SR quantities)
                        elif f in [490,491,537,538,539,540]:
                            z_all    = r.myfields[459][d].grdata    # Height (3D+time) [m]
                            uRad_all = r.myfields[496][d].grdata    # SR radial windspeed (3D+time) [m s-1]
                            rhoa_all = r.myfields[485][d].grdata    # Air density (3D+time) [kg m-3]
                            if f in [490,491]:    # Use WRF BL
                                pblh_all = r.myfields[ 62][d].grdata    # BL height (2D+time) [m]
                            elif f in [537,538]:    # Use thermodynamic BL
                                pblh_all = r.myfields[483][d].grdata    # BL height (2D+time) [m]
                            elif f in [539,540]:    # Use SR dynamic BL
                                pblh_all = r.myfields[499][d].grdata    # BL height (2D+time) [m]
                            if f in [490,537,539]:    # Quantity of interest is specific enthalpy
                                qoi_all = r.myfields[464][d].grdata*1000    # Specific enthalpy (3D+time) [J kg-1]
                            elif f in [491,538,540]:    # Quantity of interest is specific angular momentum
                                qoi_all = r.myfields[486][d].grdata*1e6    # Specific ang mom (3D+time) [m2 s-1]
                            radflux_all = rhoa_all*uRad_all*qoi_all    # Radial advective flux at all gridpoints, [W m-2] or [kg s-2]
                            BLvintflux_all = []
                            for t in range(np.shape(z_all)[0]):    # Loop over timesteps
                                z       =       z_all[t,:,:,:]    # 3D [m]
                                radflux = radflux_all[t,:,:,:]    # 3D, [W m-2] or [kg s-2]
                                pblh = np.array([pblh_all[t,:,:] for k in range(np.shape(z)[0])])    # 2D [m] projected to all vertical levels
                                # Calculate vertical segment lengths for integration
                                zbelow = np.full_like(z,np.nan)    # Height of lower edge of segments [m]
                                zabove = np.full_like(z,np.nan)    # Height of upper edge of segments [m]
                                zbelow[0,:,:] = 0.0    # Surface
                                zbelow[1:,:,:] = 0.5*(z[1:,:,:] + z[:-1,:,:])
                                zabove[:-1,:,:] = zbelow[1:,:,:]
                                zabove[-1,:,:] = z[-1,:,:]    # Neglect section above highest level
                                dz = zabove - zbelow    # Segment lengths for vertical gridpoints [m]
                                at_pblh = np.logical_and(zbelow <= pblh,zabove > pblh)    # Segments intersected by PBLH
                                dz_orig = np.copy(dz[at_pblh])    # Original lengths of segments intersected by PBLH
                                dz[at_pblh] = pblh[at_pblh] - zbelow[at_pblh]    # Shorten segments intersected by PBLH
                                dz_new = np.copy(dz[at_pblh])    # New (shortened) lengths of segments intersected by PBLH
                                dz[zbelow > pblh] = np.nan    # Eliminate segments above PBLH
                                print('%.12f'%np.nanmin(dz_new/dz_orig)+'    %.12f'%np.nanmax(dz_new/dz_orig)\
                                       +'    %.12f'%np.nanmax(np.abs(pblh[0,:,:] - np.nansum(dz,axis=0)))\
                                       +'    %.12f'%np.nanmin(pblh)+'    %.12f'%np.nanmin(dz_orig)+'    %.12f'%np.nanmin(dz_new)\
                                       +'    %d'%np.nansum(zbelow == pblh))
                                #print(pblh[zbelow == pblh])
                                #print(zbelow[zbelow == pblh])
                                # Integrate vertically
                                BLvintflux_all.append(np.nansum(radflux*dz,axis=0))    # Integrate vertically, [W m-1] or [kg m s-2]
                            r.myfields[f][d].grdata = np.array(BLvintflux_all)/1e6    # [MW m-1] or [Gg m s-2]

                        # Field 492: Tangential windspeed at LML (Earth-relative)
                        elif f == 492:
                            r.myfields[492][d].grdata = r.myfields[457][d].grdata[:,0,:,:]

                        # Field 493: Radial windspeed at LML (Earth-relative)
                        elif f == 493:
                            r.myfields[493][d].grdata = r.myfields[458][d].grdata[:,0,:,:]

                        # Field 494: Pressure in mb
                        elif f == 494:
                            r.myfields[494][d].grdata = r.myfields[461][d].grdata/100

                        # Field 497: Inflow angle, storm-relative [deg]
                        elif f == 497:
                            uTan =  r.myfields[495][d].grdata    # Tangential windspeed [m s-1]
                            uRad = -r.myfields[496][d].grdata    # Radial windspeed, inflow is positive [m s-1]
                            r.myfields[497][d].grdata = np.arctan2(uRad,uTan)*180/np.pi    # Inflow angle [deg]

                        # Field 498: Inflow angle at LML, storm-relative [deg]
                        elif f == 498:
                            infang = r.myfields[497][d].grdata    # 3d inflow angle [deg]
                            r.myfields[498][d].grdata = np.array([infang[t,0,:,:] for t in range(np.shape(infang)[0])])

                        # Field 500: Tangential windspeed at LML (storm-relative)
                        elif f == 500:
                            r.myfields[500][d].grdata = r.myfields[495][d].grdata[:,0,:,:]

                        # Field 501: Radial windspeed at LML (storm-relative)
                        elif f == 501:
                            r.myfields[501][d].grdata = r.myfields[496][d].grdata[:,0,:,:]

                        # Field 502: Pressure at LML in mb
                        elif f == 502:
                            r.myfields[502][d].grdata = r.myfields[45][d].grdata/100

                        # Fields 503 and 504: X and Y storm-centered coordinates (rotated to be motion-relative)
                        elif f == 503:
                            x = r.myfields[10][d].grdata    # Unrotated X coordinates [km]
                            y = r.myfields[11][d].grdata    # Unrotated Y coordinates [km]
                            strmdir = r.strmdir    # Storm motion direction [rad CCW from East]
                            theta = np.array(strmdir) - np.pi/2    # Rotation angle [rad]
                            xRot = np.array([ x[t,:,:]*np.cos(theta[t]) + y[t,:,:]*np.sin(theta[t]) for t in range(len(strmdir))])    # X coordinates rotated to be motion-relative [km]
                            yRot = np.array([-x[t,:,:]*np.sin(theta[t]) + y[t,:,:]*np.cos(theta[t]) for t in range(len(strmdir))])    # Y coordinates rotated to be motion-relative [km]
                            r.myfields[503][d].grdata = xRot
                            r.myfields[504][d].grdata = yRot

                        # Fields 505 - 509: Horizontal slices through WRF 3D fields
                        elif f in [505,506,507,508,509]:
                            indx3D = wg.field_info[f][4][0]    # Index of 3D field for slicing
                            field3D = r.myfields[indx3D][d].grdata    # 3D field for slicing
                            if f == 505:
                                lev = wg.WRFsliceAlevels[0]
                            elif f == 506:
                                lev = wg.WRFsliceAlevels[1]
                            elif f == 507:
                                lev = wg.WRFsliceAlevels[2]
                            elif f == 508:
                                lev = wg.WRFsliceAlevels[3]
                            elif f == 509:
                                lev = wg.WRFsliceAlevels[4]
                            r.myfields[f][d].grdata = field3D[:,lev,:,:]

                        # Field 510: 3D Windspeed [m s-1]
                        elif f == 510:
                            r.myfields[f][d].grdata = (r.myfields[448][d].grdata**2 + r.myfields[449][d].grdata**2)**0.5

                        # Field 512: Specific available energy for heat transfer due to temperature change; no wetbulb contribution [J kg-1] (SPRWRT calcs)
                        elif f == 512:
                            cp_sw = 4200.    # Specific heat capacity of seawater [J kg-1 K-1]
                            r.myfields[f][d].grdata = cp_sw*r.myfields[381][d].grdata

                        # Field 513: Specific available energy for heat transfer due to temperature change; wetbulb contribution only [J kg-1] (SPRWRT calcs)
                        elif f == 513:
                            r.myfields[f][d].grdata = r.myfields[384][d].grdata - r.myfields[512][d].grdata

                        # Field 515: t_0 - t_10Npr (global calculations) [K]
                        elif f == 515:
                            r.myfields[f][d].grdata = r.myfields[99][d].grdata - r.myfields[514][d].grdata

                        # Field 521: Virtual potential temperature at LML [K]
                        elif f == 521:
                            r.myfields[f][d].grdata = r.myfields[482][d].grdata[:,0,:,:]

                        # Field 522: Relative humidity [%]
                        elif f == 522:
                            r.myfields[f][d].grdata = satratio(r.myfields[481][d].grdata,r.myfields[461][d].grdata,r.myfields[462][d].grdata,2)*100

                        # Field 523: Specific enthalpy at LML [kJ kg-1]
                        elif f == 523:
                            r.myfields[f][d].grdata = r.myfields[464][d].grdata[:,0,:,:]

                        # Field 524: Vertical velocity at LML [m s-1]
                        elif f == 524:
                            r.myfields[f][d].grdata = r.myfields[463][d].grdata[:,0,:,:]

                        # Field 525: Radial advective flux of enthalpy [kW m-2]
                        elif f == 525:
                            rhoa = r.myfields[485][d].grdata    # Air density [kg m-3]
                            uRad = r.myfields[496][d].grdata    # Radial windspeed (storm relative) [m s-1]
                            k    = r.myfields[464][d].grdata*1000    # Specific enthalpy [J kg-1]
                            r.myfields[f][d].grdata = rhoa*uRad*k/1e3    # [kW m-2]

                        # Field 526: Radial advective flux of enthalpy at LML [kW m-2]
                        elif f == 526:
                            r.myfields[f][d].grdata = r.myfields[525][d].grdata[:,0,:,:]

                        # Field 527: Vertical advective flux of enthalpy [kW m-2]
                        elif f == 527:
                            rhoa  = r.myfields[485][d].grdata    # Air density [kg m-3]
                            wvert = r.myfields[463][d].grdata    # Vertical velocity [m s-1]
                            k     = r.myfields[464][d].grdata*1000    # Specific enthalpy [J kg-1]
                            r.myfields[f][d].grdata = rhoa*wvert*k/1e3    # [kW m-2]

                        # Field 528: Vertical advective flux of enthalpy at LML [kW m-2]
                        elif f == 528:
                            r.myfields[f][d].grdata = r.myfields[527][d].grdata[:,0,:,:]

                        # Field 529: SS-based active whitecap fraction per DLM17 [-]
                        elif f == 529:
                            dcp   = r.myfields[24][d].grdata    # Dominant phase speed [m s-1]
                            ustar = r.myfields[21][d].grdata    # Friction velocity [m s-1]
                            swh   = r.myfields[22][d].grdata    # Significant wave height [m]
                            r.myfields[f][d].grdata = whitecapActive_DLM17(dcp,ustar,swh)

                        # Field 530: Wind-based total whitecap fraction from Chris Fairall [-]
                        elif f == 530:
                            wspd10 = r.myfields[6][d].grdata    # 10m windspeed [m s-1]
                            r.myfields[f][d].grdata = whitecap_CF210504(wspd10)

                        # Field 531: WRF PBL height [km]
                        elif f == 531:
                            r.myfields[f][d].grdata = r.myfields[62][d].grdata/1000

                        # Field 532: Thermodynamic BL height [km]
                        elif f == 532:
                            r.myfields[f][d].grdata = r.myfields[483][d].grdata/1000

                        # Field 533: Dynamic BL height (storm-relative) [km]
                        elif f == 533:
                            r.myfields[f][d].grdata = r.myfields[499][d].grdata/1000

                        # Field 534: Radial advective flux of angular momentum [Mg s-2]
                        elif f == 534:
                            rhoa = r.myfields[485][d].grdata    # Air density [kg m-3]
                            uRad = r.myfields[496][d].grdata    # Radial windspeed (storm relative) [m s-1]
                            M    = r.myfields[486][d].grdata*1e6    # Specific angular momentum [m2 s-1]
                            r.myfields[f][d].grdata = rhoa*uRad*M/1e3    # [Mg s-2]

                        # Field 535: Radial advective flux of angular momentum at LML [Mg s-2]
                        elif f == 535:
                            r.myfields[f][d].grdata = r.myfields[534][d].grdata[:,0,:,:]

                        # Field 536: Specific angular momentum at LML times 1e-6 [m2 s-1]
                        elif f == 536:
                            r.myfields[f][d].grdata = r.myfields[486][d].grdata[:,0,:,:]

                        # Field 541: Specific humidity [g kg-1]
                        elif f == 541:
                            r.myfields[f][d].grdata = r.myfields[462][d].grdata*1000

                        # Field 542: Specific humidity at LML [g kg-1]
                        elif f == 542:
                            r.myfields[f][d].grdata = r.myfields[541][d].grdata[:,0,:,:]

                        # Field 543: Spray mass flux (Nans removed) [kg m-2 s-1]
                        elif f == 543:
                            mask = r.myfields[3][d].grdata[:,:,:]    # WRF landmask (1=land,2=sea)
                            r.myfields[f][d].grdata = np.where(np.logical_and(mask == 2,\
                                    np.isnan(r.myfields[93][d].grdata[:,:,:])),1e-20,r.myfields[93][d].grdata[:,:,:])

                        # Field 544: Spray heat flux due to temperature change (Nans removed) [W m-2]
                        elif f == 544:
                            mask = r.myfields[3][d].grdata[:,:,:]    # WRF landmask (1=land,2=sea)
                            r.myfields[f][d].grdata = np.where(np.logical_and(mask == 2,\
                                    np.isnan(r.myfields[83][d].grdata[:,:,:])),1e-20,r.myfields[83][d].grdata[:,:,:])

                        # Field 545: Spray LHF (Nans removed) [W m-2]
                        elif f == 545:
                            mask = r.myfields[3][d].grdata[:,:,:]    # WRF landmask (1=land,2=sea)
                            r.myfields[f][d].grdata = np.where(np.logical_and(mask == 2,\
                                    np.isnan(r.myfields[86][d].grdata[:,:,:])),1e-20,r.myfields[86][d].grdata[:,:,:])

                        # Field 546: Net spray SHF (Nans removed) [W m-2]
                        elif f == 546:
                            mask = r.myfields[3][d].grdata[:,:,:]    # WRF landmask (1=land,2=sea)
                            r.myfields[f][d].grdata = np.where(np.logical_and(mask == 2,\
                                    np.isnan(r.myfields[467][d].grdata[:,:,:])),1e-20,r.myfields[467][d].grdata[:,:,:])

                        # Field 549: Vertical advective flux of enthalpy [W m-2]
                        elif f == 549:
                            r.myfields[f][d].grdata = r.myfields[527][d].grdata*1000

                        # Fields 550 - 569: WRF fields filtered by RMW and z limits
                        elif f in [550,551,552,553,554,555,556,557,558,559,\
                                   560,561,562,563,564,565,566,567,568,569]:
                            RMWlims = {550:[wg.WRFfield_RMWfilt_A_defs[1],wg.WRFfield_RMWfilt_A_defs[2],wg.WRFfield_RMWfilt_A_defs[3]],
                                       551:[wg.WRFfield_RMWfilt_B_defs[1],wg.WRFfield_RMWfilt_B_defs[2],wg.WRFfield_RMWfilt_B_defs[3]],
                                       552:[wg.WRFfield_RMWfilt_C_defs[1],wg.WRFfield_RMWfilt_C_defs[2],wg.WRFfield_RMWfilt_C_defs[3]],
                                       553:[wg.WRFfield_RMWfilt_D_defs[1],wg.WRFfield_RMWfilt_D_defs[2],wg.WRFfield_RMWfilt_D_defs[3]],
                                       554:[wg.WRFfield_RMWfilt_E_defs[1],wg.WRFfield_RMWfilt_E_defs[2],wg.WRFfield_RMWfilt_E_defs[3]],
                                       555:[wg.WRFfield_RMWfilt_F_defs[1],wg.WRFfield_RMWfilt_F_defs[2],wg.WRFfield_RMWfilt_F_defs[3]],
                                       556:[wg.WRFfield_RMWfilt_G_defs[1],wg.WRFfield_RMWfilt_G_defs[2],wg.WRFfield_RMWfilt_G_defs[3]],
                                       557:[wg.WRFfield_RMWfilt_H_defs[1],wg.WRFfield_RMWfilt_H_defs[2],wg.WRFfield_RMWfilt_H_defs[3]],
                                       558:[wg.WRFfield_RMWfilt_I_defs[1],wg.WRFfield_RMWfilt_I_defs[2],wg.WRFfield_RMWfilt_I_defs[3]],
                                       559:[wg.WRFfield_RMWfilt_J_defs[1],wg.WRFfield_RMWfilt_J_defs[2],wg.WRFfield_RMWfilt_J_defs[3]],
                                       560:[wg.WRFfield_RMWfilt_K_defs[1],wg.WRFfield_RMWfilt_K_defs[2],wg.WRFfield_RMWfilt_K_defs[3]],
                                       561:[wg.WRFfield_RMWfilt_L_defs[1],wg.WRFfield_RMWfilt_L_defs[2],wg.WRFfield_RMWfilt_L_defs[3]],
                                       562:[wg.WRFfield_RMWfilt_M_defs[1],wg.WRFfield_RMWfilt_M_defs[2],wg.WRFfield_RMWfilt_M_defs[3]],
                                       563:[wg.WRFfield_RMWfilt_N_defs[1],wg.WRFfield_RMWfilt_N_defs[2],wg.WRFfield_RMWfilt_N_defs[3]],
                                       564:[wg.WRFfield_RMWfilt_O_defs[1],wg.WRFfield_RMWfilt_O_defs[2],wg.WRFfield_RMWfilt_O_defs[3]],
                                       565:[wg.WRFfield_RMWfilt_P_defs[1],wg.WRFfield_RMWfilt_P_defs[2],wg.WRFfield_RMWfilt_P_defs[3]],
                                       566:[wg.WRFfield_RMWfilt_Q_defs[1],wg.WRFfield_RMWfilt_Q_defs[2],wg.WRFfield_RMWfilt_Q_defs[3]],
                                       567:[wg.WRFfield_RMWfilt_R_defs[1],wg.WRFfield_RMWfilt_R_defs[2],wg.WRFfield_RMWfilt_R_defs[3]],
                                       568:[wg.WRFfield_RMWfilt_S_defs[1],wg.WRFfield_RMWfilt_S_defs[2],wg.WRFfield_RMWfilt_S_defs[3]],
                                       569:[wg.WRFfield_RMWfilt_T_defs[1],wg.WRFfield_RMWfilt_T_defs[2],wg.WRFfield_RMWfilt_T_defs[3]]}
                            indxWRFfield = wg.field_info[f][4][1]    # Index of field to filter
                            RbyRMW   = r.myfields[375][d].grdata    # 2D + time
                            WRFfield = r.myfields[indxWRFfield][d].grdata    # Field to filter, could be 2D + time or 3D + time
                            keepR = np.logical_and(RbyRMW >= RMWlims[f][0],RbyRMW <= RMWlims[f][1])    # 2D + time
                            if WRFfield.ndim == 3:    # 2D + time
                                r.myfields[f][d].grdata = np.where(keepR,WRFfield,np.nan)
                            elif WRFfield.ndim == 4:    # 3D + time
                                keepR3D = np.array([[keepR[t,:,:] for k in range(np.shape(WRFfield)[1])] for t in range(np.shape(WRFfield)[0])])
                                keepZ = (r.myfields[459][d].grdata <= RMWlims[f][2]*1000)
                                keep = np.logical_and(keepR3D,keepZ)
                                r.myfields[f][d].grdata = np.where(keep,WRFfield,np.nan)

                        # Field 572: Cloud water mixing ratio [g kg-1]
                        elif f == 572:
                            r.myfields[f][d].grdata = r.myfields[570][d].grdata*1000

                        # Field 573: Rain water mixing ratio [g kg-1]
                        elif f == 573:
                            r.myfields[f][d].grdata = r.myfields[571][d].grdata*1000

                        # Field 576: Air density at LML [kg m-3]
                        elif f == 576:
                            r.myfields[f][d].grdata = r.myfields[485][d].grdata[:,0,:,:]

                        # Field 577: Surface buoyancy flux [m2 s-3]
                        elif f == 577:
                            g = 9.81    # Acceleration due to gravity [m s-2]
                            cp = 1004.67    # Specific heat capacity of air [J kg-1 K-1]
                            Lv = 2.43e6    # Latent heat of vap for water [J kg-1]
                            HS1     = r.myfields[  7][d].grdata    # Surface sensible heat flux [W m-2]
                            HL1     = r.myfields[  8][d].grdata    # Surface latent heat flux [W m-2]
                            th_LML  = r.myfields[ 46][d].grdata    # Potential temperature at LML [K]
                            thv_LML = r.myfields[521][d].grdata    # Virtual potential temperature at LML [K]
                            rho_LML = r.myfields[576][d].grdata    # Air density at LML [kg m-3]
                            r.myfields[f][d].grdata = g/rho_LML/cp/thv_LML*(HS1 + 0.61*cp*th_LML/Lv*HL1)

                        # Fields 578,579,580,583: Vertical gradients
                        elif f in [578,579,580,583,594,595]:
                            indx = {578:448,
                                    579:449,
                                    580:482,
                                    583:510,
                                    594:495,
                                    595:496}
                            z = r.myfields[459][d].grdata    # Height [m]
                            var = r.myfields[indx[f]][d].grdata    # Variable to differentiate [m s-1 or K]
                            dvardz = np.full_like(var,np.nan)
                            for t in range(np.shape(var)[0]):    # Loop over time
                                for j in range(np.shape(var)[2]):    # Loop over latitude
                                    for i in range(np.shape(var)[3]):    # Loop over longitude
                                        dvardz[t,:,j,i] = np.gradient(var[t,:,j,i],z[t,:,j,i])
                            r.myfields[f][d].grdata = dvardz

                        # Field 581: Gradient Richardson number [-]
                        elif f == 581:
                            g = 9.81    # Acceleration due to gravity [m s-2]
                            thv    = r.myfields[482][d].grdata    # Virtual potential temperature [K]
                            dudz   = r.myfields[578][d].grdata    # Vertical gradient in zonal windspeed [s-1]
                            dvdz   = r.myfields[579][d].grdata    # Vertical gradient in meridional windspeed [s-1]
                            dthvdz = r.myfields[580][d].grdata    # Vertical gradient in thetav [K m-1]
                            r.myfields[f][d].grdata = g/thv*dthvdz/(dudz**2 + dvdz**2)

                        # Field 582: |dU/dz| (Resultant of derivative of vector) [s-1]
                        elif f == 582:
                            r.myfields[f][d].grdata = np.sqrt(r.myfields[578][d].grdata**2 + r.myfields[579][d].grdata**2)

                        # Field 584: Obukhov stability length [m]
                        elif f == 584:
                            r.myfields[f][d].grdata = -r.myfields[78][d].grdata**3/0.4/r.myfields[577][d].grdata

                        # Field 585: Stability parameter zeta
                        elif f == 585:
                            z_all = r.myfields[459][d].grdata    # Height [m], 3D + time
                            L_all = r.myfields[584][d].grdata    # Obukhov stability length [m], 2D + time
                            zeta_all = []
                            for t in range(np.shape(z_all)[0]):    # Loop over time
                                z = z_all[t,:,:,:]
                                L = np.array([L_all[t,:,:] for k in range(np.shape(z_all)[1])])
                                zeta_all.append(z/L)
                            r.myfields[f][d].grdata = np.array(zeta_all)

                        # Fields 586,587,588,596,597: Diagnosed YSU PBL Computations
                        elif f in [586,587,588,596,597]:
                            b = 6.8    # Coefficient in minusGammaM
                            kappa = 0.4    # von Karman constant
                            pblh    = r.myfields[ 62][d].grdata    # WRF PBLH [m], 2D + time
                            ustar   = r.myfields[ 78][d].grdata    # Friction velocity [m s-1], 2D + time
                            Bflux   = r.myfields[577][d].grdata    # Surface buoyancy flux [m2 s-3], 2D + time
                            L       = r.myfields[584][d].grdata    # Obukhov length [m], 2D + time
                            z       = r.myfields[459][d].grdata    # Height [m], 3D + time
                            rho     = r.myfields[485][d].grdata    # Air density [kg m-3], 3D + time
                            dUtotdz = r.myfields[583][d].grdata    # Vertical gradient in total windspeed [s-1], 3D + time
                            dUtandz = r.myfields[594][d].grdata    # Vertical gradient in tangential windspeed [s-1], 3D + time
                            dUraddz = r.myfields[595][d].grdata    # Vertical gradient in radial windspeed [s-1], 3D + time
                            # Make calculations on 2D fields
                            phiM = np.where(Bflux >= 0.,(1 - 16*0.1*pblh/L)**-0.25,1 + 5*0.1*pblh/L)    # Stability profile function
                            wstarb = np.where(Bflux >= 0.,(Bflux*pblh)**(1/3),0.)    # Convective velocity scale [m s-1], must be >= 0
                            ws0 = (ustar**3 + phiM*kappa*wstarb**3*0.5)**(1/3)    # Mixed layer velocity scale at z = 0.5*pblh [m s-1]
                            minusGammaM = -b*(-ustar**2)/ws0/pblh    # Momentum countergradient term [s-1], a "resultant" quantity
                            if f == 587:
                                r.myfields[f][d].grdata = minusGammaM
                            # Convert 2D fields to 3D
                            pblh_k = []
                            ustar_k = []
                            phiM_k = []
                            wstarb_k = []
                            minusGammaM_k = []
                            for t in range(np.shape(z)[0]):    # Loop over time
                                pblh_k.append(       np.array([pblh[t,:,:]        for k in range(np.shape(z)[1])]))
                                ustar_k.append(      np.array([ustar[t,:,:]       for k in range(np.shape(z)[1])]))
                                phiM_k.append(       np.array([phiM[t,:,:]        for k in range(np.shape(z)[1])]))
                                wstarb_k.append(     np.array([wstarb[t,:,:]      for k in range(np.shape(z)[1])]))
                                minusGammaM_k.append(np.array([minusGammaM[t,:,:] for k in range(np.shape(z)[1])]))
                            pblh_k = np.array(pblh_k)
                            ustar_k = np.array(ustar_k)
                            phiM_k = np.array(phiM_k)
                            wstarb_k = np.array(wstarb_k)
                            minusGammaM_k = np.array(minusGammaM_k)
                            # Calculate eddy diffusivity
                            ws = (ustar_k**3 + phiM_k*kappa*wstarb_k**3*z/pblh_k)**(1/3)    # Mixed layer velocity scale [m s-1]
                            KM = kappa*ws*z*(1 - z/pblh_k)**2    # Eddy diffusivity for momentum [m2 s-1]
                            KM[z > pblh_k] = 0.
                            if f == 586:
                                r.myfields[f][d].grdata = KM
                            # Calculate directional and "resultant" shear stresses
                            MomFluxTot = -KM*(dUtotdz + minusGammaM_k)    # "Resultant" kinematic momentum flux (positive upwards) [m2 s-2]
                            MomFluxTan = -KM*dUtandz    # Tangential kinematic momentum flux [m2 s-2]
                            MomFluxRad = -KM*dUraddz    # Radial kinematic momentum flux [m2 s-2]
                            if f == 588:
                                r.myfields[f][d].grdata = -rho*MomFluxTot    # "Resultant" total stress (positive for downward mom flux) [Pa]
                            if f == 596:
                                r.myfields[f][d].grdata = -rho*MomFluxTan    # Local turbulent tangential stress (pos in +theta direction) [Pa]
                            if f == 597:
                                r.myfields[f][d].grdata = -rho*MomFluxRad    # Local turbulent radial stress (pos in +r direction) [Pa]

                        # Field 589: Total accumulated rainfall [mm]
                        elif f == 589:
                            totaccum = r.myfields[66][d].grdata + r.myfields[67][d].grdata    # RAINC + RAINNC [mm]
                            r.myfields[f][d].grdata = totaccum

                        # Field 590: Total accumulated rainfall within 500km of storm center [mm]
                        elif f == 590:
                            Rlim = 500    # Limit for accumulating rainfall [km]
                            totaccum = r.myfields[66][d].grdata + r.myfields[67][d].grdata    # RAINC + RAINNC [mm]
                            intaccum = np.full_like(totaccum,0)    # Rain accumulated during interval [mm]
                            intaccum[1:,:,:] = totaccum[1:,:,:] - totaccum[:-1,:,:]
                            Rstorm = r.myfields[12][d].grdata    # Distance to storm center [km]
                            intaccumFILT = np.where(Rstorm <= Rlim,intaccum,0)    # Remove rainfall outside limit
                            totaccumFILT = np.full_like(totaccum,0)
                            for i in range(1,np.shape(totaccum)[0]):
                                totaccumFILT[i,:,:] = totaccumFILT[i-1,:,:] + intaccumFILT[i,:,:]
                            totaccumFILT[0,:,:] = 1e-20    # Set not equal to zero to make plotting happy
                            print('\n\n')
                            print(r.run_path)
                            print('Total accumulated rainfall sum:')
                            print(np.nansum(totaccumFILT[-1,:,:]))
                            print('\n\n')
                            r.myfields[f][d].grdata = totaccumFILT

                        # Field 591, 592: Max value of selected field through time using all 3 WRF domains
                        elif f in [591,592]:
                            indx = {591: 6,
                                    592:54}
                            Rlim = 500.    # Limit for selecting max values
                            # Get fields and define grids for max field
                            lon_d01_All = r.myfields[1][1].grdata    # d01 longitude [deg]
                            lon_d02_All = r.myfields[1][2].grdata    # d02 longitude [deg]
                            lon_d03_All = r.myfields[1][3].grdata    # d03 longitude [deg]
                            lat_d01_All = r.myfields[2][1].grdata    # d01 latitude [deg]
                            lat_d02_All = r.myfields[2][2].grdata    # d02 latitude [deg]
                            lat_d03_All = r.myfields[2][3].grdata    # d03 latitude [deg]
                            field_d01_All = np.where(r.myfields[12][1].grdata <= Rlim,r.myfields[indx[f]][1].grdata,np.nan)    # Selected field on d01, filtered by Rlim
                            field_d02_All = np.where(r.myfields[12][2].grdata <= Rlim,r.myfields[indx[f]][2].grdata,np.nan)    # Selected field on d01, filtered by Rlim
                            field_d03_All = np.where(r.myfields[12][3].grdata <= Rlim,r.myfields[indx[f]][3].grdata,np.nan)    # Selected field on d01, filtered by Rlim
                            lon_vect = np.arange(wg.lonlatlims_MaxField[0],wg.lonlatlims_MaxField[1]+0.01,0.01)
                            lat_vect = np.arange(wg.lonlatlims_MaxField[2],wg.lonlatlims_MaxField[3]+0.01,0.01)
                            wg.lat_MaxField,wg.lon_MaxField = np.meshgrid(lat_vect,lon_vect,indexing='ij')    # Create lon/lat grids for field of max values
                            field_max_All = []
                            # Loop over timesteps
                            for t in range(np.shape(lon_d01_All)[0]):
                                # Get fields for this timestep
                                lon_d01   =   lon_d01_All[t,:,:][~np.isnan(field_d01_All[t,:,:])]
                                lat_d01   =   lat_d01_All[t,:,:][~np.isnan(field_d01_All[t,:,:])]
                                field_d01 = field_d01_All[t,:,:][~np.isnan(field_d01_All[t,:,:])]
                                lon_d02   =   lon_d02_All[t,:,:][~np.isnan(field_d02_All[t,:,:])]
                                lat_d02   =   lat_d02_All[t,:,:][~np.isnan(field_d02_All[t,:,:])]
                                field_d02 = field_d02_All[t,:,:][~np.isnan(field_d02_All[t,:,:])]
                                lon_d03   =   lon_d03_All[t,:,:][~np.isnan(field_d03_All[t,:,:])]
                                lat_d03   =   lat_d03_All[t,:,:][~np.isnan(field_d03_All[t,:,:])]
                                field_d03 = field_d03_All[t,:,:][~np.isnan(field_d03_All[t,:,:])]
                                # Perform interpolation
                                lonlat_d01_forInterp = np.transpose(np.array([lon_d01.flatten(),lat_d01.flatten()]))
                                lonlat_d02_forInterp = np.transpose(np.array([lon_d02.flatten(),lat_d02.flatten()]))
                                lonlat_d03_forInterp = np.transpose(np.array([lon_d03.flatten(),lat_d03.flatten()]))
                                field_d01_forInterp = field_d01.flatten()
                                field_d02_forInterp = field_d02.flatten()
                                field_d03_forInterp = field_d03.flatten()
                                field_d01_remapped = griddata(lonlat_d01_forInterp,field_d01_forInterp,(ug.lon_MaxField,ug.lat_MaxField),method='linear')
                                field_d02_remapped = griddata(lonlat_d02_forInterp,field_d02_forInterp,(ug.lon_MaxField,ug.lat_MaxField),method='linear')
                                field_d03_remapped = griddata(lonlat_d03_forInterp,field_d03_forInterp,(ug.lon_MaxField,ug.lat_MaxField),method='linear')
                                # Combine fields for all 3 domains for this timestep
                                field_max_now = np.full_like(field_d01_remapped,np.nan)
                                field_max_now[~np.isnan(field_d01_remapped)] = field_d01_remapped[~np.isnan(field_d01_remapped)]
                                field_max_now[~np.isnan(field_d02_remapped)] = field_d02_remapped[~np.isnan(field_d02_remapped)]
                                field_max_now[~np.isnan(field_d03_remapped)] = field_d03_remapped[~np.isnan(field_d03_remapped)]
                                # Add to field_max_All
                                if t == 0:
                                    field_max_All.append(field_max_now)
                                else:
                                    max_update = np.fmax(field_max_All[-1],field_max_now)
                                    field_max_All.append(max_update)
                            field_max_All = np.array(field_max_All)
                            print('\n')
                            print(r.run_path)
                            print('Max field sum (All 3 domains):')
                            print(np.nansum(field_max_All[-1,:,:]))
                            print('\n')
                            r.myfields[f][d].grdata = field_max_All

                        # Field 593: Max value of selected field through time using only the selected domain
                        elif f in [593]:
                            indx = {593:6}
                            Rlim = 500.    # Limit for selecting max values
                            field_All = np.where(r.myfields[12][d].grdata <= Rlim,r.myfields[indx[f]][d].grdata,np.nan)    # Selected field on selected domain, filtered by Rlim
                            field_max_All = []
                            for t in range(np.shape(field_All)[0]):
                                field_max_now = field_All[t,:,:]
                                if t == 0:
                                    field_max_All.append(field_max_now)
                                else:
                                    max_update = np.fmax(field_max_All[-1],field_max_now)
                                    field_max_All.append(max_update)
                            field_max_All = np.array(field_max_All)
                            print('\n')
                            print(r.run_path)
                            print('Max field sum (single domain):')
                            print(np.nansum(field_max_All[-1,:,:]))
                            print('\n')
                            r.myfields[f][d].grdata = field_max_All

                        # Field 598: Product of SR uRad and equivalent potential temperature [m s-1 K]
                        elif f == 598:
                            r.myfields[f][d].grdata = r.myfields[496][d].grdata*r.myfields[575][d].grdata

                        # Field 599: Nondimensional depth version 2 (dwn*depth/pi) [-]
                        elif f == 599:
                            r.myfields[f][d].grdata = r.myfields[100][d].grdata/np.pi

                        # Field 600: Magnitude of storm-relative secondary circulation windspeed [m s-1]
                        elif f == 600:
                            r.myfields[f][d].grdata = np.sqrt(r.myfields[463][d].grdata**2 + r.myfields[496][d].grdata**2)
                                                          
                        # Field 601: Product of SR u2nd and equivalent potential temperature [m s-1 K]
                        elif f == 601:
                            r.myfields[f][d].grdata = r.myfields[600][d].grdata*r.myfields[575][d].grdata

                        # Field 706: KE per unit volume at 10m [N m-2]
                        elif f == 706:
                            rho_10 = 1.0    # Density at 10m, will update later [kg m-3]
                            U10 = r.myfields[6][d].grdata    # 10m windspeed [m s-1]
                            KE = 0.5*rho_10*U10**2    # KE per unit volume [N m-2]
                            r.myfields[f][d].grdata = KE

                        # Fields 707 - 715, 734 - 736: Ratios of heat fluxes
                        elif f in [707,708,709,710,711,712,713,714,715,734,735,736]:
                            indx = {707:[467,469],
                                    708:[ 86,470],
                                    709:[ 83,471],
                                    710:[467,  7],
                                    711:[ 86,  8],
                                    712:[ 83, 56],
                                    713:[ 81,465],
                                    714:[ 82,466],
                                    715:[447,468],
                                    734:[469,  7],
                                    735:[470,  8],
                                    736:[471, 56]}
                            Htop = r.myfields[indx[f][0]][d].grdata    # Value in ratio top
                            Hbot = r.myfields[indx[f][1]][d].grdata    # Value in ratio bottom
                            Htop[np.isnan(Htop)] =  0.0    # Remove nans
                            ratio = Htop/Hbot
                            ratio[np.abs(ratio) > 3] = np.nan
                            if f in [710,711,712,713,714,715,734,735,736]:    # Convert to percent
                                ratio = ratio*100
                            r.myfields[f][d].grdata = ratio

                        # Fields 716 - 721, 724 - 727: Heat transfer coefficients, 10Npr conditions, and available energies (same calcs as spraylibs_v4)
                        elif f in [716,717,718,719,720,721,724,725,726,727,730,731]:

                            # 0. Constants and imported fields
                            cp_a = 1004.67    # Specific heat capacity of air [J kg-1 K-1]
                            cp_sw = 4200    # Specific heat capacity of seawater [J kg-1 K-1]
                            Lv = 2.43e6    # Latent heat of vap for water at 30C [J kg-1]
                            Rdry = 287.    # Dry air gas constant [J kg-1 K-1]
                            kappa = 0.41    # von Karman constant [-]
                            xs = 0.035    # Mass fraction of salt in seawater [-]
                            rho_sw = 1030    # Density of seawater [kg m-3]
                            rho_dry = 2160    # Density of chrystalline salt [kg m-3]
                            nu = 2    # Number of ions into which NaCl dissociates [-]
                            Phi_s = 0.924    # Practical osmotic coefficient at molality of 0.6 [-]
                            Mw = 18.02    # Molecular weight of water [g mol-1]
                            Ms = 58.44    # Molecular weight of salt [g mol-1]
                            g = 9.81    # Acceleration due to gravity [m s-2]
                            Pr_a = 0.71    # Prandtl number for air [-]
                            Sc_a = 0.60    # Schmidt number for air [-]
                            H_tol = 0.01    # Tolerance for convergence for heat fluxes [W m-2]
                            z_1     = r.myfields[ 40][d].grdata    # Height of LML [m]
                            U_1     = r.myfields[ 54][d].grdata    # Total windspeed at LML [m s-1]
                            th_1    = r.myfields[ 46][d].grdata    # Potential temperature at LML [K]
                            q_1     = r.myfields[ 49][d].grdata    # Specific humidity at LML [kg kg-1]
                            p_0     = r.myfields[ 72][d].grdata    # Surface pressure [Pa]
                            ustar   = r.myfields[ 78][d].grdata    # Friction velocity (WRF) [m s-1]
                            t_0     = r.myfields[ 15][d].grdata    # Surface temperature [K]
                            dHS1spr = r.myfields[ 81][d].grdata    # Spray modification to SHF [W m-2]
                            dHL1spr = r.myfields[ 82][d].grdata    # Spray modification to LHF [W m-2]
                            H_S0pr  = r.myfields[465][d].grdata    # Bulk SHF with spray effect removed [W m-2]
                            H_L0pr  = r.myfields[466][d].grdata    # Bulk LHF with spray effect removed [W m-2]

                            # 1. Background calcs
                            th2t = (p_0/1e5)**0.286    # Factor converting potential temperature to temperature [-]
                            rdryBYr0 = (xs*rho_sw/rho_dry)**(1/3)    # Ratio of rdry to r0 [-]
                            y0 = -nu*Phi_s*Mw/Ms*rho_dry/rho_sw*rdryBYr0**3/(1 - rho_dry/rho_sw*rdryBYr0**3)    # y for surface seawater [-]
                            q_0 = qsat0(t_0,p_0)*(1 + y0)    # Specific humidity at surface (accounting for salt) [kg kg-1]
                            t_1 = th_1*th2t    # Temperature at z_1 [K]
                            th_0 = t_0/th2t    # Surface potential temperature [K]
                            t_mean = 0.5*(t_1+t_0)    # Approx mean air temp [K]
                            tC_mean = t_mean - 273.15    # Approx mean air temp [C]
                            q_mean = 0.5*(q_1+q_0)    # Approx mean air spec hum [kg kg-1]
                            rho_a = p_0/(Rdry*t_mean*(1.+0.61*q_mean))    # Air density [kg m-3]
                            nu_a = 1.326e-5*(1.+6.542e-3*tC_mean+8.301e-6*tC_mean**2-4.84e-9*tC_mean**3)    # Kin visc of air [m2 s-1]
                            gammaWB = 240.97*17.502/(tC_mean+240.97)**2    # gamma = (dqsat/dT)/qsat [K-1], per Buck (1981) correlation
                            G_S = rho_a*cp_a*kappa*ustar    # Dimensional group for SHF [W m-2 K-1]
                            G_L = rho_a*Lv*kappa*ustar    # Dimensional group for LHF [W m-2 K-1]

                            # 2. Interfacial heat fluxes and related parameters
                            L = np.full_like(z_1,-1e12)    # Obukhov stability length [m]
                            z0       = np.full_like(z_1,np.nan)    # Momentum roughness length [m]
                            z0t      = np.full_like(z_1,np.nan)    # Thermal roughness length [m]
                            z0q      = np.full_like(z_1,np.nan)    # Moisture roughness length [m]
                            H_S0pr_C = np.full_like(z_1,np.nan)    # SHF without spray, calculated [W m-2]
                            H_L0pr_C = np.full_like(z_1,np.nan)    # LHF without spray, calculated [W m-2]
                            NC = ~np.isnan(z_1)    # Non-converged gridpoints
                            firstCalc = True
                            count = 0
                            while np.nansum(NC) > 0:
                                count += 1
                                if count > 100:
                                    L[NC] = -1e12    # Where iteration did not converge, revert to no stability
                                    break
                                print('Interfacial HF iteration %d: %d non-converged points' % (count,np.nansum(NC)))
                                H_S0pr_C_prev = np.copy(H_S0pr_C[NC])    # Non-converged values from previous iteration [W m-2]
                                H_L0pr_C_prev = np.copy(H_L0pr_C[NC])    # Non-converged values from previous iteration [W m-2]
                                thvstar = -H_S0pr_C[NC]*kappa/G_S[NC] - 0.61*th_1[NC]*H_L0pr_C[NC]*kappa/G_L[NC]    # Flux scale for thv [K]
                                if firstCalc:
                                    firstCalc = False
                                else:
                                    L[NC] = ustar[NC]**2/(kappa*g/th_1[NC]*thvstar)
                                psiM_1 = stabIntM(z_1[NC]/L[NC])    # Stability integral for momentum at z_1 [-]
                                z0[NC] = z_1[NC]/np.exp(kappa*U_1[NC]/ustar[NC] + psiM_1)
                                Restar = ustar[NC]*z0[NC]/nu_a[NC]    # Roughness Reynolds number [-]
                                z0t[NC] = z0[NC]/np.exp(kappa*(7.3*Restar**0.25*Pr_a**0.5 - 5))    # Per Garratt (1992) Eq. 4.14
                                z0q[NC] = z0[NC]/np.exp(kappa*(7.3*Restar**0.25*Sc_a**0.5 - 5))    # Per Garratt (1992) Eq. 4.15

                                psiH_1 = stabIntH(z_1[NC]/L[NC])
                                H_S0pr_C[NC] = G_S[NC]*(th_0[NC] - th_1[NC])/(np.log(z_1[NC]/z0t[NC]) - psiH_1)
                                H_L0pr_C[NC] = G_L[NC]*( q_0[NC] -  q_1[NC])/(np.log(z_1[NC]/z0q[NC]) - psiH_1)
                                NC[NC] = np.where(np.logical_and(np.abs(H_S0pr_C[NC] - H_S0pr_C_prev) < H_tol,\
                                                                 np.abs(H_L0pr_C[NC] - H_L0pr_C_prev) < H_tol),False,True)
                            L[np.abs(L) == np.inf] = -1e12    # Where L went to +/- infinity, revert to neutral
                            # Run one more time with final values of L
                            psiM_1 = stabIntM(z_1/L)
                            z0 = z_1/np.exp(kappa*U_1/ustar + psiM_1)
                            Restar = ustar*z0/nu_a
                            z0t = z0/np.exp(kappa*(7.3*Restar**0.25*Pr_a**0.5 - 5))
                            z0q = z0/np.exp(kappa*(7.3*Restar**0.25*Sc_a**0.5 - 5))
                            psiH_1 = stabIntH(z_1/L)
                            H_S0pr_C = G_S*(th_0 - th_1)/(np.log(z_1/z0t) - psiH_1)
                            H_L0pr_C = G_L*( q_0 -  q_1)/(np.log(z_1/z0q) - psiH_1)

                            # 3. Calculate 10m neutral conditions and available energies
                            U_10N   = ustar/kappa*np.log(10/z0)    # 10m neutral windspeed [m s-1]
                            t_10Npr = (th_0 - H_S0pr_C/G_S*np.log(10/z0t))*th2t    # 10m neutral sprayless air temperature [K]
                            q_10Npr =   q_0 - H_L0pr_C/G_L*np.log(10/z0q)    # 10m neutral sprayless air spec hum [kg kg-1]
                            s_10Npr = satratio(t_10Npr,p_0,q_10Npr,0.99999)    # 10m neutral saturation ratio without spray [-]
                            betaWB_10Npr = 1/(1 + Lv*gammaWB*(1 + y0)/cp_a*qsat0(t_10Npr,p_0))    # 10m neutral WB coeff without spray [-]
                            wetdep_10Npr = (1 - s_10Npr/(1 + y0))*(1 - betaWB_10Npr)/gammaWB    # 10m neutral WB depression without spray [K]
                            reqBYr0_10Npr = (xs*(1 + nu*Phi_s*Mw/Ms/(1 - s_10Npr)))**(1/3)    # 10m neutral req/r0 without spray [-]
                            a_T = cp_sw*(t_0 - t_10Npr + wetdep_10Npr)    # Available energy for heat transfer due to temp change [J kg-1]
                            a_R = Lv*(1 - reqBYr0_10Npr**3)    # Available energy for heat transfer due to size change [J kg-1]
                            
                            # 4. Calculate heat transfer coefficients
                            whichH0pr = 'calc'    # 'model' to use UWIN-CM output and 'calc' to use calculated values
                            if whichH0pr == 'model':
                                HS0pr = H_S0pr
                                HL0pr = H_L0pr
                            elif whichH0pr == 'calc':
                                HS0pr = H_S0pr_C
                                HL0pr = H_L0pr_C
                            HK0pr = HS0pr + HL0pr
                            HS1   = HS0pr + dHS1spr
                            HL1   = HL0pr + dHL1spr
                            HK1   = HS1 + HL1
                            Ch10N   =   HS1/(rho_a*cp_a*U_10N*(t_0 - t_10Npr))    # 10m neutral SH transfer coefficient [-]
                            Ch10Npr = HS0pr/(rho_a*cp_a*U_10N*(t_0 - t_10Npr))    # 10m neutral SH transfer coefficient without spray [-]
                            Cq10N   =   HL1/(rho_a*Lv*U_10N*(q_0 - q_10Npr))    # 10m neutral LH transfer coefficient [-]
                            Cq10Npr = HL0pr/(rho_a*Lv*U_10N*(q_0 - q_10Npr))    # 10m neutral LH transfer coefficient without spray [-]
                            Ck10N   =   HK1/(rho_a*U_10N*(cp_a*(t_0 - t_10Npr) + Lv*(q_0 - q_10Npr)))    # 10m neutral E transfer coeff [-]
                            Ck10Npr = HK0pr/(rho_a*U_10N*(cp_a*(t_0 - t_10Npr) + Lv*(q_0 - q_10Npr)))    # 10m neutral E transfer coeff without spray [-]
                            # Assign to output
                            if f == 716:
                                r.myfields[f][d].grdata = Ch10N
                            elif f == 717:
                                r.myfields[f][d].grdata = Ch10Npr
                            elif f == 718:
                                r.myfields[f][d].grdata = Cq10N
                            elif f == 719:
                                r.myfields[f][d].grdata = Cq10Npr
                            elif f == 720:
                                r.myfields[f][d].grdata = Ck10N
                            elif f == 721:
                                r.myfields[f][d].grdata = Ck10Npr
                            elif f == 724:
                                r.myfields[f][d].grdata = U_10N
                            elif f == 725:
                                r.myfields[f][d].grdata = t_0 - t_10Npr
                            elif f == 726:
                                r.myfields[f][d].grdata = q_0 - q_10Npr
                            elif f == 727:
                                r.myfields[f][d].grdata = s_10Npr - y0
                            elif f == 730:
                                r.myfields[f][d].grdata = a_T
                            elif f == 731:
                                r.myfields[f][d].grdata = a_R

                        # Field 722: WRF model layer thickness [m]
                        elif f == 722:
                            phi_SZ = r.myfields[450][d].grdata + r.myfields[451][d].grdata    # Z-staggered geopotential [m2 s-2]
                            z_SZ = phi_SZ/9.81    # Z-staggered height [m]
                            dz = z_SZ[:,1:,:,:] - z_SZ[:,:-1,:,:]    # Layer thickness [m]
                            r.myfields[f][d].grdata = dz

                        # Field 723: Equivalent radar reflectivity Ze [-]
                        elif f == 723:
                            dBZ = r.myfields[574][d].grdata    # Logarithmic reflectivity [dBZ]
                            Ze = 10**(dBZ/10)    # Equivalent reflectivity [-]
                            r.myfields[f][d].grdata = Ze

                        # Fields 732 and 733: Mean heat transfer efficiencies [-]
                        elif f == 732:
                            r.myfields[f][d].grdata = r.myfields[83][d].grdata/r.myfields[730][d].grdata/r.myfields[93][d].grdata    # ETbar
                        elif f == 733:
                            r.myfields[f][d].grdata = r.myfields[85][d].grdata/r.myfields[731][d].grdata/r.myfields[93][d].grdata    # ERbar

                        # Fields 737 - 742: Droplet number spectrum [m-2 s-1 um-1]
                        elif f in [737,738,739,740,741,742]:
                            indx = {737:[117,wc.SprayData.r0[0]],
                                    738:[147,wc.SprayData.r0[1]],
                                    739:[177,wc.SprayData.r0[2]],
                                    740:[207,wc.SprayData.r0[3]],
                                    741:[618,wc.SprayData.r0[4]],
                                    742:[666,wc.SprayData.r0[5]]}
                            dmdr0 = r.myfields[indx[f][0]][d].grdata    # Droplet mass spectrum [kg m-2 s-1 um-1]
                            r0 = indx[f][1]    # Droplet radius vector [m]
                            rho_sw = 1030    # Density of seawater [kg m-3]
                            V = 4/3*np.pi*r0**3    # Droplet volume vector [m3]
                            V2D = np.array([np.full_like(dmdr0[0][0,:,:],Vi) for Vi in V])    # Volume vector extended across 2D field
                            for t in range(len(dmdr0)):
                                r.myfields[f][d].grdata.append(dmdr0[t]/V2D/rho_sw)    # Number spectrum

                        # Field 752: Magnitude of current speed at uppermost level of KC 1D mixed layer model [m s-1]
                        elif f == 752:
                            r.myfields[f][d].grdata = (r.myfields[750][d].grdata**2 + r.myfields[751][d].grdata**2)**0.5

                        # Field 766: Ratio of wave energy dissipation flux to SWH [W m-3]
                        elif f == 766:
                            r.myfields[766][d].grdata = r.myfields[547][d].grdata/r.myfields[548][d].grdata

                        # Fields 767-778,781-786: Spray mass flux contributed by select portions of SSGF [kg m-2 s-1]
                        elif f in [767,768,769,770,771,772,773,774,775,776,777,778,781,782,783,784,785,786]:
                            indx = {767:[0,117,50. ,'LE'],
                                    768:[1,147,50. ,'LE'],
                                    769:[2,177,50. ,'LE'],
                                    770:[3,207,50. ,'LE'],
                                    771:[4,618,50. ,'LE'],
                                    772:[5,666,50. ,'LE'],
                                    773:[0,117,100.,'LE'],
                                    774:[1,147,100.,'LE'],
                                    775:[2,177,100.,'LE'],
                                    776:[3,207,100.,'LE'],
                                    777:[4,618,100.,'LE'],
                                    778:[5,666,100.,'LE'],
                                    781:[0,117,100.,'GT'],
                                    782:[1,147,100.,'GT'],
                                    783:[2,177,100.,'GT'],
                                    784:[3,207,100.,'GT'],
                                    785:[4,618,100.,'GT'],
                                    786:[5,666,100.,'GT']}
                            r0 = np.copy(wc.SprayData.r0[indx[f][0]])*1e6    # Droplet radius vector [um]
                            delta_r0 = np.copy(wc.SprayData.delta_r0[indx[f][0]])*1e6    # Droplet bin width vector [um]
                            if indx[f][3] == 'LE':
                                delta_r0[r0 >  indx[f][2]] = 0.    # Remove contributions for droplets > selected max
                            elif indx[f][3] == 'GT':
                                delta_r0[r0 <= indx[f][2]] = 0.    # Remove contributions for droplets <= selected max
                            dmdr0 = r.myfields[indx[f][1]][d].grdata    # Droplet mass spectrum [kg m-2 s-1 um-1]
                            MsprSUBSET = []
                            for t in range(len(dmdr0)):
                                dmdr0_t = dmdr0[t]
                                MsprSUBSET_t = np.full_like(dmdr0_t[0,:,:],np.nan)
                                dims = np.shape(dmdr0_t)
                                for i in range(dims[1]):
                                    for j in range(dims[2]):
                                        MsprSUBSET_t[i,j] = np.dot(dmdr0_t[:,i,j],delta_r0)    # [kg m-2 s-1]
                                MsprSUBSET.append(MsprSUBSET_t)
                            r.myfields[f][d].grdata = np.array(MsprSUBSET)*1000    # [g m-2 s-1]








def calcRMW(r):

    U10 = r.myfields[6][3].grdata    # d03 10m windspeed [m s-1]
    Rstorm = r.myfields[12][3].grdata    # d03 distance to storm center [km]
    numbins = 200
    Rmax = 200.    # Max R for taking average [km]
    binwid = (Rmax-0)/numbins
    bintol = binwid/2
    Rbin = np.linspace(0+bintol,Rmax-bintol,numbins)    # Bin centers
    U10maxAzimAvg = []
    RMW = []
    for t in range(np.shape(U10)[0]):
        Rindx = np.round((Rstorm[t,:,:]-0)/(Rmax-0)*numbins-0.5)
        Rindx[Rindx < 0] = np.nan
        Rindx[Rindx > numbins-1] = np.nan
        U10mean = np.full((numbins,),np.nan)
        for b in range(numbins):
            thisbin = (Rindx == b)
            U10mean[b] = np.nanmean(U10[t,:,:][thisbin])    # Azim mean wind profile [m s-1]
        U10maxAzimAvg.append(np.nanmax(U10mean))
        RMW.append(Rbin[np.nanargmax(U10mean)])
    r.U10maxAzimAvg = np.array(U10maxAzimAvg)
    r.RMW = np.array(RMW)

