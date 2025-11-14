import numpy as np
import woacmpy_v1.woacmpy_global as wg
import coare_spray_v1.ssgf as csp


# ====================== Class definitions for WOACMPY ======================


class Run:    # Output for one model run

    AllRuns = []

    def __init__(self,run_path,w2c_path,w2c_filename,sprwrt_path=None,thermo_path=None):

        self.run_path = run_path    # Path to uwincm run
        self.w2c_path = w2c_path    # Path to vortex tracking results
        self.w2c_filename = w2c_filename    # Filename of vortex tracking results
        self.sprwrt_path = sprwrt_path    # Path to previously written spray calculation output
        self.thermo_path = thermo_path    # Path for thermodynamic fields used for global spray analysis
        Run.AllRuns.append(self)

        self.tag = ''    # Short name used for legend entries, titles, etc.
        self.track = []
        self.lon0 = []    # Storm center, longitude
        self.lat0 = []    # Storm center, latitude
        self.lon0_6hrly = []    # Storm center, longitude, 6-hourly
        self.lat0_6hrly = []    # Storm center, latitude, 6-hourly
        self.strmdir = []    # Storm direction [rad CCW from East]
        self.mslp = []    # Minimum sea level pressure [mb]
        self.strmspeed = []    # Storm translation speed [m s-1]
        self.U10maxAzimAvg = []    # Maximum azimuthally-averaged windspeed (per d03) [m s-1]
        self.RMW = []    # Radius of maximum azimuthally-averaged windspeed (per d03) [m]
        self.startTime = None    # Start time for run
        self.endTime = None    # End time for run
        self.timedel = None    # Timestepping information for run
        self.n_timesteps = None    # Number of timesteps
        self.steplist = None    # List used to count through run's timesteps
        self.time = []    # Simulation dateTime series
        self.time_6hrly = []    # Simulation dateTime series, 6-hourly
        self.thing0 = []    # Unspecified item the run stores for the plotting routine
        self.thing1 = []    # Unspecified item the run stores for the plotting routine
        self.thing2 = []    # Unspecified item the run stores for the plotting routine
        self.thing3 = []    # Unspecified item the run stores for the plotting routine
        self.thing4 = []    # Unspecified item the run stores for the plotting routine
        self.thing5 = []    # Unspecified item the run stores for the plotting routine
        self.thing6 = []    # Unspecified item the run stores for the plotting routine
        self.thing7 = []    # Unspecified item the run stores for the plotting routine
        self.thing8 = []    # Unspecified item the run stores for the plotting routine
        self.thing9 = []    # Unspecified item the run stores for the plotting routine
        self.strmname = None    # Storm name
        self.strmtag = None    # Short storm name tag
        self.myobs = []    # Storm observations object
        
        # Initialize matrix of switches for importing fields
        #        [ ID, ATMd01, ATMd02, ATMd03, WAVd01, WAVd02, WAVd03, OCNd01, OCNd02, OCNd03]
        self.field_impswitches = np.array(\
                [[  n,      0,      0,      0,      0,      0,      0,      0,      0,      0] for n in range(len(wg.field_info))])

        # Initialize matrix of fields
        #        [ ID, ATMd01, ATMd02, ATMd03, WAVd01, WAVd02, WAVd03, OCNd01, OCNd02, OCNd03]
        self.myfields = \
                [[  n,     [],     [],     [],     [],     [],     [],     [],     [],     []] for n in range(len(wg.field_info))]


class Field:    # A field of model data

    AllNativeFields = []

    def __init__(self):

        self.grdata = []    # Gridded model data
        self.filters = []    # Filters to apply to this data
        self.grdata_filt = []    # Arrays of filtered gridded model data


class Fig:    # A figure created by the analysis

    AllFigs = []

    def __init__(self,fig_frames,fig_params):

        Fig.AllFigs.append(self)
        self.myframes = []
        for f in fig_frames:
            new_frame = Frame(f)
            self.myframes.append(new_frame)
        self.figobj = []
        self.gs = []
        self.type = fig_params[0]
        self.size = fig_params[1]
        self.grid = fig_params[2]
        self.title = fig_params[3]
        self.subadj = fig_params[4]
        self.figtag = fig_params[5]
        self.marks = fig_params[6]


class Frame:    # A frame in a figure

    def __init__(self,framedat):

        self.type = framedat[0][0]
        self.typeparams = framedat[0][1]
        self.fldname = framedat[1]
        self.runs = framedat[2]
        self.doms = framedat[3]
        self.colors = framedat[4]
        self.gsindx = framedat[5][0]
        self.scales = framedat[5][1]
        self.scinot = framedat[5][2]
        self.limits = framedat[5][3]
        self.labels = framedat[5][4]
        self.title = framedat[5][5]
        self.fontsize = framedat[6]
        self.filter = framedat[7][0]
        self.filtparams = framedat[7][1]
        self.legloc = framedat[8][0]
        self.legtext = framedat[8][1]
        self.legncol = framedat[8][2]

        self.fldindx = []
        for i in self.fldname:
            for f in wg.field_info:
                if f[1] == i:
                    self.fldindx.append(f[0])
        self.axobj = []
        self.filtindx = []
        if self.type == 'TimeSeries':
            if self.labels[0] is None:
                self.labels[0] = 'Time'
            if self.labels[1] is None:
                self.labels[1] = wg.field_info[self.fldindx[0]][5]
        elif self.type == 'SpecProf':
            if self.typeparams[3] == 'Spec':
                if self.labels[0] is None:
                    self.labels[0] = 'Droplet Radius at Formation $r_0$ [$\mu m$]'
                if self.labels[1] is None:
                    self.labels[1] = wg.field_info[self.fldindx[1]][5]
            elif self.typeparams[3] == 'Prof':
                if self.labels[0] is None:
                    self.labels[0] = wg.field_info[self.fldindx[1]][5]
                self.labels[1] = 'Height [$m$]'
        elif self.type == 'Map':
            pass
        elif self.type in ['HycVert','WRFOcnVert']:
            if self.labels[0] is None:
                if self.typeparams[0] == 'SameLat':
                    self.labels[0] = 'Longitude [$\degree E$]'
                elif self.typeparams[0] == 'SameLon':
                    self.labels[0] = 'Latitude [$\degree N$]'
            if self.labels[1] is None:
                self.labels[1] = 'Depth [$m$]'
        elif self.type == 'BulkTCProps':
            if self.labels[0] is None:
                self.labels[0] = '***User must specify***'
            if self.labels[1] is None:
                self.labels[1] = '***User must specify***'
        elif self.type == 'WrfVert':
            if self.labels[0] is None:
                self.labels[0] = 'Distance to Storm Center [$km$]'
            if self.labels[1] is None:
                self.labels[1] = 'Height [$km$]'
        else:
            if self.labels[0] is None:
                self.labels[0] = wg.field_info[self.fldindx[0]][5]
            if self.labels[1] is None:
                self.labels[1] = wg.field_info[self.fldindx[1]][5]


class SprayData:    # Stores offline spray calculation parameters

    # Key defining which version of spray libraries to use:
    #     'spr4_uwincm' - use spraylibs_v4 libraries and UWIN-CM model fields
    #     'spr4_umwmglb' - use spraylibs_v4 libraries and global UMWM model fields
    #     'csp1_uwincm' - use coare_spray_v1 libraries and UWIN-CM model fields
    # This must be defined by user every time.
    whichspray = []
    # Original parameter list for spraylibs_v4
    sourcestrength = []    # This is the variable fs, user must provide every time.
    r0 = []    # Droplet radius vector, default input is hard-coded 25-value default vector from spray code.
    delta_r0 = []    # Droplet r0 width vector, default input is hard-coded 25-value default vector from spray code.
    SSGFname = []    # Which SSGF to use, user must provide every time.
    feedback = []    # Include subgrid feedback?  Default input of True includes feedback -- recommended.
    profiles = []    # Calculate vertical thermo profiles?  These are optional diagnostics.  Default input of False turns this off.
    zRvaries = []    # Use height-varying zR?  Using this is very slow and doesn't make much difference.  Default input of False turns this off -- recommended.
    stability = []    # Include stability effects?  This should usually be included (True).  Not used for 'csp1_uwincm'.  User must specify.
    sprayLB = []    # Lower bound to calculate spray heat fluxes.  10 m s-1 is current best practice.  User must provide every time.
    fdbkfsolve = []    # Method for solving for feedback.  Default input of 'iterIG' uses initial guess with one iteration -- recommended.
    fdbkcrzyOPT = []    # Option for handling poorly behaved feedback points.  This is not really used anymore and default input of 0 should be used.
    showfdbkcrzy = []    # Set to True to show points where feedback is having problems.  Not really used anymore and default input of False should be used.
    scaleSSGF = []    # If True, then scale SSGF using chi1 and chi2.  Seldom used.  Default value of False keeps this turned off.
    chi1 = []    # Factor scaling small droplet end of SSGF.  Seldom used.  Default input is None.
    chi2 = []    # Factor scaling large droplet end of SSGF.  Seldom used.  Default input is None.
    which_z0tq = []    # Which model to use for interfacial roughness lengths.  User must provide every time.
    # Additional parameters for coare_spray_v1
    param_delspr_Wi = []    # True to parameterize delspr from winds, False to use input swh.  Default input is False.
    which_stress = []    # Which option to use for interfacial stress.  User must provide every time.
    use_gf = []    # Use COARE gust factor physics?  Should be True when using COARE and probably False when providing input ustar.  User must specify.
    z_ref = []    # Height to calculate spray surface layer thermo changes.  Default value of -1 uses mid-spray-layer height.
    
    def __init__(self,\
            whichspray      = [],\
            sourcestrength  = [],\
            r0              = [csp.r0_default      ,csp.r0_default      ,csp.r0_default      ,csp.r0_default      ,csp.r0_default      ,csp.r0_default      ],\
            delta_r0        = [csp.delta_r0_default,csp.delta_r0_default,csp.delta_r0_default,csp.delta_r0_default,csp.delta_r0_default,csp.delta_r0_default],\
            SSGFname        = [],\
            feedback        = [True ,True ,True ,True ,True ,True],\
            profiles        = [False,False,False,False,False,False],\
            zRvaries        = [False,False,False,False,False,False],\
            stability       = [],\
            sprayLB         = [],\
            fdbkfsolve      = ['iterIG','iterIG','iterIG','iterIG','iterIG','iterIG'],\
            fdbkcrzyOPT     = [0,0,0,0,0,0],\
            showfdbkcrzy    = [False,False,False,False,False,False],\
            scaleSSGF       = [False,False,False,False,False,False],\
            chi1            = [None ,None ,None ,None ,None ,None],\
            chi2            = [None ,None ,None ,None ,None ,None],\
            which_z0tq      = [],\
            param_delspr_Wi = [False,False,False,False,False,False],\
            which_stress    = [],\
            use_gf          = [],\
            z_ref           = [-1,-1,-1,-1,-1,-1]):

        SprayData.whichspray = whichspray
        SprayData.sourcestrength = sourcestrength
        SprayData.r0 = r0
        SprayData.delta_r0 = delta_r0
        SprayData.SSGFname = SSGFname
        SprayData.feedback = feedback
        SprayData.profiles = profiles
        SprayData.zRvaries = zRvaries
        SprayData.stability = stability
        SprayData.sprayLB = sprayLB
        SprayData.fdbkfsolve = fdbkfsolve
        SprayData.fdbkcrzyOPT = fdbkcrzyOPT
        SprayData.showfdbkcrzy = showfdbkcrzy
        SprayData.scaleSSGF = scaleSSGF
        SprayData.chi1 = chi1
        SprayData.chi2 = chi2
        SprayData.which_z0tq = which_z0tq
        SprayData.param_delspr_Wi = param_delspr_Wi
        SprayData.which_stress = which_stress
        SprayData.use_gf = use_gf
        SprayData.z_ref = z_ref


