import woacmpy_v1.woacmpy_classes as wc
import woacmpy_v1.woacmpy_observations as wo
import woacmpy_v1.woacmpy_global as wg
import woacmpy_v1.woacmpy_track as wt


# ====================== File paths and definitions for accessing model data for WOACMPY ===========


def use_run(name,startTime,endTime,timedel):

    if name[:6] == 'Harvey':
        if name == 'Harvey_230116_full_nospray':
            run_path = '/home/orca3/bwbarr/runs/harvey/run_230116_full_nospray' 
            w2c_path = '/home/orca3/bwbarr/runs/harvey/run_230116_full_nospray/w2c' 
            w2c_filename = 'harvey_1.log.csv' 
            run = wc.Run(run_path,w2c_path,w2c_filename)
        elif name == 'Harvey_230116_full_wispray':
            run_path = '/home/orca3/bwbarr/runs/harvey/run_230116_full_wispray' 
            w2c_path = '/home/orca3/bwbarr/runs/harvey/run_230116_full_wispray/w2c' 
            w2c_filename = 'harvey_1.log.csv' 
            run = wc.Run(run_path,w2c_path,w2c_filename)
        run.strmname = 'Harvey'
        run.strmtag = 'har'

    elif name[:7] == 'Michael':
        if name == 'Michael_230116_full_nospray_CTL':
            run_path = '/home/orca3/bwbarr/runs/michael/run_230116_full_nospray_CTL' 
            w2c_path = '/home/orca3/bwbarr/runs/michael/run_230116_full_nospray_CTL/w2c'
            w2c_filename = 'michael_1.log.csv' 
            run = wc.Run(run_path,w2c_path,w2c_filename)
        elif name == 'Michael_230116_full_wispray_CTL':
            run_path = '/home/orca3/bwbarr/runs/michael/run_230116_full_wispray_CTL' 
            w2c_path = '/home/orca3/bwbarr/runs/michael/run_230116_full_wispray_CTL/w2c'
            w2c_filename = 'michael_1.log.csv' 
            run = wc.Run(run_path,w2c_path,w2c_filename)
        run.strmname = 'Michael'
        run.strmtag = 'mic'

    elif name[:8] == 'Florence':
        if name == 'Florence_230116_full_nospray':
            run_path = '/home/orca3/bwbarr/runs/florence/run_230116_full_nospray'
            w2c_path = '/home/orca3/bwbarr/runs/florence/run_230116_full_nospray/w2c'
            w2c_filename = 'florence_1.log.csv'
            run = wc.Run(run_path,w2c_path,w2c_filename)
        elif name == 'Florence_230116_full_wispray':
            run_path = '/home/orca3/bwbarr/runs/florence/run_230116_full_wispray'
            w2c_path = '/home/orca3/bwbarr/runs/florence/run_230116_full_wispray/w2c'
            w2c_filename = 'florence_1.log.csv'
            run = wc.Run(run_path,w2c_path,w2c_filename)
        elif name == 'Florence_230116_full_wispray_MsprX2':
            run_path = '/home/orca3/bwbarr/runs/florence/run_230116_full_wispray_MsprX2'
            w2c_path = '/home/orca3/bwbarr/runs/florence/run_230116_full_wispray_MsprX2/w2c'
            w2c_filename = 'florence_1.log.csv'
            run = wc.Run(run_path,w2c_path,w2c_filename)
        elif name == 'Florence_230116_full_wispray_MsprX4':
            run_path = '/home/orca3/bwbarr/runs/florence/run_230116_full_wispray_MsprX4'
            w2c_path = '/home/orca3/bwbarr/runs/florence/run_230116_full_wispray_MsprX4/w2c'
            w2c_filename = 'florence_1.log.csv'
            run = wc.Run(run_path,w2c_path,w2c_filename)
        elif name == 'Florence_230116_full_wispray_Scale05v10':
            run_path = '/home/orca3/bwbarr/runs/florence/run_230116_full_wispray_Scale05v10'
            w2c_path = '/home/orca3/bwbarr/runs/florence/run_230116_full_wispray_Scale05v10/w2c'
            w2c_filename = 'florence_1.log.csv'
            run = wc.Run(run_path,w2c_path,w2c_filename)
        elif name == 'Florence_230116_full_wispray_Scale10v05':
            run_path = '/home/orca3/bwbarr/runs/florence/run_230116_full_wispray_Scale10v05'
            w2c_path = '/home/orca3/bwbarr/runs/florence/run_230116_full_wispray_Scale10v05/w2c'
            w2c_filename = 'florence_1.log.csv'
            run = wc.Run(run_path,w2c_path,w2c_filename)
        elif name == 'Florence_230116_full_wispray_Scale01v10_MsprX2':
            run_path = '/home/orca3/bwbarr/runs/florence/run_230116_full_wispray_Scale01v10_MsprX2'
            w2c_path = '/home/orca3/bwbarr/runs/florence/run_230116_full_wispray_Scale01v10_MsprX2/w2c'
            w2c_filename = 'florence_1.log.csv'
            run = wc.Run(run_path,w2c_path,w2c_filename)
        elif name == 'Florence_230116_full_wispray_Scale01v10':
            run_path = '/home/orca3/bwbarr/runs/florence/run_230116_full_wispray_Scale01v10'
            w2c_path = '/home/orca3/bwbarr/runs/florence/run_230116_full_wispray_Scale01v10/w2c'
            w2c_filename = 'florence_1.log.csv'
            run = wc.Run(run_path,w2c_path,w2c_filename)
        elif name == 'Florence_230116_full_wispray_Scale10v01':
            run_path = '/home/orca3/bwbarr/runs/florence/run_230116_full_wispray_Scale10v01'
            w2c_path = '/home/orca3/bwbarr/runs/florence/run_230116_full_wispray_Scale10v01/w2c'
            w2c_filename = 'florence_1.log.csv'
            run = wc.Run(run_path,w2c_path,w2c_filename)
        elif name == 'Florence_230116_full_wispray_Scale10v01_MsprX05':
            run_path = '/home/orca3/bwbarr/runs/florence/run_230116_full_wispray_Scale10v01_MsprX05'
            w2c_path = '/home/orca3/bwbarr/runs/florence/run_230116_full_wispray_Scale10v01_MsprX05/w2c'
            w2c_filename = 'florence_1.log.csv'
            run = wc.Run(run_path,w2c_path,w2c_filename)
        elif name == 'Florence_230116_full_wispray_ENS222':
            run_path = '/home/orca3/bwbarr/runs/florence/run_230116_full_wispray_ENS222'
            w2c_path = '/home/orca3/bwbarr/runs/florence/run_230116_full_wispray_ENS222/w2c'
            w2c_filename = 'florence_1.log.csv'
            run = wc.Run(run_path,w2c_path,w2c_filename)
        elif name == 'Florence_230116_full_wispray_ENS221':
            run_path = '/home/orca3/bwbarr/runs/florence/run_230116_full_wispray_ENS221'
            w2c_path = '/home/orca3/bwbarr/runs/florence/run_230116_full_wispray_ENS221/w2c'
            w2c_filename = 'florence_1.log.csv'
            run = wc.Run(run_path,w2c_path,w2c_filename)
        elif name == 'Florence_230116_full_wispray_ENS220':
            run_path = '/home/orca3/bwbarr/runs/florence/run_230116_full_wispray_ENS220'
            w2c_path = '/home/orca3/bwbarr/runs/florence/run_230116_full_wispray_ENS220/w2c'
            w2c_filename = 'florence_1.log.csv'
            run = wc.Run(run_path,w2c_path,w2c_filename)
        elif name == 'Florence_230116_full_wispray_ENS219':
            run_path = '/home/orca3/bwbarr/runs/florence/run_230116_full_wispray_ENS219'
            w2c_path = '/home/orca3/bwbarr/runs/florence/run_230116_full_wispray_ENS219/w2c'
            w2c_filename = 'florence_1.log.csv'
            run = wc.Run(run_path,w2c_path,w2c_filename)
        elif name == 'Florence_230116_full_wispray_ENS218':
            run_path = '/home/orca3/bwbarr/runs/florence/run_230116_full_wispray_ENS218'
            w2c_path = '/home/orca3/bwbarr/runs/florence/run_230116_full_wispray_ENS218/w2c'
            w2c_filename = 'florence_1.log.csv'
            run = wc.Run(run_path,w2c_path,w2c_filename)
        run.strmname = 'Florence'
        run.strmtag = 'flo'

    elif name[:6] == 'Fanapi':
        pass

    elif name[:6] == 'Dorian':
        if name == 'Dorian_211118_AWO1.3_wider_d02_relo2_WRFreinit19083012_retry19090200':
            run_path = '/home/orca3/bwbarr/runs/dorian/run_211118_AWO1.3_wider_d02_relo2_WRFreinit19083012_retry19090200'
            w2c_path = '/home/orca3/bwbarr/runs/dorian/run_211118_AWO1.3_wider_d02_relo2_WRFreinit19083012_retry19090200/w2c_new'
            w2c_filename = 'dorian_1.log.csv'
            run = wc.Run(run_path,w2c_path,w2c_filename)
        elif name == 'Dorian_230116_WRFreinit19083012_full_nospray':
            run_path = '/home/orca3/bwbarr/runs/dorian/run_230116_WRFreinit19083012_full_nospray'
            w2c_path = '/home/orca3/bwbarr/runs/dorian/run_230116_WRFreinit19083012_full_nospray/w2c'
            w2c_filename = 'dorian_1.log.csv'
            run = wc.Run(run_path,w2c_path,w2c_filename)
        elif name == 'Dorian_230116_WRFreinit19083012_full_wispray':
            run_path = '/home/orca3/bwbarr/runs/dorian/run_230116_WRFreinit19083012_full_wispray'
            w2c_path = '/home/orca3/bwbarr/runs/dorian/run_230116_WRFreinit19083012_full_wispray/w2c'
            w2c_filename = 'dorian_1.log.csv'
            run = wc.Run(run_path,w2c_path,w2c_filename)
        run.strmname = 'Dorian'
        run.strmtag = 'dor'

    elif name[:5] == 'Irene':
        if name == 'Irene_230415_copyDKS_full_nospray':
            run_path = '/home/orca3/bwbarr/runs/irene/run_230415_copyDKS_full_nospray'
            w2c_path = '/home/orca3/bwbarr/runs/irene/run_230415_copyDKS_full_nospray/w2c'
            w2c_filename = 'irene_1.log.csv'
            run = wc.Run(run_path,w2c_path,w2c_filename)
        elif name == 'Irene_230415_copyDKS_full_wispray':
            run_path = '/home/orca3/bwbarr/runs/irene/run_230415_copyDKS_full_wispray'
            w2c_path = '/home/orca3/bwbarr/runs/irene/run_230415_copyDKS_full_wispray/w2c'
            w2c_filename = 'irene_1.log.csv'
            run = wc.Run(run_path,w2c_path,w2c_filename)
        run.strmname = 'Irene'
        run.strmtag = 'ire'

    elif name[:7] == 'UMWMGLB':
        if name == 'UMWMGLB_210624_ccmp_global':
            run_path = '/home/orca3/bwbarr/runs/umwm3_solo/210624_ccmp_global'
            w2c_path = None
            w2c_filename = None
            thermo_path = '/home/orca/bwbarr/analysis/210625_UMWM_global_spray/220803_process_ERA5/netCDFfiles'
            run = wc.Run(run_path,w2c_path,w2c_filename,thermo_path=thermo_path)
        elif name == 'UMWMGLB_210624_ccmp_global_SSBased':
            run_path = '/home/orca3/bwbarr/runs/umwm3_solo/210624_ccmp_global'
            w2c_path = None
            w2c_filename = None
            thermo_path = '/home/orca/bwbarr/analysis/210625_UMWM_global_spray/220803_process_ERA5/netCDFfiles'
            sprwrt_path = '/home/orca/bwbarr/analysis/uwincmpy_saved_ncdf/220909_globalsprayHFs/SSBased'
            run = wc.Run(run_path,w2c_path,w2c_filename,thermo_path=thermo_path,sprwrt_path=sprwrt_path)
        elif name == 'UMWMGLB_210624_ccmp_global_WiBased':
            run_path = '/home/orca3/bwbarr/runs/umwm3_solo/210624_ccmp_global'
            w2c_path = None
            w2c_filename = None
            thermo_path = '/home/orca/bwbarr/analysis/210625_UMWM_global_spray/220803_process_ERA5/netCDFfiles'
            sprwrt_path = '/home/orca/bwbarr/analysis/uwincmpy_saved_ncdf/220909_globalsprayHFs/WiBased'
            run = wc.Run(run_path,w2c_path,w2c_filename,thermo_path=thermo_path,sprwrt_path=sprwrt_path)
        run.strmname = 'Global'
        run.strmtag = 'glb'

    elif name[:8] == 'Nor18Jan':
        if name == 'Nor18Jan_230824_KComl_noML':
            run_path = '/srv/seolab/bbarr/runs/wrf/run_230824_KComl_noML'
            w2c_path = None
            w2c_filename = None
            run = wc.Run(run_path,w2c_path,w2c_filename)
        elif name == 'Nor18Jan_230824_KComl_KC':
            run_path = '/srv/seolab/bbarr/runs/wrf/run_230824_KComl_KC'
            w2c_path = None
            w2c_filename = None
            run = wc.Run(run_path,w2c_path,w2c_filename)
        elif name == 'Nor18Jan_231010_KComl_KC':
            run_path = '/srv/seolab/bbarr/runs/wrf/run_231010_KComl_KC'
            w2c_path = None
            w2c_filename = None
            run = wc.Run(run_path,w2c_path,w2c_filename)
        run.strmname = 'Nor18Jan'
        run.strmtag = 'n18jan'

    # Define time-stepping information and track
    run.startTime = startTime
    run.endTime = endTime
    run.timedel = timedel
    if timedel[0] == 'hours':
        run.n_timesteps = int((endTime - startTime).total_seconds()/3600/timedel[1] + 1)
    elif timedel[0] == 'minutes':
        run.n_timesteps = int((endTime - startTime).total_seconds()/60/timedel[1] + 1)
    run.steplist = range(run.n_timesteps)
    if wg.useSRinfo:
        run.track = wt.atcf_csv(run.w2c_path+'/w2c/'+run.w2c_filename)

    # Load observations
    if wg.global_UMWM == False:
        if run.strmname not in wo.StormObs.StormNamesUsed: 
            newobs = wo.StormObs(run.strmname,run.startTime,run.endTime)
        run.myobs = wo.StormObs.ObsObjectsUsed[wo.StormObs.StormNamesUsed.index(run.strmname)]

    return run


