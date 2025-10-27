import numpy as np
from netCDF4 import Dataset
import woacmpy_v1.woacmpy_global as wg


# =================== Functions for writing fields to netcdf files =======================

def write_fields(r,d):
    # Function to write selected fields to netcdf for provided run r and domain d.
    # Only works for fields on domains 1,2,3.  If plotting spray spectra, default number
    # of radius bins (25) is required.  3D fields (i.e., height-varying) not supported.
    print('Writing to NetCDF file... \n')
    for t in r.steplist:
        ncwrt = Dataset(wg.write_dir+'/writeflds_'+r.strmtag+'_'+r.time[t].isoformat().replace('T','_')+'.nc','w',format='NETCDF3_CLASSIC')
        ncwrt.createDimension('Time',size=1)
        dims2D = np.shape(r.myfields[1][d].grdata[t,:,:])
        ncwrt.createDimension('south_north',size=dims2D[0])
        ncwrt.createDimension('west_east',size=dims2D[1])
        ncwrt.createDimension('r0',size=25)
        for f in wg.write_fld_indx:
            if f in [117,147]:    # [Time,r0,south_north,west_east]
                ncwrt.createVariable(wg.field_info[f][1],datatype='f4',dimensions=('Time','r0','south_north','west_east'))[:] = np.array([r.myfields[f][d].grdata[t]])
            else:    # [Time,south_north,west_east]
                ncwrt.createVariable(wg.field_info[f][1],datatype='f4',dimensions=('Time','south_north','west_east'))[:] = np.array([r.myfields[f][d].grdata[t,:,:]])
    
def set_WRTIN_fields():
    # Set which fields in wg.field_info should come from WRTIN files
    for f in wg.write_fld_indx:
        wg.field_info[f][2] = 'WRTIN'    # Set to come from written fields file
        wg.field_info[f][3] = wg.field_info[f][1]    # Set key to match nc file


