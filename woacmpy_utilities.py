import woacmpy_v1.woacmpy_global as wg


# ====================== Utility functions for WOACMPY ============================

def indx(fldname):

    # Get the index of a field name from wg.field_info
    for f in wg.field_info:
        if f[1] == fldname:
            fldindx = f[0]
            break
    return fldindx


