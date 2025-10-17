import numpy as np
import matplotlib.pylab as plt
import scipy, scipy.spatial
#from .poisson_disk_sampling import generate_poisson_disk_sampling_points_2d
from scipy.ndimage.filters import gaussian_filter1d

def calc_dist(X, Y, xi, yi):

    #return np.sqrt((X - xi)**2 + (Y - yi)**2)
    return (X - xi)**2 + (Y - yi)**2  # This is faster since I only care about nearest point.


def bk_interp_uv(X, Y, u, v, xi, yi):
    
    """
    Weighted interp of data on grid (X,Y) at point (xi, yi).
    """

    import numpy as np

    X = np.array(X)
    Y = np.array(Y)
    u = np.array(u)
    v = np.array(v)
    
    
    dist = calc_dist(X,Y,xi,yi)
    indx2 = np.unravel_index(np.argmin(dist, axis=None), dist.shape)
    
    #
    #        10           11
    #          x
    #     00          01
    #

    
    indx00j = indx2[0]
    indx00i = indx2[1]
    
    if (X[indx00j, indx00i] > xi and indx00i > 0):
        indx00i -= 1

    if (Y[indx00j, indx00i] > yi and indx00j > 0):
        indx00j -= 1

    if (indx00j >= X.shape[0]-1 or indx00i >= X.shape[1]-1):
        return (u[indx00j, indx00i], v[indx00j, indx00i])

    else:

        X00, X01, X10, X11 = X[indx00j:indx00j+2, indx00i:indx00i+2].flatten()
        Y00, Y01, Y10, Y11 = Y[indx00j:indx00j+2, indx00i:indx00i+2].flatten()
        u00, u01, u10, u11 = u[indx00j:indx00j+2, indx00i:indx00i+2].flatten()
        v00, v01, v10, v11 = v[indx00j:indx00j+2, indx00i:indx00i+2].flatten()
        
        ## HACK add 0.001 to avoid divide by zero.
        dist00 = np.sqrt((xi - X00)**2 + (yi - Y00)**2) + 0.001
        dist10 = np.sqrt((xi - X10)**2 + (yi - Y10)**2) + 0.001
        dist01 = np.sqrt((xi - X01)**2 + (yi - Y01)**2) + 0.001
        dist11 = np.sqrt((xi - X11)**2 + (yi - Y11)**2) + 0.001


    
        sum_inverse_dist = (1.0/dist00 + 1.0/dist10 + 1.0/dist01 + 1.0/dist11)

        sum_weighted_values_u = (u00 / (dist00)
                               + (u10 / (dist10))
                               + (u01 / (dist01))
                               + (u11 / (dist11)))

        sum_weighted_values_v = (v00 / (dist00)
                               + (v10 / (dist10))
                               + (v01 / (dist01))
                               + (v11 / (dist11)))

        weighted_value_u = sum_weighted_values_u / sum_inverse_dist
        weighted_value_v = sum_weighted_values_v / sum_inverse_dist

        return (weighted_value_u, weighted_value_v)



def do_euler_integration(this_x, this_y, this_u, this_v, DT):

    if abs(this_u) < 0.001 and abs(this_v) < 0.001:
        return (0, 0)

    else:
        this_dx = DT * this_u / np.sqrt(this_u**2 + this_v**2)
        this_dy = DT * this_v / np.sqrt(this_u**2 + this_v**2)
        
        return (this_dx, this_dy)


def do_rk4_integration(this_x, this_y, this_u, this_v, DT, interp_function):

    if abs(this_u) < 0.001 and abs(this_v) < 0.001:
        return (0, 0)

    k1_u = DT * this_u / np.sqrt(this_u**2 + this_v**2)
    k1_v = DT * this_v / np.sqrt(this_u**2 + this_v**2)

    this_u2, this_v2 = interp_function(this_x + 0.5*k1_u, this_y + 0.5*k1_v)
    if abs(this_u2) < 0.001 and abs(this_v2) < 0.001:
        return (0, 0)
            
    k2_u = DT * this_u2 / np.sqrt(this_u2**2 + this_v2**2) 
    k2_v = DT * this_v2 / np.sqrt(this_u2**2 + this_v2**2)

    this_u3, this_v3 = interp_function(this_x + 0.5*k2_u, this_y + 0.5*k2_v)
    if abs(this_u3) < 0.001 and abs(this_v3) < 0.001:
        return (0, 0)

    k3_u = DT * this_u3 / np.sqrt(this_u3**2 + this_v3**2) 
    k3_v = DT * this_v3 / np.sqrt(this_u3**2 + this_v3**2)

    this_u4, this_v4 = interp_function(this_x + k3_u, this_y + k3_v)
    if abs(this_u4) < 0.001 and abs(this_v4) < 0.001:
        return (0, 0)

    k4_u = DT * this_u4 / np.sqrt(this_u4**2 + this_v4**2) 
    k4_v = DT * this_v4 / np.sqrt(this_u4**2 + this_v4**2)
            
    this_dx = (k1_u + 2.0*k2_u + 2.0*k3_u + k4_u) / 6.0
    this_dy = (k1_v + 2.0*k2_v + 2.0*k3_v + k4_v) / 6.0

    return (this_dx, this_dy)


def get_nearest_neighbor_index(x, y, X, Y):

    # Nearest neighbor using an arbitrarily shapped quadrilateral
    #
    #         10           11
    #                  X
    #
    #                   01
    #     00

    X = np.array(X)
    Y = np.array(Y)    
    dist = calc_dist(X,Y,x,y)
    indx2 = np.unravel_index(np.argmin(dist, axis=None), dist.shape)
    return (indx2[1], indx2[0]) # other functions are expecting (i,j) here.


def calculate_a_curly_vector(x_seed, y_seed, X, Y, u, v, scale_factor = 5.0, time_step=0.1, verbose=False
                             , method='rk4', direction = 'forward', interp='linear'
                             , min_length = 0.0, max_length = 1000.0):

    DT = time_step #* scale_factor

    # Check sizes.
    shape_X = X.shape
    ny = shape_X[0]
    nx = shape_X[1]

    
    # Start the curly vector at the data point.
    xx = [x_seed]
    yy = [y_seed]
    current_curly_vector_length = 0;


    # Get subset of points where the curly vector could possibly lie.
    # It is a pretty gross estimate. 110% of the curly vector length in each direction.
    # Note: interpolation needs at least two points in each direction.

    ii, jj = get_nearest_neighbor_index(x_seed, y_seed, X, Y)
    
    this_curly_vector_length = np.sqrt(u[jj,ii]**2 + v[jj,ii]**2) / scale_factor

    if (this_curly_vector_length < min_length):
        this_curly_vector_length = min_length
    if (this_curly_vector_length > max_length):
        this_curly_vector_length = max_length
    
    ii_subset_indx = np.logical_and(X[jj,:] > X[jj,ii] - 1.0*this_curly_vector_length
                                    , X[jj,:] < X[jj,ii] + 1.0*this_curly_vector_length)
    jj_subset_indx = np.logical_and(Y[:,ii] > Y[jj,ii] - 1.0*this_curly_vector_length
                                    , Y[:,ii] < Y[jj,ii] + 1.0*this_curly_vector_length)
    
    
    if (np.sum(ii_subset_indx) <= 1 or np.sum(jj_subset_indx) <= 1):
        ii_subset_indx = range(max(0, ii-1), min(ii+1, nx))
        jj_subset_indx = range(max(0, jj-1), min(jj+1, ny))

        if verbose:
            print("WARNING: Curly vector at ({0}, {1}) shorter than one grid cell.".format(jj, ii))
                
    X_subset = X[jj_subset_indx,:][:,ii_subset_indx]
    Y_subset = Y[jj_subset_indx,:][:,ii_subset_indx]
    u_subset = np.nan_to_num(u[jj_subset_indx,:][:,ii_subset_indx])
    v_subset = np.nan_to_num(v[jj_subset_indx,:][:,ii_subset_indx])

    
    # Define interp function using this subset of points.
    if interp == 'linear':
        def interp_2d_uv(xi, yi):
            return bk_interp_uv(X_subset, Y_subset, u_subset, v_subset, xi, yi)

    else:
        # Default to nearest neighbor.
        def interp_2d_uv(xi, yi):
            i_nn, j_nn = get_nearest_neighbor_index(xi, yi, X_subset, Y_subset)
            return (u_subset[j_nn, i_nn], v_subset[j_nn, i_nn])

        
    count = 0
    while (current_curly_vector_length < this_curly_vector_length):

        count += 1
        if count > 1000:
            if verbose:
                print("WARNING: Exceeded 1000 iterations at ({0}, {1}).".format(jj,ii), flush=True)
            break
                

        if (direction == 'forward' or direction == 'both'):

            this_x = xx[-1]
            this_y = yy[-1]
            this_u, this_v = interp_2d_uv(this_x, this_y)

            if (method.lower() == 'euler'):
                this_dx, this_dy = do_euler_integration(this_x, this_y, this_u, this_v, DT)
            else:
                # Defaults to 4th Order Runge Kutta
                this_dx, this_dy = do_rk4_integration(this_x, this_y, this_u, this_v, DT, interp_2d_uv)

            xx = np.append(xx, xx[-1] + this_dx)
            yy = np.append(yy, yy[-1] + this_dy)
            current_curly_vector_length += np.sqrt(this_dx**2 +  this_dy**2)

        if (direction == 'backward' or direction == 'both'):

            this_x = xx[0]
            this_y = yy[0]
            this_u, this_v = interp_2d_uv(this_x, this_y)

            if (method.lower() == 'euler'):
                this_dx, this_dy = do_euler_integration(this_x, this_y, this_u, this_v, DT)
            else:
                # Defaults to 4th Order Runge Kutta
                this_dx, this_dy = do_rk4_integration(this_x, this_y, this_u, this_v, DT, interp_2d_uv)

            xx = np.insert(xx, 0, xx[0] - this_dx)
            yy = np.insert(yy, 0, yy[0] - this_dy)
            current_curly_vector_length += np.sqrt(this_dx**2 +  this_dy**2)

                
    return (xx,yy)



def draw_arrow_head(xx, yy, linewidth=1, color='k'
                    , arrowhead_angle = 30.0, arrowhead_length_factor = 0.3
                    , arrowhead_min_length = 0.0, arrowhead_max_length = 1000.0
                    , closed = False,zorder=1):

    LW = linewidth
    COL = color

    ## Get length
    this_curly_vector_length = 0.0
    for ii in range(len(xx)-1):

        this_dx = xx[ii+1] - xx[ii]
        this_dy = yy[ii+1] - yy[ii]
        this_curly_vector_length += np.sqrt(this_dx**2 +  this_dy**2)
    
    arrowhead_length = this_curly_vector_length * arrowhead_length_factor
    if arrowhead_length < arrowhead_min_length:
        arrowhead_length = arrowhead_min_length
    if arrowhead_length > arrowhead_max_length:
        arrowhead_length = arrowhead_max_length
            
    n_arrow_points = yy.shape[0]
    how_far_back = min(3, len(yy))
    theta = np.arctan2(yy[n_arrow_points - how_far_back] - yy[n_arrow_points-1],
                       xx[n_arrow_points - how_far_back] - xx[n_arrow_points-1])

    psi_right = theta + arrowhead_angle * 3.14159/180.0
    psi_left  = theta - arrowhead_angle * 3.14159/180.0
    
    x_right = xx[n_arrow_points-1] + arrowhead_length * np.cos(psi_right)
    y_right = yy[n_arrow_points-1] + arrowhead_length * np.sin(psi_right)
    
    x_left = xx[n_arrow_points-1] + arrowhead_length * np.cos(psi_left)
    y_left = yy[n_arrow_points-1] + arrowhead_length * np.sin(psi_left)
    

    if closed:

        x_fill = [xx[n_arrow_points-1], x_right, x_left]
        y_fill = [yy[n_arrow_points-1], y_right, y_left]
        plt.fill(x_fill, y_fill, color=COL,zorder=zorder)

    else:
        plt.plot([xx[n_arrow_points-1], x_right],
                 [yy[n_arrow_points-1], y_right], color=COL, linewidth=LW,zorder=zorder)
        plt.plot([xx[n_arrow_points-1], x_left],
                 [yy[n_arrow_points-1], y_left], color=COL, linewidth=LW,zorder=zorder)

    

######################################################################################
######################################################################################
######################################################################################
######################################################################################
######################################################################################


def curly_vectors(X, Y, u, v, scale_factor = 5.0, time_step=0.1 , skip=1, linewidth=1, color='k'
                  , arrowhead_angle = 30.0, arrowhead_length_factor = 0.3
                  , arrowhead_min_length = 0.0, arrowhead_max_length = 1000.0, verbose=False
                  , method='rk4', direction = 'forward'
                  , exclude_points_x = [-1], exclude_points_y = [-1]
                  , seed_points = [], interp='linear'
                  , smoothing = False, min_length = 0.0, max_length = 1000.0
                  , arrowhead_close = False
                  , min_speed = 0.0, max_speed = 1000.0,zorder=1):

    """
    curly_vectors(X, Y, u, v, scale_factor = 5.0, time_step=0.1 , skip=1, linewidth=1, color='k'
                  , arrowhead_angle = 30.0, arrowhead_length_factor = 0.3
                  , arrowhead_min_length = 0.0, arrowhead_max_length = 1000.0, verbose=False)

    - X and Y should be "meshgrid" like 2-d coordinates. Monotonic irregular rectangular grid 
      like WRF grid also works.

    - scale_factor is vector magnitude corresponding to data coordinate distance of 1. Curly vector
      length is the speed divided by the scale_factor.
    
    - time_step defaults to 0.1. It is not a physical time.
      Smaller values will be more precise but take longer.
    
    - By default, it will use all grid points as curly vector seeds, which may be way too dense!
      Use the "skip" option to N for every Nth point. (Note: the path integration still uses
      the full resolution data).
    
    - linewidth and color are the same as in matplotlib plot. In fact, the curves are drawn
      using matplotlib plot. 

    - Arrow heads: They are drawn at an angle of arrowhead_angle from the end of each curly vector.
      They are drawn with a length of arrowhead_length_factor times the curly vector length.
      The arrowhead length can be overridden with arrowhead_min_length and/or arrowhead_max_length.
      These lengths are in data coordinates.

    - verbose (default false): display row count as curly vectors get done. To make sure it is
      running if it seems to be taking forever.

    - method = [rk4] | euler
      4th Order Runge Kutta  |  Euler integration (faster but may be less accurate)
      (Not sensitive to caps. method = nonsense will default to rk4.)

    - interp = [linear] | nn (or anything else)
      if linear, use linear interp. Otherwise, nearest neighbor. Faster for high res data.
    
    - direction = [forward] | backward | both
      Integration direction. Default is forward.

    - exclude_points_x, exclude_points_y
      Points to exclude in the x and y direction. Useful for nested domains to avoid 
      duplicating multiple sets of curly vectors in the inner domain(s).

    - seed_points: specify a list of seed points [(x1,y1), (x2, y2), ...]
      If left blank, the function reverts to grid points with "skip"
    -----------------------------------------------------------------------------
    Contact: Brandon Kerns bkerns@uw.edu

    """

    import numpy as np
    import matplotlib.pylab as plt
    
    LW = linewidth
    COL = color

    # Check sizes.
    shape_X = X.shape
    ny = shape_X[0]
    nx = shape_X[1]
    
    if verbose:
        print("Data size: ny={0} by nx={1}.".format(ny, nx), flush = True)

        if (method.lower() == 'euler'):
            print("Integration method: Euler.", flush=True)
        else:
            print("Integration method: 4th order Runge-Kutta (rk4).", flush=True)
            
    # Collect the list_of_points to include.
    list_of_points = seed_points
    if len(seed_points) < 1:

        if verbose:
            print('Seed points not specified, so using grid points with skip.')
            
        seed_points = []
        
        for jj in range(0, ny, skip):
            
            for ii in range(0, nx, skip):

                if (not np.isfinite(u[jj,ii]) or not np.isfinite(v[jj,ii])):
                    continue

                if ii in exclude_points_x and jj in exclude_points_y:
                    continue


                seed_points.append((X[jj,ii], Y[jj,ii]))


    # Process each of the list_of_points

    for iii in range(len(seed_points)):
        if verbose and ((iii+1) % 100 == 0):
            print('Doing: ', str(iii), ' of ' + str(len(seed_points)) + " points.", flush=True)


        i_nn, j_nn = get_nearest_neighbor_index(seed_points[iii][0], seed_points[iii][1], X, Y)
        u_nn = u[j_nn, i_nn]
        v_nn = v[j_nn, i_nn]

        # Skip this one if it is NaN or very small.
        if not np.isfinite(u_nn):
            continue
        if not np.isfinite(v_nn):
            continue
        if np.sqrt(u_nn**2 + v_nn**2) < min_speed:
            continue
        if np.sqrt(u_nn**2 + v_nn**2) > max_speed:
            continue
        
        xx,yy = calculate_a_curly_vector(seed_points[iii][0], seed_points[iii][1], X, Y, u, v
                                         , scale_factor = scale_factor
                                         , time_step = time_step
                                         , verbose = verbose
                                         , method = method, direction = direction
                                         , interp = interp
                                         , min_length = min_length
                                         , max_length = max_length)

        if smoothing and len(xx) >= 3:
            xx = gaussian_filter1d(xx, 1, mode='reflect')
            yy = gaussian_filter1d(yy, 1, mode='reflect')
                    
        if (len(xx) > 1):
            # Plot the curve
            plt.plot(xx,yy,color=COL, linewidth=LW,zorder=zorder)

            # Now add the arrow at the end.
            draw_arrow_head(xx, yy, color=COL, linewidth=LW
                            , arrowhead_angle = arrowhead_angle
                            , arrowhead_length_factor = arrowhead_length_factor
                            , arrowhead_min_length = arrowhead_min_length
                            , arrowhead_max_length = arrowhead_max_length
                            , closed = arrowhead_close,zorder=zorder)
            
            
######################################################################################
######################################################################################
######################################################################################
######################################################################################
######################################################################################

## Example script.
if (__name__ == "__main__"):

    x = np.arange(0,20.1,0.1)
    y = np.arange(-5.0,10.1,0.1)

    X, Y = np.meshgrid(x, y)

    u = np.cos(X) + Y
    v = 5.0*np.sin(X)

    plt.close("all")
    fig = plt.figure(figsize=(8,5))
    ax = fig.add_subplot(121)

    spd = np.sqrt(u**2 + v**2)
    H=ax.pcolormesh(x,y,spd)
    ax.quiver(x[::10], y[::10], u[::10,::10], v[::10,::10], color='b', width=0.002)

    ## Try gridpoint sampling method.
    curly_vectors(X, Y, u, v, scale_factor = 5.0, skip=10, linewidth=1, color='k', verbose=False, interp='nn')

    
    ## Try Poisson disk sampling method.
    ax2 = fig.add_subplot(122)

    seed_points = generate_poisson_disk_sampling_points_2d([0, 20, -5, 10], 0.8, start_points=[], verbose=False, do_plots=False)

    H2=ax2.pcolormesh(x,y,spd)
    ax2.quiver(x[::10], y[::10], u[::10,::10], v[::10,::10], color='b', width=0.002)

    curly_vectors(X, Y, u, v, scale_factor = 5.0, linewidth=1, color='k', verbose=False, arrowhead_min_length = 0.5, method='euler', seed_points = seed_points, direction='both', interp='nn')

    #plt.colorbar(H)
    ax.set_aspect('equal')
    ax.set_ylim(y[0],y[-1])
    ax.set_xlim(x[0],x[-1])

    ax2.set_aspect('equal')
    ax2.set_ylim(y[0],y[-1])
    ax2.set_xlim(x[0],x[-1])
    
    #plt.show()
    outFileBase = "test_bk_curly_vectors"
    plt.savefig(outFileBase + '_waves.png' ,bbox_inches='tight', dpi=150)
