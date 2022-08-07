
import numpy as np
from bisect import bisect
import pytz
import math
import scipy.optimize as opt

NaN = float("NaN")

def lerp(a,b,f):
    a = np.array(a)
    b = np.array(b)

    return a*f+b*(1-f)

def lerp2(a,b,c, f, d2):
    
    if abs(f + 1) < 0.001: return np.array(a)*1
    if abs(f) < 0.001: return np.array(b)*1

    a = np.array(a)
    b = np.array(b)
    c = np.array(c)

    k0 = b 
    k2 = (a*d2 + c - b - b*d2)/(d2+d2**2)
    k1 = (c-a*d2**2 - b + b*d2**2)/(d2+d2**2)

    return k2*f*f+k1*f+k0

def getValuesAtTimeLiniar(index,values,time):
    i = bisect(index, time)
    if i == 0:
        return values[0].copy()

    if i >= len(values):
        return values[-1].copy()

    d = index[i]-index[i-1]
    return lerp(values[i], values[i-1], (time-index[i-1])/d)

def getValuesAtTime(index,values,time):
    try:
        i = bisect(index, time)
        if i == 0 or i >= len(values):
            return np.full(values[0].shape,NaN)
        if i >= len(values) - 1:
            d = index[i]-index[i-1]
            return lerp(values[i], values[i-1], (time-index[i-1])/d)

    except:
        return index[i-1]
    d = index[i]-index[i-1]
    v1 = lerp2(values[i-1], values[i], values[i+1], (time-index[i-1])/d - 1, (index[i+1]-index[i])/d)
   
    return v1

def calc_haversine(lat1, lon1, lat2, lon2):
  """Calculates the great circle distance between two points
  on the earth. Inputs are array-like and specified in decimal degrees.
  """
  RADIUS = 6_367_000
  lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
  dlat = lat2 - lat1
  dlon = lon2 - lon1
  a = np.sin(dlat/2)**2 + \
      np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
  dist = 2 * RADIUS * np.arcsin(a**0.5)
  return dist

def ecef2lla(x, y, z):
  # x, y and z are scalars or vectors in meters
  x = np.array([x]).reshape(np.array([x]).shape[-1], 1)
  y = np.array([y]).reshape(np.array([y]).shape[-1], 1)
  z = np.array([z]).reshape(np.array([z]).shape[-1], 1)

  a=6378137
  a_sq=a**2
  e = 8.181919084261345e-2
  e_sq = 6.69437999014e-3

  f = 1/298.257223563
  b = a*(1-f)

  # calculations:
  r = np.sqrt(x**2 + y**2)
  ep_sq  = (a**2-b**2)/b**2
  ee = (a**2-b**2)
  f = (54*b**2)*(z**2)
  g = r**2 + (1 - e_sq)*(z**2) - e_sq*ee*2
  c = (e_sq**2)*f*r**2/(g**3)
  s = (1 + c + np.sqrt(c**2 + 2*c))**(1/3.)
  p = f/(3.*(g**2)*(s + (1./s) + 1)**2)
  q = np.sqrt(1 + 2*p*e_sq**2)
  r_0 = -(p*e_sq*r)/(1+q) + np.sqrt(0.5*(a**2)*(1+(1./q)) - p*(z**2)*(1-e_sq)/(q*(1+q)) - 0.5*p*(r**2))
  u = np.sqrt((r - e_sq*r_0)**2 + z**2)
  v = np.sqrt((r - e_sq*r_0)**2 + (1 - e_sq)*z**2)
  z_0 = (b**2)*z/(a*v)
  h = u*(1 - b**2/(a*v))
  phi = np.arctan((z + ep_sq*z_0)/r)
  lambd = np.arctan2(y, x)

  return phi*180/np.pi, lambd*180/np.pi, h
  
def WGS84_to_ECEF(lat, lon, alt):
    # convert to radians
    rad_lat = lat * (np.pi / 180.0)
    rad_lon = lon * (np.pi / 180.0)
    a    = 6378137.0
    # f is the flattening factor
    finv = 298.257223563
    f = 1 / finv   
    # e is the eccentricity
    e2 = 1 - (1 - f) * (1 - f)    
    # N is the radius of curvature in the prime vertical
    N = a / np.sqrt(1 - e2 * np.sin(rad_lat) * np.sin(rad_lat))
    x = (N + alt) * np.cos(rad_lat) * np.cos(rad_lon)
    y = (N + alt) * np.cos(rad_lat) * np.sin(rad_lon)
    z = (N * (1 - e2) + alt)        * np.sin(rad_lat)
    return x, y, z
    

def rotate_sat(sat, dist):
    res = np.zeros((len(sat),3))
    tm = dist/299792458
    ang = math.pi/(12*60*60)*tm
    res[:,2] = sat[:,2]
    res[:,0] = np.cos(ang)*sat[:,0]+np.sin(ang)*sat[:,1]
    res[:,1] = -np.sin(ang)*sat[:,0]+np.cos(ang)*sat[:,1]
    return res


from scipy.optimize import minimize
import numpy as np

def cost_function(x_hat, weights, sat_pos, sat_types, pr):
    rows = np.abs(weights * (np.linalg.norm(sat_pos - x_hat[:3], axis=1) + x_hat[sat_types+3] - pr))
    return np.sum(rows)

def calc_pos_fix2(_sat_pos, _pr, _weights, index, _sat_types, x0):
    pr      = _pr[index]
    sat_pos = _sat_pos[index]
    weights = _weights[index]
    sat_types = _sat_types[index]

    valid = (pr > 10000000)
    pr      = pr[valid]
    sat_pos = sat_pos[valid]
    weights = weights[valid]
    sat_types = sat_types[valid]

    valid = (pr < 40000000)
    pr      = pr[valid]
    sat_pos = sat_pos[valid]
    weights = weights[valid]
    sat_types = sat_types[valid]


    n = len(pr)
    if n < 4:
        return x0, []

    Fx_pos = pr_residual(sat_pos, pr, sat_types, n, weights=weights)
    x0 = np.array(x0)
    errs = np.abs(Fx_pos(x0))
    valid = (errs < 1000)
    if np.sum(valid) >= 4 and np.sum(valid) > n*2/3:
        pr      = pr[valid]
        sat_pos = sat_pos[valid]
        weights = weights[valid]
        sat_types = sat_types[valid]
        n = len(pr)

    Fx_pos = pr_residual(sat_pos, pr, sat_types, n, weights=weights)
    opt_pos = minimize(cost_function, x0, args=(weights, sat_pos, sat_types, pr)).x
    return opt_pos, Fx_pos(opt_pos, sat_types, weights=1)

def pr_residual_speed(sat_dir, pr, sat_types, n, weights=1):
    # solve for pos
    def Fx_pos(x_hat, sat_types = sat_types, weights=weights):
        rows = np.abs(weights * (np.sum(sat_dir*x_hat[:3], axis=-1) + x_hat[sat_types+3] - pr))**0.2
        return rows
    return Fx_pos

def calc_speed(_sat_dirs, _pr, _weights, index, _sat_types, x0):
    '''
    Calculates gps fix with WLS optimizer
    returns:
    0 -> list with positions
    1 -> pseudorange errs
    '''

    pr      = _pr[index]
    sat_pos = _sat_dirs[index]
    weights = _weights[index]
    sat_types = _sat_types[index]

    n = len(pr)
    if n < 8:
        return x0, []

    Fx_pos = pr_residual_speed(sat_pos, pr, sat_types, n, weights=weights)
    x0 = np.array(x0)
    opt_pos = opt.least_squares(Fx_pos, x0).x
    errs = Fx_pos(opt_pos, sat_types, weights=1)
    opt_pos2 = np.zeros((9))
    if True:
        for i in range(5):
            if(np.sum(sat_types == i) == 0):
                continue
            opt_pos2[i+3] = np.median(pr[sat_types == i])
        errs2 = Fx_pos(opt_pos2, sat_types, weights=1)



    return opt_pos, errs


def calc_pos_fix(_sat_pos, _pr, _weights, index, _sat_types, x0):
    '''
    Calculates gps fix with WLS optimizer
    returns:
    0 -> list with positions
    1 -> pseudorange errs
    '''

    pr      = _pr[index]
    sat_pos = _sat_pos[index]
    weights = _weights[index]
    sat_types = _sat_types[index]

    valid = (pr > 10000000)
    pr      = pr[valid]
    sat_pos = sat_pos[valid]
    weights = weights[valid]
    sat_types = sat_types[valid]

    valid = (pr < 40000000)
    pr      = pr[valid]
    sat_pos = sat_pos[valid]
    weights = weights[valid]
    sat_types = sat_types[valid]


    n = len(pr)
    if n < 6:
        return x0, np.array([3000])

    Fx_pos,Fx_pos2 = pr_residual(sat_pos, pr, sat_types, n, weights=weights)
    x0 = np.array(x0)
    errs = np.abs(Fx_pos(x0))
    valid = (errs < 1000)
    if np.sum(valid) >= 4 and np.sum(valid) > n*2/3:
        pr      = pr[valid]
        sat_pos = sat_pos[valid]
        weights = weights[valid]
        sat_types = sat_types[valid]
        n = len(pr)
        Fx_pos,Fx_pos2 = pr_residual(sat_pos, pr, sat_types, n, weights=weights)
    else:
        x0=np.zeros((9))

    opt_pos = opt.least_squares(Fx_pos, x0).x

    opt_pos = opt.minimize(Fx_pos2, opt_pos, method='nelder-mead',
              options={'xatol': 1e-8, 'disp': False}).x


    return opt_pos, Fx_pos(opt_pos, sat_types, weights=1)


def pr_residual(sat_pos, pr, sat_types, n, weights=1):
    # solve for pos
    def Fx_pos(x_hat, sat_types = sat_types, weights=weights):
        rows = np.abs(weights * (np.linalg.norm(sat_pos - x_hat[:3], axis=1) + x_hat[sat_types+3] - pr))
        return rows
    def Fx_pos2(x_hat, sat_types = sat_types, weights=weights):
        rows = np.abs(weights * (np.linalg.norm(sat_pos - x_hat[:3], axis=1) + x_hat[sat_types+3] - pr))
        return np.sum(rows)
    return Fx_pos,Fx_pos2


def pr_shift(sat_dirs, shifts, num, weights=1):
    # solve for pos
    def Fx_shift(x_hat, weights=weights):
        cur_shifts = np.sum(sat_dirs*x_hat[:3], axis=1)
        rows = np.abs(cur_shifts - shifts - x_hat[3])
        rows = sorted(rows, key=abs)
        rows = rows[:num]
        rows.append(abs(x_hat[2]))
        return rows
    return Fx_shift

def calc_shift_fromsat(sat_dir_in, lens_in, x0=[0, 0, 0, 0], y0 = [0, 0, 0, 0], weights=1):

    lens_in = np.reshape(lens_in, (-1))
    index = ~np.isnan(lens_in)
    lens = lens_in[index]
    sat_dir = sat_dir_in[index]

    n = len(lens)
    if n < 6:
        return [NaN, NaN, NaN, NaN], []

    ones = np.ones((n,1))
    sat_dir = np.concatenate((sat_dir,ones), axis = -1)

    num = 2*n//3 
    err = 1e7
    if num < 6:
        num = 6


    res = [NaN, NaN, NaN, NaN]
    err_Res = []
    for _ in range(250):
        indexes = np.random.choice(n, 4, replace=False)
        mat = sat_dir[indexes]
        vec = lens[indexes]
        try:
            mat = np.linalg.inv(mat)
        except:
            continue
        x_hat = np.dot(mat, vec)
        x_hat = np.reshape(x_hat,(1,4))

        cur_shifts = np.sum(sat_dir*x_hat, axis=1)
        rows = np.abs(cur_shifts - lens)
        err_cur = rows
        rows = np.sort(rows)
        rows = rows[:num]
        #rows.append(abs(x_hat[2]))
        if -np.sum(rows < 0.1) + abs(x_hat[0,2])*20 < err:
            err = -np.sum(rows < 0.1) + abs(x_hat[0,2])*20
            res = x_hat
            err_Res = err_cur
            if err < -num + 1:
                break

            
    ones = np.ones((256,1))
    sat_dir_copy = np.concatenate((sat_dir_in.copy(),ones), axis = -1)
    cur_shifts = np.sum(sat_dir_copy*res, axis=1)
    rows = np.abs(cur_shifts - lens_in)
    return res, rows

def calc_shift_fromsat2d(sat_dir_in, lens_in, x0=[0, 0, 0, 0], y0 = [0, 0, 0, 0], weights=1):

    lens_in = np.reshape(lens_in, (-1))
    index = ~np.isnan(lens_in)
    lens = lens_in[index]
    sat_dir = sat_dir_in[index]

    n = len(lens)
    if n < 6:
        return [NaN, NaN, NaN, NaN], []

    sat_dir[:,2] = np.ones((n))

    num = 2*n//3 
    err = 1e7
    if num < 6:
        num = 6


    res = [NaN, NaN, NaN, NaN]
    err_Res = []
    for _ in range(250):
        indexes = np.random.choice(n, 3, replace=False)
        mat = sat_dir[indexes]
        vec = lens[indexes]
        try:
            mat = np.linalg.inv(mat)
        except:
            continue
        x_hat = np.dot(mat, vec)
        x_hat = np.reshape(x_hat,(1,3))

        cur_shifts = np.sum(sat_dir*x_hat, axis=1)
        rows = np.abs(cur_shifts - lens)
        err_cur = rows
        rows = np.sort(rows)
        rows = rows[:num]
        #rows.append(abs(x_hat[2]))
        if -np.sum(rows < 0.1) < err:
            err = -np.sum(rows < 0.1)
            res = x_hat
            err_Res = err_cur
            if err < -num + 1:
                break

            
    sat_dir_copy = sat_dir_in.copy()
    sat_dir_copy[:,2] = np.ones((256))
    cur_shifts = np.sum(sat_dir_copy*res, axis=1)
    rows = np.abs(cur_shifts - lens_in)
    return np.array([res[0,0], res[0,1], 0, res[0,2]]), rows
