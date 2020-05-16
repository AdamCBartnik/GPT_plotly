import numpy as np
import copy
from .ParticleGroupExtension import ParticleGroupExtension, divide_particles
import numpy.polynomial.polynomial as poly

def postprocess_screen(screen, **params):
    
    need_copy_params = ['take_slice', 'cylindrical_copies', 'remove_correlation', 'kill_zero_weight', 'radial_aperture']
    need_copy = any([p in params for p in need_copy_params])
    
    if (need_copy):
        screen = copy.deepcopy(screen)
    
    if ('radial_aperture' in params):
        cutoff_radius = params['radial_aperture']
        screen = radial_aperture(screen, cutoff_radius)
    
    if ('kill_zero_weight' in params):
        if (params['kill_zero_weight']):
            screen = kill_zero_weight(screen)
    
    if ('take_slice' in params):
        (take_slice_var, slice_index, n_slices) = params['take_slice']
        if (n_slices > 1):
            screen = take_slice(screen, take_slice_var, slice_index, n_slices)
            
    if ('cylindrical_copies' in params):
        cylindrical_copies_n = params['cylindrical_copies']
        if (cylindrical_copies_n > 0):
            screen = add_cylindrical_copies(screen, params['cylindrical_copies'])
               
    if ('remove_correlation' in params):
        (remove_correlation_var1, remove_correlation_var2, remove_correlation_n) = params['remove_correlation']
        if (remove_correlation_n >= 0):
            screen = remove_correlation(screen, remove_correlation_var1, remove_correlation_var2, remove_correlation_n)
        
    return screen


def kill_zero_weight(screen):
    weight = screen.weight
    
    screen.x = screen.x[weight>0]
    screen.y = screen.y[weight>0]
    screen.z = screen.z[weight>0]
    screen.px = screen.px[weight>0]
    screen.py = screen.py[weight>0]
    screen.pz = screen.pz[weight>0]
    screen.t = screen.t[weight>0]
    screen.status = screen.status[weight>0]

    if (hasattr(screen, 'id')):
        screen.id = screen.id[weight>0]
        
    screen.weight = screen.weight[weight>0] # do last
    
    return screen


def radial_aperture(screen, cutoff_radius):
    r = screen.r
    screen.weight[r>cutoff_radius] = 0
    return screen

def take_slice(screen, take_slice_var, slice_index, n_slices):
    p_list, edges, density_norm = divide_particles(screen, nbins=n_slices, key=take_slice_var)
    if (slice_index>=0 and slice_index<len(p_list)):
        return p_list[slice_index]
    else:
        return screen


def remove_correlation(screen, var1, var2, max_power):

    x = getattr(screen,var1)
    y = getattr(screen,var2)
    
    c = poly.polyfit(x, y, max_power)
    y_new = poly.polyval(x, c)
    
    setattr(screen, var2, y-y_new)
    
    return screen


def add_cylindrical_copies(screen, n_copies):
    species = screen.species
    npart = len(screen.x)
    
    x = np.matlib.repmat(screen.x, 1, n_copies).reshape(npart*n_copies)
    y = np.matlib.repmat(screen.y, 1, n_copies).reshape(npart*n_copies)
    z = np.matlib.repmat(screen.z, 1, n_copies).reshape(npart*n_copies)
    px = np.matlib.repmat(screen.px, 1, n_copies).reshape(npart*n_copies)
    py = np.matlib.repmat(screen.py, 1, n_copies).reshape(npart*n_copies)
    pz = np.matlib.repmat(screen.pz, 1, n_copies).reshape(npart*n_copies)
    t = np.matlib.repmat(screen.t, 1, n_copies).reshape(npart*n_copies)
    status = np.matlib.repmat(screen.status, 1, n_copies).reshape(npart*n_copies)
    weight = np.matlib.repmat(screen.weight, 1, n_copies).reshape(npart*n_copies)
    
    theta = np.linspace(0,2*np.pi,npart+1)
    theta = theta[:-1]
    theta = np.matlib.repmat(theta, n_copies, 1).T.reshape(npart*n_copies)
    
    costh = np.cos(theta);
    sinth = np.sin(theta);
    
    px_new = px*costh - py*sinth
    py_new = px*sinth + py*costh
    px = px_new
    py = py_new

    x_new = x*costh - y*sinth
    y_new = x*sinth + y*costh
    x = x_new
    y = y_new
    
    weight = weight/n_copies
    
    data = dict(
        species=species,
        x=x,
        y=y,
        z=z,
        px=px,
        py=py,
        pz=pz,
        t=t,
        status=status,
        weight=weight
    )
    
    return ParticleGroupExtension(data=data)


