import numpy as np
import copy
from .ParticleGroupExtension import ParticleGroupExtension, divide_particles
import numpy.polynomial.polynomial as poly
from random import shuffle

def postprocess_screen(screen, **params):
    need_copy_params = ['take_slice', 'take_range', 'cylindrical_copies', 'remove_correlation', 'kill_zero_weight', 
                        'radial_aperture', 'remove_spinning', 'include_ids', 'random_N', 'first_N', 'clip_to_charge']
    need_copy = any([p in params for p in need_copy_params])
    
    if ('need_copy' in params):
        need_copy = params['need_copy']
    
    if (need_copy):
        screen = copy.deepcopy(screen)
    
    if ('kill_zero_weight' in params):
        if (params['kill_zero_weight']):
            screen = kill_zero_weight(screen)
    
    if ('include_ids' in params):
        ids = params['include_ids']
        if (len(ids) > 0):
            screen = include_ids(screen, ids)
    
    if ('take_range' in params):
        (take_range_var, range_min, range_max) = params['take_range']
        if (range_min < range_max):
            screen = take_range(screen, take_range_var, range_min, range_max)
    
    if ('take_slice' in params):
        (take_slice_var, slice_index, n_slices) = params['take_slice']
        if (n_slices > 1):
            screen = take_slice(screen, take_slice_var, slice_index, n_slices)
            
    if ('clip_to_charge' in params):
        target_charge = params['clip_to_charge']
        if (target_charge > 0):
            clip_to_charge(screen, target_charge, verbose=False)
            
    if ('cylindrical_copies' in params):
        cylindrical_copies_n = params['cylindrical_copies']
        if (cylindrical_copies_n > 0):
            screen = add_cylindrical_copies(screen, params['cylindrical_copies'])
            
    if ('remove_spinning' in params):
        if (params['remove_spinning']):
            screen = remove_spinning(screen)
               
    if ('remove_correlation' in params):
        (remove_correlation_var1, remove_correlation_var2, remove_correlation_n) = params['remove_correlation']
        if (remove_correlation_n >= 0):
            screen = remove_correlation(screen, remove_correlation_var1, remove_correlation_var2, remove_correlation_n)
    
    if ('random_N' in params):
        N = params['random_N']
        if (N > 0):
            screen = random_N(screen, N, random=True)
    else:
        if ('first_N' in params):
            N = params['first_N']
            if (N > 0):
                screen = random_N(screen, N, random=False)
        
    return screen


# Returns IDs of the N nearest particles to center_particle_id in the ndim dimensional phase space
# "Nearest" is determined by changing coordinates to ones with sigma_matrix = identity_matrix
def id_of_nearest_N(screen_input, center_particle_id, N, ndim=4):
    screen = copy.deepcopy(screen_input)
    
    if (ndim == 6):
        screen.drift_to_t()
    
    x = screen.x
    px = screen.px
    w = screen.weight
    pid = screen.id
    
    if (center_particle_id not in pid):
        print('Cannot find center particle')
        return np.array([])
    
    if (ndim == 2):
        x = x - np.sum(x*w)/np.sum(w)
        px = px - np.sum(px*w)/np.sum(w)
        u0 = np.vstack((x, px))
    if (ndim == 4):
        y = screen.y
        py = screen.py
        x = x - np.sum(x*w)/np.sum(w)
        px = px - np.sum(px*w)/np.sum(w)
        y = y - np.sum(y*w)/np.sum(w)
        py = py - np.sum(py*w)/np.sum(w)
        u0 = np.vstack((x, px, y, py))
    if (ndim == 6):
        y = screen.y
        py = screen.py
        z = screen.z
        pz = screen.pz
        
        x = x - np.sum(x*w)/np.sum(w)
        px = px - np.sum(px*w)/np.sum(w)
        y = y - np.sum(y*w)/np.sum(w)
        py = py - np.sum(py*w)/np.sum(w)
        z = z - np.sum(z*w)/np.sum(w)
        pz = pz - np.sum(pz*w)/np.sum(w)
        u0 = np.vstack((x, px, y, py, z, pz))
    
    sigma_matrix = np.cov(u0, aweights=w)
            
    # Change into round phase space coordinates
    (E, V) = np.linalg.eig(sigma_matrix)
    u1 = np.diag(1.0/np.sqrt(E)) @ np.linalg.solve(V, u0)
        
    u1_cen = u1[:, pid == center_particle_id]
    d = np.sum((u1 - u1_cen)**2, 0)
    sorted_index = np.argsort(d)
        
    return pid[sorted_index[0:N]]
    

    
# Returns a screen with either only the first N or a random N particles remaining
def random_N(screen, N, random=True):
    alive_ids = screen.id[screen.weight > 0]
    if (random):
        shuffle(alive_ids)
    if (N < len(alive_ids)):
        alive_ids = alive_ids[0:N]
    return include_ids(screen, alive_ids)


# Returns a screen with only the particles with id = ids remaining
def include_ids(screen, ids):
    id_to_index = {id : i for i,id in enumerate(screen.id)}
    ids_to_zero = np.setdiff1d(screen.id, ids, assume_unique=True)
    ind_to_zero = [id_to_index[id] for id in ids_to_zero]
    screen.weight[ind_to_zero] = 0.0
    
    return kill_zero_weight(screen)


# Removes a screen without the x-py, y-px correlations associated with particles spinning in a solenoid
def remove_spinning(screen):
    x = copy.copy(screen.x)
    px = copy.copy(screen.px)
    y = copy.copy(screen.y)
    py = copy.copy(screen.py)
    w = screen.weight

    sumw = np.sum(w)

    x = x - np.sum(x*w)/sumw
    px = px - np.sum(px*w)/sumw
    y = y - np.sum(y*w)/sumw
    py = py - np.sum(py*w)/sumw

    x2 = np.sum(x*x*w)/sumw
    y2 = np.sum(y*y*w)/sumw

    xpy = np.sum(x*py*w)/sumw
    ypx = np.sum(y*px*w)/sumw

    C1 = -ypx/y2
    C2 = -xpy/x2
    
    screen.px = screen.px + C1*screen.y
    screen.py = screen.py + C2*screen.x
        
    return screen


# Removes particles that have zero weight from the distribution
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


# Removes particles that are outside of a given range of a variable
def take_range(screen, take_range_var, range_min, range_max):
    x = getattr(screen, take_range_var) 
    
    if (take_range_var in ['x','y','z','t']):
        # Subtract mean
        x = x - sum(x*screen.weight)/sum(screen.weight)
    
    out_of_range = np.logical_or(x < range_min, x > range_max)
    
    if (np.count_nonzero(out_of_range) < len(out_of_range)):
        screen.weight[out_of_range] = 0.0
    
    return kill_zero_weight(screen)

    
# Takes n_slices slices over the full range of the variable take_slice_var, and then returns a screen with the particles in the slice_index'th slice
def take_slice(screen, take_slice_var, slice_index, n_slices):
    p_list, edges, density_norm = divide_particles(screen, nbins=n_slices, key=take_slice_var)
    if (slice_index>=0 and slice_index<len(p_list)):
        return p_list[slice_index]
    else:
        return screen


# Removes a polynomial correlation in the var1-var2 phase space. Subtracts from var2 to remove correlation.
def remove_correlation(screen, var1, var2, max_power):

    x = getattr(screen,var1)
    y = getattr(screen,var2)
    
    c = poly.polyfit(x, y, max_power)
    y_new = poly.polyval(x, c)
    
    setattr(screen, var2, y-y_new)
    
    return screen


def clip_to_charge(PG, clipping_charge, verbose=False):
    min_final_particles = 3
    
    r_i = np.argsort(PG.r)
    r = PG.r[r_i]
    w = PG.weight[r_i]
    w_sum = np.cumsum(w)
    if (clipping_charge >= w_sum[-1]):
        n_clip = -1
    else:
        n_clip = np.argmax(w_sum > clipping_charge)
    if (n_clip < (min_final_particles-1) and n_clip > -1):
        n_clip = min_final_particles-1
    r_cut = r[n_clip]
    PG.weight[PG.r>r_cut] = 0
    if (verbose):
        print(f'Clipping at r = {r_cut}')
    PG = kill_zero_weight(PG)


# Duplicates all particles n_copies times, uniformly rotated around the z-axis. Useful for making pretty plots when the screen is cylindrically symmetric 
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


