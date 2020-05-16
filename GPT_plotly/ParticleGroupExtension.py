from pmd_beamphysics import ParticleGroup
from pmd_beamphysics.units import unit, PARTICLEGROUP_UNITS
from pmd_beamphysics.statistics import norm_emit_calc
from matplotlib import pyplot as plt
import numpy.polynomial.polynomial as poly
import scipy.optimize as optimize
import numpy as np
import copy

class ParticleGroupExtension(ParticleGroup):
    n_slices = 50
    slice_key = 't'
    
    def __init__(self, input_particle_group=None, data=None):
        if (input_particle_group):
            data={}
            for key in input_particle_group._settable_keys:
                data[key] = copy.copy(input_particle_group[key])  # is deepcopy needed?
        
        super().__init__(data=data)
        
        if ('sqrt_norm_emit_4d' not in PARTICLEGROUP_UNITS.keys()):
            PARTICLEGROUP_UNITS['sqrt_norm_emit_4d'] = unit('m')

        if ('slice_emit_x' not in PARTICLEGROUP_UNITS.keys()):
            PARTICLEGROUP_UNITS['slice_emit_x'] = unit('m')

        if ('slice_emit_y' not in PARTICLEGROUP_UNITS.keys()):
            PARTICLEGROUP_UNITS['slice_emit_y'] = unit('m')
            
        if ('core_emit_x' not in PARTICLEGROUP_UNITS.keys()):
            PARTICLEGROUP_UNITS['core_emit_x'] = unit('m')

        if ('core_emit_y' not in PARTICLEGROUP_UNITS.keys()):
            PARTICLEGROUP_UNITS['core_emit_y'] = unit('m')
        
        if ('core_emit_4d' not in PARTICLEGROUP_UNITS.keys()):
            PARTICLEGROUP_UNITS['core_emit_4d'] = unit('m')

        if ('slice_emit_4d' not in PARTICLEGROUP_UNITS.keys()):
            PARTICLEGROUP_UNITS['slice_emit_4d'] = unit('m')
            
        if ('action_x' not in PARTICLEGROUP_UNITS.keys()):
            PARTICLEGROUP_UNITS['action_x'] = unit('m')
        
        if ('action_y' not in PARTICLEGROUP_UNITS.keys()):
            PARTICLEGROUP_UNITS['action_y'] = unit('m')
            
        if ('action_r' not in PARTICLEGROUP_UNITS.keys()):
            PARTICLEGROUP_UNITS['action_r'] = unit('m')

        if ('rp' not in PARTICLEGROUP_UNITS.keys()):
            PARTICLEGROUP_UNITS['rp'] = unit('rad')
            
    @property
    def rp(self):
        return self.pr/self.pz 
            
    @property
    def core_emit_x(self):
        return core_emit_calc(self.x, self.xp, self.weight)
    
    @property
    def core_emit_y(self):
        return core_emit_calc(self.y, self.yp, self.weight)
    
    @property
    def core_emit_4d(self):
        return core_emit_calc_4d(self.x, self.xp, self.y, self.yp, self.weight)
        
    @property
    def sqrt_norm_emit_4d(self):
        return np.sqrt(norm_emit_calc(self, planes=['x', 'y']))

    @property
    def slice_emit_x(self):
        (p_list, _, _) = divide_particles(self, nbins = self.n_slices, key=self.slice_key)
        return slice_emit(p_list, 'norm_emit_x')

    @property
    def slice_emit_y(self):
        (p_list, _, _) = divide_particles(self, nbins = self.n_slices, key=self.slice_key)
        return slice_emit(p_list, 'norm_emit_y')

    @property
    def slice_emit_4d(self):
        (p_list, _, _) = divide_particles(self, nbins = self.n_slices, key=self.slice_key)
        return slice_emit(p_list, 'sqrt_norm_emit_4d')

    @property
    def action_x(self):
        sig = self.cov('x', 'xp')
        emit = np.sqrt(np.linalg.det(sig))
        beta = sig[0,0] / emit
        alpha = -sig[0,1] / emit
        gamma = sig[1,1] / emit
        return gamma*self.x*self.x + 2.0*alpha*self.x*self.xp + beta*self.xp*self.xp
    
    @property
    def action_y(self):
        sig = self.cov('y', 'yp')
        emit = np.sqrt(np.linalg.det(sig))
        beta = sig[0,0] / emit
        alpha = -sig[0,1] / emit
        gamma = sig[1,1] / emit
        return gamma*self.y*self.y + 2.0*alpha*self.y*self.yp + beta*self.yp*self.yp
    
    @property
    def action_r(self):
        sig = self.cov('r', 'rp')
        emit = np.sqrt(np.linalg.det(sig))
        beta = sig[0,0] / emit
        alpha = -sig[0,1] / emit
        gamma = sig[1,1] / emit
        return gamma*self.r*self.r + 2.0*alpha*self.r*self.rp + beta*self.rp*self.rp
    

#-----------------------------------------
# helper functions for ParticleGroupExtension class

def slice_emit(p_list, key):
    min_particles = 5
    weights = np.array([0.0 for p in p_list])
    emit = np.array([0.0 for p in p_list])
    for p_i, p in enumerate(p_list):
        if (p.n_particle >= min_particles):
            emit[p_i] = p[key]
            weights[p_i] = p['charge']
    weights = weights/np.sum(weights)
    avg_emit = np.sum(emit*weights)
    
    return avg_emit



def convert_gpt_data(gpt_data_input):
    gpt_data = copy.deepcopy(gpt_data_input)  # This is lazy, should just make a new GPT()
    for i, pmd in enumerate(gpt_data_input.particles):
        gpt_data.particles[i] = ParticleGroupExtension(input_particle_group=pmd)  # This copies the data again
    for tout in gpt_data.tout:
        tout.drift_to_z() # Turn all the touts into quasi-screens
    return gpt_data



def divide_particles(particle_group, nbins = 100, key='t'):
    """
    Splits a particle group into even slices of 'key'. Returns a list of particle groups. 
    """
    x = getattr(particle_group, key) 
    
    if (key == 'r'):
        x = x*x
        xmin = 0  # force r=0 as min, could use min(x) here, optionally
        xmax = max(x)
        dx = (xmax-xmin)/(nbins-1)
        edges = np.linspace(xmin, xmax + 0.01*dx, nbins+1) # extends slightly further than max(r2)
        dx = edges[1]-edges[0]
    else:
        dx = (max(x)-min(x))/(nbins-1)
        edges = np.linspace(min(x) - 0.01*dx, max(x) + 0.01*dx, nbins+1) # extends slightly further than range(r2)
        dx = edges[1]-edges[0]
    
    which_bins = np.digitize(x, edges)-1
    
    if (key == 'r'):
        x = np.sqrt(x)
        edges = np.sqrt(edges)
            
    # Split particles
    plist = []
    for bin_i in range(nbins):
        chunk = which_bins==bin_i
        # Prepare data
        data = {}
        #keys = ['x', 'px', 'y', 'py', 'z', 'pz', 't', 'status', 'weight'] 
        for k in particle_group._settable_array_keys:
            data[k] = getattr(particle_group, k)[chunk]
        # These should be scalars
        data['species'] = particle_group.species
        
        # New object
        p = ParticleGroupExtension(data=data)
        plist.append(p)
    
    # normalization for sums of particle properties, = 1 / histogram bin width
    if (key == 'r'):
        density_norm = 1.0/(np.pi*(edges[1]**2 - edges[0]**2))
    else:
        density_norm = 1.0/(edges[1] - edges[0])
    
    return plist, edges, density_norm



def core_emit_calc_4d(x, xp, y, yp, w, show_fit=False):

    x = copy.copy(x)
    xp = copy.copy(xp)
    y = copy.copy(y)
    yp = copy.copy(yp)
    
    sumw = np.sum(w)
    
    x = x - np.sum(x*w)/sumw
    xp = xp - np.sum(xp*w)/sumw
    y = y - np.sum(y*w)/sumw
    yp = yp - np.sum(yp*w)/sumw
    
    x2 = np.sum(x*x*w)/sumw
    y2 = np.sum(y*y*w)/sumw

    u2 = (x2+y2)/2.0

    xpy = np.sum(x*yp*w)/sumw
    ypx = np.sum(y*xp*w)/sumw

    L = (xpy-ypx)/2.0

    C = -L/u2

    xp = xp - C*y
    yp = yp + C*x

    ec4x = core_emit_calc(x, xp, w, show_fit=show_fit)
    ec4y = core_emit_calc(y, yp, w, show_fit=show_fit)

    return 0.5*(ec4x+ec4y)


def core_emit_calc(x, xp, w, show_fit=False):

    x = copy.copy(x)
    xp = copy.copy(xp)
    
    emit_change_factor = 3 # fit data in range where emittance changes by less than this factor
    
    min_particle_count = 10000   # minimum number of particles required to compute a core emittance
    average_count_per_bin = int(min([len(x)/50, 1000]))

    if (len(x) < min_particle_count):
        raise ValueError('Too few particles to calculate core emittance.')

    x = x - np.sum(x*w)/np.sum(w)
    xp = xp - np.sum(xp*w)/np.sum(w)
    
    u0 = np.vstack((x, xp))
    sigma_matrix = np.cov(u0, aweights=w)
            
    if (np.linalg.det(sigma_matrix) < 1e-21):
        print('Possible zero emittance found, assuming core emittance is zero.')
        return 0

    # Change into better (round phase space) coordinates
    (_, V) = np.linalg.eig(sigma_matrix)
    u1 = np.linalg.solve(V, u0)

    # Now get the sigma matrix in the new coordinates
    sigma_matrix = np.cov(u1, aweights=w)
    
    r = np.sqrt(np.array([1.0/np.diag(sigma_matrix)]).dot(u1**2))[0]    
    dr = np.sort(r)[average_count_per_bin-1] # first dr includes exactly average_count_per_bin particles
    
    rbin = np.arange(0, np.max(r), dr)
    
    rhor = np.histogram(r, bins=rbin)[0]
        
    rbin = rbin[0:-1] + 0.5*(rbin[1] - rbin[0])
    rhonorm = np.trapz(rhor, rbin)
    
    rho = rhor / (rbin * rhonorm * 2 * np.pi * np.sqrt(np.prod(np.diag(sigma_matrix))));
            
    emit_in_range = rho > np.max(rho) / emit_change_factor
    max_fit_r = np.max(rbin[emit_in_range])
    plot_range = rbin < max_fit_r
    
    core_eps = 1.0/(4.0 * np.pi * rho[plot_range])
    rbin_fit = rbin[plot_range]   
        
    best_fit = poly.polyfit(rbin_fit, core_eps, 2);
    
    ec = best_fit[0]
    
    if (show_fit):
        plt.figure()
        p_list = []
        leg_list = []
        
        line_handle, = plt.plot(rbin[plot_range], core_eps, 'o')
        p_list.append(line_handle)
        leg_list.append('Data')
        
        r_plot = np.linspace(0, np.max(rbin_fit), 300)
        line_handle, = plt.plot(r_plot, poly.polyval(r_plot, best_fit), '-')
        p_list.append(line_handle)
        leg_list.append('Fit')
                                
        plt.xlim([0, np.max(rbin_fit)])
        plt.ylim([0, 1.1*np.max(core_eps)])
        plt.xlabel('Normalized radius^2');
        plt.ylabel('Emittance');
        plt.legend(p_list, leg_list)
    
        
    
    return ec