from pmd_beamphysics import ParticleGroup
from pmd_beamphysics.units import unit, PARTICLEGROUP_UNITS
from pmd_beamphysics.statistics import norm_emit_calc
import numpy as np
from .tools import divide_particles


def add_to_particlegroup():
    if (not hasattr(ParticleGroup, 'sqrt_norm_emit_4d')):
        ParticleGroup.sqrt_norm_emit_4d = sqrt_norm_emit_4d
        
    if (not hasattr(ParticleGroup, 'n_slices')):
        ParticleGroup.n_slices = 50
                
    if (not hasattr(ParticleGroup, 'slice_key')):
        ParticleGroup.slice_key = 't'
        
    if (not hasattr(ParticleGroup, 'slice_emit_x')):
        ParticleGroup.slice_emit_x = slice_emit_x
        
    if (not hasattr(ParticleGroup, 'slice_emit_y')):
        ParticleGroup.slice_emit_y = slice_emit_y
        
    if (not hasattr(ParticleGroup, 'slice_emit_4d')):
        ParticleGroup.slice_emit_4d = slice_emit_4d
        
    if ('sqrt_norm_emit_4d' not in PARTICLEGROUP_UNITS.keys()):
        PARTICLEGROUP_UNITS['sqrt_norm_emit_4d'] = unit('m')
        
    if ('slice_emit_x' not in PARTICLEGROUP_UNITS.keys()):
        PARTICLEGROUP_UNITS['slice_emit_x'] = unit('m')
        
    if ('slice_emit_y' not in PARTICLEGROUP_UNITS.keys()):
        PARTICLEGROUP_UNITS['slice_emit_y'] = unit('m')
        
    if ('slice_emit_4d' not in PARTICLEGROUP_UNITS.keys()):
        PARTICLEGROUP_UNITS['slice_emit_4d'] = unit('m')
    
    return


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