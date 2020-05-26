import yaml, copy
import numpy as np
from distgen import Generator
from distgen.tools import update_nested_dict


def get_cathode_particlegroup(settings, DISTGEN_INPUT_FILE, verbose=True, distgen_verbose=False, id_start=1):
    distgen_input = yaml.safe_load(open(DISTGEN_INPUT_FILE))
    for k, v in settings.items():
        distgen_input = update_nested_dict(distgen_input, {k:v}, verbose=verbose, create_new=False)
    gen = Generator(distgen_input,verbose=distgen_verbose)
    gen.run()
    PG = gen.particles

    if ('cathode:sigma_xy' in settings):
        sigma_xy = settings.pop('cathode:sigma_xy') # remove from dictionary to avoid recursion problem
        sig_ratio = sigma_xy/(0.5*(PG['sigma_x'] + PG['sigma_y']))
        settings_1 = copy.copy(settings)
        
        var_list = ['r_dist:sigma_xy:value', 'r_dist:truncation_radius_right:value', 'r_dist:truncation_radius_left:value']
        for var in var_list:
            if (var in settings):
                settings_1[var] = settings[var] * sig_ratio
        PG = get_cathode_particlegroup(settings_1, DISTGEN_INPUT_FILE, verbose=verbose, distgen_verbose=distgen_verbose, id_start=id_start)
        if (verbose):
            print(f'Rescaling sigma_xy from {sx} -> {settings["cathode:sigma_xy"]}. Acheived: {PG["sigma_x"]}')
        return PG
    
    PG.assign_id()
    PG.id = np.arange(id_start,id_start+gen['n_particle'])
    
    return PG

    

def get_coreshield_particlegroup(settings, DISTGEN_INPUT_FILE, verbose=True, distgen_verbose=False):
    
    if ('coreshield:n_core' in settings):
        n_core = settings['coreshield:n_core']
        
    if ('coreshield:n_shield' in settings):
        n_shield = settings['coreshield:n_shield']
        
    core_charge_fraction = 0.5
    if ('coreshield:core_charge_fraction' in settings):
        core_charge_fraction = settings['coreshield:core_charge_fraction']       
    
    sigma_xy = None
    if ('cathode:sigma_xy' in settings):
        sigma_xy = settings.pop('cathode:sigma_xy') # Remove from dictionary so that calls to get_cathode_particlegroup do not see it
    
    PG = get_cathode_particlegroup(settings, DISTGEN_INPUT_FILE, verbose=False)
        
    sx = PG['sigma_x']
    sig_ratio = 1.0
    if (sigma_xy is not None):
        sig_ratio = sigma_xy/sx
    r_i = np.argsort(PG.r)
    r = PG.r[r_i]
    w = PG.weight[r_i]
    w_sum = np.cumsum(w)
    n_core_orig = np.argmax(w_sum > core_charge_fraction * w_sum[-1])
    n_shield_orig = len(w)-n_core_orig
    r_cut = r[n_core_orig]
    
    if (n_core is None):
        n_core = n_core_orig
        
    if (n_shield is None):
        n_shield = n_shield_orig
        
    settings_1 = copy.copy(settings)
    settings_1['r_dist:sigma_xy:value'] = sig_ratio*settings['r_dist:sigma_xy:value']
    settings_1['r_dist:truncation_radius_right:value'] = sig_ratio*settings['r_dist:truncation_radius_right:value']
    settings_1['r_dist:truncation_radius_left:value'] = sig_ratio*r_cut
    settings_1['r_dist:truncation_radius_left:units'] = 'm'
    settings_1['n_particle'] = n_shield
    settings_1['total_charge:value'] = (1.0-core_charge_fraction) * settings['total_charge:value']
    
    settings_2 = copy.copy(settings)
    settings_2['r_dist:sigma_xy:value'] = sig_ratio*settings['r_dist:sigma_xy:value']
    settings_2['r_dist:truncation_radius_right:value'] = sig_ratio*r_cut
    settings_2['r_dist:truncation_radius_right:units'] = 'm'
    settings_2['r_dist:truncation_radius_left:value'] = sig_ratio*settings['r_dist:truncation_radius_left:value']
    settings_2['n_particle'] = n_core
    settings_2['total_charge:value'] = core_charge_fraction * settings['total_charge:value']
    
    PG_shield = get_cathode_particlegroup(settings_1, DISTGEN_INPUT_FILE, verbose=False, id_start=n_core+1)
    PG_core = get_cathode_particlegroup(settings_2, DISTGEN_INPUT_FILE, verbose=False, id_start=1)
    
    PG = PG_core+PG_shield
        
    if (verbose):
        if (sigma_xy is not None):
            print(f'Rescaling sigma_xy from {sx} -> {sigma_xy}. Acheived: {PG["sigma_x"]}')
        
    return PG



