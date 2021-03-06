import time, os, copy
import numpy as np
from gpt import GPT
from gpt.gpt_phasing import gpt_phasing
from distgen import Generator
from distgen.writers import write_gpt
from .tools import get_screen_data
from .postprocess import kill_zero_weight, clip_to_charge
from .cathode_particlegroup import get_coreshield_particlegroup, get_cathode_particlegroup
from gpt.merit import default_gpt_merit
from gpt.gpt_distgen import fingerprint_gpt_with_distgen
from pint import UnitRegistry

def multirun_gpt_with_particlegroup(settings=None,
                             gpt_input_file=None,
                             input_particle_group=None,
                             workdir=None, 
                             use_tempdir=True,
                             gpt_bin='$GPT_BIN',
                             timeout=2500,
                             auto_phase=False,
                             verbose=False,
                             gpt_verbose=False,
                             asci2gdf_bin='$ASCI2GDF_BIN'
                             ):
    """
    Run gpt with particles from ParticleGroup. 
    
        settings: dict with keys that are in gpt input file.    
        
    """

    unit_registry = UnitRegistry()
    
    # Call simpler evaluation if there is no input_particle_group:
    if (input_particle_group is None):
        raise ValueError('Must supply input_particle_group')
    
    if(verbose):
        print('Run GPT with ParticleGroup:') 

    # Make gpt and generator objects
    G = GPT(gpt_bin=gpt_bin, input_file=gpt_input_file, initial_particles=input_particle_group, workdir=workdir, use_tempdir=use_tempdir)
    G.timeout=timeout
    G.verbose = verbose

    
    # Set inputs
    if settings:
        for k, v in settings.items():
            G.set_variable(k,v)
    else:
        raise ValueError('Must supply settings')
            
    G.set_variable('multi_run',0)
            
    if ('clipping_charge' in settings):
        raise ValueError('clipping_charge is deprecated, please specify value and units instead.')
    if ('final_charge' in settings):
        raise ValueError('final_charge is deprecated, please specify value and units instead.')    
    
    # Run
    if(auto_phase): 

        if(verbose):
            print('\nAuto Phasing >------\n')
        t1 = time.time()

        # Create the distribution used for phasing
        if(verbose):
            print('****> Creating initial distribution for phasing...')

        phasing_beam = get_distgen_beam_for_phasing_from_particlegroup(input_particle_group, n_particle=10, verbose=verbose)
        phasing_particle_file = os.path.join(G.path, 'gpt_particles.phasing.gdf')
        write_gpt(phasing_beam, phasing_particle_file, verbose=verbose, asci2gdf_bin=asci2gdf_bin)
    
        if(verbose):
            print('<**** Created initial distribution for phasing.\n')    

        G.write_input_file()   # Write the unphased input file

        phased_file_name, phased_settings = gpt_phasing(G.input_file, path_to_gpt_bin=G.gpt_bin[:-3], path_to_phasing_dist=phasing_particle_file, verbose=verbose)
        G.set_variables(phased_settings)
        t2 = time.time()

        if(verbose):
            print(f'Time Ellapsed: {t2-t1} sec.')
            print('------< Auto Phasing\n')

    if ('t_restart' not in settings):
        raise ValueError('t_restart must be supplied')
    t_restart = settings['t_restart']
    
    G.set_variable('multi_run',1)
    G.set_variable('last_run',2)
    G.set_variable('t_start', 0.0)
    G.set_variable('t_restart', t_restart)           

    # If here, either phasing successful, or no phasing requested
    G.run(gpt_verbose=gpt_verbose)
                        
    # Remove touts and screens that are after t_restart
    t_restart_with_fudge = t_restart + 1.0e-18 # slightly larger that t_restart to avoid floating point comparison problem
    G.output['n_tout'] = np.count_nonzero(G.stat('mean_t', 'tout') <= t_restart_with_fudge)
    G.output['n_screen'] = np.count_nonzero(G.stat('mean_t', 'screen') <= t_restart_with_fudge)
    for p in reversed(G.particles):
        if (p['mean_t'] > t_restart_with_fudge):
            G.particles.remove(p)
    
    #G_all = copy.deepcopy(G)
    G_all = G
        
    if (verbose):
        print(f'Looking for tout at t = {t_restart}')
    restart_particles = get_screen_data(G, tout_t = t_restart, use_extension=False, verbose=verbose)[0]
    
    if ('clipping_charge:value' in settings and 'clipping_charge:units' in settings):
        clipping_charge = settings['clipping_charge:value'] * unit_registry.parse_expression(settings['clipping_charge:units'])
        clipping_charge = clipping_charge.to('coulomb').magnitude
        clip_to_charge(restart_particles, clipping_charge)
        
    G = GPT(gpt_bin=gpt_bin, input_file=gpt_input_file, initial_particles=restart_particles, workdir=workdir, use_tempdir=use_tempdir)
    G.timeout = timeout
    G.verbose = verbose
    for k, v in settings.items():
        G.set_variable(k,v)
    G.set_variables(phased_settings)
    G.set_variable('multi_run',2)
    G.set_variable('last_run',2)
    G.set_variable('t_start', t_restart)
    G.run(gpt_verbose=gpt_verbose)
        
    G_all.output['particles'][G_all.output['n_tout']:G_all.output['n_tout']] = G.tout
    G_all.output['particles'] = G_all.output['particles'] + G.screen
    G_all.output['n_tout'] = G_all.output['n_tout']+G.output['n_tout']
    G_all.output['n_screen'] = G_all.output['n_screen']+G.output['n_screen']
    
    if ('final_charge:value' in settings and 'final_charge:units' in settings and len(G_all.screen)>0):
        final_charge = settings['final_charge:value'] * unit_registry.parse_expression(settings['final_charge:units'])
        final_charge = final_charge.to('coulomb').magnitude
        clip_to_charge(G_all.screen[-1], final_charge)
        
    if (input_particle_group['sigma_t'] == 0.0):
        # Initial distribution is a tout
        if (G_all.output['n_tout'] > 0):
            # Don't include the cathode if there are no other screens. Screws up optimizations of "final" screen when there is an error
            G_all.output['particles'].insert(0, input_particle_group)
            G_all.output['n_tout'] = G_all.output['n_tout']+1
    else:
        # Initial distribution is a screen
        if (G_all.output['n_screen'] > 0):
            # Don't include the cathode if there are no other screens. Screws up optimizations of "final" screen when there is an error
            G_all.output['particles'].insert(G_all.output['n_tout'], input_particle_group)
            G_all.output['n_screen'] = G_all.output['n_screen']+1
        
    return G_all

    

def evaluate_multirun_gpt_with_particlegroup(settings,
                                             archive_path=None,
                                             merit_f=None, 
                                             gpt_input_file=None,
                                             distgen_input_file=None,
                                             workdir=None, 
                                             use_tempdir=True,
                                             gpt_bin='$GPT_BIN',
                                             timeout=2500,
                                             auto_phase=False,
                                             verbose=False,
                                             gpt_verbose=False,
                                             asci2gdf_bin='$ASCI2GDF_BIN'):    
    """
    Will raise an exception if there is an error. 
    """
    if ('final_charge' in settings and 'coreshield:core_charge_fraction' not in settings):
        settings['coreshield:core_charge_fraction'] = 0.5
        
    if ('coreshield' not in settings):
        input_particle_group = get_cathode_particlegroup(settings, distgen_input_file, verbose=verbose)
    else:
        input_particle_group = get_coreshield_particlegroup(settings, distgen_input_file, verbose=verbose)
    
    G = multirun_gpt_with_particlegroup(settings=settings,
                             gpt_input_file=gpt_input_file,
                             input_particle_group=input_particle_group,
                             workdir=workdir, 
                             use_tempdir=use_tempdir,
                             gpt_bin=gpt_bin,
                             timeout=timeout,
                             auto_phase=auto_phase,
                             verbose=verbose,
                             gpt_verbose=gpt_verbose,
                             asci2gdf_bin=asci2gdf_bin)
                        
    if merit_f:
        output = merit_f(G)
    else:
        output = default_gpt_merit(G)
    
    for k in settings:
        output[k] = settings[k]
    
    if output['error']:
        raise ValueError('error occured!')
        
    #Recreate Generator object for fingerprint, proper archiving
    # TODO: make this cleaner
    gen = Generator()
    
    output['fingerprint'] = G.fingerprint()    
    
    if archive_path:
        path = tools.full_path(archive_path)
        assert os.path.exists(path), f'archive path does not exist: {path}'
        archive_file = os.path.join(path, fingerprint+'.h5')
        output['archive'] = archive_file
        
        # Call the composite archive method
        archive_gpt_with_distgen(G, gen, archive_file=archive_file)          
        
    return output




def run_gpt_with_particlegroup(settings=None,
                             gpt_input_file=None,
                             input_particle_group=None,
                             workdir=None, 
                             use_tempdir=True,
                             gpt_bin='$GPT_BIN',
                             timeout=2500,
                             auto_phase=False,
                             verbose=False,
                             gpt_verbose=False,
                             asci2gdf_bin='$ASCI2GDF_BIN'
                             ):
    """
    Run gpt with particles from ParticleGroup. 
    
        settings: dict with keys that are in gpt input file.    
        
    """

    # Call simpler evaluation if there is no input_particle_group:
    if (input_particle_group is None):
        return run_gpt(settings=settings, 
                       gpt_input_file=gpt_input_file, 
                       workdir=workdir,
                       use_tempdir=use_tempdir,
                       gpt_bin=gpt_bin, 
                       timeout=timeout, 
                       verbose=verbose)
    
    if(verbose):
        print('Run GPT with ParticleGroup:') 

    unit_registry = UnitRegistry()
        
    # Make gpt and generator objects
    G = GPT(gpt_bin=gpt_bin, input_file=gpt_input_file, initial_particles=input_particle_group, workdir=workdir, use_tempdir=use_tempdir)
    G.timeout=timeout
    G.verbose = verbose

    # Set inputs
    if settings:
        for k, v in settings.items():
            G.set_variable(k,v)
            
    if ('final_charge' in settings):
        raise ValueError('final_charge is deprecated, please specify value and units instead.')
            
    # Run
    if(auto_phase): 

        if(verbose):
            print('\nAuto Phasing >------\n')
        t1 = time.time()

        # Create the distribution used for phasing
        if(verbose):
            print('****> Creating initial distribution for phasing...')

        phasing_beam = get_distgen_beam_for_phasing_from_particlegroup(input_particle_group, n_particle=10, verbose=verbose)
        phasing_particle_file = os.path.join(G.path, 'gpt_particles.phasing.gdf')
        write_gpt(phasing_beam, phasing_particle_file, verbose=verbose, asci2gdf_bin=asci2gdf_bin)
    
        if(verbose):
            print('<**** Created initial distribution for phasing.\n')    

        G.write_input_file()   # Write the unphased input file

        phased_file_name, phased_settings = gpt_phasing(G.input_file, path_to_gpt_bin=G.gpt_bin[:-3], path_to_phasing_dist=phasing_particle_file, verbose=verbose)
        G.set_variables(phased_settings)
        t2 = time.time()

        if(verbose):
            print(f'Time Ellapsed: {t2-t1} sec.')
            print('------< Auto Phasing\n')


    # If here, either phasing successful, or no phasing requested
    G.run(gpt_verbose=gpt_verbose)
    
    if ('final_charge:value' in settings and 'final_charge:units' in settings and len(G.screen)>0):
        final_charge = settings['final_charge:value'] * unit_registry.parse_expression(settings['final_charge:units'])
        final_charge = final_charge.to('coulomb').magnitude
        clip_to_charge(G.screen[-1], final_charge)
    
    if (input_particle_group['sigma_t'] == 0.0):
        # Initial distribution is a tout
        if (G.output['n_tout'] > 0):
            G.output['particles'].insert(0, input_particle_group)
            G.output['n_tout'] = G.output['n_tout']+1
    else:
        # Initial distribution is a screen
        if (G.output['n_screen'] > 0):
            G.output['particles'].insert(G.output['n_tout'], input_particle_group)
            G.output['n_screen'] = G.output['n_screen']+1
    
    
    return G





def evaluate_run_gpt_with_particlegroup(settings,
                                         archive_path=None,
                                         merit_f=None, 
                                         gpt_input_file=None,
                                         distgen_input_file=None,
                                         workdir=None, 
                                         use_tempdir=True,
                                         gpt_bin='$GPT_BIN',
                                         timeout=2500,
                                         auto_phase=False,
                                         verbose=False,
                                         gpt_verbose=False,
                                         asci2gdf_bin='$ASCI2GDF_BIN'):    
    """
    Will raise an exception if there is an error. 
    """
    if ('final_charge' in settings and 'coreshield:core_charge_fraction' not in settings):
        settings['coreshield:core_charge_fraction'] = 0.5
        
    if ('coreshield' not in settings):
        input_particle_group = get_cathode_particlegroup(settings, distgen_input_file, verbose=verbose)
    else:
        input_particle_group = get_coreshield_particlegroup(settings, distgen_input_file, verbose=verbose)
    
    G = run_gpt_with_particlegroup(settings=settings,
                         gpt_input_file=gpt_input_file,
                         input_particle_group=input_particle_group,
                         workdir=workdir, 
                         use_tempdir=use_tempdir,
                         gpt_bin=gpt_bin,
                         timeout=timeout,
                         auto_phase=auto_phase,
                         verbose=verbose,
                         gpt_verbose=gpt_verbose,
                         asci2gdf_bin=asci2gdf_bin)
        
    if merit_f:
        output = merit_f(G)
    else:
        output = default_gpt_merit(G)
    
    for k in settings:
        output[k] = settings[k]
    
    if output['error']:
        raise ValueError('error occured!')
        
    #Recreate Generator object for fingerprint, proper archiving
    # TODO: make this cleaner
    gen = Generator()
    
    output['fingerprint'] = G.fingerprint()    
    
    if archive_path:
        path = tools.full_path(archive_path)
        assert os.path.exists(path), f'archive path does not exist: {path}'
        archive_file = os.path.join(path, fingerprint+'.h5')
        output['archive'] = archive_file
        
        # Call the composite archive method
        archive_gpt_with_distgen(G, gen, archive_file=archive_file)          
        
    return output




def get_distgen_beam_for_phasing_from_particlegroup(PG, n_particle=10, verbose=False):

    variables = ['x', 'y', 'z', 'px', 'py', 'pz', 't']

    transforms = { f'avg_{var}':{'type': f'set_avg {var}', f'avg_{var}': { 'value': PG['mean_'+var], 'units':  PG.units(var).unitSymbol  } } for var in variables }

    phasing_distgen_input = {'n_particle':10, 'random_type':'hammersley', 'transforms':transforms,
                             'total_charge':{'value':0.0, 'units':'C'},
                             'start': {'type':'time', 'tstart':{'value': 0.0, 'units': 's'}},}
    
    gen = Generator(phasing_distgen_input, verbose=verbose) 
    pbeam = gen.beam()

    return pbeam


