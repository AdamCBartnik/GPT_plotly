import time, os
from gpt import GPT
from gpt.gpt_phasing import gpt_phasing
from distgen import Generator
from distgen.writers import write_gpt


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

    # Make gpt and generator objects
    G = GPT(gpt_bin=gpt_bin, input_file=gpt_input_file, initial_particles=input_particle_group, workdir=workdir, use_tempdir=use_tempdir)
    G.timeout=timeout
    G.verbose = verbose

    # Set inputs
    if settings:
        for k, v in settings.items():
            G.set_variable(k,v)
            
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
    
    if (input_particle_group['sigma_t'] == 0.0):
        # Initial distribution is a tout
        G.output['particles'].insert(0, input_particle_group)
        G.output['n_tout'] = G.output['n_tout']+1
    else:
        # Initial distribution is a screen
        G.output['particles'].insert(G.output['n_tout'], input_particle_group)
        G.output['n_screen'] = G.output['n_screen']+1
    
    
    return G




def get_distgen_beam_for_phasing_from_particlegroup(PG, n_particle=10, verbose=False):

    variables = ['x', 'y', 'z', 'px', 'py', 'pz', 't']

    transforms = { f'avg_{var}':{'type': f'set_avg {var}', f'avg_{var}': { 'value': PG['mean_'+var], 'units':  PG.units(var).unitSymbol  } } for var in variables }

    phasing_distgen_input = {'n_particle':10, 'random_type':'hammersley', 'transforms':transforms,
                             'total_charge':{'value':0.0, 'units':'C'},
                             'start': {'type':'time', 'tstart':{'value': 0.0, 'units': 's'}},}
    
    gen = Generator(phasing_distgen_input, verbose=verbose) 
    pbeam = gen.beam()

    return pbeam

