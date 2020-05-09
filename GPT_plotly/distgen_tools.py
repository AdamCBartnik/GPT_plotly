import yaml
from distgen import Generator
from distgen.tools import update_nested_dict

def run_distgen_with_settings(settings, DISTGEN_INPUT_FILE):
    distgen_input = yaml.safe_load(open(DISTGEN_INPUT_FILE))
    for k, v in settings.items():
        distgen_input = update_nested_dict(distgen_input, {k:v}, verbose=True, create_new=False)
    gen = Generator(distgen_input,verbose=False)
    gen.run()
    return gen.particles
