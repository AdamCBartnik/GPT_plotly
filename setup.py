
from setuptools import setup, find_packages
from os import path, environ

cur_dir = path.abspath(path.dirname(__file__))

with open(path.join(cur_dir, 'requirements.txt'), 'r') as f:
    requirements = f.read().split()



setup(
    name='GPT_plotly',
    version = 'v0.2.0',
    packages=find_packages(), 
    package_dir={'GPT_plotly':'GPT_plotly'},
    url='https://github.com/AdamCBartnik/GPT_plotly',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    install_requires=requirements,
    include_package_data=True,
    python_requires='>=3.6'
)
