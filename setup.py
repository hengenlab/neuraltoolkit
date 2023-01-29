from setuptools import setup
import subprocess

with open("README.md", 'r') as f:
    long_description = f.read()

try:
    process = subprocess.check_output(['git', 'describe'], shell=False)
    __git_version__ = process.strip().decode('ascii')
except Exception as e:
    print(e)
    __git_version__ = 'unknown'
print("__git_version__ ", __git_version__)
try:
    with open('neuraltoolkit/_version.py', 'w') as fp:
        fp.write("__git_version__ = '" + str(__git_version__) + "'")
except Exception as e:
    print("Error : ", e)

setup(
   name='neuraltoolkit',
   version='0.3.0',
   description='A powerful and fast set of tools for loading data, filtering,\
                processing, working with data formats, and basic utilities for\
                electrophysiology and behavioral data.',
   license="",
   keywords='neuraltoolkit, neuroscience, electrophysiology',
   package_dir={'neuraltoolkit': 'neuraltoolkit'},
   author='\
           (Hengen Lab Washington University in St. Louis)',
   author_email='',
   maintainer='Kiran Bhaskaran-Nair,\
           (Hengen Lab Washington University in St. Louis)',
   maintainer_email='',
   url="https://github.com/hengenlab/neuraltoolkit",
   download_url="https://github.com/hengenlab/neuraltoolkit",
   packages=['neuraltoolkit'],
   install_requires=['ipython', 'numpy', 'matplotlib', 'seaborn', 'pandas',
                     'opencv-python', 'scipy', 'h5py', 'tables'],
   classifiers=[
        'Natural Language :: English',
        'Programming Language :: Python :: 3',
    ],
   scripts=[
           ]
)
