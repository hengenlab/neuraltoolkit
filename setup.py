from setuptools import setup

with open("README.md", 'r') as f:
    long_description = f.read()

setup(
   name='neuraltoolkit',
   version='0.2.0',
   description='A powerful and fast set of tools for loading data, filtering,\
                processing, working with data formats, and basic utilities for\
                electrophysiology and behavioral data.',
   license="",
   long_description=long_description,
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
                     'opencv-python', 'scipy', 'h5py'],
   classifiers=[
        'Development Status :: 1 - Pre-Alpha',
        'Natural Language :: English',
        'Programming Language :: Python :: 3',
    ],
   scripts=[
           ]
)
