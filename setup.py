from setuptools import setup

#from setuptools import find_packages  
#from os import path
#from codecs import open  # To use a consistent encoding
#here = path.abspath(path.dirname(__file__))
#
# Get the long description from the relevant file
#with open(path.join(here, 'README'), encoding='utf-8') as f:
#    long_description = f.read()

from ttim import __version__

l_d = ''
try:
   import pypandoc
   l_d = pypandoc.convert('README.md', 'rst')
except:
   pass  


setup(
    name='ttim',

    # Versions should comply with PEP440.  For a discussion on single-sourcing
    # the version across setup.py and the project code, see
    # http://packaging.python.org/en/latest/tutorial.html#version
    version = __version__,

    description = 'Transient multi-layer analytic element model',
    long_description = l_d,

    # The project's main homepage.
    url = 'https://github.com/mbakker7/ttim',

    # Author details
    author='Mark Bakker',
    author_email='markbak@gmail.com',

    # Choose your license
    license = 'MIT',

    # See https://PyPI.python.org/PyPI?%3Aaction=list_classifiers
    classifiers=[
        'Development Status :: 5 - Production/Stable',
        # Indicate who your project is intended for
        #'Intended Audience :: Groundwater Modelers',
        # Pick yor license as you wish (should match "license" above)
        'License :: OSI Approved :: MIT License',
        # Specify the Python versions you support here. In particular, ensure
        # that you indicate whether you support Python 2, Python 3 or both.
        'Programming Language :: Python :: 2.7'
        ],
    platforms='Windows, Mac OS-X',
    install_requires=['numpy>=1.9', 'matplotlib>=1.4'],
    packages=['ttim'],
    include_package_data = True,
    package_data = {'ttim': ['bessel.f95', 'bessel.so', 'bessel.pyd', 'invlap.f90', 'invlap.so', 'invlap.pyd']}
    )