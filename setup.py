
import sys

import setuptools

from numpy.distutils.core import setup
from numpy.distutils.extension import Extension

# from setuptools import find_packages
# from os import path
# from codecs import open  # To use a consistent encoding
# here = path.abspath(path.dirname(__file__))
#
#  Get the long description from the relevant file
# with open(path.join(here, 'README'), encoding='utf-8') as f:
#     long_description = f.read()

version = open("VERSION").readline().strip()

l_d = ''
try:
    import pypandoc
    l_d = pypandoc.convert('README.md', 'rst')
except ImportError:
    pass

include_package_data = True
if sys.platform.startswith('win') or sys.platform.startswith('darwin'):
    ext_modules = []
else:
    ext_modules = [
        Extension('ttim.bessel',
                  ['ttim/bessel.pyf',
                   'ttim/bessel.f95']),
        Extension('ttim.invlap',
                  ['ttim/invlap.pyf',
                   'ttim/invlap.f90'])
    ]

setup(
    name='ttim',

    # Versions should comply with PEP440.  For a discussion on single-sourcing
    # the version across setup.py and the project code, see
    # http://packaging.python.org/en/latest/tutorial.html#version
    version=version,

    description='Transient multi-layer analytic element model',
    long_description=l_d,

    # The project's main homepage.
    url='https://github.com/mbakker7/ttim',

    # Author details
    author='Mark Bakker',
    author_email='markbak@gmail.com',

    # Choose your license
    license='MIT',

    # See https://PyPI.python.org/PyPI?%3Aaction=list_classifiers
    classifiers=[
        'Development Status :: 3 - Alpha',
        #  Indicate who your project is intended for
        # 'Intended Audience :: Groundwater Modelers',
        #  Pick yor license as you wish (should match "license" above)
        'License :: OSI Approved :: MIT License',
        #  Specify the Python versions you support here. In particular, ensure
        #  that you indicate whether you support Python 2, Python 3 or both.
        'Programming Language :: Python :: 3.6'
        ],
    platforms='Windows, Mac OS-X',
    install_requires=['numpy>=1.12', 'matplotlib>=2.0', 'lmfit>=0.9'],
    packages=['ttim'],
    include_package_data=include_package_data,
    package_data={'ttim': ['bessel.f95',
                           'invlap.f90',
                           'invlap.cpython-36m-darwin.so',
                           'bessel.cpython-36m-darwin.so',
                           'invlap.cp36-win_amd64.pyd',
                           'bessel.cp36-win_amd64.pyd']},
    data_files=[('ttim', ['VERSION'])],
    ext_modules=ext_modules,
    )
