#!/usr/bin/env python
# -*- coding: utf-8 -*-

import warnings
import os
import io
import re

try:
  from setuptools import setup
  from setuptools.command.install import install
  setup
except ImportError:
  from distutils.core import setup
  setup

# Get the long description from the README
def readme():
  with open('README.rst') as f:
    return f.read()

# Read, version funcs taken from:
# https://github.com/ellisonbg/altair/blob/master/setup.py
def read(path, encoding='utf-8'):
    path = os.path.join(os.path.dirname(__file__), path)
    with io.open(path, encoding=encoding) as fp:
        return fp.read()

def version(path):
    """
    Obtain the packge version from a python file e.g. pkg/__init__.py
    See <https://packaging.python.org/en/latest/single_source_version.html>.
    """
    version_file = read(path)
    version_match = re.search(r"""^__version__ = ['"]([^'"]*)['"]""",
                              version_file, re.M)
    if version_match:
        return version_match.group(1)
    raise RuntimeError("Unable to find version string.")


VERSION = version('trappist/__init__.py')

# Setup!
setup(name = 'trappist',
      version = VERSION,
      description = 'Constraining the stellar evolution of TRAPPIST-1 using MCMC',
      long_description = readme(),
      classifiers = [
                      'Development Status :: 3 - Alpha',
                      'License :: OSI Approved :: MIT License',
                      'Programming Language :: Python',
                      'Programming Language :: Python :: 3.6',
                      'Topic :: Scientific/Engineering :: Astronomy',
                    ],
      url = 'https://github.com/dflemin3/trappist',
      author = 'David P. Fleming',
      author_email = 'dflemin3@uw.edu',
      license = 'MIT',
      packages = ['trappist'],
      install_requires = [
                          'numpy',
                          'matplotlib >= 3.0.0',
                          'scipy',
                          'emcee == 3.0rc2',
                          'corner',
                          'vplot'
                          ],
      include_package_data = True,
      zip_safe = False)
