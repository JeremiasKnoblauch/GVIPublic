#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function
from setuptools import setup

setup(name='gpflow-monitor',
      version="0.1",
      author="Mark van der Wilk",
      author_email="mark@prowler.io",
      description="An optimisation monitoring framework tailored to GPflow.",
      license="Apache License 2.0",
      url="http://github.com/markvdw/gpflow-monitor",
      ext_modules=[],
      packages=["gpflow_monitor"],
      test_suite='testing',
      install_requires=['gpflow>=0.9.8'])
