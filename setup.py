#! /usr/bin/python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from setuptools import setup, find_packages


MAJOR = 0
MINOR = 0
PATCH = 1
PRE_RELEASE = 'alpha'
# Use the following formatting: (major, minor, patch, prerelease)
VERSION = (MAJOR, MINOR, PATCH, PRE_RELEASE)

long_description = "tlx2onnx is a toolkit for converting trained model of TensorLayerX to ONNX. \n\n"
long_description += "Usage: export(tlx_model, input_spec=input, path='model.onnx') \n"
long_description += "GitHub: https://github.com/tensorlayer/TLX2ONNX \n"
long_description += "Email: tensorlayer@gmail.com"

setup(
    name="tlx2onnx",
    version='.'.join(map(str, VERSION[:3])) + ''.join(VERSION[3:]),
    author="TLX2ONNX Contributors",
    author_email="tensorlayer@gmail.com",
    description="a toolkit for converting trained model of TensorLayerX to ONNX.",
    long_description=long_description,
    long_description_content_type="text/plain",
    url="https://github.com/tensorlayer/TLX2ONNX",
    packages=find_packages(),
    install_requires=['tensorlayerx>=0.5.1', 'onnx<=1.11.0'],
    classifiers=[
        # How mature is this project? Common values are
        #  1 - Planning 2 - Pre-Alpha 3 - Alpha 4 - Beta 5 - Production/Stable 6 - Mature 7 - Inactive
        'Development Status :: 3 - Alpha',

        # Specify the Python versions you support here.
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.5",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",

        # Indicate who your project is intended for
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Information Technology",

        # Pick your license as you wish (should match "license" above)
        "License :: OSI Approved :: Apache Software License",

        # Indicate what your project relates to
        'Topic :: Scientific/Engineering :: Artificial Intelligence',

        # Additionnal Settings
        "Operating System :: POSIX",
        "Operating System :: Microsoft :: Windows",
    ],
    license='Apache 2.0',
    )
