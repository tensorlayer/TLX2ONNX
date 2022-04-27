#! /usr/bin/python
# -*- coding: utf-8 -*-


MAJOR = 0
MINOR = 0
PATCH = 1
PRE_RELEASE = ''
# Use the following formatting: (major, minor, patch, prerelease)
VERSION = (MAJOR, MINOR, PATCH, PRE_RELEASE)

__shortversion__ = '.'.join(map(str, VERSION[:3]))
__version__ = '.'.join(map(str, VERSION[:3])) + ''.join(VERSION[3:])
__author__ = "TensorLayerX Contributors"
__producer__ = "tlx2onnx"
__description__ = 'This package converts TensorlayerX models into ONNX for use with any inference engine supporting ONNX.'
__repository_url__ = 'https://github.com/tensorlayer/TLX2ONNX'
__download_url__ = 'https://github.com/tensorlayer/TLX2ONNX'
__license__ = 'apache'
__keywords__ = 'tensorlayerx, onnx, deep learning'


from .main import export
