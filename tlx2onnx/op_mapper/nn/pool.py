#! /usr/bin/python
# -*- coding: utf-8 -*-

from onnx import helper, numpy_helper
from collections import OrderedDict
import tensorlayerx as tlx
from tlx2onnx.op_mapper.datatype_mapping import NP_TYPE_TO_TENSOR_TYPE
from tlx2onnx.op_mapper.op_mapper import OpMapper
from tlx2onnx.common import make_node