#! /usr/bin/python
# -*- coding: utf-8 -*-

from onnx import helper, TensorProto, numpy_helper
from ..op_mapper import OpMapper
from ...common import make_node, to_numpy
from ..datatype_mapping import STR_TYPE_TO_TENSOR_TYPE

@OpMapper('Dropout')
class Dropout():
    # supports v1-v12

    @classmethod
    def version_1(cls, node, **kwargs):
        onnx_node = []
        onnx_value = []
        onnx_init = []