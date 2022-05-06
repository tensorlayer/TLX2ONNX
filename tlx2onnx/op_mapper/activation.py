#! /usr/bin/python
# -*- coding: utf-8 -*-

from onnx import helper
from .op_mapper import OpMapper
from ..common import make_node


@OpMapper(["ReLU"])
class Relu():

    @classmethod
    def version_1(cls, node = None, **kwargs):
        if node is not None :
            # TODO : use act as a layer node.
            pass
        return make_node("Relu", **kwargs)




