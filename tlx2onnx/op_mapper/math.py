#! /usr/bin/python
# -*- coding: utf-8 -*-

from onnx import helper, numpy_helper
from collections import OrderedDict
import tensorlayerx as tlx
from .datatype_mapping import NP_TYPE_TO_TENSOR_TYPE
import numpy as np
from .op_mapper import OpMapper

def order_repeated_field(repeated_proto, key_name, order):
    order = list(order)
    repeated_proto.sort(key=lambda x: order.index(getattr(x, key_name)))

def make_node(op_type, inputs, outputs, name=None, doc_string=None, domain=None, **kwargs):
    node = helper.make_node(op_type, inputs, outputs, name, doc_string, domain, **kwargs)
    if doc_string == '':
        node.doc_string = ''
    order_repeated_field(node.attribute, 'name', kwargs.keys())
    return node

#TODO : LOG, ADD, MATMUL,....... CONVERTER