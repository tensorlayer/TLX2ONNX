#! /usr/bin/python
# -*- coding: utf-8 -*-

from onnx import helper
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

@OpMapper(["ReLU"])
class Relu():

    @classmethod
    def version_1(cls, node = None, **kwargs):
        if node is not None :
            # TODO : use act as a layer node.
            pass
        return make_node("Relu", **kwargs)




