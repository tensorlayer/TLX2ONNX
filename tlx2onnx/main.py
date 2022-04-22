#! /usr/bin/python
# -*- coding: utf-8 -*-

import tensorlayerx as tlx
from onnx import helper, TensorProto
import onnx
from .topology import construct_topology

_layer = tlx.nn
tlx_layer_to_operator = {
    _layer.Conv1D: convert_tlx_conv1d,
    _layer.Conv2D: convert_tlx_conv2d,
    _layer.Conv3D: convert_tlx_conv3d,
    _layer.BatchNorm: convert_tlx_batch_norm,
}


def make_graph(*args, doc_string=None, **kwargs):
    graph = helper.make_graph(*args, doc_string=doc_string, **kwargs)
    if doc_string == '':
        graph.doc_string = ''
    return graph


def export(model, input_spec, path=None, export_params=False):
    """

    Parameters
    ----------
    model
    input_spec
    path
    export_params

    Returns
    -------

    """


    _topology = construct_topology(model, input_spec)

    onnx_ondes = []
    onnx_values = []
    onnx_weights = []

    for tlx_node in _topology:
        onnx_node, onnx_value, onnx_weight =tlx_layer_to_operator(tlx_node)
        onnx_ondes.extend(onnx_ondes)
        onnx_values.extend(onnx_value)
        onnx_weights.extend(onnx_weight)

    graph = make_graph(
        name='torch-jit-export',
        inputs=[helper.make_tensor_value_info('images', TensorProto.FLOAT, shape=[1, 3, 416, 416])],
        outputs=[helper.make_tensor_value_info('output', TensorProto.FLOAT, shape=[1, 3549, 85])],
        initializer=onnx_weights,
        value_info=onnx_values,
    )

    onnx.save(graph, path)





