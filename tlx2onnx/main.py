#! /usr/bin/python
# -*- coding: utf-8 -*-

import tensorlayerx as tlx
from onnx import helper, TensorProto
import onnx
from .topology import construct_topology
from .op_mapper import *

_layer = tlx.nn
tlx_layer_to_operator = {
    _layer.Conv1d: convert_tlx_conv1d,
    _layer.Conv2d: convert_tlx_conv2d,
    _layer.Conv3d: convert_tlx_conv3d,
    _layer.ConvTranspose1d: convert_tlx_conv_transpose1d,
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


    memory = construct_topology(model, input_spec)
    input_shape = memory[next(iter(memory))]['out_tensors']
    output_shape = memory[list(memory.keys())[-1]]['out_tensors']
    input_name = memory[next(iter(memory))]['out_nodes_name']
    output_name = memory[list(memory.keys())[-1]]['out_nodes_name']

    onnx_ondes = []
    onnx_values = []
    onnx_weights = []

    for key in memory.keys():
        if memory[key]['node'].layer.__class__.__name__ not in tlx.nn.inputs.__all__:
            onnx_node, onnx_value, onnx_weight =tlx_layer_to_operator[memory[key]['node'].layer](memory[key])
            onnx_ondes.extend(onnx_node)
            onnx_values.extend(onnx_value)
            onnx_weights.extend(onnx_weight)
        else:
            pass

    graph = make_graph(
        name='tlx-graph-export',
        inputs=[helper.make_tensor_value_info(input_name, TensorProto.FLOAT, shape=input_shape)],
        outputs=[helper.make_tensor_value_info(output_name, TensorProto.FLOAT, shape=output_shape)],
        initializer=onnx_weights,
        value_info=onnx_values,
        nodes=onnx_ondes
    )

    onnx.save(graph, path)





