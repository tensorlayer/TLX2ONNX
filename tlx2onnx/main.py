#! /usr/bin/python
# -*- coding: utf-8 -*-

import tensorlayerx as tlx
from onnx import helper, TensorProto
import onnx
from .topology import construct_topology

_layer = tlx.nn
tlx_layer_to_operator = {
    _layer.UpSampling1D: convert_keras_upsample_1d,
    _layer.UpSampling2D: convert_keras_upsample_2d,
    _layer.UpSampling3D: convert_keras_upsample_3d,
    _layer.BatchNormalization: convert_keras_batch_normalization,
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





