#! /usr/bin/python
# -*- coding: utf-8 -*-

import tensorlayerx as tlx
from onnx import helper
from .topology import construct_topology
import onnx
from .op_mapper.op_mapper import OpMapper
from .common import make_graph, logging
from .op_mapper.datatype_mapping import NP_TYPE_TO_TENSOR_TYPE

def export(model, input_spec, path=None, enable_onnx_checker=True, opset_version = 9, dynamic_axes = None ,auto_update_opset=True):
    """

    Parameters
    ----------
    model : object
        TensorLayerX instantiate the net object.
    input_spec : tensor
        TensorLayerX Input.
    path : string
        ONNX file saving path
    enable_onnx_checker : bool
        Whether to enable ONNX model checker.
    opset_version : int
        The version of the default (ai.onnx) opset to target. Must be >= 7 and <= 17.
    dynamic_axes : list or tuple
        To specify axes of tensors as dynamic.
        By default the exported model will have fixed shapes of all input and output tensors.

    Returns
    -------
        ONNX model file

    Examples
    ---------
    >>> class NET(Module):
    >>> net = NET()
    >>> net.set_eval()
    >>> input = tlx.nn.Input([10, 50, 50, 32], name='input')
    >>> onnx_model = export(net, input_spec=input, path='vgg.onnx')

    """


    memory = construct_topology(model, input_spec)
    input_shape = memory[next(iter(memory))]['out_tensors'][0]
    output_shape = memory[list(memory.keys())[-1]]['out_tensors'][0]
    input_name = memory[next(iter(memory))]['out_nodes_name'][0]
    output_name = memory[list(memory.keys())[-1]]['out_nodes_name'][0]
    input_dtype = memory[next(iter(memory))]['in_dtype']
    output_dtype = memory[list(memory.keys())[-1]]['out_dtype']
    onnx_nodes = []
    onnx_values = []
    onnx_weights = []
    if auto_update_opset:
        opset_version = OpMapper.update_opset_version(memory, opset_version)
    else:
        OpMapper.check_support_version(memory, opset_version)

    for key in memory.keys():
        if memory[key]['node'].layer.__class__.__name__ not in tlx.nn.inputs.__all__:
            onnx_node, onnx_value, onnx_weight =OpMapper.mapping(memory[key], opset_version)
            onnx_nodes.extend(onnx_node)
            onnx_values.extend(onnx_value)
            onnx_weights.extend(onnx_weight)
        else:
            pass

    # Make Graph
    if dynamic_axes is None:
        graph = make_graph(
            name='tlx-graph-export',
            inputs=[helper.make_tensor_value_info(input_name, NP_TYPE_TO_TENSOR_TYPE[input_dtype], shape=input_shape)],
            outputs=[helper.make_tensor_value_info(output_name, NP_TYPE_TO_TENSOR_TYPE[output_dtype], shape=output_shape)],
            initializer=onnx_weights,
            value_info=onnx_values,
            nodes=onnx_nodes
        )
    else:
        graph = make_graph(
            name='tlx-graph-export',
            inputs=[helper.make_tensor_value_info(input_name, NP_TYPE_TO_TENSOR_TYPE[input_dtype], shape=input_shape)],
            outputs=[helper.make_tensor_value_info(output_name, NP_TYPE_TO_TENSOR_TYPE[output_dtype], shape=output_shape)],
            initializer=onnx_weights,
            nodes=onnx_nodes
        )
    # Make model
    model_def = helper.make_model(
        graph,
        producer_name='onnx-mode'
    )
    if dynamic_axes is not None:
        for i in dynamic_axes:
            model_def.graph.input[0].type.tensor_type.shape.dim[i].dim_param = '?'
            model_def.graph.output[0].type.tensor_type.shape.dim[i].dim_param = '?'
    if enable_onnx_checker:
        onnx.checker.check_model(model_def)
    onnx.save(model_def, path)
    logging.info("ONNX model saved in {}".format(path))
    return model_def


