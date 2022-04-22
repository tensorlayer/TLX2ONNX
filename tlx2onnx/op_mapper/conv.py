#! /usr/bin/python
# -*- coding: utf-8 -*-

def convert_tlx_conv(tlx_node):
    """

    Parameters
    ----------
    tlx_node

    Returns
    -------

    """

    inputs = tlx_node.in_tensors
    outputs = tlx_node.out_tensors
    node_name = tlx_node.node_name
    in_nodes_name = tlx_node.in_nodes_name
    out_nodes_name = tlx_node.out_nodes_name

    innodes = tlx_node.in_nodes
    outnodes = tlx_node.out_nodes
    node_param = tlx_node.layer._params

    return onnx_node, onnx_value, onnx_init

