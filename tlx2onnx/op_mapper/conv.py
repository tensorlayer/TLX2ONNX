#! /usr/bin/python
# -*- coding: utf-8 -*-

def convert_tlx_conv(tlx_node):
    """

    Parameters
    ----------
    tlx_node:node dict {node: node,
                        in_tensors: node inputs,
                        out_tensors: node outputs,
                        in_nodes_name: node inputs name,
                        out_nodes_name: node outputs name}

    Returns
    -------

    """

    inputs = tlx_node['in_tensors']
    outputs = tlx_node['out_tensors']
    in_nodes_name = tlx_node['in_nodes_name']
    out_nodes_name = tlx_node['out_nodes_name']
    cur_node = tlx_node['node']

    node_name = cur_node.node_name
    innodes = tlx_node.in_nodes
    outnodes = tlx_node.out_nodes
    node_param = tlx_node.layer._params

    return onnx_node, onnx_value, onnx_init

