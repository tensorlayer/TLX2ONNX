#! /usr/bin/python
# -*- coding: utf-8 -*-
# The computed graph of TLX returns the information needed to build ONNX.


import tensorlayerx as tlx

def memory_node_info(node):
    node_info = {}
    if node.layer.__class__.__name__ in tlx.nn.inputs.__all__:
        node_info['in_tensors'] = None
        node_info['out_tensors'] = list(node.out_tensors[0].shape)
        node_info['node'] = node
        node_info['in_nodes_name'] = None
        node_info['out_nodes_name'] = [node.node_name + str(idx) for idx, inode in enumerate(node.out_tensors)]
    else:
        node_info['in_tensors'] = list(node.in_tensors[0].shape)
        node_info['out_tensors'] = list(node.out_tensors[0].shape)
        node_info['node'] = node
        node_info['in_nodes_name'] = [node.in_nodes[0].node_name + str(idx) for inode, idx in
                                      zip(node.in_nodes, node.in_tensors_idxes)]
        node_info['out_nodes_name'] = [node.node_name + str(idx) for idx, inode in enumerate(node.out_tensors)]
    return node_info


def construct_topology(model, inputs):
    """

    Parameters
    ----------
    model: TensorLayerX model,
    inputs

    Returns
    -------

    """

    node_by_depth, all_layers = model.build_graph(inputs)

    memory = dict()
    # get each layer's by going through the graph in depth order
    for depth, nodes in enumerate(node_by_depth):
        if depth == 0:
            if isinstance(inputs, list):
                assert len(inputs) == len(nodes)
                for idx, node in enumerate(nodes):
                    memory[node.node_name] = memory_node_info(node)
            else:
                memory[nodes[0].node_name] = memory_node_info(nodes[0])
        else:
            for node in nodes:
                memory[node.node_name] = memory_node_info(node)
    return memory

