#! /usr/bin/python
# -*- coding: utf-8 -*-
# The computed graph of TLX returns the information needed to build ONNX.


import tensorlayerx as tlx

def memory_node_info(node):
    node_info = {}
    if node.layer.__class__.__name__ in tlx.nn.inputs.__all__:
        node_info['in_tensors'] = None
        node_info['out_tensors'] = [list(out_tensor.shape) for out_tensor in node.out_tensors]
        node_info['node'] = node
        node_info['in_nodes_name'] = None
        node_info['out_nodes_name'] = [node.node_name + str(idx) for idx, onode in enumerate(node.out_tensors)]
        node_info['dtype'] = tlx.convert_to_numpy(node.out_tensors[0]).dtype
        node_info['attr'] = node.attr
    else:
        node_info['in_tensors'] = [list(in_tensor.shape) for in_tensor, idx in zip(node.in_tensors, node.in_tensors_idxes)]
        node_info['out_tensors'] = [list(out_tensor.shape) for out_tensor in node.out_tensors]
        node_info['node'] = node
        node_info['in_nodes_name'] = [inode.node_name + str(idx) for inode, idx in zip(node.in_nodes, node.in_tensors_idxes)]
        node_info['out_nodes_name'] = [node.node_name + str(id) for id, onode in enumerate(node.out_tensors)]
        node_info['dtype'] = tlx.convert_to_numpy(node.in_tensors[0]).dtype
        node_info['attr'] = node.attr
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


