#! /usr/bin/python
# -*- coding: utf-8 -*-

from onnx import helper, TensorProto, numpy_helper
from ..op_mapper import OpMapper
from ...common import make_node, to_numpy
from ..datatype_mapping import NP_TYPE_TO_TENSOR_TYPE
import numpy as np


@OpMapper('OneHot')
class OneHot():
    # supports v9-v11

    @classmethod
    def version_9(cls, node, **kwargs):
        onnx_node = []
        onnx_value = []
        onnx_init = []

        # input, output
        x = node['in_nodes_name'][0]
        x_shape = node['in_tensors'][0]
        y = node['out_nodes_name'][0]
        out_shape = node['out_tensors'][0]
        out_v = helper.make_tensor_value_info(node['out_nodes_name'][0], NP_TYPE_TO_TENSOR_TYPE[node['dtype']],
                                              shape=out_shape)
        onnx_value.append(out_v)

        # attr
        _depth = node['node'].layer.depth
        _on_value = node['node'].layer.on_value
        _off_value = node['node'].layer.off_value
        axis = node['node'].layer.axis

        # create attr
        values = numpy_helper.from_array(np.array([_off_value, _on_value], dtype=np.int64), name='values')
        onnx_init.append(values)
        depth = numpy_helper.from_array(np.array(_depth, dtype=np.int64), name='depth')
        onnx_init.append(depth)

        if node['dtype'] == np.int64:
            x_hot = helper.make_tensor_value_info(y + '_hot', TensorProto.INT64, shape=out_shape)
            onnx_value.append(x_hot)
            oht_node, out = make_node('OneHot', inputs=[x, 'depth', 'values'], outputs=[y + '_hot'], axis=axis)
            onnx_node.append(oht_node)
        else:
            # make cast input to int64
            cxv = helper.make_tensor_value_info(x + '_cast', TensorProto.INT64, shape=x_shape)
            onnx_value.append(cxv)
            cxn, x = make_node('Cast', inputs=[x], outputs=[x + '_cast'], to=TensorProto.INT64)
            onnx_node.append(cxn)

            x_hot = helper.make_tensor_value_info(y + '_hot', TensorProto.INT64, shape=out_shape)
            onnx_value.append(x_hot)
            oht_node, out = make_node('OneHot', inputs=[x, 'depth', 'values'], outputs=[y + '_hot'], axis=axis)
            onnx_node.append(oht_node)

        # cast output to dtype
        out_node, _ = make_node('Cast', inputs=[out], outputs=[y], to=TensorProto.FLOAT)
        onnx_node.append(out_node)

        return onnx_node, onnx_value, onnx_init


@OpMapper('Embedding')
class Embedding():
    # supports v1-v15

    @classmethod
    def version_1(cls, node, **kwargs):
        onnx_node = []
        onnx_value = []
        onnx_init = []

        # input, output
        x = node['in_nodes_name'][0]
        x_shape = node['in_tensors'][0]
        y = node['out_nodes_name'][0]
        out_shape = node['out_tensors'][0]
        out_v = helper.make_tensor_value_info(node['out_nodes_name'][0], NP_TYPE_TO_TENSOR_TYPE[node['dtype']],
                                              shape=out_shape)
        onnx_value.append(out_v)

        # attr
        embeddings = node['node'].layer.embeddings
        e_weights = numpy_helper.from_array(arr=to_numpy(embeddings), name='embeddings')
        onnx_init.append(e_weights)

        # make gather node
        g_node, _ = make_node('Gather', inputs=['embeddings', x], outputs=[y])
        onnx_node.append(g_node)

        return onnx_node, onnx_value, onnx_init