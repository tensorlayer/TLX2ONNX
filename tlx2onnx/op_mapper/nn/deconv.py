#! /usr/bin/python
# -*- coding: utf-8 -*-

from onnx import helper, TensorProto, numpy_helper
from ..op_mapper import OpMapper
from ...common import make_node, to_numpy
from ..datatype_mapping import NP_TYPE_TO_TENSOR_TYPE
from ...common import tlx_act_2_onnx


@OpMapper(['ConvTranspose1d', 'ConvTranspose2d', 'ConvTranspose3d'])
class ConvTranspose():
    # supports v1-v12

    @classmethod
    def version_1(cls, node, **kwargs):
        onnx_node = []
        onnx_value = []
        onnx_init = []

        x = node['in_nodes_name'][0]
        # TODO How to compatible multiple framework parameter names
        y = node['node'].layer.name + '/weights'
        weights = numpy_helper.from_array(arr=to_numpy(node['node'].layer.W), name=y)
        onnx_init.append(weights)

        dilations = node['attrs']['dilation']
        kernel_shape = node['attrs']['kernel_size']
        strides = node['attrs']['stride']
        pads = node['attrs']['padding']
        data_format = node['attrs']['data_format']

        if len(pads) == 2 or len(pads) == 3:
            pads = pads + pads
        elif len(pads) == 4:
            pads = [pads[i] for i in [0, 2, 1, 3]]
        elif len(pads) == 6:
            pads = [pads[i] for i in [0, 2, 4, 1, 3, 5]]

        if node['node'].layer.b_init is not None:
            b = numpy_helper.from_array(arr=to_numpy(node['node'].layer.b), name=node['node'].layer.name + '/b')
            onnx_init.append(b)
            b_name = node['node'].layer.name + '/b'

        if node['node'].layer.b_init is not None:
            # add bias
            if node['node'].layer.act is not None:
                # add activation
                pass
            else:
                # none activation
                pass
        else:
            # bias = None
            if node['node'].layer.act is not None:
                # add activation
                pass
            else:
                # none activation
                pass


