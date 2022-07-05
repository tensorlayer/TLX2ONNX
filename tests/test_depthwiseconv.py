#! /usr/bin/python
# -*- coding: utf-8 -*-

import os
os.environ["TL_BACKEND"] = 'torch'
import tensorlayerx as tlx
from tensorlayerx.nn import Module
from tensorlayerx.nn import DepthwiseConv2d
from tlx2onnx.main import export
import onnxruntime as rt
import numpy as np


class MLP(Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.dpconv1 = DepthwiseConv2d(kernel_size = (2, 2), stride = (1, 1), dilation = 2, padding='SAME',
                                       depth_multiplier = 2, data_format='channels_first')

        self.dpconv2 = DepthwiseConv2d(kernel_size=(2, 2), stride=(1, 1), dilation=2, padding='VALID',
                                       depth_multiplier=2, data_format='channels_first')
    def forward(self, x):
        x = self.dpconv1(x)
        x = self.dpconv2(x)
        return x

net = MLP()
net.set_eval()
input = tlx.nn.Input(shape=(4, 3, 10, 10))
onnx_model = export(net, input_spec=input, path='dwconv.onnx')
print("tlx out", net(input))

# Infer Model
sess = rt.InferenceSession('dwconv.onnx')

input_name = sess.get_inputs()[0].name
output_name = sess.get_outputs()[0].name

input_data = np.array(input, dtype=np.float32)

result = sess.run([output_name], {input_name: input_data})
print("onnx out", result)