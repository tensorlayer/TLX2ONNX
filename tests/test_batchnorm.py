#! /usr/bin/python
# -*- coding: utf-8 -*-
# TODO The output of TLX and ONNX is inconsistent.
import os
os.environ["TL_BACKEND"] = 'tensorflow'
import tensorlayerx as tlx
from tensorlayerx.nn import Module
from tensorlayerx.nn import Conv2d, BatchNorm2d
from tlx2onnx.main import export
import onnxruntime as rt
import numpy as np


class MLP(Module):
    def __init__(self):
        super(MLP, self).__init__()
        # weights init
        self.conv1 = Conv2d(out_channels=16, kernel_size=3, stride=1, padding=(2, 2), in_channels=3, data_format='channels_last', act = tlx.nn.ReLU)
        self.bn = BatchNorm2d(data_format='channels_last')


    def forward(self, x):
        x = self.conv1(x)
        x = self.bn(x)
        return x

net = MLP()
net.set_eval()
input = tlx.nn.Input(shape=(4, 20, 20, 3))
onnx_model = export(net, input_spec=input, path='batchnorm.onnx')
print("tlx out", net(input))

# Infer Model
sess = rt.InferenceSession('batchnorm.onnx')

input_name = sess.get_inputs()[0].name
output_name = sess.get_outputs()[0].name

input_data = tlx.nn.Input(shape=(4, 20, 20, 3))
input_data = np.array(input_data, dtype=np.float32)

result = sess.run([output_name], {input_name: input_data})
print("onnx out", result)