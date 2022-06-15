#! /usr/bin/python
# -*- coding: utf-8 -*-

import os
os.environ["TL_BACKEND"] = 'tensorflow'
import tensorlayerx as tlx
from tensorlayerx.nn import Module
from tensorlayerx.nn import Conv2d, ConvTranspose2d
from tlx2onnx.main import export
import onnxruntime as rt
import numpy as np


class MLP(Module):
    def __init__(self):
        super(MLP, self).__init__()
        # weights init
        self.conv1 = Conv2d(out_channels=16, kernel_size=3, stride=1, padding=(2, 2), in_channels=3, data_format='channels_last', act = tlx.nn.ReLU)
        self.deconv1 = ConvTranspose2d(out_channels=3, kernel_size=3, stride=1, padding=(2, 2), act=tlx.nn.ReLU, dilation=1, data_format='channels_last')
        self.conv2 = Conv2d(out_channels=16, kernel_size=3, stride=1, padding=(2, 2), in_channels=3, data_format='channels_last', act=tlx.nn.ReLU)


    def forward(self, x):
        x = self.conv1(x)
        x = self.deconv1(x)
        x = self.conv2(x)
        return x

net = MLP()
input = tlx.nn.Input(shape=(4, 20, 20, 3))
onnx_model = export(net, input_spec=input, path='deconv_model.onnx')
print(net(input).shape)

# Infer Model
sess = rt.InferenceSession('deconv_model.onnx')

input_name = sess.get_inputs()[0].name
output_name = sess.get_outputs()[0].name

input_data = tlx.nn.Input(shape=(4, 20, 20, 3))
input_data = np.array(input_data, dtype=np.float32)

result = sess.run([output_name], {input_name: input_data})
print(np.shape(result))