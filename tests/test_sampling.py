#! /usr/bin/python
# -*- coding: utf-8 -*-

import os
os.environ["TL_BACKEND"] = 'torch'
import tensorlayerx as tlx
from tensorlayerx.nn import Module
from tensorlayerx.nn import UpSampling2d, DownSampling2d
from tlx2onnx.main import export
import onnxruntime as rt
import numpy as np


class MLP(Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.upsampling = UpSampling2d(scale=(2, 2), method='bilinear', data_format='channels_first')
        self.downsampling = DownSampling2d(scale=(2, 2), method='bilinear', data_format='channels_first')

    def forward(self, x):
        x = self.upsampling(x)
        x = self.downsampling(x)
        return x


net = MLP()
input = tlx.nn.Input(shape=(3, 3, 5, 5), init=tlx.initializers.RandomNormal())
net.set_eval()
output = net(input)
print("tlx out", output.shape)
onnx_model = export(net, input_spec=input, path='sampling.onnx')

# Infer Model
sess = rt.InferenceSession('sampling.onnx')

input_name = sess.get_inputs()[0].name
output_name = sess.get_outputs()[0].name

input_data = np.array(input, dtype=np.float32)

result = sess.run([output_name], {input_name: input_data})
print('onnx out', np.shape(result))