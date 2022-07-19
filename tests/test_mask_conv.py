#! /usr/bin/python
# -*- coding: utf-8 -*-

import os
os.environ["TL_BACKEND"] = 'tensorflow'
import tensorlayerx as tlx
from tensorlayerx.nn import Module
from tensorlayerx.nn import MaskedConv3d
from tlx2onnx.main import export
import onnxruntime as rt
import numpy as np


class MLP(Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.mask_conv = MaskedConv3d(mask_type='B', out_channels=32, kernel_size=(1, 1, 1), stride=(2, 2, 2), act=tlx.ReLU, name='conv3d_2',
                                      in_channels=3, padding='SAME')

    def forward(self, x):
        x = self.mask_conv(x)
        return x

net = MLP()
input = tlx.nn.Input(shape=(5, 10, 10, 10, 3))
net.set_eval()
output = net(input)
print("tlx out", output)
onnx_model = export(net, input_spec=input, path='maskconv.onnx')

# Infer Model
sess = rt.InferenceSession('maskconv.onnx')

input_name = sess.get_inputs()[0].name
output_name = sess.get_outputs()[0].name

input_data = np.array(input, dtype=np.float32)

result = sess.run([output_name], {input_name: input_data})
print('onnx out', result)