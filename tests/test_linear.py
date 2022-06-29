#! /usr/bin/python
# -*- coding: utf-8 -*-

import os
os.environ["TL_BACKEND"] = 'tensorflow'
import tensorlayerx as tlx
from tensorlayerx.nn import Module
from tensorlayerx.nn import Linear, Dropout, Flatten, ReLU6
from tlx2onnx.main import export
import onnxruntime as rt
import numpy as np


class MLP(Module):
    def __init__(self):
        super(MLP, self).__init__()
        # weights init
        self.flatten = Flatten()
        self.line1 = Linear(in_features=32, out_features=64, act=tlx.nn.LeakyReLU(0.3))
        self.d1 = Dropout()
        self.line2 = Linear(in_features=64, out_features=128, b_init=None, act=tlx.nn.ReLU)
        self.relu6 = ReLU6()
        self.line3 = Linear(in_features=128, out_features=10, act=tlx.nn.ReLU)

    def forward(self, x):
        x = self.flatten(x)
        z = self.line1(x)
        z = self.d1(z)
        z = self.line2(z)
        z = self.relu6(z)
        z = self.line3(z)
        return z

net = MLP()
input = tlx.nn.Input(shape=(3, 2, 2, 8))
net.set_eval()
output = net(input)
print("tlx out", output)
onnx_model = export(net, input_spec=input, path='linear_model.onnx')

# Infer Model
sess = rt.InferenceSession('linear_model.onnx')

input_name = sess.get_inputs()[0].name
output_name = sess.get_outputs()[0].name

input_data = tlx.nn.Input(shape=(3, 2, 2, 8))
input_data = np.array(input_data, dtype=np.float32)

result = sess.run([output_name], {input_name: input_data})
print('onnx out',result)