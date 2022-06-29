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

class MatMul(tlx.nn.Module):
    def __init__(self):
        super(MatMul, self).__init__()
        self.matmul = tlx.ops.MatMul()
        self.transpose_X = False
        self.transpose_Y = False
        self._built = True

    def forward(self, a, b):
        z = self.matmul(a, b)
        if not self._nodes_fixed and self._build_graph:
            self._add_node([a, b], z)
            self._nodes_fixed = True
        return z

class MLP(Module):
    def __init__(self):
        super(MLP, self).__init__()
        # weights init
        self.flatten = Flatten()
        self.line1 = Linear(in_features=32, out_features=64, act=tlx.nn.LeakyReLU(0.3))
        self.d1 = Dropout()
        self.line2 = Linear(in_features=64, out_features=128, b_init=None)
        self.relu6 = ReLU6()
        self.line3 = Linear(in_features=128, out_features=10, act=tlx.nn.ReLU)
        self.line4 = Linear(in_features=128, out_features=10)
        self.mat = MatMul()

    def forward(self, x):
        x = self.flatten(x)
        z = self.line1(x)
        z = self.d1(z)
        z = self.line2(z)
        z = self.relu6(z)
        z1 = self.line3(z)
        z2 = self.line4(z)
        z = self.mat(z1, z2)
        return z

net = MLP()
net.set_eval()
input = tlx.nn.Input(shape=(10, 2, 2, 8))
print("tlx output", net(input))
onnx_model = export(net, input_spec=input, path='linear_model.onnx')

# Infer Model
sess = rt.InferenceSession('linear_model.onnx')

input_name = sess.get_inputs()[0].name
output_name = sess.get_outputs()[0].name

input_data = tlx.nn.Input(shape=(10, 2, 2, 8))
input_data = np.array(input_data, dtype=np.float32)

result = sess.run([output_name], {input_name: input_data})
print("onnx output", result)