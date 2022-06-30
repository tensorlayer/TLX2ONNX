#! /usr/bin/python
# -*- coding: utf-8 -*-

import os
os.environ["TL_BACKEND"] = 'tensorflow'
import tensorlayerx as tlx
from tensorlayerx.nn import Module
from tensorlayerx.nn import ReLU, LeakyReLU, ELU, Tanh, Softmax, Softplus, Sigmoid, ReLU6, \
    PRelu, Mish, Swish, LeakyReLU6
from tlx2onnx.main import export
import onnxruntime as rt
import numpy as np


class MLP(Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.relu = ReLU()
        self.leakyrelu = LeakyReLU()
        self.elu = ELU()
        self.tanh = Tanh()
        self.softmax = Softmax()
        self.softplus = Softplus()
        self.sigmoid = Sigmoid()
        self.relu6 = ReLU6()
        self.prelu = PRelu()
        self.mish = Mish()
        self.swish = Swish()
        self.lrelu6 = LeakyReLU6()

    def forward(self, x):
        z = self.relu(x)
        z = self.leakyrelu(z)
        z = self.elu(z)
        z = self.tanh(z)
        z = self.softmax(z)
        z = self.softplus(z)
        z = self.sigmoid(z)
        z = self.relu6(z)
        z = self.prelu(z)
        z = self.mish(z)
        z = self.swish(z)
        z = self.lrelu6(z)
        return z

net = MLP()
net.set_eval()
input = tlx.nn.Input(shape=(4, 5, 5, 3))
onnx_model = export(net, input_spec=input, path='activation.onnx')
print("tlx out", net(input))

# Infer Model
sess = rt.InferenceSession('activation.onnx')

input_name = sess.get_inputs()[0].name
output_name = sess.get_outputs()[0].name

input_data = tlx.nn.Input(shape=(4, 5, 5, 3))
input_data = np.array(input_data, dtype=np.float32)

result = sess.run([output_name], {input_name: input_data})
print("onnx out", result)