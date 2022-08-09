#! /usr/bin/python
# -*- coding: utf-8 -*-

import os
os.environ["TL_BACKEND"] = 'tensorflow'
import tensorlayerx as tlx
from tensorlayerx.nn import Module
from tensorlayerx.nn import LayerNorm
from tlx2onnx.main import export
import onnxruntime as rt
import numpy as np


class NET(Module):
    def __init__(self):
        super(NET, self).__init__()
        self.layernorm = LayerNorm([50, 50, 32], act=tlx.nn.ReLU)

    def forward(self, x):
        x = self.layernorm(x)
        return x

net = NET()
print(type(net))
net.set_eval()
input = tlx.nn.Input(shape=(10, 50, 50, 32))
onnx_model = export(net, input_spec=input, path='layernorm.onnx', enable_onnx_checker=False)
print("tlx out", input)

# Infer Model
sess = rt.InferenceSession('layernorm.onnx')

input_name = sess.get_inputs()[0].name
output_name = sess.get_outputs()[0].name

input_data = np.array(input, dtype=np.float32)

result = sess.run([output_name], {input_name: input_data})
print("onnx out", result)
