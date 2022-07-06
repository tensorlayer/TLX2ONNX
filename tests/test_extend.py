#! /usr/bin/python
# -*- coding: utf-8 -*-

import os
os.environ["TL_BACKEND"] = 'tensorflow'
import tensorlayerx as tlx
from tensorlayerx.nn import Module
from tensorlayerx.nn import ExpandDims, Tile
from tlx2onnx.main import export
import onnxruntime as rt
import numpy as np


class NET(Module):
    def __init__(self):
        super(NET, self).__init__()
        self.expand = ExpandDims(axis=2)

    def forward(self, x):
        x = self.expand(x)
        return x

net = NET()
net.set_eval()
input = tlx.nn.Input(shape=(10, 3, 5, 6))
onnx_model = export(net, input_spec=input, path='extend.onnx')
print("tlx out", net(input).shape)

# Infer Model
sess = rt.InferenceSession('extend.onnx')

input_name = sess.get_inputs()[0].name
output_name = sess.get_outputs()[0].name

input_data = np.array(input, dtype=np.float32)

result = sess.run([output_name], {input_name: input_data})
print("onnx out", np.shape(result))

################################  Tile   #############################################
class Tile_M(Module):
    def __init__(self):
        super(Tile_M, self).__init__()
        self.expand = Tile(multiples=[2, 3])

    def forward(self, x):
        x = self.expand(x)
        return x

net = Tile_M()
net.set_eval()
input = tlx.nn.Input(shape=(10, 9))
tile_model = export(net, input_spec=input, path='tile.onnx')
print("tlx out", net(input).shape)

# Infer Model
sess = rt.InferenceSession('tile.onnx')

input_name = sess.get_inputs()[0].name
output_name = sess.get_outputs()[0].name

input_data = np.array(input, dtype=np.float32)

result = sess.run([output_name], {input_name: input_data})
print("onnx out", np.shape(result))