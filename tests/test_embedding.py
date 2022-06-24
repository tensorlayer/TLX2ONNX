#! /usr/bin/python
# -*- coding: utf-8 -*-

import os
os.environ["TL_BACKEND"] = 'tensorflow'
import tensorlayerx as tlx
from tensorlayerx.nn import Module
from tensorlayerx.nn import OneHot
from tlx2onnx.main import export
import onnxruntime as rt
import numpy as np


class Model(Module):
    def __init__(self):
        super(Model, self).__init__()
        self.onehot = OneHot(depth=10)

    def forward(self, x):
        z = self.onehot(x)
        return z

net = Model()
input = tlx.nn.Input([10], dtype=tlx.int32)
onnx_model = export(net, input_spec=input, path='onehot.onnx')

# Infer Model
sess = rt.InferenceSession('onehot.onnx')

input_name = sess.get_inputs()[0].name
output_name = sess.get_outputs()[0].name

input_data = np.array(input, dtype=np.int32)

result = sess.run([output_name], {input_name: input_data})
print(result)