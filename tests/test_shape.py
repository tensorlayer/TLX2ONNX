#! /usr/bin/python
# -*- coding: utf-8 -*-

import os
os.environ["TL_BACKEND"] = 'tensorflow'
import tensorlayerx as tlx
from tensorlayerx.nn import Module
from tensorlayerx.nn import Transpose, Reshape
from tlx2onnx.main import export
import onnxruntime as rt
import numpy as np


class MLP(Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.trans1 = Transpose(perm=[0, 1, 2, 3])
        self.trans2 = Transpose(perm=[2, 0, 1, 3])
        self.reshpe = Reshape(shape=(2, 3, 16))

    def forward(self, x):
        z = self.trans1(x)
        z = self.trans2(z)
        z = self.reshpe(z)
        return z

net = MLP()
input = tlx.nn.Input(shape=(3, 2, 2, 8))
onnx_model = export(net, input_spec=input, path='shape_op.onnx')

# Infer Model
sess = rt.InferenceSession('shape_op.onnx')

input_name = sess.get_inputs()[0].name
output_name = sess.get_outputs()[0].name

input_data = tlx.nn.Input(shape=(3, 2, 2, 8))
input_data = np.array(input_data, dtype=np.float32)

result = sess.run([output_name], {input_name: input_data})
print(result)