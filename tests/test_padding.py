#! /usr/bin/python
# -*- coding: utf-8 -*-

import os
os.environ["TL_BACKEND"] = 'tensorflow'
import tensorlayerx as tlx
from tensorlayerx.nn import Module
from tensorlayerx.nn import PadLayer, ZeroPad1d, ZeroPad2d, ZeroPad3d
from tlx2onnx.main import export
import onnxruntime as rt
import numpy as np

class CustomModel(Module):
    def __init__(self):
        super(CustomModel, self).__init__(name="custom")
        self.pad = PadLayer([[1, 2], [3, 4], [5, 6], [7, 8]], "REFLECT", name='inpad')
        self.pad2d = ZeroPad2d(padding=((2, 2), (3, 3)), data_format='channels_last')

    def forward(self, inputs):
        x = self.pad(inputs)
        x = self.pad2d(x)
        return x

net = CustomModel()
input = tlx.nn.Input(shape=(5, 5, 10, 10), init=tlx.initializers.RandomNormal())
net.set_eval()
output = net(input)
print("tlx out", output.shape)
onnx_model = export(net, input_spec=input, path='padding.onnx')

# Infer Model
sess = rt.InferenceSession('padding.onnx')

input_name = sess.get_inputs()[0].name
output_name = sess.get_outputs()[0].name

input_data = np.array(input, dtype=np.float32)

result = sess.run([output_name], {input_name: input_data})
print('onnx out', np.shape(result))