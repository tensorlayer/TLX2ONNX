#! /usr/bin/python
# -*- coding: utf-8 -*-

import os
os.environ["TL_BACKEND"] = 'tensorflow'
import tensorlayerx as tlx
from tensorlayerx.nn import Module
from tensorlayerx.nn import GaussianNoise
from tlx2onnx.main import export
import onnxruntime as rt
import numpy as np

class CustomModel(Module):
    def __init__(self):
        super(CustomModel, self).__init__(name="custom")
        self.noise = GaussianNoise()

    def forward(self, inputs):
        x = self.noise(inputs)
        return x

net = CustomModel()
input = tlx.nn.Input(shape=(3, 20), init=tlx.initializers.RandomNormal())
net.set_eval()
output = net(input)
print("tlx out", output)
onnx_model = export(net, input_spec=input, path='noise.onnx')

# Infer Model
sess = rt.InferenceSession('noise.onnx')

input_name = sess.get_inputs()[0].name
output_name = sess.get_outputs()[0].name

input_data = np.array(input, dtype=np.float32)

result = sess.run([output_name], {input_name: input_data})
print('onnx out', result)