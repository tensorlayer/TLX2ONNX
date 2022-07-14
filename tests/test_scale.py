#! /usr/bin/python
# -*- coding: utf-8 -*-

import os
os.environ["TL_BACKEND"] = 'torch'
import tensorlayerx as tlx
from tensorlayerx.nn import Module
from tensorlayerx.nn import Scale
from tlx2onnx.main import export
import onnxruntime as rt
import numpy as np

class CustomModel(Module):
    def __init__(self):
        super(CustomModel, self).__init__(name="custom")
        self.scale = Scale(init_scale=0.5)

    def forward(self, inputs):
        outputs = self.scale(inputs)
        return outputs

net = CustomModel()
input = tlx.nn.Input(shape=(3, 20), init=tlx.initializers.RandomNormal())
net.set_eval()
output = net(input)
print("tlx out", output)
onnx_model = export(net, input_spec=input, path='sacle.onnx')

# Infer Model
sess = rt.InferenceSession('sacle.onnx')

input_name = sess.get_inputs()[0].name
output_name = sess.get_outputs()[0].name

input_data = np.array(input, dtype=np.float32)

result = sess.run([output_name], {input_name: input_data})
print('onnx out', result)