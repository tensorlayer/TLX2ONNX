#! /usr/bin/python
# -*- coding: utf-8 -*-
import os
# os.environ["TL_BACKEND"] = 'tensorflow'
# os.environ["TL_BACKEND"] = 'paddle'
os.environ["TL_BACKEND"] = 'torch'
# os.environ["TL_BACKEND"] = 'mindspore'
import tensorlayerx as tlx
from tensorlayerx.nn import Module
from tensorlayerx.nn import SubpixelConv2d
from tlx2onnx.main import export
import onnxruntime as rt
import numpy as np
tlx.set_seed(42)

class conv(Module):

    def __init__(self):
        super(conv, self).__init__()
        self.conv = SubpixelConv2d(scale=2, data_format="channels_last")

    def forward(self, x):

        x = self.conv(x)
        return x

model = conv()
input = tlx.nn.Input(shape= (1, 2, 2, 8), init=tlx.nn.initializers.HeUniform())
model.set_eval()
output = model(input)
print("SubpixelConv2d tlx output", output)
onnx_model = export(model, input_spec=input, path='SubpixelConv2d.onnx', opset_version = 11)

sess = rt.InferenceSession('SubpixelConv2d.onnx')
input_name = sess.get_inputs()[0].name
output_name = sess.get_outputs()[0].name

input_data = np.array(input, dtype=np.float32)
result = sess.run([output_name], {input_name: input_data})
print("SubpixelConv2d onnx output", result)