#! /usr/bin/python
# -*- coding: utf-8 -*-

import os
os.environ["TL_BACKEND"] = 'tensorflow'
import tensorlayerx as tlx
from tensorlayerx.nn import Module
from tensorlayerx.nn import GlobalMaxPool2d, GlobalAvgPool2d, Linear
from tlx2onnx.main import export
import onnxruntime as rt
import numpy as np

################################# Test GlobalAvgPool2d  ################################################
class ModelAvg(Module):
    def __init__(self):
        super(ModelAvg, self).__init__()
        self.globalmax = GlobalAvgPool2d(data_format='channels_first')
        self.line = Linear(out_features=10)

    def forward(self, x):
        x = self.globalmax(x)
        x = self.line(x)
        return x

net = ModelAvg()
net.set_eval()
input = tlx.nn.Input(shape=(5, 6, 3, 3))
onnx_model_avg = export(net, input_spec=input, path='globalavg_model.onnx')
print("tlx output", net(input))

# Infer Model
sess = rt.InferenceSession('globalavg_model.onnx')

input_name = sess.get_inputs()[0].name
output_name = sess.get_outputs()[0].name

input_data = tlx.nn.Input(shape=(5, 6, 3, 3))
input_data = np.array(input_data, dtype=np.float32)

result = sess.run([output_name], {input_name: input_data})
print("onnx output", result)

################################# Test GlobalMaxPool2d  ################################################
class ModelMax(Module):
    def __init__(self):
        super(ModelMax, self).__init__()
        self.globalmax = GlobalMaxPool2d(data_format='channels_first')
        self.line = Linear(out_features=10)

    def forward(self, x):
        x = self.globalmax(x)
        x = self.line(x)
        return x

net = ModelMax()
net.set_eval()
input = tlx.nn.Input(shape=(5, 6, 3, 3))
onnx_model_max = export(net, input_spec=input, path='globalmax_model.onnx')
print("tlx output", net(input))

# Infer Model
sess = rt.InferenceSession('globalmax_model.onnx')

input_name = sess.get_inputs()[0].name
output_name = sess.get_outputs()[0].name

input_data = tlx.nn.Input(shape=(5, 6, 3, 3))
input_data = np.array(input_data, dtype=np.float32)

result = sess.run([output_name], {input_name: input_data})
print("onnx output", result)