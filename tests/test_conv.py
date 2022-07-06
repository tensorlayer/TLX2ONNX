#! /usr/bin/python
# -*- coding: utf-8 -*-

import os
os.environ["TL_BACKEND"] = 'tensorflow'
# os.environ["TL_BACKEND"] = 'paddle'
# os.environ["TL_BACKEND"] = 'torch'
# os.environ["TL_BACKEND"] = 'mindspore'
import tensorlayerx as tlx
from tensorlayerx.nn import Module
from tensorlayerx.nn import (Conv2d, Conv1d, Conv3d)
from tlx2onnx.main import export
import onnxruntime as rt
import numpy as np


############################################ test 2d ###########################################################
class CNN(Module):

    def __init__(self):
        super(CNN, self).__init__()
        # weights init
        W_init = tlx.nn.initializers.truncated_normal(stddev=5e-2)
        b_init2 = tlx.nn.initializers.constant(value=0.1)
        self.conv1 = Conv2d(64, (5, 5), (1, 1), padding=(2,2), W_init=W_init, b_init=b_init2, name='conv1', in_channels=3, data_format='channels_last', act = tlx.nn.ReLU)
    def forward(self, x):
        z = self.conv1(x)
        return z

net = CNN()
input = tlx.nn.Input(shape=(1, 32, 32, 3))
net.set_eval()
output = net(input)
print("conv2d tlx output", output)
onnx_model = export(net, input_spec=input, path='conv2d_model.onnx')

# Infer Model
sess = rt.InferenceSession('conv2d_model.onnx')

input_name = sess.get_inputs()[0].name
output_name = sess.get_outputs()[0].name

input_data = tlx.nn.Input(shape=(1, 32, 32, 3))
input_data = np.array(input_data, dtype=np.float32)

result = sess.run([output_name], {input_name: input_data})
print("conv2d onnx output", result)


############################################ test 1d ###########################################################
class CNN1d(Module):

    def __init__(self):
        super(CNN1d, self).__init__()
        # weights init
        W_init = tlx.nn.initializers.truncated_normal(stddev=5e-2)
        b_init2 = tlx.nn.initializers.constant(value=0.1)
        self.conv1 = Conv1d(64, 5, 1, padding=2, W_init=W_init, b_init=b_init2, name='conv1', in_channels=3, data_format='channels_last', act = tlx.nn.ReLU)
    def forward(self, x):
        z = self.conv1(x)
        return z

net = CNN1d()
input = tlx.nn.Input(shape=(1, 32, 3))
net.set_eval()
output = net(input)
print("conv1d tlx output", output)
onnx_model = export(net, input_spec=input, path='conv1d_model.onnx')

# Infer Model
sess = rt.InferenceSession('conv1d_model.onnx')

input_name = sess.get_inputs()[0].name
output_name = sess.get_outputs()[0].name

input_data = tlx.nn.Input(shape=(1, 32, 3))
input_data = np.array(input_data, dtype=np.float32)

result = sess.run([output_name], {input_name: input_data})
print("conv1d onnx output", result)


############################################ test 3d ###########################################################
class CNN3d(Module):

    def __init__(self):
        super(CNN3d, self).__init__()
        # weights init
        W_init = tlx.nn.initializers.truncated_normal(stddev=5e-2)
        b_init2 = tlx.nn.initializers.constant(value=0.1)
        self.conv1 = Conv3d(64, 5, 1, padding=2, W_init=W_init, b_init=b_init2, name='conv1', in_channels=3, data_format='channels_last', act = tlx.nn.ReLU)
    def forward(self, x):
        z = self.conv1(x)
        return z

net = CNN3d()
input = tlx.nn.Input(shape=(1, 32, 32, 32, 3))
net.set_eval()
output = net(input)
print("conv3d tlx output", output)
onnx_model = export(net, input_spec=input, path='conv3d_model.onnx')

# Infer Model
sess = rt.InferenceSession('conv3d_model.onnx')

input_name = sess.get_inputs()[0].name
output_name = sess.get_outputs()[0].name

input_data = tlx.nn.Input(shape=(1, 32, 32, 32, 3))
input_data = np.array(input_data, dtype=np.float32)

result = sess.run([output_name], {input_name: input_data})
print("conv3d onnx output", result)