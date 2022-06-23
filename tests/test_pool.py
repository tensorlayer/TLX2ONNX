#! /usr/bin/python
# -*- coding: utf-8 -*-

import os
os.environ["TL_BACKEND"] = 'tensorflow'
# os.environ["TL_BACKEND"] = 'paddle'
# os.environ["TL_BACKEND"] = 'torch'
# os.environ["TL_BACKEND"] = 'mindspore'
import tensorlayerx as tlx
from tensorlayerx.nn import Module
from tensorlayerx.nn import (Conv1d, MaxPool1d, AvgPool1d)
from tensorlayerx.nn import (Conv2d, MaxPool2d, AvgPool2d)
from tensorlayerx.nn import (Conv3d, MaxPool3d, AvgPool3d)
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
        self.conv1 = Conv2d(64, (3, 3), (1, 1), padding=(2, 2), W_init=W_init, b_init=b_init2, name='conv1', in_channels=3, data_format='channels_last', act = tlx.nn.ReLU)
        self.pool1 = MaxPool2d(kernel_size=(3, 3), stride=(1, 1), padding='SAME', data_format='channels_last')
        self.conv2 = Conv2d(128, (3, 3), (1, 1), padding=(2, 2), W_init=W_init, b_init=b_init2, name='conv2', in_channels=64, data_format='channels_last')
        self.pool2 = AvgPool2d(kernel_size=(3, 3), stride=(1, 1), padding='SAME', data_format='channels_last')
    def forward(self, x):
        z = self.conv1(x)
        z = self.pool1(z)
        z = self.conv2(z)
        z = self.pool2(z)
        return z

net = CNN()
input = tlx.nn.Input(shape=(1, 32, 32, 3))
onnx_model = export(net, input_spec=input, path='conv_model.onnx')

# Infer Model
sess = rt.InferenceSession('conv_model.onnx')

input_name = sess.get_inputs()[0].name
output_name = sess.get_outputs()[0].name

input_data = tlx.nn.Input(shape=(1, 32, 32, 3))
input_data = np.array(input_data, dtype=np.float32)

result = sess.run([output_name], {input_name: input_data})
print(result)

############################################ test 1d ###########################################################
class CNN1d(Module):

    def __init__(self):
        super(CNN1d, self).__init__()
        # weights init
        W_init = tlx.nn.initializers.truncated_normal(stddev=5e-2)
        b_init2 = tlx.nn.initializers.constant(value=0.1)
        self.conv1 = Conv1d(64, 3, 1, padding=2, W_init=W_init, b_init=b_init2, name='conv1', in_channels=3, data_format='channels_last', act = tlx.nn.ReLU)
        self.pool1 = MaxPool1d(kernel_size=3, stride=1, padding='SAME', data_format='channels_last')
        self.conv2 = Conv1d(128, 3, 1, padding=2, W_init=W_init, b_init=b_init2, name='conv2', in_channels=64, data_format='channels_last')
        self.pool2 = AvgPool1d(kernel_size=3, stride=1, padding='SAME', data_format='channels_last')
    def forward(self, x):
        z = self.conv1(x)
        z = self.pool1(z)
        z = self.conv2(z)
        z = self.pool2(z)
        return z

net = CNN1d()
input = tlx.nn.Input(shape=(1, 32, 3))
onnx_model_1d = export(net, input_spec=input, path='conv_model_1d.onnx')

# Infer Model
sess = rt.InferenceSession('conv_model_1d.onnx')

input_name = sess.get_inputs()[0].name
output_name = sess.get_outputs()[0].name

input_data = tlx.nn.Input(shape=(1, 32, 3))
input_data = np.array(input_data, dtype=np.float32)

result = sess.run([output_name], {input_name: input_data})
print(result)


############################################ test 3d ###########################################################
class CNN(Module):

    def __init__(self):
        super(CNN, self).__init__()
        # weights init
        W_init = tlx.nn.initializers.truncated_normal(stddev=5e-2)
        b_init2 = tlx.nn.initializers.constant(value=0.1)
        self.conv1 = Conv3d(64, (3, 3, 3), (1, 1, 1), padding=(2, 2, 2), W_init=W_init, b_init=b_init2, name='conv1', in_channels=3, data_format='channels_last', act = tlx.nn.ReLU)
        self.pool1 = MaxPool3d(kernel_size=(3, 3, 3), stride=(1, 1, 1), padding='SAME', data_format='channels_last')
        self.conv2 = Conv3d(128, (3, 3, 3), (1, 1, 1), padding=(2, 2, 2), W_init=W_init, b_init=b_init2, name='conv2', in_channels=64, data_format='channels_last')
        self.pool2 = AvgPool3d(kernel_size=(3, 3, 3), stride=(1, 1, 1), padding='SAME', data_format='channels_last')
    def forward(self, x):
        z = self.conv1(x)
        z = self.pool1(z)
        z = self.conv2(z)
        z = self.pool2(z)
        return z

net = CNN()
input = tlx.nn.Input(shape=(1, 32, 32, 32, 3))
onnx_model = export(net, input_spec=input, path='conv_model_3d.onnx')

# Infer Model
sess = rt.InferenceSession('conv_model_3d.onnx')

input_name = sess.get_inputs()[0].name
output_name = sess.get_outputs()[0].name

input_data = tlx.nn.Input(shape=(1, 32, 32, 32, 3))
input_data = np.array(input_data, dtype=np.float32)

result = sess.run([output_name], {input_name: input_data})
print(result)
