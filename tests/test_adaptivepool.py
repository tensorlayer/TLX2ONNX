#! /usr/bin/python
# -*- coding: utf-8 -*-

import os
os.environ["TL_BACKEND"] = 'tensorflow'
# os.environ["TL_BACKEND"] = 'paddle'
# os.environ["TL_BACKEND"] = 'torch'
# os.environ["TL_BACKEND"] = 'mindspore'
import tensorlayerx as tlx
from tensorlayerx.nn import Module
from tensorlayerx.nn import (AdaptiveAvgPool1d,AdaptiveAvgPool2d,AdaptiveAvgPool3d)
from tensorlayerx.nn import (AdaptiveMaxPool1d,AdaptiveMaxPool2d,AdaptiveMaxPool3d)
from tlx2onnx.main import export
import onnxruntime as rt
import numpy as np


############################################ test 2d ###########################################################
class Adaptiveavgpool2d(Module):

    def __init__(self):
        super(Adaptiveavgpool2d, self).__init__()
        self.pool1 = AdaptiveAvgPool2d(output_size=16, data_format='channels_last')

    def forward(self, x):
        z = self.pool1(x)
        return z

net = Adaptiveavgpool2d()
input = tlx.nn.Input(shape=(1, 32, 32, 3))
onnx_model = export(net, input_spec=input, path='Adaptiveavgpool2d.onnx')

# Infer Model
sess = rt.InferenceSession('Adaptiveavgpool2d.onnx')

input_name = sess.get_inputs()[0].name
output_name = sess.get_outputs()[0].name

input_data = tlx.nn.Input(shape=(1, 32, 32, 3))
input_data = np.array(input_data, dtype=np.float32)

result = sess.run([output_name], {input_name: input_data})
print('Adaptiveavgpool2d result', result[0].shape)


class Adaptivemaxpool2d(Module):

    def __init__(self):
        super(Adaptivemaxpool2d, self).__init__()
        self.pool1 = AdaptiveMaxPool2d(output_size=16, data_format='channels_last')

    def forward(self, x):
        z = self.pool1(x)
        return z


net = Adaptivemaxpool2d()
input = tlx.nn.Input(shape=(1, 32, 32, 3))
onnx_model = export(net, input_spec=input, path='Adaptivemaxpool2d.onnx')

# Infer Model
sess = rt.InferenceSession('Adaptivemaxpool2d.onnx')

input_name = sess.get_inputs()[0].name
output_name = sess.get_outputs()[0].name

input_data = tlx.nn.Input(shape=(1, 32, 32, 3))
input_data = np.array(input_data, dtype=np.float32)

result = sess.run([output_name], {input_name: input_data})
print('Adaptivemaxpool2d result', result[0].shape)



############################################ test 1d ###########################################################
class Adaptiveavgpool1d(Module):

    def __init__(self):
        super(Adaptiveavgpool1d, self).__init__()
        self.pool1 = AdaptiveAvgPool1d(output_size=16, data_format='channels_last')

    def forward(self, x):
        z = self.pool1(x)
        return z

net = Adaptiveavgpool1d()
input = tlx.nn.Input(shape=(1, 32, 3))
onnx_model_1d = export(net, input_spec=input, path='Adaptiveavgpool1d.onnx')

# Infer Model
sess = rt.InferenceSession('Adaptiveavgpool1d.onnx')

input_name = sess.get_inputs()[0].name
output_name = sess.get_outputs()[0].name

input_data = tlx.nn.Input(shape=(1, 32, 3))
input_data = np.array(input_data, dtype=np.float32)

result = sess.run([output_name], {input_name: input_data})
print('Adaptiveavgpool1d result', result[0].shape)



class Adaptivemaxpool1d(Module):

    def __init__(self):
        super(Adaptivemaxpool1d, self).__init__()
        self.pool1 = AdaptiveMaxPool1d(output_size=16, data_format='channels_last')

    def forward(self, x):
        z = self.pool1(x)
        return z

net = Adaptiveavgpool1d()
input = tlx.nn.Input(shape=(1, 32, 3))
onnx_model_1d = export(net, input_spec=input, path='Adaptivemaxpool1d.onnx')

# Infer Model
sess = rt.InferenceSession('Adaptivemaxpool1d.onnx')

input_name = sess.get_inputs()[0].name
output_name = sess.get_outputs()[0].name

input_data = tlx.nn.Input(shape=(1, 32, 3))
input_data = np.array(input_data, dtype=np.float32)

result = sess.run([output_name], {input_name: input_data})
print('Adaptivemaxpool1d result', result[0].shape)


############################################ test 3d ###########################################################
class Adaptiveavgpool3d(Module):

    def __init__(self):
        super(Adaptiveavgpool3d, self).__init__()
        self.pool1 = AdaptiveAvgPool3d(output_size=16, data_format='channels_last')

    def forward(self, x):
        z = self.pool1(x)
        return z

net = Adaptiveavgpool3d()
input = tlx.nn.Input(shape=(1, 32, 32, 32, 3))
onnx_model_1d = export(net, input_spec=input, path='Adaptiveavgpool3d.onnx')

# Infer Model
sess = rt.InferenceSession('Adaptiveavgpool3d.onnx')

input_name = sess.get_inputs()[0].name
output_name = sess.get_outputs()[0].name

input_data = tlx.nn.Input(shape=(1, 32, 32, 32, 3))
input_data = np.array(input_data, dtype=np.float32)

result = sess.run([output_name], {input_name: input_data})
print('Adaptiveavgpool3d result', result[0].shape)



class Adaptivemaxpool3d(Module):

    def __init__(self):
        super(Adaptivemaxpool3d, self).__init__()
        self.pool1 = AdaptiveMaxPool3d(output_size=16, data_format='channels_last')

    def forward(self, x):
        z = self.pool1(x)
        return z

net = Adaptiveavgpool3d()
input = tlx.nn.Input(shape=(1, 32, 32, 32, 3))
onnx_model_1d = export(net, input_spec=input, path='Adaptivemaxpool3d.onnx')

# Infer Model
sess = rt.InferenceSession('Adaptivemaxpool3d.onnx')

input_name = sess.get_inputs()[0].name
output_name = sess.get_outputs()[0].name

input_data = tlx.nn.Input(shape=(1, 32, 32, 32, 3))
input_data = np.array(input_data, dtype=np.float32)

result = sess.run([output_name], {input_name: input_data})
print('Adaptivemaxpool3d result', result[0].shape)