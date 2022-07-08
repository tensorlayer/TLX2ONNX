#! /usr/bin/python
# -*- coding: utf-8 -*-

import os
os.environ["TL_BACKEND"] = 'tensorflow'
import tensorlayerx as tlx
from tensorlayerx.nn import Module
from tensorlayerx.nn import Linear, Concat, Elementwise
from tlx2onnx.main import export
import onnxruntime as rt
import numpy as np

class CustomModel(Module):
    def __init__(self):
        super(CustomModel, self).__init__(name="custom")
        self.linear1 = Linear(in_features=20, out_features=10, act=tlx.ReLU, name='relu1_1')
        self.linear2 = Linear(in_features=20, out_features=10, act=tlx.ReLU, name='relu2_1')
        self.concat = Concat(concat_dim=1, name='concat_layer')

    def forward(self, inputs):
        d1 = self.linear1(inputs)
        d2 = self.linear2(inputs)
        outputs = self.concat([d1, d2])
        return outputs

net = CustomModel()
input = tlx.nn.Input(shape=(3, 20), init=tlx.initializers.RandomNormal())
net.set_eval()
output = net(input)
print("tlx out", output)
onnx_model = export(net, input_spec=input, path='concat.onnx')

# Infer Model
sess = rt.InferenceSession('concat.onnx')

input_name = sess.get_inputs()[0].name
output_name = sess.get_outputs()[0].name

input_data = np.array(input, dtype=np.float32)

result = sess.run([output_name], {input_name: input_data})
print('onnx out', result)

#####################################  Elementwise  ###################################################
class CustomModel2(Module):
    def __init__(self):
        super(CustomModel2, self).__init__(name="custom")
        self.linear1 = Linear(in_features=10, out_features=10, act=tlx.ReLU, name='relu1_1')
        self.linear2 = Linear(in_features=10, out_features=10, act=tlx.ReLU, name='relu2_1')
        self.linear3 = Linear(in_features=10, out_features=10, act=tlx.ReLU, name='relu3_1')
        self.element = Elementwise(combine_fn=tlx.matmul, name='concat')

    def forward(self, inputs):
        d1 = self.linear1(inputs)
        d2 = self.linear2(inputs)
        d3 = self.linear3(inputs)
        outputs = self.element([d1, d2, d3])
        return outputs

net = CustomModel2()
input = tlx.nn.Input(shape=(10, 10), init=tlx.initializers.RandomNormal())
net.set_eval()
output = net(input)
print("tlx out", output)
onnx_model2 = export(net, input_spec=input, path='elementwise.onnx')

# Infer Model
sess = rt.InferenceSession('elementwise.onnx')

input_name = sess.get_inputs()[0].name
output_name = sess.get_outputs()[0].name

input_data = np.array(input, dtype=np.float32)

result = sess.run([output_name], {input_name: input_data})
print('onnx out', result)