#! /usr/bin/python
# -*- coding: utf-8 -*-

import os
os.environ["TL_BACKEND"] = 'tensorflow'
import tensorlayerx as tlx
from tensorlayerx.nn import Module
from tensorlayerx.nn import Linear, Stack, UnStack
from tlx2onnx.main import export
import onnxruntime as rt
import numpy as np

class CustomModel(Module):
    def __init__(self):
        super(CustomModel, self).__init__(name="custom")

        self.l1 = Linear(10, name='dense1')
        self.l2 = Linear(10, name='dense2')
        self.l3 = Linear(10, name='dense3')
        self.stack = Stack(axis=1)
        self.unstack = UnStack(axis=1)
        self.stack1 = Stack(axis=1)

    def forward(self, inputs):
        out1 = self.l1(inputs)
        out2 = self.l2(inputs)
        out3 = self.l3(inputs)
        outputs = self.stack([out1, out2, out3])
        o1, o2, o3 = self.unstack(outputs)
        outputs = self.stack1([o1, o2, o3])
        return outputs

net = CustomModel()
input = tlx.nn.Input(shape=(10, 784), init=tlx.initializers.RandomNormal())
net.set_eval()
output = net(input)
print("tlx out", output)
onnx_model = export(net, input_spec=input, path='stack.onnx')

# Infer Model
sess = rt.InferenceSession('stack.onnx')

input_name = sess.get_inputs()[0].name
output_name = sess.get_outputs()[0].name

input_data = np.array(input, dtype=np.float32)

result = sess.run([output_name], {input_name: input_data})
print('onnx out', result)