#! /usr/bin/python
# -*- coding: utf-8 -*-

import os
os.environ["TL_BACKEND"] = 'tensorflow'
import tensorlayerx as tlx
from tensorlayerx.nn import Module
from tensorlayerx.nn import OneHot, Embedding
from tlx2onnx.main import export
import onnxruntime as rt
import numpy as np

###################################  OneHot  ##############################################
class Model(Module):
    def __init__(self):
        super(Model, self).__init__()
        self.onehot = OneHot(depth=10)

    def forward(self, x):
        z = self.onehot(x)
        return z

net = Model()
input = tlx.nn.Input([10], dtype=tlx.int64)
onnx_model = export(net, input_spec=input, path='onehot.onnx')

# Infer Model
sess = rt.InferenceSession('onehot.onnx')

input_name = sess.get_inputs()[0].name
output_name = sess.get_outputs()[0].name

input_data = np.array(input, dtype=np.int64)

result = sess.run([output_name], {input_name: input_data})
print(result)

###################################### Embedding  #################################################
class Model_E(Module):
    def __init__(self):
        super(Model_E, self).__init__()
        self.embedding = Embedding(num_embeddings=1000, embedding_dim=50, name='embed')

    def forward(self, x):
        z = self.embedding(x)
        return z

net = Model_E()
input = tlx.nn.Input([10, 100], dtype=tlx.int64)
onnx_model_e = export(net, input_spec=input, path='embedding.onnx')

# Infer Model
sess = rt.InferenceSession('embedding.onnx')

input_name = sess.get_inputs()[0].name
output_name = sess.get_outputs()[0].name

input_data = np.array(input, dtype=np.int64)

result = sess.run([output_name], {input_name: input_data})
print(np.shape(result))
