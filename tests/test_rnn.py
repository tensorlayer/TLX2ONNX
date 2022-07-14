#! /usr/bin/python
# -*- coding: utf-8 -*-
import os
# os.environ["TL_BACKEND"] = 'tensorflow'
# os.environ["TL_BACKEND"] = 'paddle'
os.environ["TL_BACKEND"] = 'torch'
# os.environ["TL_BACKEND"] = 'mindspore'
import tensorlayerx as tlx
from tensorlayerx.nn import Module
from tensorlayerx.nn import  RNN, LSTM, GRU, Linear
from tlx2onnx.main import export
import onnxruntime as rt
import numpy as np
tlx.set_seed(42)

class Rnn(Module):

    def __init__(self):
        super(Rnn, self).__init__()
        self.rnn = RNN(input_size=5, hidden_size=5, act='relu', bidirectional=False, num_layers=4)
    def forward(self, x):
        x, _ = self.rnn(x)
        return x
model = Rnn()
input = tlx.nn.Input(shape=[1, 5, 5])
model.set_eval()
output = model(input)
print("RNN tlx output", output)
onnx_model = export(model, input_spec=input, path='rnn.onnx')

sess = rt.InferenceSession('rnn.onnx')
input_name = sess.get_inputs()[0].name
output_name = sess.get_outputs()[0].name

input_data = np.array(input, dtype=np.float32)
result = sess.run([output_name], {input_name: input_data})
print("RNN onnx output", result)
print("==========================================================================================================")


class Lstm(Module):

    def __init__(self):
        super(Lstm, self).__init__()
        self.rnn = LSTM(input_size=5, hidden_size=5, bidirectional=True, num_layers=4)
    def forward(self, x):
        x, _ = self.rnn(x)
        return x
model = Lstm()
input = tlx.nn.Input(shape=[1, 5, 5])
model.set_eval()
output = model(input)
print("LSTM tlx output", output)
onnx_model = export(model, input_spec=input, path='lstm.onnx')

sess = rt.InferenceSession('lstm.onnx')
input_name = sess.get_inputs()[0].name
output_name = sess.get_outputs()[0].name

input_data = np.array(input, dtype=np.float32)
result = sess.run([output_name], {input_name: input_data})
print("LSTM onnx output", result)
print("==========================================================================================================")

class Gru(Module):

    def __init__(self):
        super(Gru, self).__init__()
        self.rnn = GRU(input_size=5, hidden_size=5, bidirectional=True, num_layers=4)
    def forward(self, x):
        x, _ = self.rnn(x)
        return x

model = Gru()
input = tlx.nn.Input(shape=[1, 5, 5])
model.set_eval()
output = model(input)
print("GRU tlx output", output)
onnx_model = export(model, input_spec=input, path='gru.onnx')

sess = rt.InferenceSession('gru.onnx')
input_name = sess.get_inputs()[0].name
output_name = sess.get_outputs()[0].name

input_data = np.array(input, dtype=np.float32)
result = sess.run([output_name], {input_name: input_data})
print("GRU onnx output", result)