#! /usr/bin/python
# -*- coding: utf-8 -*-

import os
os.environ["TL_BACKEND"] = 'tensorflow'
import tensorlayerx as tlx
from tensorlayerx.nn import Module
from tensorlayerx.nn import Linear
from tlx2onnx.main import export


class MLP(Module):
    def __init__(self):
        super(MLP, self).__init__()
        # weights init
        self.line1 = Linear(in_features=32, out_features=64, act=tlx.nn.ReLU)
        self.line2 = Linear(in_features=64, out_features=128, b_init=None)
        self.line3 = Linear(in_features=128, out_features=10, act=tlx.nn.ReLU)

    def forward(self, x):
        z = self.line1(x)
        z = self.line2(z)
        z = self.line3(z)
        return z

net = MLP()
input = tlx.nn.Input(shape=(3, 32))
onnx_model = export(net, input_spec=input, path='linear_model.onnx')