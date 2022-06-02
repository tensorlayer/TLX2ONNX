#! /usr/bin/python
# -*- coding: utf-8 -*-

import os
os.environ["TL_BACKEND"] = 'tensorflow'
import tensorlayerx as tlx
from tensorlayerx.nn import Module
from tensorlayerx.nn import Linear, Dropout, Flatten, ReLU6
from tlx2onnx.main import export


class MLP(Module):
    def __init__(self):
        super(MLP, self).__init__()
        # weights init
        self.flatten = Flatten()
        self.line1 = Linear(in_features=32, out_features=64, act=tlx.nn.LeakyReLU(0.3))
        self.d1 = Dropout()
        self.line2 = Linear(in_features=64, out_features=128, b_init=None)
        self.relu6 = ReLU6()
        self.line3 = Linear(in_features=128, out_features=10, act=tlx.nn.ReLU)

    def forward(self, x):
        x = self.flatten(x)
        z = self.line1(x)
        z = self.d1(z)
        z = self.line2(z)
        z = self.relu6(z)
        z = self.line3(z)
        return z

net = MLP()
input = tlx.nn.Input(shape=(3, 2, 2, 8))
onnx_model = export(net, input_spec=input, path='linear_model.onnx')