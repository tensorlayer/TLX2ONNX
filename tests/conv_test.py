import os
os.environ["TL_BACKEND"] = 'tensorflow'
import tensorlayerx as tlx
from tensorlayerx.nn import Module
from tensorlayerx.nn import (Conv2d)
from tlx2onnx.main import export
class CNN(Module):

    def __init__(self):
        super(CNN, self).__init__()
        # weights init
        W_init = tlx.nn.initializers.truncated_normal(stddev=5e-2)
        b_init2 = tlx.nn.initializers.constant(value=0.1)
        self.conv1 = Conv2d(64, (5, 5), (1, 1), padding='SAME', W_init=W_init, b_init=b_init2, name='conv1', in_channels=3, act=tlx.ReLU)

    def forward(self, x):
        z = self.conv1(x)
        return z

net = CNN()
input = tlx.nn.Input(shape=(1,32,32,3))
onnx_model = export(net, input_spec=input)
print(onnx_model)
