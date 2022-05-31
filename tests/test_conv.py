import os
# os.environ["TL_BACKEND"] = 'tensorflow'
# os.environ["TL_BACKEND"] = 'paddle'
os.environ["TL_BACKEND"] = 'torch'
# os.environ["TL_BACKEND"] = 'mindspore'
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
        self.conv1 = Conv2d(64, (5, 5), (1, 1), padding=(2,2), W_init=W_init, b_init=b_init2, name='conv1', in_channels=3, data_format='channels_last', act = tlx.nn.ReLU)
        self.conv2 = Conv2d(128, (5, 5), (1, 1), padding=(2,2), W_init=W_init, b_init=b_init2, name='conv2', in_channels=64, data_format='channels_last')
    def forward(self, x):
        z = self.conv1(x)
        z = self.conv2(z)
        return z

net = CNN()
input = tlx.nn.Input(shape=(1,32, 32,3))
onnx_model = export(net, input_spec=input, path='conv_model.onnx')



