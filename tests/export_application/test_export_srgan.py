import os
os.environ['TL_BACKEND'] = 'tensorflow'
from tensorlayerx.nn import Module
import tensorlayerx as tlx
from tensorlayerx.nn import Conv2d, BatchNorm2d, Elementwise, SubpixelConv2d, UpSampling2d, Flatten, Sequential
from tensorlayerx.nn import Linear, MaxPool2d
from tlx2onnx import export
import numpy as np
import onnxruntime
import matplotlib.pyplot as plt
import cv2
W_init = tlx.initializers.TruncatedNormal(stddev=0.02)
G_init = tlx.initializers.TruncatedNormal(mean=1.0, stddev=0.02)

class Add(Module):
    def __init__(self):
        super(Add, self).__init__()
        self._built = True

    def forward(self, a, b):
        z = tlx.ops.add(a, b)
        if not self._nodes_fixed and self._build_graph:
            self._add_node([a, b], z)
            self._nodes_fixed = True
        return z


class ResidualBlock(Module):

    def __init__(self):
        super(ResidualBlock, self).__init__()
        self.conv1 = Conv2d(
            out_channels=64, kernel_size=(3, 3), stride=(1, 1), act=None, padding='SAME', W_init=W_init,
            data_format='channels_last', b_init=None
        )
        self.bn1 = BatchNorm2d(num_features=64, act=tlx.ReLU, gamma_init=G_init, data_format='channels_last')
        self.conv2 = Conv2d(
            out_channels=64, kernel_size=(3, 3), stride=(1, 1), act=None, padding='SAME', W_init=W_init,
            data_format='channels_last', b_init=None
        )
        self.bn2 = BatchNorm2d(num_features=64, act=None, gamma_init=G_init, data_format='channels_last')
        self.add = Add()

    def forward(self, x):
        temp = x
        z = self.conv1(x)
        z = self.bn1(z)
        z = self.conv2(z)
        z = self.bn2(z)
        out = self.add(temp,z)
        return out

class SRGAN_g(Module):
    """ Generator in Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial Network
    feature maps (n) and stride (s) feature maps (n) and stride (s)
    """

    def __init__(self):
        super(SRGAN_g, self).__init__()
        self.conv1 = Conv2d(
            out_channels=64, kernel_size=(3, 3), stride=(1, 1), act=tlx.ReLU, padding='SAME', W_init=W_init,
            data_format='channels_last'
        )
        self.residual_block = self.make_layer()
        self.conv2 = Conv2d(
            out_channels=64, kernel_size=(3, 3), stride=(1, 1), padding='SAME', W_init=W_init,
            data_format='channels_last', b_init=None
        )
        self.bn1 = BatchNorm2d(num_features=64, act=None, gamma_init=G_init, data_format='channels_last')
        self.conv3 = Conv2d(out_channels=256, kernel_size=(3, 3), stride=(1, 1), padding='SAME', W_init=W_init, data_format='channels_last')
        self.subpiexlconv1 = SubpixelConv2d(data_format='channels_last', scale=2, act=tlx.ReLU)
        self.conv4 = Conv2d(out_channels=256, kernel_size=(3, 3), stride=(1, 1), padding='SAME', W_init=W_init, data_format='channels_last')
        self.subpiexlconv2 = SubpixelConv2d(data_format='channels_last', scale=2, act=tlx.ReLU)
        self.conv5 = Conv2d(3, kernel_size=(1, 1), stride=(1, 1), act=tlx.Tanh, padding='SAME', W_init=W_init, data_format='channels_last')
        self.add = Add()

    def make_layer(self):
        layer_list = []
        for i in range(16):
            layer_list.append(ResidualBlock())
        return Sequential(layer_list)

    def forward(self, x):
        x = self.conv1(x)
        temp = x
        x = self.residual_block(x)
        x = self.conv2(x)
        x = self.bn1(x)
        x = self.add(x, temp)
        # x = x + temp
        x = self.conv3(x)
        x = self.subpiexlconv1(x)
        x = self.conv4(x)
        x = self.subpiexlconv2(x)
        x = self.conv5(x)

        return x


class SRGAN_g2(Module):
    """ Generator in Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial Network
    feature maps (n) and stride (s) feature maps (n) and stride (s)
    96x96 --> 384x384
    Use Resize Conv
    """

    def __init__(self):
        super(SRGAN_g2, self).__init__()
        self.conv1 = Conv2d(
            out_channels=64, kernel_size=(3, 3), stride=(1, 1), act=None, padding='SAME', W_init=W_init,
            data_format='channels_last'
        )
        self.residual_block = self.make_layer()
        self.conv2 = Conv2d(
            out_channels=64, kernel_size=(3, 3), stride=(1, 1), padding='SAME', W_init=W_init,
            data_format='channels_last', b_init=None
        )
        self.bn1 = BatchNorm2d(act=None, gamma_init=G_init, data_format='channels_last')
        self.upsample1 = UpSampling2d(data_format='channels_last', scale=(2, 2), method='bilinear')
        self.conv3 = Conv2d(
            out_channels=64, kernel_size=(3, 3), stride=(1, 1), padding='SAME', W_init=W_init,
            data_format='channels_last', b_init=None
        )
        self.bn2 = BatchNorm2d(act=tlx.ReLU, gamma_init=G_init, data_format='channels_last')
        self.upsample2 = UpSampling2d(data_format='channels_last', scale=(4, 4), method='bilinear')
        self.conv4 = Conv2d(
            out_channels=32, kernel_size=(3, 3), stride=(1, 1), padding='SAME', W_init=W_init,
            data_format='channels_last', b_init=None
        )
        self.bn3 = BatchNorm2d(act=tlx.ReLU, gamma_init=G_init, data_format='channels_last')
        self.conv5 = Conv2d(
            out_channels=3, kernel_size=(1, 1), stride=(1, 1), act=tlx.Tanh, padding='SAME', W_init=W_init
        )

    def make_layer(self):
        layer_list = []
        for i in range(16):
            layer_list.append(ResidualBlock())
        return Sequential(layer_list)

    def forward(self, x):
        x = self.conv1(x)
        temp = x
        x = self.residual_block(x)
        x = self.conv2(x)
        x = self.bn1(x)
        x = x + temp
        x = self.upsample1(x)
        x = self.conv3(x)
        x = self.bn2(x)
        x = self.upsample2(x)
        x = self.conv4(x)
        x = self.bn3(x)
        x = self.conv5(x)
        return x


class SRGAN_d2(Module):
    """ Discriminator in Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial Network
    feature maps (n) and stride (s) feature maps (n) and stride (s)
    """

    def __init__(self, ):
        super(SRGAN_d2, self).__init__()
        self.conv1 = Conv2d(
            out_channels=64, kernel_size=(3, 3), stride=(1, 1), act=tlx.LeakyReLU(negative_slope=0.2), padding='SAME',
            W_init=W_init, data_format='channels_last'
        )
        self.conv2 = Conv2d(
            out_channels=64, kernel_size=(3, 3), stride=(2, 2), act=tlx.LeakyReLU(negative_slope=0.2), padding='SAME',
            W_init=W_init, data_format='channels_last', b_init=None
        )
        self.bn1 = BatchNorm2d(gamma_init=G_init, data_format='channels_last')
        self.conv3 = Conv2d(
            out_channels=128, kernel_size=(3, 3), stride=(1, 1), act=tlx.LeakyReLU(negative_slope=0.2), padding='SAME',
            W_init=W_init, data_format='channels_last', b_init=None
        )
        self.bn2 = BatchNorm2d(gamma_init=G_init, data_format='channels_last')
        self.conv4 = Conv2d(
            out_channels=128, kernel_size=(3, 3), stride=(2, 2), act=tlx.LeakyReLU(negative_slope=0.2), padding='SAME',
            W_init=W_init, data_format='channels_last', b_init=None
        )
        self.bn3 = BatchNorm2d(gamma_init=G_init, data_format='channels_last')
        self.conv5 = Conv2d(
            out_channels=256, kernel_size=(3, 3), stride=(1, 1), act=tlx.LeakyReLU(negative_slope=0.2), padding='SAME',
            W_init=W_init, data_format='channels_last', b_init=None
        )
        self.bn4 = BatchNorm2d(gamma_init=G_init, data_format='channels_last')
        self.conv6 = Conv2d(
            out_channels=256, kernel_size=(3, 3), stride=(2, 2), act=tlx.LeakyReLU(negative_slope=0.2), padding='SAME',
            W_init=W_init, data_format='channels_last', b_init=None
        )
        self.bn5 = BatchNorm2d(gamma_init=G_init, data_format='channels_last')
        self.conv7 = Conv2d(
            out_channels=512, kernel_size=(3, 3), stride=(1, 1), act=tlx.LeakyReLU(negative_slope=0.2), padding='SAME',
            W_init=W_init, data_format='channels_last', b_init=None
        )
        self.bn6 = BatchNorm2d(gamma_init=G_init, data_format='channels_last')
        self.conv8 = Conv2d(
            out_channels=512, kernel_size=(3, 3), stride=(2, 2), act=tlx.LeakyReLU(negative_slope=0.2), padding='SAME',
            W_init=W_init, data_format='channels_last', b_init=None
        )
        self.bn7 = BatchNorm2d(gamma_init=G_init, data_format='channels_last')
        self.flat = Flatten()
        self.dense1 = Linear(out_features=1024, act=tlx.LeakyReLU(negative_slope=0.2))
        self.dense2 = Linear(out_features=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.bn1(x)
        x = self.conv3(x)
        x = self.bn2(x)
        x = self.conv4(x)
        x = self.bn3(x)
        x = self.conv5(x)
        x = self.bn4(x)
        x = self.conv6(x)
        x = self.bn5(x)
        x = self.conv7(x)
        x = self.bn6(x)
        x = self.conv8(x)
        x = self.bn7(x)
        x = self.flat(x)
        x = self.dense1(x)
        x = self.dense2(x)
        logits = x
        n = tlx.sigmoid(x)
        return n, logits


class SRGAN_d(Module):

    def __init__(self, dim=64):
        super(SRGAN_d, self).__init__()
        self.conv1 = Conv2d(
            out_channels=dim, kernel_size=(4, 4), stride=(2, 2), act=tlx.LeakyReLU, padding='SAME', W_init=W_init,
            data_format='channels_last'
        )
        self.conv2 = Conv2d(
            out_channels=dim * 2, kernel_size=(4, 4), stride=(2, 2), act=None, padding='SAME', W_init=W_init,
            data_format='channels_last', b_init=None
        )
        self.bn1 = BatchNorm2d(num_features=dim * 2, act=tlx.LeakyReLU, gamma_init=G_init, data_format='channels_last')
        self.conv3 = Conv2d(
            out_channels=dim * 4, kernel_size=(4, 4), stride=(2, 2), act=None, padding='SAME', W_init=W_init,
            data_format='channels_last', b_init=None
        )
        self.bn2 = BatchNorm2d(num_features=dim * 4, act=tlx.LeakyReLU, gamma_init=G_init, data_format='channels_last')
        self.conv4 = Conv2d(
            out_channels=dim * 8, kernel_size=(4, 4), stride=(2, 2), act=None, padding='SAME', W_init=W_init,
            data_format='channels_last', b_init=None
        )
        self.bn3 = BatchNorm2d(num_features=dim * 8, act=tlx.LeakyReLU, gamma_init=G_init, data_format='channels_last')
        self.conv5 = Conv2d(
            out_channels=dim * 16, kernel_size=(4, 4), stride=(2, 2), act=None, padding='SAME', W_init=W_init,
            data_format='channels_last', b_init=None
        )
        self.bn4 = BatchNorm2d(num_features=dim * 16, act=tlx.LeakyReLU, gamma_init=G_init, data_format='channels_last')
        self.conv6 = Conv2d(
            out_channels=dim * 32, kernel_size=(4, 4), stride=(2, 2), act=None, padding='SAME', W_init=W_init,
            data_format='channels_last', b_init=None
        )
        self.bn5 = BatchNorm2d(num_features=dim * 32, act=tlx.LeakyReLU, gamma_init=G_init, data_format='channels_last')
        self.conv7 = Conv2d(
            out_channels=dim * 16, kernel_size=(1, 1), stride=(1, 1), act=None, padding='SAME', W_init=W_init,
            data_format='channels_last', b_init=None
        )
        self.bn6 = BatchNorm2d(num_features=dim * 16, act=tlx.LeakyReLU, gamma_init=G_init, data_format='channels_last')
        self.conv8 = Conv2d(
            out_channels=dim * 8, kernel_size=(1, 1), stride=(1, 1), act=None, padding='SAME', W_init=W_init,
            data_format='channels_last', b_init=None
        )
        self.bn7 = BatchNorm2d(num_features=dim * 8, act=None, gamma_init=G_init, data_format='channels_last')
        self.conv9 = Conv2d(
            out_channels=dim * 2, kernel_size=(1, 1), stride=(1, 1), act=None, padding='SAME', W_init=W_init,
            data_format='channels_last', b_init=None
        )
        self.bn8 = BatchNorm2d(num_features=dim * 2, act=tlx.LeakyReLU, gamma_init=G_init, data_format='channels_last')
        self.conv10 = Conv2d(
            out_channels=dim * 2, kernel_size=(3, 3), stride=(1, 1), act=None, padding='SAME', W_init=W_init,
            data_format='channels_last', b_init=None
        )
        self.bn9 = BatchNorm2d(num_features=dim * 2, act=tlx.LeakyReLU, gamma_init=G_init, data_format='channels_last')
        self.conv11 = Conv2d(
            out_channels=dim * 8, kernel_size=(3, 3), stride=(1, 1), act=None, padding='SAME', W_init=W_init,
            data_format='channels_last', b_init=None
        )
        self.bn10 = BatchNorm2d(num_features=dim * 8, gamma_init=G_init, data_format='channels_last')
        self.add = Elementwise(combine_fn=tlx.add, act=tlx.LeakyReLU)
        self.flat = Flatten()
        self.dense = Linear(out_features=1, W_init=W_init)

    def forward(self, x):

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.bn1(x)
        x = self.conv3(x)
        x = self.bn2(x)
        x = self.conv4(x)
        x = self.bn3(x)
        x = self.conv5(x)
        x = self.bn4(x)
        x = self.conv6(x)
        x = self.bn5(x)
        x = self.conv7(x)
        x = self.bn6(x)
        x = self.conv8(x)
        x = self.bn7(x)
        temp = x
        x = self.conv9(x)
        x = self.bn8(x)
        x = self.conv10(x)
        x = self.bn9(x)
        x = self.conv11(x)
        x = self.bn10(x)
        x = self.add([temp, x])
        x = self.flat(x)
        x = self.dense(x)

        return x


class Vgg19_simple_api(Module):

    def __init__(self):
        super(Vgg19_simple_api, self).__init__()
        """ conv1 """
        self.conv1 = Conv2d(out_channels=64, kernel_size=(3, 3), stride=(1, 1), act=tlx.ReLU, padding='SAME')
        self.conv2 = Conv2d(out_channels=64, kernel_size=(3, 3), stride=(1, 1), act=tlx.ReLU, padding='SAME')
        self.maxpool1 = MaxPool2d(kernel_size=(2, 2), stride=(2, 2), padding='SAME')
        """ conv2 """
        self.conv3 = Conv2d(out_channels=128, kernel_size=(3, 3), stride=(1, 1), act=tlx.ReLU, padding='SAME')
        self.conv4 = Conv2d(out_channels=128, kernel_size=(3, 3), stride=(1, 1), act=tlx.ReLU, padding='SAME')
        self.maxpool2 = MaxPool2d(kernel_size=(2, 2), stride=(2, 2), padding='SAME')
        """ conv3 """
        self.conv5 = Conv2d(out_channels=256, kernel_size=(3, 3), stride=(1, 1), act=tlx.ReLU, padding='SAME')
        self.conv6 = Conv2d(out_channels=256, kernel_size=(3, 3), stride=(1, 1), act=tlx.ReLU, padding='SAME')
        self.conv7 = Conv2d(out_channels=256, kernel_size=(3, 3), stride=(1, 1), act=tlx.ReLU, padding='SAME')
        self.conv8 = Conv2d(out_channels=256, kernel_size=(3, 3), stride=(1, 1), act=tlx.ReLU, padding='SAME')
        self.maxpool3 = MaxPool2d(kernel_size=(2, 2), stride=(2, 2), padding='SAME')
        """ conv4 """
        self.conv9 = Conv2d(out_channels=512, kernel_size=(3, 3), stride=(1, 1), act=tlx.ReLU, padding='SAME')
        self.conv10 = Conv2d(out_channels=512, kernel_size=(3, 3), stride=(1, 1), act=tlx.ReLU, padding='SAME')
        self.conv11 = Conv2d(out_channels=512, kernel_size=(3, 3), stride=(1, 1), act=tlx.ReLU, padding='SAME')
        self.conv12 = Conv2d(out_channels=512, kernel_size=(3, 3), stride=(1, 1), act=tlx.ReLU, padding='SAME')
        self.maxpool4 = MaxPool2d(kernel_size=(2, 2), stride=(2, 2), padding='SAME')  # (batch_size, 14, 14, 512)
        """ conv5 """
        self.conv13 = Conv2d(out_channels=512, kernel_size=(3, 3), stride=(1, 1), act=tlx.ReLU, padding='SAME')
        self.conv14 = Conv2d(out_channels=512, kernel_size=(3, 3), stride=(1, 1), act=tlx.ReLU, padding='SAME')
        self.conv15 = Conv2d(out_channels=512, kernel_size=(3, 3), stride=(1, 1), act=tlx.ReLU, padding='SAME')
        self.conv16 = Conv2d(out_channels=512, kernel_size=(3, 3), stride=(1, 1), act=tlx.ReLU, padding='SAME')
        self.maxpool5 = MaxPool2d(kernel_size=(2, 2), stride=(2, 2), padding='SAME')  # (batch_size, 7, 7, 512)
        """ fc 6~8 """
        self.flat = Flatten()
        self.dense1 = Linear(out_features=4096, act=tlx.ReLU)
        self.dense2 = Linear(out_features=4096, act=tlx.ReLU)
        self.dense3 = Linear(out_features=1000, act=tlx.identity)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.maxpool1(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.maxpool2(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.conv7(x)
        x = self.conv8(x)
        x = self.maxpool3(x)
        x = self.conv9(x)
        x = self.conv10(x)
        x = self.conv11(x)
        x = self.conv12(x)
        x = self.maxpool4(x)
        conv = x
        x = self.conv13(x)
        x = self.conv14(x)
        x = self.conv15(x)
        x = self.conv16(x)
        x = self.maxpool5(x)
        x = self.flat(x)
        x = self.dense1(x)
        x = self.dense2(x)
        x = self.dense3(x)

        return x, conv


net = SRGAN_g()
net.init_build(tlx.nn.Input(shape=(1, 96, 96, 3)))
net.load_weights('model/g.npz', format='npz_dict')
input = tlx.nn.Input(shape=(1, 96, 96, 3))

onnx_model = export(net, input_spec=input, path='srgan.onnx')

sess = onnxruntime.InferenceSession('srgan.onnx')

input_name = sess.get_inputs()[0].name
output_name = sess.get_outputs()[0].name

# TODO DYNAMIC INPUT SHAPE
valid_hr_img = tlx.vision.load_image('data/0882.png')
valid_lr_img = np.asarray(valid_hr_img)
hr_size = [valid_hr_img.shape[0], valid_hr_img.shape[1]]
valid_lr_img = cv2.resize(valid_lr_img, dsize=(hr_size[1]//4, hr_size[0]//4))
lr_size = [valid_hr_img.shape[0]//4, valid_hr_img.shape[1]//4]
input = (valid_lr_img / 127.5) - 1
input = np.asarray(input, dtype=np.float32)
input = input[np.newaxis, :, :, :]
output = sess.run([output_name], {input_name : input})
output = np.asarray((output + 1) * 127.5, dtype=np.uint8)

plt.figure()
plt.subplot(1,3,1)
plt.plot(valid_hr_img)

plt.subplot(1,3,2)
plt.plot(valid_lr_img)

plt.subplot(output)
plt.show()