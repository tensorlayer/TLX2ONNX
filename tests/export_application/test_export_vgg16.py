#! /usr/bin/python
# -*- coding: utf-8 -*-

import tensorlayerx as tlx
from tlx2onnx import export
import os
import numpy as np
from tensorlayerx import logging
from tensorlayerx.files import assign_weights, maybe_download_and_extract
from tensorlayerx.nn import (BatchNorm, Conv2d, Linear, Flatten, Sequential, MaxPool2d)
from tensorlayerx.nn import Module
import onnxruntime as rt
from tests.export_application.imagenet_classes import class_names
import tensorflow as tf
import onnx
from onnx import shape_inference

__all__ = [
    'VGG',
    'vgg16'
]

layer_names = [
    ['conv1_1', 'conv1_2'], 'pool1', ['conv2_1', 'conv2_2'], 'pool2',
    ['conv3_1', 'conv3_2', 'conv3_3', 'conv3_4'], 'pool3', ['conv4_1', 'conv4_2', 'conv4_3', 'conv4_4'], 'pool4',
    ['conv5_1', 'conv5_2', 'conv5_3', 'conv5_4'], 'pool5', 'flatten', 'fc1_relu', 'fc2_relu', 'outputs'
]

cfg = {
    'A': [[64], 'M', [128], 'M', [256, 256], 'M', [512, 512], 'M', [512, 512], 'M', 'F', 'fc1', 'fc2', 'O'],
    'B': [[64, 64], 'M', [128, 128], 'M', [256, 256], 'M', [512, 512], 'M', [512, 512], 'M', 'F', 'fc1', 'fc2', 'O'],
    'D':
        [
            [64, 64], 'M', [128, 128], 'M', [256, 256, 256], 'M', [512, 512, 512], 'M', [512, 512, 512], 'M', 'F',
            'fc1', 'fc2', 'O'
        ],
    'E':
        [
            [64, 64], 'M', [128, 128], 'M', [256, 256, 256, 256], 'M', [512, 512, 512, 512], 'M', [512, 512, 512, 512],
            'M', 'F', 'fc1', 'fc2', 'O'
        ],
}

mapped_cfg = {
    'vgg11': 'A',
    'vgg11_bn': 'A',
    'vgg13': 'B',
    'vgg13_bn': 'B',
    'vgg16': 'D',
    'vgg16_bn': 'D',
    'vgg19': 'E',
    'vgg19_bn': 'E'
}

model_urls = {
    'vgg16': 'http://www.cs.toronto.edu/~frossard/vgg16/',
}

model_saved_name = {'vgg16': 'vgg16_weights.npz', 'vgg19': 'vgg19.npy'}


class VGG(Module):

    def __init__(self, layer_type, batch_norm=False, end_with='outputs', name=None):
        super(VGG, self).__init__(name=name)
        self.end_with = end_with

        config = cfg[mapped_cfg[layer_type]]
        self.make_layer = make_layers(config, batch_norm, end_with)

    def forward(self, inputs):
        """
        inputs : tensor
            Shape [None, 224, 224, 3], value range [0, 1].
        """

        # inputs = inputs * 255 - tlx.convert_to_tensor(np.array([123.68, 116.779, 103.939], dtype=np.float32).reshape([1, 1, 1, 3]))
        out = self.make_layer(inputs)


        return out


def make_layers(config, batch_norm=False, end_with='outputs'):
    layer_list = []
    is_end = False
    for layer_group_idx, layer_group in enumerate(config):
        if isinstance(layer_group, list):
            for idx, layer in enumerate(layer_group):
                layer_name = layer_names[layer_group_idx][idx]
                n_filter = layer
                if idx == 0:
                    if layer_group_idx > 0:
                        in_channels = config[layer_group_idx - 2][-1]
                    else:
                        in_channels = 3
                else:
                    in_channels = layer_group[idx - 1]
                layer_list.append(
                    Conv2d(
                        out_channels=n_filter, kernel_size=(3, 3), stride=(1, 1), act=tlx.ReLU, padding='SAME',
                        in_channels=in_channels, name=layer_name
                    )
                )
                if batch_norm:
                    layer_list.append(BatchNorm(num_features=n_filter))
                if layer_name == end_with:
                    is_end = True
                    break
        else:
            layer_name = layer_names[layer_group_idx]
            if layer_group == 'M':
                layer_list.append(MaxPool2d(kernel_size=(2, 2), stride=(2, 2), padding='SAME', name=layer_name))
            elif layer_group == 'O':
                layer_list.append(Linear(out_features=1000, in_features=4096, name=layer_name))
            elif layer_group == 'F':
                layer_list.append(Flatten(name='flatten'))
            elif layer_group == 'fc1':
                layer_list.append(Linear(out_features=4096, act=tlx.ReLU, in_features=512 * 7 * 7, name=layer_name))
            elif layer_group == 'fc2':
                layer_list.append(Linear(out_features=4096, act=tlx.ReLU, in_features=4096, name=layer_name))
            if layer_name == end_with:
                is_end = True
        if is_end:
            break
    return Sequential(layer_list)


def restore_model(model, layer_type):
    logging.info("Restore pre-trained weights")
    # download weights
    maybe_download_and_extract(model_saved_name[layer_type], 'model', model_urls[layer_type])
    weights = []
    npz = np.load(os.path.join('model', model_saved_name[layer_type]), allow_pickle=True)
    # get weight list
    for val in sorted(npz.items()):
        logging.info("  Loading weights %s in %s" % (str(val[1].shape), val[0]))
        weights.append(val[1])
        if len(model.all_weights) == len(weights):
            break
    assign_weights(weights, model)
    del weights


def vgg16(pretrained=False, end_with='outputs', mode='dynamic', name=None):
    if mode == 'dynamic':
        model = VGG(layer_type='vgg16', batch_norm=False, end_with=end_with, name=name)
    if pretrained:
        restore_model(model, layer_type='vgg16')
    return model



input = tlx.nn.Input(shape=(1, 224, 224, 3))
net = vgg16(pretrained=True)
net.set_eval()
onnx_model = export(net, input_spec=input, path='vgg.onnx')

# Infer Model
sess = rt.InferenceSession('vgg.onnx')

input_name = sess.get_inputs()[0].name
output_name = sess.get_outputs()[0].name

# Preprocess input data
img = tlx.vision.load_image('data/tiger.jpeg')
img = tlx.vision.transforms.Resize((224, 224))(img).astype(np.float32) / 255
inputs = img * 255 - tlx.convert_to_tensor(np.array([123.68, 116.779, 103.939], dtype=np.float32))
input_data = inputs[np.newaxis, :, :, :]
input_data = np.array(input_data, dtype=np.float32)
# Infer output
result = sess.run([output_name], {input_name: input_data})

print(np.shape(result))
result = np.squeeze(result, axis = (0, 1))
probs = tf.nn.softmax(result).numpy()
print(probs)
preds = (np.argsort(probs)[::-1])[0:5]
for p in preds:
    print(class_names[p], probs[p])

# # Debug Infer
# def get_tensor_shape(tensor):
#     dims = tensor.type.tensor_type.shape.dim
#     n = len(dims)
#     return [dims[i].dim_value for i in range(n)]
#
#
# def runtime_infer(onnx_model):
#     graph = onnx_model.graph
#     graph.output.insert(0, graph.input[0])
#     for i, tensor in enumerate(graph.value_info):
#         graph.output.insert(i + 1, tensor)
#     model_file = "temp.onnx"
#     onnx.save(onnx_model, model_file)
#
#     sess = rt.InferenceSession(model_file)
#     input_name = sess.get_inputs()[0].name
#     # preprocess input
#     img = tlx.vision.load_image('data/tiger.jpeg')
#     img = tlx.vision.transforms.Resize((224, 224))(img).astype(np.float32) / 255
#     inputs = img * 255 - tlx.convert_to_tensor(np.array([123.68, 116.779, 103.939], dtype=np.float32))
#     input_data = inputs[np.newaxis, :, :, :]
#     input_data = np.array(input_data, dtype=np.float32)
#
#     outputs = {}
#     for out in sess.get_outputs():
#         tensor = sess.run([out.name], {input_name: input_data})
#         outputs[str(out.name)] = np.array(tensor[0]).shape
#         if out.name == '_inputlayer_1_node_00_t_t':
#             print(out.name, tensor, np.shape(tensor))
#     # os.remove(model_file)
#     return outputs
#
#
# def infer_shapes(model_file, running_mode=False):
#     onnx_model = onnx.load(model_file)
#     onnx.checker.check_model(onnx_model)
#     inferred_onnx_model = shape_inference.infer_shapes(onnx_model)
#
#     save_path = model_file[:-5] + "_new.onnx"
#     onnx.save(inferred_onnx_model, save_path)
#     print("Model is saved in:", save_path)
#
#     outputs = {}
#     if running_mode:
#         outputs = runtime_infer(inferred_onnx_model)
#     else:
#         graph = inferred_onnx_model.graph
#         # only 1 input tensor
#         tensor = graph.input[0]
#         outputs[str(tensor.name)] = get_tensor_shape(tensor)
#         # process tensor
#         for tensor in graph.value_info:
#             outputs[str(tensor.name)] = get_tensor_shape(tensor)
#         # output tensor
#         for tensor in graph.output:
#             outputs[str(tensor.name)] = get_tensor_shape(tensor)
#     return outputs
#
# if __name__ == '__main__':
#     model_1 = "vgg.onnx"
#     outputs = infer_shapes(model_1, True)
#     print(outputs)

