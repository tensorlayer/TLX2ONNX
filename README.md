# TLX2ONNX
ONNX Model Exporter for TensorLayerX. It's updated on Both [OpenI](https://git.openi.org.cn/OpenI/TLX2ONNX) and [Github](https://github.com/tensorlayer/TLX2ONNX/). You can get a [free GPU](https://git.openi.org.cn/OpenI/TLX2ONNX/debugjob?debugListType=all) on OpenI to use this project.

## Introduction

TLX2ONNX enables users to convert models from TensorLayerX to ONNX.

- Supported operators. TLX2ONNX can stably export models to ONNX Opset 9~11, and partialy support lower version opset. More details please refer to [Operator list](OP_LIST.md).
- Supported Layers. You can find officially verified Layers by TLX2ONNX/tests in [TLX2ONNX/test](https://github.com/tensorlayer/TLX2ONNX/tree/main/tests).

## Installation

#### Via Pip
```bash
pip install tlx2onnx
```
 
#### From Source
```bash
 git clone https://github.com/tensorlayer/TLX2ONNX.git
 cd TLX2ONNX
 python setup.py install
```

## Usage
TLX2ONNX can convert models built using TensorLayerX Module Subclass and Layers, and the Layers support list can be found in [Operator list](OP_LIST.md).

```python
#! /usr/bin/python
# -*- coding: utf-8 -*-

import tensorlayerx as tlx
from tensorlayerx.nn import Module
from tensorlayerx.nn import Linear, Dropout, Flatten, ReLU6
from tlx2onnx import export
import onnxruntime as rt
import numpy as np

class MLP(Module):
    def __init__(self):
        super(MLP, self).__init__()
        # weights init
        self.flatten = Flatten()
        self.line1 = Linear(in_features=32, out_features=64, act=tlx.nn.LeakyReLU(0.3))
        self.d1 = Dropout()
        self.line2 = Linear(in_features=64, out_features=128, b_init=None, act=tlx.nn.ReLU)
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
net.eval()
input = tlx.nn.Input(shape=(3, 2, 2, 8))
onnx_model = export(net, input_spec=input, path='linear_model.onnx')

# Infer Model
sess = rt.InferenceSession('linear_model.onnx')
input_name = sess.get_inputs()[0].name
output_name = sess.get_outputs()[0].name
input_data = tlx.nn.Input(shape=(3, 2, 2, 8))
input_data = np.array(input_data, dtype=np.float32)
result = sess.run([output_name], {input_name: input_data})
print(result)
```

# Citation

If you find TensorLayerX useful for your project, please cite the following papers：

```
@article{tensorlayer2017,
    author  = {Dong, Hao and Supratak, Akara and Mai, Luo and Liu, Fangde and Oehmichen, Axel and Yu, Simiao and Guo, Yike},
    journal = {ACM Multimedia},
    title   = {{TensorLayer: A Versatile Library for Efficient Deep Learning Development}},
    url     = {http://tensorlayer.org},
    year    = {2017}
}

@inproceedings{tensorlayer2021,
  title={TensorLayer 3.0: A Deep Learning Library Compatible With Multiple Backends},
  author={Lai, Cheng and Han, Jiarong and Dong, Hao},
  booktitle={2021 IEEE International Conference on Multimedia \& Expo Workshops (ICMEW)},
  pages={1--3},
  year={2021},
  organization={IEEE}
}
```
