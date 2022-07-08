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

The following is an example of converting a multi-layer perceptron. You can get the code from [here](https://github.com/tensorlayer/TLX2ONNX/tree/main/tests/test_merge.py).
```python
import os
os.environ["TL_BACKEND"] = 'tensorflow'
import tensorlayerx as tlx
from tensorlayerx.nn import Module
from tensorlayerx.nn import Linear, Concat, Elementwise
from tlx2onnx.main import export
import onnxruntime as rt
import numpy as np

class CustomModel(Module):
    def __init__(self):
        super(CustomModel, self).__init__(name="custom")
        self.linear1 = Linear(in_features=20, out_features=10, act=tlx.ReLU, name='relu1_1')
        self.linear2 = Linear(in_features=20, out_features=10, act=tlx.ReLU, name='relu2_1')
        self.concat = Concat(concat_dim=1, name='concat_layer')

    def forward(self, inputs):
        d1 = self.linear1(inputs)
        d2 = self.linear2(inputs)
        outputs = self.concat([d1, d2])
        return outputs

net = CustomModel()
input = tlx.nn.Input(shape=(3, 20), init=tlx.initializers.RandomNormal())
net.set_eval()
output = net(input)
print("tlx out", output)
onnx_model = export(net, input_spec=input, path='concat.onnx')

# Infer Model
sess = rt.InferenceSession('concat.onnx')
input_name = sess.get_inputs()[0].name
output_name = sess.get_outputs()[0].name
input_data = np.array(input, dtype=np.float32)
result = sess.run([output_name], {input_name: input_data})
print('onnx out', result)
```
The converted onnx file can be viewed via Netron.

<p align="center"><img src="https://git.openi.org.cn/laich/pose_data/raw/commit/7ac74f03dbfdd8e023cdb205cd415a8571ebb91a/onnxfile.png" width="580"\></p>


The converted results have almost no loss of accuracy. 
And the graph show the input and output sizes of each layer, which is very helpful for checking the model.


# Citation

If you find TensorLayerX or TLX2ONNX useful for your project, please cite the following papers：

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
