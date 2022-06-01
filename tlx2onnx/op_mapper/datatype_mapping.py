# SPDX-License-Identifier: Apache-2.0

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from onnx import TensorProto, SequenceProto
from typing import Text, Any
import numpy as np  # type: ignore

TENSOR_TYPE_TO_NP_TYPE = {
    TensorProto.FLOAT: np.dtype('float32'),
    TensorProto.UINT8: np.dtype('uint8'),
    TensorProto.INT8: np.dtype('int8'),
    TensorProto.UINT16: np.dtype('uint16'),
    TensorProto.INT16: np.dtype('int16'),
    TensorProto.INT32: np.dtype('int32'),
    TensorProto.INT64: np.dtype('int64'),
    TensorProto.BOOL: np.dtype('bool'),
    TensorProto.FLOAT16: np.dtype('float16'),
    TensorProto.DOUBLE: np.dtype('float64'),
    TensorProto.COMPLEX64: np.dtype('complex64'),
    TensorProto.COMPLEX128: np.dtype('complex128'),
    TensorProto.UINT32: np.dtype('uint32'),
    TensorProto.UINT64: np.dtype('uint64'),
    TensorProto.STRING: np.dtype(np.object)
}

NP_TYPE_TO_TENSOR_TYPE = {v: k for k, v in TENSOR_TYPE_TO_NP_TYPE.items()}

TENSOR_TYPE_TO_STORAGE_TENSOR_TYPE = {
    TensorProto.FLOAT: TensorProto.FLOAT,
    TensorProto.UINT8: TensorProto.INT32,
    TensorProto.INT8: TensorProto.INT32,
    TensorProto.UINT16: TensorProto.INT32,
    TensorProto.INT16: TensorProto.INT32,
    TensorProto.INT32: TensorProto.INT32,
    TensorProto.INT64: TensorProto.INT64,
    TensorProto.BOOL: TensorProto.INT32,
    TensorProto.FLOAT16: TensorProto.UINT16,
    TensorProto.BFLOAT16: TensorProto.UINT16,
    TensorProto.DOUBLE: TensorProto.DOUBLE,
    TensorProto.COMPLEX64: TensorProto.FLOAT,
    TensorProto.COMPLEX128: TensorProto.DOUBLE,
    TensorProto.UINT32: TensorProto.UINT32,
    TensorProto.UINT64: TensorProto.UINT64,
    TensorProto.STRING: TensorProto.STRING,
}

STORAGE_TENSOR_TYPE_TO_FIELD = {
    TensorProto.FLOAT: 'float_data',
    TensorProto.INT32: 'int32_data',
    TensorProto.INT64: 'int64_data',
    TensorProto.UINT16: 'int32_data',
    TensorProto.DOUBLE: 'double_data',
    TensorProto.COMPLEX64: 'float_data',
    TensorProto.COMPLEX128: 'double_data',
    TensorProto.UINT32: 'uint64_data',
    TensorProto.UINT64: 'uint64_data',
    TensorProto.STRING: 'string_data',
    TensorProto.BOOL: 'int32_data',
}

STORAGE_ELEMENT_TYPE_TO_FIELD = {
    SequenceProto.TENSOR: 'tensor_values',
    SequenceProto.SPARSE_TENSOR: 'sparse_tensor_values',
    SequenceProto.SEQUENCE: 'sequence_values',
    SequenceProto.MAP: 'map_values'
}

STR_TYPE_TO_TENSOR_TYPE = {
    'float32': TensorProto.FLOAT,
    'uint8': TensorProto.UINT8,
    'int8': TensorProto.INT8,
    'uint16': TensorProto.UINT16,
    'int16': TensorProto.INT16,
    'int32': TensorProto.INT32,
    'int64': TensorProto.INT64,
    'bool': TensorProto.BOOL,
    'float16': TensorProto.FLOAT16,
    'float64': TensorProto.DOUBLE,
    'complex64': TensorProto.COMPLEX64,
    'complex128': TensorProto.COMPLEX128,
    'uint32': TensorProto.UINT32,
    'uint64': TensorProto.UINT64,
}