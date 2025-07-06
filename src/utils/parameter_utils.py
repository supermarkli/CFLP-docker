import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from src.grpc.generated import federation_pb2

def serialize_parameters(parameters):
    """
    将模型参数序列化为 protobuf NumpyArray 字典。
    """
    serialized = {}
    for key, value in parameters.items():
        if isinstance(value, np.ndarray):
            numpy_array = federation_pb2.NumpyArray(
                data=value.tobytes(),
                shape=list(value.shape),
                dtype=str(value.dtype)
            )
            serialized[key] = numpy_array
        else:
            arr = np.array([value])
            numpy_array = federation_pb2.NumpyArray(
                data=arr.tobytes(),
                shape=[1],
                dtype=str(arr.dtype)
            )
            serialized[key] = numpy_array
    return serialized


def deserialize_parameters(parameters, param_mapping=None):
    """
    将 protobuf NumpyArray 字典反序列化为 numpy 参数字典。
    param_mapping: 可选参数名映射字典（如 {'weights': 'coef_'}）
    """
    deserialized = {}
    mapping = param_mapping or {}
    for key, value in parameters.items():
        mapped_key = mapping.get(key, key)
        try:
            dtype = np.dtype(value.dtype)
            arr = np.frombuffer(value.data, dtype=dtype).reshape(value.shape)
            if len(value.shape) == 1 and value.shape[0] == 1:
                deserialized[mapped_key] = arr[0]
            else:
                deserialized[mapped_key] = arr
        except (ValueError, TypeError) as e:
            raise ValueError(f"Failed to deserialize parameter {key}: {str(e)}")
    return deserialized 