import pickle
import numpy as np
from collections import OrderedDict
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.grpc.generated import federation_pb2

def serialize_parameters(parameters):
    """
    将 OrderedDict 类型的模型参数字典序列化为 bytes。
    """
    return pickle.dumps(parameters)

def deserialize_parameters(serialized_parameters):
    """
    将 bytes 反序列化为 OrderedDict 类型的模型参数字典。
    """
    return pickle.loads(serialized_parameters) 