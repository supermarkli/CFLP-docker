import os
import sys
import time
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from .base_strategy import ClientStrategy
from src.grpc.generated import federation_pb2
from src.utils.parameter_utils import serialize_parameters
from src.utils.logging_config import get_logger

logger = get_logger()


class NoneClientStrategy(ClientStrategy):
    def prepare_update_request(self, current_round, model_parameters, metrics):
        """创建参数更新消息（明文）"""
        # 注意：BASE模式无加密，此处仅记录序列化时间作为"加密"阶段
        # 实际加密时间在 client_grpc.py 中统一测量
        serialized_params = serialize_parameters(model_parameters)
        
        training_metrics = federation_pb2.TrainingMetrics(
            test_acc=metrics.get('test_acc', 0.0),
            test_num=metrics.get('test_num', 0),
            auc=metrics.get('auc', 0.0),
            loss=metrics.get('loss', 0.0),
            train_num=metrics.get('train_num', 0)
        )
        
        model_params_proto = federation_pb2.ModelParameters(
            parameters=serialized_params
        )

        params_and_metrics = federation_pb2.ParametersAndMetrics(
            parameters=model_params_proto,
            metrics=training_metrics
        )
        
        plaintext_payload = federation_pb2.PlaintextPayload(
            parameters_and_metrics=params_and_metrics
        )
        
        return federation_pb2.ClientUpdate(
            client_id=self.client.client_id,
            round=current_round,
            plaintext=plaintext_payload
        ) 