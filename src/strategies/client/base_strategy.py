from abc import ABC, abstractmethod


class ClientStrategy(ABC):
    def __init__(self, client_instance):
        self.client = client_instance

    @abstractmethod
    def prepare_update_request(self, current_round, model_parameters, metrics):
        """
        根据给定的隐私模式，准备 ClientUpdate 消息。

        Args:
            current_round (int): 当前的训练轮次。
            model_parameters (dict): 模型参数。
            metrics (dict): 训练指标。

        Returns:
            federation_pb2.ClientUpdate: 准备好的更新请求。
        """
        pass 