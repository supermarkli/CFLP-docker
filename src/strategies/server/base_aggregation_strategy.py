from abc import ABC, abstractmethod


class AggregationStrategy(ABC):
    def __init__(self, server_instance):
        self.server = server_instance

    @abstractmethod
    def aggregate(self, request, context):
        """
        处理来自客户端的更新并进行聚合。

        Args:
            request: 来自客户端的 ClientUpdate 消息。
            context: gRPC 上下文。

        Returns:
            federation_pb2.ServerUpdate: 对客户端的响应。
        """
        pass

    @abstractmethod
    def prepare_setup_response(self, request):
        """
        为客户端准备注册和设置阶段的响应消息。

        Args:
            request: 来自客户端的 ClientInfo 消息。

        Returns:
            federation_pb2.SetupResponse: 准备好的设置响应。
        """
        pass

    @abstractmethod
    def aggregate_parameters(self, round_num):
        """
        聚合指定轮次的所有客户端参数。

        Args:
            round_num: 当前轮次。

        Returns:
            聚合后的模型参数。
        """
        pass

    @abstractmethod
    def evaluate_metrics(self, round_num):
        """
        评估指定轮次的所有客户端指标。

        Args:
            round_num: 当前轮次。
        """
        pass 