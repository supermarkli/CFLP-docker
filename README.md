# CFLP-docker: 可配置的隐私联邦学习平台

CFLP-docker (Configurable Federated Learning Platform) 是一个基于 Docker 和 gRPC 的联邦学习框架，专为研究和比较不同的隐私增强技术 (Privacy-Enhancing Technologies, PETs) 而设计。它提供了一个模块化的平台，让开发者和研究人员可以轻松地在多种隐私保护策略下进行联邦学习实验。

## 主要特性

- **模块化隐私策略**: 支持多种开箱即用的隐私保护方案，并可通过配置文件一键切换：
  - `none`: 标准联邦学习，用于性能基准测试。
  - `he`: 基于 Paillier 同态加密 (Homomorphic Encryption) 的方案，在客户端加密梯度后上传。
  - `mpc`: 基于 Shamir 秘密共享的安全多方计算 (Multi-Party Computation) 方案，允许多方协同计算而不泄露各自的私有数据。
  - `tee`: 基于可信执行环境 (Trusted Execution Environment) 的模拟方案，数据在服务器端的安全区域内进行处理。
- **容器化部署**: 使用 Docker 和 Docker Compose，一键即可启动整个联邦学习环境，包括一个中心服务器和多个客户端，极大地简化了部署和测试流程。
- **gRPC 通信**: 客户端与服务器之间采用高性能的 gRPC 框架进行通信，协议在 `federation.proto` 中清晰定义，保证了通信的效率和稳定性。
- **高度可配置**: 所有的实验参数，从联邦学习的轮次、客户端数量到特定隐私方案的参数（如加密密钥长度、MPC 秘密共享阈值等），都可以在一个中心化的 `default.yaml` 文件中进行配置。
- **可扩展性**: 项目结构清晰，易于扩展。可以方便地添加新的隐私保护策略或自定义模型。

## 系统架构

本平台采用经典的星型联邦学习拓扑结构，其所有组件均被容器化，并通过 Docker Compose 进行编排。

其核心架构由以下几部分组成：

- **服务容器 (Services)**:
    - **一个 `fl-server` 容器**: 作为中心协调者，负责管理整个联邦学习的生命周期，包括模型分发、客户端协调和安全聚合。
    - **多个 `fl-client` 容器**: 模拟独立的联邦学习参与方（如`fl-client-1`, `fl-client-2` 等）。每个客户端都持有自己的私有数据集，并独立进行本地训练。

- **网络 (Networking)**:
    - 所有容器都连接到一个名为 `federated-network` 的自定义 Docker bridge 网络中。
    - 在此网络内，客户端可以通过服务名 (`server`) 直接与服务端容器通信，无需关心底层的IP地址。服务器的 gRPC 服务运行在 `50051` 端口。

- **数据卷挂载 (Volume Mounts)**:
    - **数据集 (`data/`)**: 每个客户端容器都挂载了其专属的数据子目录 (e.g., `data/client1` -> `/app/data`)，这精确地模拟了联邦学习中数据不出本地的“数据孤岛”场景。
    - **证书 (`certs/`)**: `server` 和 `client` 容器分别挂载了用于 gRPC 安全通信的 TLS 证书和密钥，确保了信道安全。
    - **输出与日志 (`out/`, `logs/`)**: `server` 容器挂载了这两个目录，用于将训练过程中生成的性能图表和日志文件持久化地保存在主机上，方便分析和回顾。

- **通信协议 (Communication)**:
    - 客户端与服务器之间的所有交互都通过 gRPC 进行。`src/grpc/protos/federation.proto` 文件定义了通信的服务接口和消息格式，实现了高效、强类型的远程过程调用。

这种基于 Docker 的架构不仅保证了环境的一致性和可复现性，还极大地简化了部署和横向扩展（通过增加更多 client 服务）的复杂性。

## 核心工作流程

一次完整的联邦学习流程遵循以下步骤，该流程由 `src/grpc/protos/federation.proto` 中定义的 gRPC 服务驱动：

1.  **启动与注册 (`RegisterAndSetup`)**:
    - `docker-compose up` 启动 `server` 和所有 `client` 容器。
    - 每个客户端启动后，立即向服务器发起 `RegisterAndSetup` 请求，并告知自己的 `client_id` 和本地数据量。
    - 服务器收集所有客户端的注册信息。当达到 `federation.expected_clients` 所设定的数量时，服务器将为本次联邦学习任务选定一个**全局隐私模式**（根据 `default.yaml` 配置），并将此模式、初始模型及所需的加密材料（如HE公钥）通过 `SetupResponse` 返回给所有客户端。

2.  **同步等待 (`CheckTrainingStatus`)**:
    - 客户端进入轮询状态，周期性地调用 `CheckTrainingStatus` 来询问训练是否可以开始。
    - 服务器监控所有客户端的准备状态，当所有客户端都已准备就绪后，服务器会通过 `TrainingStatusResponse` 发出开始训练的信号。

3.  **本地训练与安全更新 (`SubmitUpdate`)**:
    - 收到开始信号后，每个客户端在自己的本地数据集上执行模型训练。
    - 训练完成后，客户端**根据服务器指定的隐私模式**，对本地模型更新（如梯度或权重）进行相应的安全处理（加密、秘密共享等）。
    - 客户端将处理后的模型更新封装在 `ClientUpdate` 消息中，通过 `SubmitUpdate` RPC 发送给服务器。这个消息体使用 `oneof` 结构来容纳不同隐私模式下的载荷。

4.  **安全聚合与模型更新**:
    - 服务器接收来自客户端的 `ClientUpdate`。
    - 服务器根据当前的隐私模式，调用 `src/strategies/server/` 中对应的聚合策略（如 `he_aggregation_strategy.py`）。
    - 聚合策略在**不暴露任何单个客户端原始数据**的前提下，对收集到的所有更新进行安全计算（例如，在密文上求和），从而生成新的全局模型。

5.  **获取新模型 (`GetGlobalModel`)**:
    - 客户端在提交更新后，可以调用 `GetGlobalModel` 来获取服务器聚合完毕的最新全局模型。
    - 获得新模型后，客户端用它来更新自己的本地模型，并准备进入下一轮训练。

6.  **循环与收敛**:
    - 上述第 2 到第 5 步循环执行，直到达到 `federation.max_rounds` 定义的最大轮次，或满足 `federation.convergence` 中定义的收敛条件。
    - 训练结束后，服务器会将最终的性能指标（如准确率、损失）和图表保存到 `logs/` 和 `out/` 目录。

## 隐私策略详解

本项目实现了四种不同的隐私保护等级，通过 `federation.privacy_mode` 进行切换。

### 1. `none` - 无保护
- **原理**: 标准的联邦平均 (FedAvg) 算法。
- **实现**: 客户端在本地训练后，将**明文**的模型权重或梯度直接发送给服务器。服务器以明文形式进行平均聚合。此模式主要用作性能基准。

### 2. `he` - 同态加密
- **原理**: 利用 Paillier 同态加密方案的加法同态特性。客户端可以在不解密的情况下，对密文进行某些计算（如加法）。
- **实现**:
    1.  服务器在启动时生成一对 Paillier 公私钥，并将**公钥**分发给所有客户端。
    2.  客户端使用公钥对本地计算出的模型梯度进行**加密**。
    3.  服务器收集所有客户端的加密梯度，并利用加法同态性，在**密文上直接求和**。
    4.  服务器使用**私钥**解密求和后的结果，得到聚合后的全局梯度，并用它来更新全局模型。
    - *库依赖*: `phe`

### 3. `mpc` - 安全多方计算
- **原理**: 基于 Shamir 秘密共享 (SSS) 方案。一个秘密值可以被拆分成多个“份额”，分发给不同参与方。只有当足够数量的份额组合在一起时，才能恢复出原始秘密。
- **实现**:
    1.  客户端将本地计算出的模型梯度，通过 SSS 算法**拆分成多个份额**。
    2.  每个客户端将自己的份额分发给其他所有客户端（在本项目中，为简化架构，所有份额先发送给服务器，由服务器代为分发和计算）。
    3.  服务器收集所有份额，并在这些份额上执行安全加法协议，计算出聚合梯度的份额。
    4.  服务器将聚合份额组合，恢复出最终的**明文**聚合梯度，并用其更新全局模型。
    - *库依赖*: `pyseltongue`

### 4. `tee` - 可信执行环境 (模拟)
- **原理**: TEE（如 Intel SGX）可以在服务器的 CPU 中创建一个硬件隔离的“安全区”（Enclave）。代码和数据在此区域内执行时，即使是服务器的操作系统也无法窥探。
- **实现 (模拟流程)**:
    1.  服务器模拟一个持有非对称密钥对的 Enclave，并将其**公钥**分发给客户端。
    2.  每个客户端生成一个一次性的**对称密钥**（如 AES），然后用服务器的公钥加密该对称密钥。
    3.  客户端使用该对称密钥加密自己的**明文梯度**。
    4.  客户端将“加密后的对称密钥”和“加密后的梯度”一同发送给服务器。
    5.  服务器模拟在 Enclave 内部，使用其私钥解密对称密钥，然后再用解密出的对称密钥解密梯度，得到明文梯度。
    6.  所有明文梯度都在 Enclave 内部被安全地聚合。

## 快速开始

### 前提条件

- [Docker](https://www.docker.com/get-started)
- [Docker Compose](https://docs.docker.com/compose/install/)

### 安装与运行

1.  **克隆项目**:
    ```bash
    git clone <your-repository-url>
    cd CFLP-docker
    ```

2.  **生成数据 (首次运行)**:
    如果 `data/client*` 目录为空，需要先为每个客户端生成它们各自的数据集。
    ```bash
    python src/data_process/generate_mnist_data.py
    ```

3.  **生成 gRPC 代码 (如果修改了 .proto 文件)**:
    如果 `src/grpc/protos/federation.proto` 文件被修改，需要重新生成客户端和服务器端的 gRPC 代码。
    ```bash
    python src/scripts/generate_grpc.py
    ```

4.  **启动联邦学习环境**:
    使用 Docker Compose 一键启动所有服务。
    ```bash
    docker-compose -f src/docker/docker-compose.yml up --build
    ```
    - 容器将会在后台启动并开始联邦学习过程。
    - 你可以通过 `docker logs -f fl-server` 查看服务器端的日志。

5.  **查看结果**:
    训练完成后，准确率和损失曲线图将保存在 `out/` 目录下。日志文件保存在 `logs/` 目录下。

6.  **停止并清理环境**:
    ```bash
    docker-compose -f src/docker/docker-compose.yml down
    ```

## 实验配置详解

要运行不同隐私策略的实验，或调整超参数，只需修改 `src/default.yaml` 文件即可。

### 切换隐私模式

修改 `federation.privacy_mode` 字段为你想要的模式：
```yaml
federation:
  # "none": 无保护，基准性能。
  # "he": 同态加密。
  # "mpc": 安全多方计算。
  # "tee": 可信执行环境。
  privacy_mode: "mpc" 
```

### 关键参数说明

- `federation.expected_clients`: 必须有多少个客户端注册成功后，训练才能开始。
- `federation.max_rounds`: 联邦学习的最大通信轮次。
- `training.learning_rate`: 客户端本地训练时使用的学习率。
- `encryption.key_size`: 用于 HE (Paillier) 或 TEE (RSA) 的密钥长度（位数）。密钥越长越安全，但计算开销越大。
- `mpc.shamir_k`: (Shamir's Threshold) 恢复MPC秘密所需的最小份额数。必须小于等于 `shamir_n`。
- `mpc.shamir_n`: (Total Shares) MPC中一个秘密被分割成的总份额数。通常等于客户端数量。
- `mpc.prime_mod`: 一个巨大的素数，用于定义MPC计算所在的有限域。必须足够大以避免计算溢出。


修改配置文件后，**重新运行步骤 4** (`docker-compose up`) 即可启动新的实验。

## 扩展性

该框架易于扩展，例如添加一种新的隐私保护方案：
1.  在 `src/strategies/client/` 下创建一个新的客户端策略文件 (e.g., `my_strategy.py`)。
2.  在 `src/strategies/server/` 下创建一个新的服务端聚合策略文件 (e.g., `my_aggregation_strategy.py`)。
3.  在 `federation.proto` 中为新策略定义相应的 `Payload`。
4.  更新服务端和客户端的策略加载逻辑，以识别新的 `privacy_mode`。

