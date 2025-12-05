# CFLP 联邦实验平台

本项目是机密联邦学习平台（Confidential Federated Learning Platform, CFLP）的联邦实验平台部分，是一个基于 Docker 和 gRPC 的联邦学习框架，专为研究和比较不同的隐私增强技术 (Privacy-Enhancing Technologies, PETs) 而设计。它提供了一个模块化的平台，让开发者和研究人员可以轻松地在多种隐私保护策略下进行联邦学习实验。

## 主要特性

- **模块化隐私策略**: 支持多种开箱即用的隐私保护方案，通过配置文件一键切换：
  - `none`: 标准联邦学习（FedAvg），明文传输，用于性能基准测试。
  - `he`: 基于 **CKKS 全同态加密**（TenSEAL 库）的方案，支持 SIMD 批处理，在密文空间直接聚合。
  - `mpc`: 基于 **Shamir 秘密共享**的安全多方计算方案，使用高性能向量化实现。
  - `tee`: 基于 **Intel TDX 硬件虚拟化隔离**的可信执行环境方案，服务端运行在 Trust Domain 中。
  - `sgx`: 基于 **Intel SGX 进程级隔离**的方案，聚合逻辑运行在 Gramine 托管的 Enclave 中。

- **容器化部署**: 使用 Docker 和 Docker Compose，一键启动完整的联邦学习环境，包括服务端、多个客户端以及 SGX 聚合器容器。

- **高性能 gRPC 通信**: 采用 gRPC 框架，支持 TLS 加密信道、Gzip 压缩、流式传输和 keepalive 机制，适应大规模加密参数传输。

- **灵活的数据分布**: 支持基于 **Dirichlet 分布**的 Non-IID 数据划分，通过 α 参数控制数据异构程度。

- **丰富的模型支持**: 内置 FedAvgCNN、ResNet18、VGG16 等模型，均适配 MNIST 和 CIFAR-10 数据集。

- **完善的训练优化**: 支持 CosineAnnealingLR 学习率调度、数据增强（RandomCrop、RandomHorizontalFlip）、混合精度训练（AMP）。

- **自动收敛检测**: 基于滑动窗口的准确率变化检测，自动判定训练收敛并终止。

- **实验可视化**: 自动生成收敛曲线、延迟分解图、内存占用对比等可视化图表。

## 威胁模型与安全边界

本平台的设计基于以下威胁模型假设：

### 信任边界

| 组件 | 信任级别 | 说明 |
|------|---------|------|
| 客户端环境 | **可信** | 客户端持有明文数据和模型，假设不会被攻击者控制 |
| 网络信道 | **不可信** | 通信可能被窃听，需要加密保护 |
| 服务端宿主 | **不可信** | 服务端操作系统和进程可能被攻击者控制 |
| TEE/Enclave | **可信** | 硬件隔离区域，即使宿主被攻破也能保护数据 |

### 各模式的安全保证

| 模式 | 传输层保护 | 应用层保护 | 服务端可见明文？ |
|------|-----------|-----------|----------------|
| `none` | TLS | 无 | ✅ 是 |
| `he` | TLS | CKKS 密文 | ❌ 否（仅见密文） |
| `mpc` | TLS | 秘密份额 | ❌ 否（需多方协作恢复） |
| `tee` | TLS | 混合加密 | ⚠️ 仅在 TD 内可见 |
| `sgx` | TLS | 混合加密 | ⚠️ 仅在 Enclave 内可见 |

### 端到端数据流保护

在 TEE/SGX 模式下，系统提供端到端的数据保护：

1. **客户端加密**: 模型更新使用 TEE/Enclave 的公钥进行混合加密后离开客户端
2. **密文穿透**: 加密数据穿过不可信的网络和服务端宿主进程
3. **隔离区解密**: 仅在硬件隔离的 TEE/Enclave 内部解密和聚合
4. **远程证明**: 客户端可验证 TEE/Enclave 的身份和代码完整性（通过 MRENCLAVE）

## 系统架构

本平台采用经典的星型联邦学习拓扑结构，所有组件均被容器化，通过 Docker Compose 进行编排。

### 部署拓扑

```
┌─────────────────────────────────────────────────────────────────┐
│                        宿主机 (Host)                              │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐              │
│  │ fl-client-1 │  │ fl-client-2 │  │ fl-client-3 │   (Docker)   │
│  │   (GPU)     │  │   (GPU)     │  │   (GPU)     │              │
│  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘              │
│         │                │                │                      │
│         └────────────────┼────────────────┘                      │
│                          │ gRPC (TLS)                            │
│                          ▼                                       │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │                    TDX VM / 普通 VM                         │  │
│  │  ┌─────────────────────────────────────────────────────┐  │  │
│  │  │                  fl-server (Docker)                  │  │  │
│  │  │  ┌─────────────────────────────────────────────┐    │  │  │
│  │  │  │  TEE 模式: 明文聚合在 TD 内执行              │    │  │  │
│  │  │  │  SGX 模式: 转发至 sgx-aggregator            │    │  │  │
│  │  │  └─────────────────────────────────────────────┘    │  │  │
│  │  └─────────────────────────────────────────────────────┘  │  │
│  │                          │ TCP Socket (仅 SGX)             │  │
│  │                          ▼                                 │  │
│  │  ┌─────────────────────────────────────────────────────┐  │  │
│  │  │           sgx-aggregator (Gramine-SGX)              │  │  │
│  │  │  ┌─────────────────────────────────────────────┐    │  │  │
│  │  │  │         SGX Enclave (硬件隔离)               │    │  │  │
│  │  │  │   - RSA 私钥驻留                            │    │  │  │
│  │  │  │   - 解密 + 聚合                             │    │  │  │
│  │  │  └─────────────────────────────────────────────┘    │  │  │
│  │  └─────────────────────────────────────────────────────┘  │  │
│  └───────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

### 核心架构组件

- **服务容器 (Services)**:
    - **`fl-server` 容器**: 中心协调者，负责管理联邦学习生命周期，包括客户端注册、状态同步、安全聚合和全局模型评估。
    - **`fl-client-*` 容器**: 模拟独立的联邦学习参与方，每个客户端持有私有数据集并独立进行本地训练。支持 GPU 加速。
    - **`sgx-aggregator` 容器**（仅 SGX 模式）: 运行在 Gramine-SGX 环境中的专用聚合器，通过硬件隔离保护明文聚合过程。

- **网络与通信**:
    - 所有容器连接到 `federated-network` 自定义 Docker bridge 网络。
    - 客户端与服务端通过 gRPC 通信（默认端口 50051），支持 TLS 加密。
    - SGX 模式下，服务端与 Enclave 通过 TCP Socket 通信（端口 8000），采用流式协议降低内存峰值。

- **数据卷挂载**:
    - `data/client*/`: 各客户端专属数据目录，模拟"数据不出本地"的联邦场景。
    - `data/global/`: 全局测试集，用于服务端评估全局模型真实性能。
    - `certs/`: TLS 证书和密钥，确保传输层安全。
    - `logs/`: 训练日志，记录每轮的延迟、资源消耗和收敛状态。
    - `out/`: 可视化图表输出目录。
    - `tdvm/`: TDX 虚拟机配置文件（libvirt XML），用于启动 Trust Domain。

### 通信协议设计

通信协议在 `src/grpc/protos/federation.proto` 中定义，核心设计思想：

1. **统一接口，多态载荷**: `ClientUpdate` 消息使用 `oneof` 封装不同隐私模式的载荷（明文、密文、份额、加密包），上层逻辑无需感知底层数据形态。

2. **流式推送替代轮询**: `SubscribeTrainingStatus` 采用服务端流式 RPC，基于条件变量实现高效的状态通知，消除轮询开销。

3. **分块流式传输**: `SubmitUpdateHeStream` 支持将大型加密参数按层分块上传，避免内存溢出。

| 接口 | 类型 | 说明 |
|------|------|------|
| `RegisterAndSetup` | Unary | 客户端注册，获取隐私模式、初始模型和加密材料 |
| `SubscribeTrainingStatus` | Server Stream | 客户端订阅训练状态，服务端**流式推送**状态变化 |
| `SubmitUpdate` | Unary | 统一的模型更新提交接口，使用 `oneof` 适配多种隐私模式 |
| `SubmitUpdateHeStream` | Client Stream | HE 模式专用，支持分块流式上传大型加密参数 |
| `GetGlobalModel` | Unary | 获取聚合后的全局模型参数 |

## 核心工作流程

一次完整的联邦学习流程如下：

### 1. 启动与注册

- Docker Compose 启动服务端和所有客户端容器。
- 每个客户端向服务端发起 `RegisterAndSetup` 请求，提交客户端 ID 和本地数据量。
- 服务端收集注册信息，达到预设客户端数后，根据配置确定全局隐私模式。
- 服务端返回 `SetupResponse`，包含：隐私模式、初始模型参数、加密材料（HE 公钥、TEE 公钥/证明报告、SGX Quote 等）。

### 2. 状态同步

- 客户端调用 `SubscribeTrainingStatus` 订阅训练状态。
- 服务端维护状态条件变量（Condition），当所有客户端就绪或状态变化时，**主动推送**通知。
- 这种基于推送的机制避免了传统轮询带来的网络开销和锁竞争。

### 3. 本地训练与安全更新

- 客户端在本地数据集上执行模型训练，支持数据增强和混合精度训练。
- 训练完成后，根据隐私模式对模型更新进行安全预处理：
  - **none**: 直接序列化明文参数
  - **he**: 使用 CKKS 加密，参数按层分块打包到密文中
  - **mpc**: 使用 Shamir 秘密共享将参数拆分为多个份额
  - **tee/sgx**: 使用混合加密（RSA-OAEP + AES-GCM），SGX 模式下参数以 float16 格式压缩
- 客户端同时记录延迟指标（训练时间、加密时间、上传大小、内存峰值），随更新一并提交。
- 客户端通过 `SubmitUpdate`（或 HE 模式的 `SubmitUpdateHeStream`）提交更新。

### 4. 安全聚合

- 服务端根据隐私模式调用相应的聚合策略：
  - **none**: 按数据量加权平均明文参数（FedAvg）
  - **he**: 在 CKKS 密文空间执行向量加法，然后用私钥解密
  - **mpc**: 聚合秘密份额后通过拉格朗日插值恢复明文
  - **tee**: 在 TD 内解密后执行明文聚合
  - **sgx**: 将加密参数流式转发至 Enclave，在硬件隔离环境中逐客户端解密并累加

### 5. 全局模型评估

- 服务端使用**全局测试集**评估聚合后的模型性能（准确率、AUC）。
- 如全局测试集不可用，则回退到客户端聚合指标（加权平均各客户端的本地评估结果）。
- 记录每轮评估结果用于收敛判定和可视化。

### 6. 收敛检测与终止

- 系统采用**滑动窗口**策略检测收敛：当最近 $w$ 轮（默认 4 轮）的准确率变化幅度小于阈值 $\delta$（默认 0.005）时，判定收敛。
- 达到最大轮次或收敛后，服务端：
  - 输出通信开销汇总（各客户端上传数据量）
  - 输出评估指标汇总表（每轮 Accuracy、AUC、Loss）
  - 生成可视化图表保存至 `out/` 目录

## 隐私策略详解

### 1. `none` - 标准联邦学习

- **原理**: 标准 FedAvg 算法，客户端发送明文模型参数。
- **用途**: 性能基准测试，评估隐私保护方案的额外开销。
- **安全性**: 仅依赖 TLS 信道加密，服务端可见所有明文参数。

### 2. `he` - CKKS 同态加密

- **原理**: 利用 CKKS 全同态加密的 **SIMD 批处理**特性，一个密文可打包数千个浮点数。
- **实现**:
    1. 服务端生成 CKKS 密钥对，将公钥上下文分发给客户端。
    2. 客户端将模型参数按 `chunk_size` 分块，每块打包到一个 CKKS 密文中。
    3. 密文支持 zlib 压缩，通过流式接口分块上传，降低内存压力。
    4. 服务端在**密文空间**执行向量加法聚合，然后用私钥解密。
- **性能优势**: 相比 Paillier 逐元素加密，CKKS 批处理显著提升加密和聚合效率。
- **安全性**: 服务端仅能见到密文，无法获取明文参数。私钥由服务端持有，因此服务端可解密结果。
- **依赖**: `tenseal`

### 3. `mpc` - Shamir 秘密共享

- **原理**: 基于 Shamir 秘密共享方案，一个秘密被拆分成 $n$ 个份额，至少需要 $k$ 个份额才能恢复。
- **实现**:
    1. 客户端将模型参数缩放为整数，使用 Shamir 算法生成份额。
    2. 采用固定 8 字节编码（适用于 $p < 2^{64}$），支持 NumPy 向量化加速。
    3. 服务端收集份额后，在有限域上聚合，通过预计算的拉格朗日系数批量恢复明文。
- **安全性**: 单个份额不泄露任何信息，需要 $k$ 个份额才能恢复。但当前实现中所有份额发送给同一服务端，适合研究场景。
- **配置参数**: `shamir_k`（恢复阈值）、`shamir_n`（总份额数）、`prime_mod`（素数域）。

### 4. `tee` - Intel TDX 可信执行环境

- **原理**: 服务端运行在 **Intel TDX (Trust Domain Extensions)** 硬件虚拟化隔离环境中。TDX 提供 VM 级别的内存加密和隔离，保护整个虚拟机免受宿主机攻击。
- **部署**: 服务端容器运行在 TDX VM（Trust Domain）内部，`tdvm/` 目录包含 libvirt XML 配置文件用于启动 TD。
- **实现**:
    1. 服务端在 TD 内生成 RSA 密钥对，并生成用于身份验证的 MRENCLAVE 指纹。
    2. 客户端使用**混合加密**: RSA-OAEP 加密对称密钥，AES-GCM 加密模型参数。
    3. 服务端在 TD 内解密并执行明文聚合。
- **远程证明**: 当前实现中远程证明使用软件模拟（基于模型哈希生成 MRENCLAVE），生产环境应集成真实的 TDX 远程证明流程。
- **安全性**: 明文仅在 TD 内可见，宿主机无法访问 TD 内存。

### 5. `sgx` - Intel SGX 进程级隔离

- **原理**: 聚合逻辑运行在真实的 **SGX Enclave** 中，由 Gramine 运行时托管。SGX 提供进程级别的硬件隔离，Enclave 内存对外完全不可见。
- **实现**:
    1. `sgx-aggregator` 容器启动时，在 Enclave 内生成 RSA 密钥对，并通过 `/dev/attestation/` 生成真实的 DCAP Quote。
    2. Quote 将公钥与 Enclave 度量（MRENCLAVE）绑定，服务端获取后分发给客户端。
    3. 客户端可验证 Quote 和 MRENCLAVE（当前开发阶段为可选），然后使用公钥进行混合加密。
    4. 参数以 **float16** 格式压缩（减少约 50% 数据量），服务端通过 TCP Socket **流式发送**至 Enclave。
    5. Enclave 内部逐个客户端解密、使用 float32 累加（保证精度），最终返回聚合结果。
- **安全特性**: 
    - RSA 私钥仅存在于 Enclave 内，即使宿主进程被攻破也无法提取。
    - MRENCLAVE 确保 Enclave 代码完整性可验证。
- **依赖**: Intel SGX 硬件、DCAP 驱动、Gramine 运行时。

## 快速开始

### 前提条件

- [Docker](https://www.docker.com/get-started) 和 [Docker Compose](https://docs.docker.com/compose/install/)
- NVIDIA GPU + CUDA（可选，用于客户端训练加速）
- Intel TDX 支持的 CPU（仅 TEE 模式需要）
- Intel SGX 支持的 CPU + DCAP 驱动（仅 SGX 模式需要）

### 安装与运行

1. **克隆项目**:
    ```bash
    git clone <your-repository-url>
    cd CFLP-docker
    ```

2. **生成数据集**:
    根据配置文件中的数据集和客户端数量，使用 Dirichlet 分布划分 Non-IID 数据：
    ```bash
    python src/data_process/data_loader.py
    ```
    这将生成：
    - `data/client*/`: 各客户端的本地训练集和验证集（`{dataset}_train.npz`, `{dataset}_val.npz`）
    - `data/global/`: 全局测试集（`{dataset}_test.npz`）

3. **生成 TLS 证书**（如需安全通信）:
    ```bash
    # 生成自签名证书（开发用途）
    mkdir -p certs/server certs/client
    openssl req -x509 -newkey rsa:4096 -keyout certs/server/server.key \
        -out certs/server/server.crt -days 365 -nodes -subj "/CN=server"
    cp certs/server/server.crt certs/client/ca.crt
    ```

4. **生成 gRPC 代码**（如果修改了 `.proto` 文件）:
    ```bash
    python src/scripts/generate_grpc.py
    ```

5. **修改配置**:
    编辑 `src/default.yaml`，设置隐私模式、模型、数据集、服务器地址等参数。

6. **启动联邦学习环境**:

    **标准模式（none/he/mpc）**:
    ```bash
    # 启动服务端
    docker-compose -f src/docker/docker-compose.server.yml up --build -d
    # 启动客户端
    docker-compose -f src/docker/docker-compose.clients.yml up --build -d
    ```

    **TEE 模式（TDX）**:
    ```bash
    # 1. 使用 libvirt 启动 TDX VM
    virsh create tdvm/lzh_td.xml
    # 2. 在 TD 内启动服务端容器
    # 3. 在宿主机启动客户端
    docker-compose -f src/docker/docker-compose.clients.yml up --build -d
    ```

    **SGX 模式**:
    ```bash
    # 启动服务端 + Enclave 聚合器
    docker-compose -f src/docker/docker-compose.sgx.yml up --build -d
    # 启动客户端
    docker-compose -f src/docker/docker-compose.clients.yml up --build -d
    ```

7. **查看日志**:
    ```bash
    docker logs -f fl-server
    ```

8. **查看结果**:
    - 收敛曲线和评估图表保存在 `out/` 目录
    - 训练日志保存在 `logs/` 目录

9. **停止并清理环境**:
    ```bash
    docker-compose -f src/docker/docker-compose.server.yml down
    docker-compose -f src/docker/docker-compose.clients.yml down
    # 或 SGX 模式:
    docker-compose -f src/docker/docker-compose.sgx.yml down
    ```

## 日志格式与指标说明

平台使用统一的日志格式，便于解析和可视化：

### 关键日志格式

```
# 客户端延迟指标
[Round {轮次}][Client {ID}][LATENCY] training={训练时间}s, encryption={加密时间}s
[Round {轮次}][Client {ID}][PAYLOAD] upload={上传大小} MB
[Round {轮次}][Client {ID}][RESOURCE] peak_memory={内存峰值} MB, cpu={CPU占用}%

# 服务端聚合指标
[Round {轮次}][LATENCY] decryption={解密时间}s
[Round {轮次}][LATENCY] aggregation={聚合时间}s
[Round {轮次}][Server][RESOURCE] peak_memory={内存峰值} MB, cpu={CPU占用}%

# 收敛状态（用于绘图）
[CONVERGENCE] round={轮次} accuracy={准确率} auc={AUC} loss={损失}
```

### 性能指标说明

| 指标 | 说明 | 影响因素 |
|------|------|---------|
| `training_time` | 客户端本地训练耗时 | 数据量、模型大小、GPU 性能 |
| `encryption_time` | 参数加密/秘密共享耗时 | 隐私模式、参数规模 |
| `decryption_time` | 服务端解密耗时 | 隐私模式、客户端数量 |
| `aggregation_time` | 聚合计算耗时 | 参数规模、聚合算法 |
| `peak_memory` | 进程峰值内存占用 | 模型大小、批次大小、加密膨胀 |
| `payload_size` | 上传数据量 | 模型大小、加密膨胀、压缩率 |

## 实验配置详解

所有实验参数在 `src/default.yaml` 中配置。

### 核心配置项

```yaml
# 数据配置
data:
  dataset: "cifar10"        # 数据集: "mnist" 或 "cifar10"
  dirichlet_alpha: 0.5      # Dirichlet 分布参数，α 越小越 Non-IID
  local_val_size: 0.1       # 客户端本地验证集比例

# 模型配置
model:
  name: "resnet18"          # 模型: "cnn", "resnet18", "vgg16"

# 联邦学习配置
federation:
  privacy_mode: "none"      # 隐私模式: "none", "he", "mpc", "tee", "sgx"
  expected_clients: 3       # 预期客户端数量
  max_rounds: 10000         # 最大训练轮次
  convergence:
    acc_delta_threshold: 0.005  # 收敛阈值
    window: 4                   # 滑动窗口大小

# gRPC 配置
grpc:
  server_host: "10.16.56.126"  # 服务端地址（客户端使用）
  server_port: 50051           # 服务端端口
  max_workers: 10              # gRPC 线程池大小

# 训练配置
training:
  batch_size: 128
  learning_rate: 0.01
  epochs: 5                 # 每轮本地训练轮数
  optimizer: sgd
  scheduler: cosine         # 学习率调度器
  estimated_rounds: 200     # 预估收敛轮数，用于调度器
  use_augmentation: true    # 数据增强
  use_amp: true             # 混合精度训练

# CKKS 同态加密配置（仅 he 模式）
encryption:
  poly_modulus_degree: 8192
  coeff_mod_bit_sizes: [60, 40, 40, 60]
  global_scale: 1099511627776.0  # 2^40
  chunk_size: 4096          # 每个密文打包的元素数

# MPC 配置（仅 mpc 模式）
mpc:
  shamir_k: 3               # 恢复阈值
  shamir_n: 3               # 总份额数
  scaling_factor: 1000000   # 浮点数缩放因子
  prime_mod: "2305843009213693951"  # 2^61 - 1

# TEE 配置（仅 tee 模式）
tee:
  expected_mrenclave: "..."  # 预期的 TD 度量值

# SGX 配置（仅 sgx 模式）
sgx:
  expected_mrenclave: "..."  # Enclave 度量值，用于远程证明
```

## 项目结构

```
CFLP-docker/
├── src/
│   ├── grpc/                    # gRPC 通信层
│   │   ├── protos/              # Protocol Buffers 定义
│   │   │   └── federation.proto # 核心协议定义
│   │   ├── generated/           # 自动生成的 gRPC 代码
│   │   ├── server_grpc.py       # 服务端实现（协调、调度、评估）
│   │   └── client_grpc.py       # 客户端实现（训练、上传、同步）
│   ├── strategies/              # 隐私策略模块（策略模式）
│   │   ├── server/              # 服务端聚合策略
│   │   │   ├── base_aggregation_strategy.py
│   │   │   ├── none_aggregation_strategy.py
│   │   │   ├── he_aggregation_strategy.py
│   │   │   ├── mpc_aggregation_strategy.py
│   │   │   ├── tee_aggregation_strategy.py
│   │   │   └── sgx_aggregation_strategy.py
│   │   └── client/              # 客户端更新策略
│   │       ├── base_strategy.py
│   │       ├── none_strategy.py
│   │       ├── he_strategy.py
│   │       ├── mpc_strategy.py
│   │       ├── tee_strategy.py
│   │       └── sgx_strategy.py
│   ├── models/                  # 神经网络模型（CNN, ResNet18, VGG16）
│   ├── data_process/            # 数据加载和 Dirichlet 划分
│   ├── sgx_aggregator/          # SGX Enclave 聚合器
│   │   ├── enclave.py           # Enclave 主逻辑
│   │   ├── Dockerfile.aggregator
│   │   └── aggregator.manifest.template
│   ├── docker/                  # Docker Compose 配置
│   │   ├── docker-compose.server.yml
│   │   ├── docker-compose.clients.yml
│   │   ├── docker-compose.sgx.yml
│   │   ├── Dockerfile.server
│   │   └── Dockerfile.client
│   ├── utils/                   # 工具函数
│   │   ├── logging_config.py    # 日志配置
│   │   ├── config_utils.py      # 配置加载
│   │   ├── parameter_utils.py   # 参数序列化/反序列化
│   │   ├── fast_shamir.py       # 高性能 Shamir 秘密共享
│   │   ├── draw.py              # 收敛曲线绘制
│   │   └── plot_experiments.py  # 实验结果可视化
│   ├── scripts/                 # 辅助脚本
│   └── default.yaml             # 全局配置文件
├── tdvm/                        # TDX VM 配置（libvirt XML）
├── data/                        # 数据目录
│   ├── client1/, client2/, ...  # 各客户端本地数据
│   └── global/                  # 全局测试集
├── certs/                       # TLS 证书
│   ├── server/                  # 服务端证书和私钥
│   └── client/                  # 客户端 CA 证书
├── logs/                        # 训练日志
└── out/                         # 可视化输出
```

## 扩展性

该框架采用**策略模式**设计，易于扩展新的隐私保护方案：

1. 在 `src/strategies/client/` 下创建客户端策略（继承 `ClientStrategy`），实现 `prepare_update_request()` 方法。
2. 在 `src/strategies/server/` 下创建服务端聚合策略（继承 `AggregationStrategy`），实现 `aggregate()`、`aggregate_parameters()` 和 `evaluate_metrics()` 方法。
3. 在 `federation.proto` 中为新策略定义相应的 Payload 消息类型，并添加到 `ClientUpdate` 的 `oneof` 中。
4. 更新 `server_grpc.py` 和 `client_grpc.py` 中的策略加载逻辑。

所有调度、通信、日志和可视化链路均可复用，新策略只需关注加密/解密和聚合逻辑。

## 可视化工具

平台提供了日志分析和可视化工具：

```bash
# 分析单个日志（生成收敛图和延迟分解图）
python src/utils/plot_experiments.py logs/server_sgx.log -o plots/

# 对比多个模式
python src/utils/plot_experiments.py \
    logs/server_none.log \
    logs/server_he.log \
    logs/server_mpc.log \
    logs/server_sgx.log \
    --compare -o plots/
```

生成的图表包括：
- **收敛曲线**: 准确率、AUC、损失随训练轮次的变化
- **延迟分解图**: 训练、加密、聚合、解密各阶段的时间占比
- **内存足迹对比**: 客户端、服务端、Enclave 的峰值内存使用
