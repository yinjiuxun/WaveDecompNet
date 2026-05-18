# 🌊 WaveDecompNet — 现代化改造与改进计划 (综合版)

> **项目定位**: 基于深度学习的时域地震信号与环境噪声分离工具
>
> **核心架构**: Encoder-Decoder + LSTM Bottleneck + Multi-Head Self-Attention
>
> **当前状态**: ⚠️ 创建于约 2021 年，Python 3.9 + PyTorch 1.9.0，依赖严重过时

---

## 📐 当前架构

```
Seismogram (3 channels)
    │
    ▼
┌─────────────────────────────┐
│  CNN Encoder (7 layers)     │  3 → 64 channels
│  [3 → 8 → 8 → 16 → 16 →   │
│            32 → 32 → 64]    │
└─────────────┬───────────────┘
              │
              ▼
┌─────────────────────────────┐
│  LSTM Bottleneck            │  64 → 32, bidirectional, 2 layers
└─────────────┬───────────────┘
              │
      ┌───────┴───────┐
      ▼               ▼
┌──────────┐    ┌──────────┐
│Decoder 1 │    │Decoder 2 │
│Earthquake│    │  Noise   │
└──────────┘    └──────────┘
      │               │
      ▼               ▼
  earthquake_pred   noise_pred
```

**Loss**: `MSE(earthquake_pred, earthquake_true) + MSE(noise_pred, noise_true)`

---

## 📑 目录

- [优先级 P0: 环境部署与依赖更新](#优先级-p0-环境部署与依赖更新)
  - [Phase 1: 审计现有依赖与代码兼容性](#phase-1-审计现有依赖与代码兼容性)
  - [Phase 2: 创建现代化环境配置](#phase-2-创建现代化环境配置)
  - [Phase 3: 在沙箱中验证环境](#phase-3-在沙箱中验证环境)
- [优先级 P1: 代码现代化适配](#优先级-p1-代码现代化适配)
  - [Phase 4: 修复 PyTorch API 兼容性问题](#phase-4-修复-pytorch-api-兼容性问题)
  - [Phase 5: 修复其他 Python 兼容性问题](#phase-5-修复其他-python-兼容性问题)
- [优先级 P2: 功能验证](#优先级-p2-功能验证)
  - [Phase 6: 测试现有预训练模型](#phase-6-测试现有预训练模型)
  - [Phase 7: 验证 Notebook](#phase-7-验证-notebook)
- [优先级 P3: 训练改进](#优先级-p3-训练改进)
  - [Phase 8: 训练流程优化](#phase-8-训练流程优化)
- [优先级 P4: 代码质量与项目结构](#优先级-p4-代码质量与项目结构)
  - [Phase 9: 项目重构与代码规范](#phase-9-项目重构与代码规范)
  - [Phase 10: 模型架构改进](#phase-10-模型架构改进)
- [优先级 P5: 评估与指标](#优先级-p5-评估与指标)
  - [Phase 11: 综合评估体系](#phase-11-综合评估体系)
- [优先级 P6: 部署与文档](#优先级-p6-部署与文档)
  - [Phase 12: 推理脚本与部署](#phase-12-推理脚本与部署)
  - [Phase 13: 文档完善](#phase-13-文档完善)
- [实施顺序](#实施顺序)
- [测试执行指南](#测试执行指南)
- [测试数据策略](#测试数据策略)
- [注意事项](#注意事项)

---

## 🔴 优先级 P0: 环境部署与依赖更新 (首要任务)

> **运行环境**: Host 机器 (本地 GPU) | **Python**: 3.12 | **虚拟环境**: venv | **PyTorch**: CUDA 版本

### Phase 1: 审计现有依赖与代码兼容性

**目标**: 全面了解代码中使用的所有依赖和 API，识别过时/弃用的调用。

#### 当前依赖审计

| 依赖 | 当前版本 (2021) | 用途 |
|------|----------------|------|
| Python | 3.9.9 | 运行时 |
| PyTorch | 1.9.0 | 深度学习框架 |
| NumPy | 1.22.0 | 数值计算 |
| SciPy | 1.7.3 | 信号处理 (signal, fft, interpolate) |
| h5py | 3.6.0 | HDF5 数据读写 |
| Matplotlib | 3.5.1 | 可视化 |
| scikit-learn | 1.0.2 | 数据分割、评估指标 |
| Pillow | 9.0.0 | 图像处理 (间接依赖) |

#### 已识别的过时 API / 潜在问题

> 🔴 **CRITICAL** — 必须立即修复
>
> 🟠 **HIGH** — 应在当前迭代中修复
>
> 🟡 **MEDIUM** — 建议修复
>
> 🟢 **LOW** — 可延后处理

| 编号 | 问题 | 严重度 | 位置 |
|------|------|--------|------|
| P0-6 | DotProductAttention softmax 维度错误 | 🔴 CRITICAL | `autoencoder_1D_models_torch.py:137` |
| P0-1 | dtype=torch.float64 在层构造器中 | 🟠 HIGH | `autoencoder_1D_models_torch.py` (多处) |
| P0-2 | torch.save(model, ...) 保存整个模型 | 🟠 HIGH | `train_model.py:93` |
| P0-3 | torch.load(...) 缺少 weights_only 参数 | 🟠 HIGH | `test_model.py:50`, `torch_tools.py:193,317` |
| P0-9 | 推理时缺少 torch.no_grad() | 🟠 HIGH | `test_model.py:60-68,123` |
| P0-10 | checkpoint 保存到 CWD 而非模型目录 | 🟠 HIGH | `torch_tools.py:193,317` |
| P0-4 | data_iter.next() 旧式调用 | 🟡 MEDIUM | `test_model.py:120` |
| P0-5 | os.mkdir() 无存在性检查 | 🟡 MEDIUM | `utilities.py:10-11` |
| P0-7 | PositionalEncoding dtype 不一致 | 🟡 MEDIUM | `autoencoder_1D_models_torch.py:148-155` |
| P0-8 | model_same 函数在第一次比较后就返回 | 🟡 MEDIUM | `torch_tools.py:324-330` |
| P0-11 | 通配符导入 | 🟢 LOW | `test_model.py:11` |
| P0-12 | 缺少 .gitignore | 🟢 LOW | 项目根目录 |
| P0-13 | 缺少 LICENSE 文件 | 🟢 LOW | 项目根目录 |
| P0-14 | environment.yml 硬编码用户路径 | 🟢 LOW | `environment.yml:43` |

#### 修复方案

**P0-6** DotProductAttention softmax 维度错误

```python
# ❌ 错误
F.softmax(scores, dim=0)  # 对 batch 维度做 softmax

# ✅ 修复
F.softmax(scores, dim=-1)  # 对序列/时间维度做 softmax
```

**P0-1** dtype=torch.float64 在层构造器中

```python
# ❌ 错误 (PyTorch 2.x 已弃用)
self.enc1 = nn.Conv1d(3, 8, 9, padding='same', dtype=torch.float64)

# ✅ 修复
self.enc1 = nn.Conv1d(3, 8, 9, padding='same')
# 实例化后统一转换
model.double()  # 或 model.to(torch.float64)
```

**P0-2 / P0-3** 保存/加载方式

```python
# ❌ 错误
torch.save(model, path)
model = torch.load(path)

# ✅ 修复
torch.save(model.state_dict(), path)
model.load_state_dict(torch.load(path, weights_only=True))
```

**P0-4** 旧式迭代器调用

```python
# ❌ 错误
data_iter.next()

# ✅ 修复
next(data_iter)
```

**P0-5** 目录创建

```python
# ❌ 错误
os.mkdir(dir_path)

# ✅ 修复
os.makedirs(dir_path, exist_ok=True)
```

---

### Phase 2: 创建现代化环境配置

**目标**: 使用 `venv` 创建虚拟环境，配置 Python 3.12 + CUDA PyTorch，确保所有依赖使用最新稳定版本。

#### 环境架构

```
┌─────────────────────────────────────────────────┐
│              Host 机器环境 (本地 GPU)              │
│                                                 │
│  Python 3.12  (系统安装)                         │
│    └── venv: WaveDecompNet/.venv                 │
│          ├── torch>=2.5.0  (CUDA 12.x)           │
│          ├── numpy>=1.26.0                       │
│          ├── scipy>=1.13.0                       │
│          ├── h5py>=3.11.0                        │
│          ├── matplotlib>=3.8.0                   │
│          ├── scikit-learn>=1.5.0                 │
│          └── pytest, torchinfo (开发依赖)         │
│                                                 │
│  训练: 在 Host 机器 GPU 上执行                    │
│  测试: 在 Host 机器上执行                         │
└─────────────────────────────────────────────────┘
```

#### 具体步骤

| 步骤 | 内容 | 说明 |
|------|------|------|
| P2-1 | 确认 Host 机器 Python 3.12 可用 | `python3.12 --version` |
| P2-2 | 确认 Host 机器 CUDA 可用 | `nvidia-smi` 查看 CUDA 版本 |
| P2-3 | 创建 venv 虚拟环境 | `python3.12 -m venv .venv` |
| P2-4 | 激活并安装依赖 | `source .venv/bin/activate && pip install -r requirements.txt` |
| P2-5 | 验证 CUDA 可用性 | `python -c "import torch; print(torch.cuda.is_available())"` |

#### 创建环境命令 (在 Host 机器上执行)

```bash
# 1. 进入项目目录
cd /home/yinjiuxun/hermes_workspace/code/WaveDecompNet

# 2. 确认 Python 3.12
python3.12 --version  # 应输出 Python 3.12.x

# 3. 确认 CUDA 驱动
nvidia-smi  # 查看 CUDA 版本

# 4. 创建虚拟环境
python3.12 -m venv .venv

# 5. 激活虚拟环境
source .venv/bin/activate

# 6. 升级 pip
pip install --upgrade pip

# 7. 安装依赖
pip install -r requirements.txt

# 8. 验证安装
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda if torch.cuda.is_available() else \"N/A\"}')"
```

#### 推荐 requirements.txt

```txt
# Core dependencies
torch>=2.5.0
numpy>=1.26.0
scipy>=1.13.0
h5py>=3.11.0
matplotlib>=3.8.0
scikit-learn>=1.5.0

# Development dependencies
pytest>=8.0.0
pytest-cov>=5.0.0
torchinfo>=1.8.0
```

#### CUDA 版本选择

| Host CUDA 版本 | PyTorch 安装命令 |
|----------------|------------------|
| CUDA 12.1+ | `pip install torch>=2.5.0 --index-url https://download.pytorch.org/whl/cu121` |
| CUDA 11.8 | `pip install torch>=2.5.0 --index-url https://download.pytorch.org/whl/cu118` |

> **注意**: PyTorch 默认通过 pip 安装 CPU 版本。如需 CUDA 支持，需指定 `--index-url`。
> 也可在 requirements.txt 中直接指定：`torch>=2.5.0 --index-url https://download.pytorch.org/whl/cu121`

---

### Phase 3: 在 Host 机器上验证环境

**目标**: 在 Host 机器上验证新环境能正确安装和运行，确保 CUDA 可用。

| 步骤 | 命令 / 操作 | 预期结果 |
|------|-------------|----------|
| P3-1 | `python --version` | Python 3.12.x |
| P3-2 | `python -c "import torch; print(torch.__version__)"` | 2.5.x+ |
| P3-3 | `python -c "import torch; print(torch.cuda.is_available())"` | True |
| P3-4 | `python -c "import numpy, scipy, h5py, sklearn; print('All imports OK')"` | 无报错 |
| P3-5 | `python -c "from autoencoder_1D_models_torch import *"` | 无报错 (修复后) |

---

## 🟠 优先级 P1: 代码现代化适配

### Phase 4: 修复 PyTorch API 兼容性问题

**目标**: 使所有代码适配 PyTorch 2.x API。

#### 修复清单

| 文件 | 修改内容 |
|------|----------|
| `autoencoder_1D_models_torch.py` | 移除所有 `dtype=torch.float64` 参数；实例化后统一调用 `.double()`；检查 `PositionalEncoding` 中的 `torch.float32` |
| `train_model.py` | `torch.save(model, ...)` → `torch.save(model.state_dict(), ...)`；添加 `weights_only=True` |
| `test_model.py` | `data_iter.next()` → `next(data_iter)`；修改模型加载逻辑；添加 `weights_only=True` |
| `torch_tools.py` | `torch.load('checkpoint.pt')` → `torch.load('checkpoint.pt', weights_only=True)`；检查 `EarlyStopping` 中的 `torch.save` / `torch.load` |

---

### Phase 5: 修复其他 Python 兼容性问题

**目标**: 确保代码在 Python 3.12 上无警告运行。

| 步骤 | 内容 |
|------|------|
| P5-1 | `utilities.py`: `os.mkdir()` → `os.makedirs(exist_ok=True)` |
| P5-2 | 检查 f-string 兼容性 (应该没问题) |
| P5-3 | 检查 h5py API 变化 (3.11+ 有一些 breaking changes) |
| P5-4 | 改进随机种子控制：添加 `set_seed()` 函数设置 `cudnn.deterministic`, `cudnn.benchmark`, `PYTHONHASHSEED` |

---

## 🟡 优先级 P2: 功能验证

### Phase 6: 测试现有预训练模型

**目标**: 确认新环境下预训练模型能正常加载和推理。

| 步骤 | 操作 |
|------|------|
| P6-1 | 在新环境中加载预训练模型 |
| P6-2 | 下载训练数据集 (或模拟测试数据) |
| P6-3 | 运行 `python test_model.py` 验证模型加载和推理 |
| P6-4 | 检查输出结果是否与预期一致 |

---

### Phase 7: 验证 Notebook

**目标**: 确保 Jupyter Notebook 在新环境中可运行。

| 步骤 | 操作 |
|------|------|
| P7-1 | 检查 `notebooks/apply_to_continuous_data.ipynb` 的依赖 |
| P7-2 | 在新环境中运行 Notebook 或逐单元格验证 |

---

## 🟢 优先级 P3: 训练改进

### Phase 8: 训练流程优化

**目标**: 改进训练流程，添加实验跟踪、梯度裁剪、更好的学习率调度等。

| 编号 | 内容 | 说明 |
|------|------|------|
| P8-1 | 添加梯度裁剪 | `torch.nn.utils.clip_grad_norm_` |
| P8-2 | 改进学习率调度器 | CosineAnnealingLR / OneCycleLR |
| P8-3 | 优化数据加载 | num_workers, pin_memory, prefetch_factor |
| P8-4 | 添加 TensorBoard 实验跟踪 | loss curves, gradients, histograms |
| P8-5 | 添加配置文件 + CLI 参数 | argparse |
| P8-6 | 用 logging 替换 print() | 结构化日志输出 |
| P8-7 | 确保 GPU 训练 | `model.to('cuda')`, 数据 `.to('cuda')` |
| P8-8 | 添加设备自动检测 | `device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')` |

---

## 🔵 优先级 P4: 代码质量与项目结构

### Phase 9: 项目重构与代码规范

**目标**: 改善项目结构、添加类型注解、文档字符串等。

#### 推荐项目结构

```
WaveDecompNet/
├── src/
│   ├── __init__.py
│   ├── models/
│   │   ├── __init__.py
│   │   ├── encoder.py
│   │   ├── decoder.py
│   │   ├── bottleneck.py
│   │   └── attention.py
│   ├── data/
│   │   ├── __init__.py
│   │   └── dataset.py
│   ├── train/
│   │   ├── __init__.py
│   │   ├── trainer.py
│   │   └── callbacks.py
│   └── utils/
│       ├── __init__.py
│       ├── logging.py
│       └── metrics.py
├── configs/
│   └── default.yaml
├── tests/
│   ├── __init__.py
│   └── ...
├── notebooks/
├── requirements.txt
├── pyproject.toml
├── .gitignore
└── README.md
```

| 编号 | 内容 | 说明 |
|------|------|------|
| P9-1 | 添加标准项目结构 | 见上方目录树 |
| P9-2 | 添加类型注解 (type hints) | 所有公开函数和类 |
| P9-3 | 添加 docstrings | Google/NumPy 风格 |
| P9-4 | 添加权重初始化 | kaiming_normal_, orthogonal_ |

---

### Phase 10: 模型架构改进

**目标**: 增强模型架构，添加残差连接、可配置瓶颈等。

| 编号 | 内容 | 说明 |
|------|------|------|
| P10-1 | 在 encoder/decoder 中添加残差/跳跃连接 | 改善梯度流 |
| P10-2 | 添加 Layer Normalization | 瓶颈后，改善小 batch 稳定性 |
| P10-3 | 使瓶颈类型可配置 | LSTM, Transformer, Dense |
| P10-4 | 在 encoder/decoder 中添加 dropout | 0.1-0.2，减少过拟合 |
| P10-5 | 添加模型摘要工具 | torchinfo |

---

## 🟣 优先级 P5: 评估与指标

### Phase 11: 综合评估体系

**目标**: 添加全面的评估指标、频域评估、SNR 分层分析等。

| 编号 | 内容 |
|------|------|
| P11-1 | 添加综合评估指标 (SNR 改善、Cross-correlation、Spectral distortion、Amplitude recovery、Onset time detection) |
| P11-2 | 添加频域评估 (Spectral angle mapper、Frequency-band specific MSE、Spectral convergence) |
| P11-3 | 添加 SNR 分层评估 (低/中/高 SNR 分别评估) |
| P11-4 | 添加消融实验框架 |
| P11-5 | 添加统计显著性检验 |

---

## 🟤 优先级 P6: 部署与文档

### Phase 12: 推理脚本与部署

**目标**: 添加独立的推理脚本、模型导出、Docker 支持等。

| 编号 | 内容 |
|------|------|
| P12-1 | 添加独立推理脚本 (predict.py) — 加载模型、接受地震数据、输出分离信号、支持批量处理 |
| P12-2 | 添加模型导出格式 (ONNX, TorchScript) |
| P12-3 | 添加 Docker 支持 |
| P12-4 | 添加 CI/CD 流水线 (GitHub Actions) |

---

### Phase 13: 文档完善

**目标**: 改善 README、添加 API 文档、教程等。

| 编号 | 内容 |
|------|------|
| P13-1 | 改善 README.md — 项目描述、安装指南、快速入门、架构图、引用、许可证、论文链接 |
| P13-2 | 添加 API 文档 (Sphinx/mkdocs) |
| P13-3 | 添加教程 notebooks |
| P13-4 | 添加 CONTRIBUTING.md |

---

## 📋 实施顺序

```
Phase 1 (审计) → Phase 2 (环境配置) → Phase 3 (环境验证)
                                                      │
Phase 4 (PyTorch 适配) → Phase 5 (Python 适配) → Phase 6 (模型测试)
                                                              │
Phase 7 (Notebook 验证) → Phase 8 (训练改进) → Phase 9 (代码质量)
                                                        │
Phase 10 (模型架构) → Phase 11 (评估体系) → Phase 12 (部署)
                                                    │
Phase 13 (文档完善)
```

### 每阶段完成后

1. ✅ 运行该阶段的所有 **[TEST]** 测试
2. ✅ 完成 **[VERIFY]** 手动检查
3. ✅ 提交到 Git (`git add` + `git commit`)
4. ✅ 确认全部通过后再进入下一阶段

---

## 🧪 测试执行指南

### 快速测试 (任何改动后)

```bash
pytest tests/test_attention_softmax.py tests/test_model_same.py -v
```

### 阶段特定测试

```bash
pytest tests/test_*.py -k "phase_name" -v
```

### 完整测试套件

```bash
pytest tests/ -v --tb=short
```

### 带覆盖率

```bash
pytest tests/ --cov=src --cov-report=html -v
```

### CI 风格 (lint + type + test)

```bash
flake8 src/ && black --check src/ && mypy src/ && pytest tests/ -v
```

### 仅集成测试

```bash
pytest tests/test_full_pipeline.py -v -s
```

---

## 📦 测试数据策略

### 自动化测试 — 小合成数据集

> 避免 CI 过慢，使用合成数据

| 项目 | 说明 |
|------|------|
| 数据量 | 100 个合成地震波形样本 (3 channels, 60 time steps) |
| 存储位置 | `tests/fixtures/synthetic_data.hdf5` |
| 训练轮数 | 最小训练 (1-3 epochs) |
| Batch size | 4 (加速) |

### 集成测试 — 真实数据

| 项目 | 说明 |
|------|------|
| 预训练模型 | `Branch_Encoder_Decoder_LSTM/` |
| 样本数据 | `notebooks/continuous_data/` |
| 标记方式 | `@pytest.mark.integration` (快速 CI 可跳过) |

---

## ⚠️ 注意事项

- **运行环境**: 所有开发、测试和训练均在 Host 机器上进行 (非 Docker 容器)
- **虚拟环境**: 使用 `venv` (`.venv` 目录)，已添加到 `.gitignore`
- **Python 版本**: 3.12 (通过 `python3.12` 命令调用)
- **CUDA**: PyTorch 使用 CUDA 版本，训练在 Host GPU 上执行
- 所有修改先提交到 Git，每完成一个 Phase 后 commit
- 预训练模型 (.pth 文件) 使用 `torch.load` 加载时，需确保 `map_location` 正确设置
- HDF5 数据文件较大，不需要重新生成，可以直接使用现有数据
- **注意**: `torch.float64` (double precision) 的使用是有意的 (地震数据需要高精度)，迁移时要保留精度
- 每个阶段的测试应在进入下一阶段之前全部通过
- 测试作为回归保护：如果未来改动破坏了某些功能，测试套件会捕获它

---

> **文档版本**: v1.1 | **最后更新**: 2026-05-18 | **环境**: Python 3.12 + venv + CUDA PyTorch | **维护者**: Jiuxun Yin
