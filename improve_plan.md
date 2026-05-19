# 🌊 WaveDecompNet — Modernization & Improvement Plan (Comprehensive)

> **Project Scope**: Deep learning-based tool for separating time-domain seismic signals from environmental noise
>
> **Core Architecture**: Encoder-Decoder + LSTM Bottleneck + Multi-Head Self-Attention
>
> **Current State**: ⚠️ Created ~2021, Python 3.9 + PyTorch 1.9.0, dependencies severely outdated

---

## 📐 Current Architecture

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

## 📑 Table of Contents

- [Priority P0: Environment Setup & Dependency Update](#priority-p0-environment-setup--dependency-update-primary-task)
  - [Phase 1: Audit Existing Dependencies & Code Compatibility](#phase-1-audit-existing-dependencies--code-compatibility)
  - [Phase 2: Create Modern Environment Configuration](#phase-2-create-modern-environment-configuration)
  - [Phase 3: Verify Environment on Host Machine](#phase-3-verify-environment-on-host-machine)
- [Priority P1: Code Modernization](#priority-p1-code-modernization)
  - [Phase 4: Fix PyTorch API Compatibility Issues](#phase-4-fix-pytorch-api-compatibility-issues)
  - [Phase 5: Fix Other Python Compatibility Issues](#phase-5-fix-other-python-compatibility-issues)
- [Priority P2: Functional Verification](#priority-p2-functional-verification)
  - [Phase 6: Test Existing Pre-trained Models](#phase-6-test-existing-pre-trained-models)
  - [Phase 7: Verify Notebooks](#phase-7-verify-notebooks)
- [Priority P3: Training Improvements](#priority-p3-training-improvements)
  - [Phase 8: Training Pipeline Optimization](#phase-8-training-pipeline-optimization)
- [Priority P4: Code Quality & Project Structure](#priority-p4-code-quality--project-structure)
  - [Phase 9: Project Refactoring & Code Standards](#phase-9-project-refactoring--code-standards)
  - [Phase 10: Model Architecture Improvements](#phase-10-model-architecture-improvements)
- [Priority P5: Evaluation & Metrics](#priority-p5-evaluation--metrics)
  - [Phase 11: Comprehensive Evaluation System](#phase-11-comprehensive-evaluation-system)
- [Priority P6: Deployment & Documentation](#priority-p6-deployment--documentation)
  - [Phase 12: Inference Script & Deployment](#phase-12-inference-script--deployment)
  - [Phase 13: Documentation Improvements](#phase-13-documentation-improvements)
- [Implementation Order](#implementation-order)
- [Test Execution Guide](#test-execution-guide)
- [Test Data Strategy](#test-data-strategy)
- [Important Notes](#important-notes)

---

## 🔴 Priority P0: Environment Setup & Dependency Update (Primary Task)

> **Runtime Environment**: Host machine (local GPU) | **Python**: 3.12 | **Virtual Environment**: venv | **PyTorch**: CUDA version

### ✅ Phase 1: Audit Existing Dependencies & Code Compatibility

**Status**: ✅ **COMPLETED** — Commit `657b298` (2025-07-20)

**Goal**: Fully understand all dependencies and APIs used in the codebase, identify outdated/deprecated calls.

#### Completed Fixes

| ID | Issue | Severity | Status |
|----|-------|----------|--------|
| P0-6 | DotProductAttention softmax dimension error | 🔴 CRITICAL | ✅ Fixed |
| P0-1 | dtype=torch.float64 in layer constructors | 🟠 HIGH | ✅ Fixed |
| P0-2 | torch.save(model, ...) saves entire model | 🟠 HIGH | ✅ Fixed |
| P0-3 | torch.load(...) missing weights_only param | 🟠 HIGH | ✅ Fixed |
| P0-9 | Missing torch.no_grad() during inference | 🟠 HIGH | ✅ Fixed |
| P0-4 | Legacy data_iter.next() call | 🟡 MEDIUM | ✅ Fixed |
| P0-5 | os.mkdir() without existence check | 🟡 MEDIUM | ✅ Fixed |
| P0-7 | PositionalEncoding dtype inconsistency | 🟡 MEDIUM | ✅ Fixed |
| P0-8 | model_same function returns after first comparison | 🟡 MEDIUM | ✅ Fixed |
| P0-11 | Wildcard imports | 🟢 LOW | ✅ Fixed |
| P0-12 | Missing .gitignore | 🟢 LOW | ✅ Fixed |
| NumPy 2.0 | np.Inf → np.inf compatibility | 🟠 HIGH | ✅ Fixed |

#### Test Results
- **17/18 tests passed**, 1 xfailed (original architecture issue - decoder stride mismatch)
- Test suite: `tests/test_phase1_fixes.py`

#### Remaining (deferred to later phases)
| ID | Issue | Severity | Deferred To |
|----|-------|----------|-------------|
| P0-10 | Checkpoint saved to CWD instead of model dir | 🟠 HIGH | Phase 8 (Training Pipeline) |
| P0-13 | Missing LICENSE file | 🟢 LOW | Phase 13 (Documentation) |
| P0-14 | environment.yml hardcoded user path | 🟢 LOW | Phase 2 (Environment Config) |

#### Current Dependency Audit

| Dependency | Current Version (2021) | Purpose |
|------------|----------------------|---------|
| Python | 3.9.9 | Runtime |
| PyTorch | 1.9.0 | Deep learning framework |
| NumPy | 1.22.0 | Numerical computing |
| SciPy | 1.7.3 | Signal processing (signal, fft, interpolate) |
| h5py | 3.6.0 | HDF5 data I/O |
| Matplotlib | 3.5.1 | Visualization |
| scikit-learn | 1.0.2 | Data splitting, evaluation metrics |
| Pillow | 9.0.0 | Image processing (indirect dependency) |

#### Identified Outdated APIs / Potential Issues

> 🔴 **CRITICAL** — Must fix immediately
>
> 🟠 **HIGH** — Should fix in current iteration
>
> 🟡 **MEDIUM** — Recommended to fix
>
> 🟢 **LOW** — Can be deferred

| ID | Issue | Severity | Location |
|----|-------|----------|----------|
| P0-6 | DotProductAttention softmax dimension error | 🔴 CRITICAL | `autoencoder_1D_models_torch.py:137` |
| P0-1 | dtype=torch.float64 in layer constructors | 🟠 HIGH | `autoencoder_1D_models_torch.py` (multiple) |
| P0-2 | torch.save(model, ...) saves entire model | 🟠 HIGH | `train_model.py:93` |
| P0-3 | torch.load(...) missing weights_only param | 🟠 HIGH | `test_model.py:50`, `torch_tools.py:193,317` |
| P0-9 | Missing torch.no_grad() during inference | 🟠 HIGH | `test_model.py:60-68,123` |
| P0-10 | Checkpoint saved to CWD instead of model dir | 🟠 HIGH | `torch_tools.py:193,317` |
| P0-4 | Legacy data_iter.next() call | 🟡 MEDIUM | `test_model.py:120` |
| P0-5 | os.mkdir() without existence check | 🟡 MEDIUM | `utilities.py:10-11` |
| P0-7 | PositionalEncoding dtype inconsistency | 🟡 MEDIUM | `autoencoder_1D_models_torch.py:148-155` |
| P0-8 | model_same function returns after first comparison | 🟡 MEDIUM | `torch_tools.py:324-330` |
| P0-11 | Wildcard imports | 🟢 LOW | `test_model.py:11` |
| P0-12 | Missing .gitignore | 🟢 LOW | Project root |
| P0-13 | Missing LICENSE file | 🟢 LOW | Project root |
| P0-14 | environment.yml hardcoded user path | 🟢 LOW | `environment.yml:43` |

#### Fix Strategies

**P0-6** DotProductAttention softmax dimension error

```python
# ❌ Bug
F.softmax(scores, dim=0)  # Softmax over batch dimension

# ✅ Fix
F.softmax(scores, dim=-1)  # Softmax over sequence/time dimension
```

**P0-1** dtype=torch.float64 in layer constructors

```python
# ❌ Bug (deprecated in PyTorch 2.x)
self.enc1 = nn.Conv1d(3, 8, 9, padding='same', dtype=torch.float64)

# ✅ Fix
self.enc1 = nn.Conv1d(3, 8, 9, padding='same')
# Unified conversion after instantiation
model.double()  # or model.to(torch.float64)
```

**P0-2 / P0-3** Save/Load approach

```python
# ❌ Bug
torch.save(model, path)
model = torch.load(path)

# ✅ Fix
torch.save(model.state_dict(), path)
model.load_state_dict(torch.load(path, weights_only=True))
```

**P0-4** Legacy iterator call

```python
# ❌ Bug
data_iter.next()

# ✅ Fix
next(data_iter)
```

**P0-5** Directory creation

```python
# ❌ Bug
os.mkdir(dir_path)

# ✅ Fix
os.makedirs(dir_path, exist_ok=True)
```

---

### Phase 2: Create Modern Environment Configuration

**Status**: ✅ **COMPLETED** — venv created, requirements.txt updated, dependencies installed (2026-05-18)

**Goal**: Use `venv` to create virtual environment, configure Python 3.12 + CUDA PyTorch, ensure all dependencies use latest stable versions.

#### Environment Architecture

```
┌─────────────────────────────────────────────────┐
│              Host Machine (Local GPU)             │
│                                                 │
│  Python 3.12  (system install)                   │
│    └── venv: WaveDecompNet/.venv                 │
│          ├── torch>=2.5.0  (CUDA 12.x)           │
│          ├── numpy>=1.26.0                       │
│          ├── scipy>=1.13.0                       │
│          ├── h5py>=3.11.0                        │
│          ├── matplotlib>=3.8.0                   │
│          ├── scikit-learn>=1.5.0                 │
│          └── pytest, torchinfo (dev dependencies)│
│                                                 │
│  Training: Runs on Host GPU                      │
│  Testing: Runs on Host machine                   │
└─────────────────────────────────────────────────┘
```

#### Step-by-Step Plan

| Step | Action | Status |
|------|--------|--------|
| P2-1 | Verify Python 3.12 available on host | ✅ |
| P2-2 | Verify CUDA available on host | ✅ |
| P2-3 | Create venv virtual environment | ✅ |
| P2-4 | Activate and install dependencies | ✅ |
| P2-5 | Verify CUDA availability | ✅ |

#### ⚠️ 常见问题与解决方案

**问题 1：Ubuntu 22.04+ 报 `externally-managed-environment` 错误**

Ubuntu 22.04+ 的系统 Python 不允许直接 `pip install`。解决方法是创建虚拟环境：

```bash
# 创建虚拟环境
python3 -m venv .venv

# 激活虚拟环境
source .venv/bin/activate

# 激活后再安装（提示符前会出现 (.venv)）
pip install -r requirements.txt
```

**问题 2：`--index-url` 导致 `No matching distribution found`**

`--index-url` 会**替换**默认的 PyPI 源，PyTorch 仓库里没有 numpy、scipy 等其他包，导致安装失败：

```bash
# ❌ 错误用法 — 替换了 PyPI，其他包找不到
pip install -r requirements.txt --index-url https://download.pytorch.org/whl/cu121

# ✅ 正确用法 — 额外索引，保留 PyPI + PyTorch 仓库
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu121
```

原理：`--extra-index-url` 把 PyTorch 仓库作为**额外**索引源，同时保留默认的 PyPI。这样 PyTorch 从专用仓库安装（带 CUDA 支持），其他包从 PyPI 安装。

#### Environment Setup Commands (Execute on Host Machine)

```bash
# 1. Navigate to project directory
cd /home/yinjiuxun/hermes_workspace/code/WaveDecompNet

# 2. Verify Python 3.12
python3.12 --version  # Should output Python 3.12.x

# 3. Verify CUDA driver
nvidia-smi  # Check CUDA version

# 4. Create virtual environment
python3.12 -m venv .venv

# 5. Activate virtual environment
source .venv/bin/activate

# 6. Upgrade pip
pip install --upgrade pip

# 7. Install dependencies (use --extra-index-url, NOT --index-url!)
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu121

# 8. Verify installation
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda if torch.cuda.is_available() else \"N/A\"}')\"
```

#### Recommended requirements.txt

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

#### CUDA Version Selection

| Host CUDA Version | PyTorch Install Command |
|-------------------|-------------------------|
| CUDA 12.1+ | `pip install torch>=2.5.0 --extra-index-url https://download.pytorch.org/whl/cu121` |
| CUDA 11.8 | `pip install torch>=2.5.0 --extra-index-url https://download.pytorch.org/whl/cu118` |

> **Note**: PyTorch defaults to CPU version via pip. For CUDA support, use `--extra-index-url` (NOT `--index-url`).
> `--index-url` replaces PyPI entirely, causing other packages to fail. `--extra-index-url` adds PyTorch as an additional source alongside PyPI.

---

### Phase 3: Verify Environment on Host Machine

**Status**: ✅ **COMPLETED** — PyTorch 2.12.0+cu130, CUDA available, all imports OK (2026-05-18)

**Goal**: Verify the new environment installs and runs correctly on the host machine, ensure CUDA is available.

#### Verification Results

| Step | Command / Action | Expected Result | Actual Result |
|------|------------------|-----------------|---------------|
| P3-1 | `python --version` | Python 3.12.x | ✅ Python 3.12 |
| P3-2 | `python -c "import torch; print(torch.__version__)"` | 2.5.x+ | ✅ 2.12.0+cu130 |
| P3-3 | `python -c "import torch; print(torch.cuda.is_available())"` | True | ✅ True |
| P3-4 | `python -c "import numpy, scipy, h5py, sklearn; print('All imports OK')"` | No errors | ✅ All imports OK |
| P3-5 | `python -c "from autoencoder_1D_models_torch import *"` | No errors (after fixes) | ✅ Passed |

> **Note**: P3-5 verified via existing test suite (17/17 tests passed on current codebase).

---

## 🟠 Priority P1: Code Modernization

### Phase 4: Fix PyTorch API Compatibility Issues

**Status**: ✅ **COMPLETED** — Verified all PyTorch 2.x API compatibility (2026-05-18)

**Goal**: Make all code compatible with PyTorch 2.x API.

> **Note**: Phase 4 fixes were already completed during Phase 1. This phase verifies that all changes are in place and tests pass.

#### Fix Checklist Verification

| File | Changes | Status | Verified By |
|------|---------|--------|-------------|
| `autoencoder_1D_models_torch.py` | Remove all `dtype=torch.float64` parameters; call `.double()` uniformly after instantiation; check `torch.float32` in `PositionalEncoding` | ✅ | TestP0_1_DtypeFloat64Removed, TestP0_7_PositionalEncodingDtype |
| `train_model.py` | `torch.save(model, ...)` → `torch.save(model.state_dict(), ...)`; add `weights_only=True` | ✅ | TestP0_2_3_SaveLoadStateDict |
| `test_model.py` | `data_iter.next()` → `next(data_iter)`; update model loading logic; add `weights_only=True` | ✅ | TestP0_4_NextIterator, TestP0_9_NoGradInference |
| `torch_tools.py` | `torch.load('checkpoint.pt')` → `torch.load('checkpoint.pt', weights_only=True)`; check `torch.save` / `torch.load` in `EarlyStopping` | ✅ | TestP0_2_3_SaveLoadStateDict |

#### Detailed Verification

**`autoencoder_1D_models_torch.py`** (Lines 1-253)
- ✅ All `nn.Conv1d`, `nn.ConvTranspose1d`, `nn.BatchNorm1d`, `nn.Linear`, `nn.LSTM` constructors have no `dtype` parameter
- ✅ `PositionalEncoding` uses `dtype=torch.float64` in `torch.arange` (intentional for seismic double precision)
- ✅ `DotProductAttention.softmax` uses `dim=-1` (P0-6 fix)

**`train_model.py`** (Lines 1-129)
- ✅ Line 93: `torch.save(model.state_dict(), ...)` — saves weights only
- ✅ No `torch.load` calls in this file (model saved, not loaded)

**`test_model.py`** (Lines 1-247)
- ✅ Line 60: `torch.load(..., weights_only=True)` — secure loading
- ✅ Line 131: `next(data_iter)` — modern iterator syntax
- ✅ Lines 69-70: `model.eval()` + `with torch.no_grad():` — proper inference mode

**`torch_tools.py`** (Lines 1-328)
- ✅ Line 85: `EarlyStopping.save_checkpoint` uses `torch.save(model.state_dict(), ...)`
- ✅ Line 193: `torch.load('checkpoint.pt', weights_only=True)`
- ✅ Line 317: `torch.load('checkpoint.pt', weights_only=True)`
- ✅ Lines 324-329: `model_same` uses `p1.data.ne(p2.data).sum() > 0` (correct logic)

#### Test Coverage

All Phase 4 items are covered by `tests/test_phase1_fixes.py`:
- 17/18 tests pass (1 xfail for known decoder stride mismatch — original architecture issue)
- Tests verify: dtype removal, state_dict save/load, weights_only, next(), makedirs, no_grad, model_same

---

### Phase 5: Fix Other Python Compatibility Issues

**Status**: ✅ **COMPLETED** — All Python 3.12 compatibility verified (2026-05-18)

**Goal**: Ensure code runs without warnings on Python 3.12.

#### Fix Checklist Verification

| Step | Action | Status | Verified By |
|------|--------|--------|-------------|
| P5-1 | `utilities.py`: `os.mkdir()` → `os.makedirs(exist_ok=True)` | ✅ Already done | TestP5_1_MakedirsExistOk (3 tests) |
| P5-2 | Check f-string compatibility | ✅ Compatible | TestP5_2_FStringCompatibility (3 tests) |
| P5-3 | Check h5py API changes (3.11+) | ✅ Compatible | TestP5_3_H5pyApiCompatibility (3 tests) |
| P5-4 | Improve random seed control: add `set_seed()` function | ✅ Implemented | TestP5_4_SetSeed (8 tests) |

#### Detailed Verification

**P5-1: `utilities.py` mkdir** (Line 9-10)
- ✅ `mkdir()` function already uses `os.makedirs(dir_path, exist_ok=True)`
- ✅ Idempotent, supports nested paths

**P5-2: f-string 兼容性**
- ✅ 仅 1 处 f-string 使用 (test_model.py:126)
- ✅ Python 3.12 完全兼容

**P5-3: h5py API 兼容性** (3.11+)
- ✅ `h5py.File(path, 'r')` — 标准读取模式
- ✅ `h5py.File(path, 'w')` — 标准写入模式
- ✅ `f.create_dataset()` — 标准 API
- ✅ `f.attrs[key]` — 标准属性访问
- ✅ 未使用已废弃参数 (swmr/libver/driver)

**P5-4: 随机种子控制** (`torch_tools.py` `set_seed()`)
- ✅ `random.seed(seed)` — Python 随机种子
- ✅ `np.random.seed(seed)` — NumPy 随机种子
- ✅ `torch.manual_seed(seed)` — PyTorch CPU 随机种子
- ✅ `torch.cuda.manual_seed_all(seed)` — PyTorch CUDA 随机种子
- ✅ `torch.backends.cudnn.deterministic = True` — cuDNN 确定性模式
- ✅ `torch.backends.cudnn.benchmark = False` — 禁用 cuDNN 自动调优
- ✅ `os.environ["PYTHONHASHSEED"] = str(seed)` — Python 哈希种子
- ✅ `train_model.py` 已更新使用 `set_seed(99)` 替代分散的种子设置

#### Test Coverage

`tests/test_phase5_fixes.py` 包含 17 个测试：
- TestP5_1_MakedirsExistOk: 3 tests (mkdir 功能验证)
- TestP5_2_FStringCompatibility: 3 tests (f-string 兼容性)
- TestP5_3_H5pyApiCompatibility: 3 tests (h5py API 兼容性)
- TestP5_4_SetSeed: 8 tests (随机种子控制)

---

## 🟡 Priority P2: Functional Verification

### Phase 6: Test Existing Pre-trained Models

**Goal**: Confirm pre-trained models load and run inference correctly in the new environment.

| Step | Action |
|------|--------|
| P6-1 | Load pre-trained model in new environment |
| P6-2 | Download training dataset (or use synthetic test data) |
| P6-3 | Run `python test_model.py` to verify model loading and inference |
| P6-4 | Check output results match expectations |

---

### Phase 7: Verify Notebooks

**Goal**: Ensure Jupyter Notebooks run in the new environment.

| Step | Action |
|------|--------|
| P7-1 | Check dependencies of `notebooks/apply_to_continuous_data.ipynb` |
| P7-2 | Run Notebook in new environment or verify cell by cell |

---

## 🟢 Priority P3: Training Improvements

### Phase 8: Training Pipeline Optimization

**Goal**: Improve training pipeline with experiment tracking, gradient clipping, better learning rate scheduling, etc.

| ID | Action | Details |
|----|--------|---------|
| P8-1 | Add gradient clipping | `torch.nn.utils.clip_grad_norm_` |
| P8-2 | Improve learning rate scheduler | CosineAnnealingLR / OneCycleLR |
| P8-3 | Optimize data loading | num_workers, pin_memory, prefetch_factor |
| P8-4 | Add TensorBoard experiment tracking | loss curves, gradients, histograms |
| P8-5 | Add config file + CLI arguments | argparse |
| P8-6 | Replace print() with logging | Structured log output |
| P8-7 | Ensure GPU training | `model.to('cuda')`, data `.to('cuda')` |
| P8-8 | Add device auto-detection | `device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')` |

---

## 🔵 Priority P4: Code Quality & Project Structure

### Phase 9: Project Refactoring & Code Standards

**Goal**: Improve project structure, add type hints, docstrings, etc.

#### Recommended Project Structure

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

| ID | Action | Details |
|----|--------|---------|
| P9-1 | Add standard project structure | See directory tree above |
| P9-2 | Add type hints | All public functions and classes |
| P9-3 | Add docstrings | Google/NumPy style |
| P9-4 | Add weight initialization | kaiming_normal_, orthogonal_ |

---

### Phase 10: Model Architecture Improvements

**Goal**: Enhance model architecture with residual connections, configurable bottleneck, etc.

| ID | Action | Details |
|----|--------|---------|
| P10-1 | Add residual/skip connections in encoder/decoder | Improve gradient flow |
| P10-2 | Add Layer Normalization | After bottleneck, improve small-batch stability |
| P10-3 | Make bottleneck type configurable | LSTM, Transformer, Dense |
| P10-4 | Add dropout in encoder/decoder | 0.1-0.2, reduce overfitting |
| P10-5 | Add model summary tool | torchinfo |

---

## 🟣 Priority P5: Evaluation & Metrics

### Phase 11: Comprehensive Evaluation System

**Goal**: Add comprehensive evaluation metrics, frequency-domain evaluation, SNR-stratified analysis, etc.

| ID | Action |
|----|--------|
| P11-1 | Add comprehensive evaluation metrics (SNR improvement, Cross-correlation, Spectral distortion, Amplitude recovery, Onset time detection) |
| P11-2 | Add frequency-domain evaluation (Spectral angle mapper, Frequency-band specific MSE, Spectral convergence) |
| P11-3 | Add SNR-stratified evaluation (Low/Medium/High SNR evaluated separately) |
| P11-4 | Add ablation study framework |
| P11-5 | Add statistical significance testing |

---

## 🟤 Priority P6: Deployment & Documentation

### Phase 12: Inference Script & Deployment

**Goal**: Add standalone inference script, model export, Docker support, etc.

| ID | Action |
|----|--------|
| P12-1 | Add standalone inference script (predict.py) — load model, accept seismic data, output separated signals, support batch processing |
| P12-2 | Add model export formats (ONNX, TorchScript) |
| P12-3 | Add Docker support |
| P12-4 | Add CI/CD pipeline (GitHub Actions) |

---

### Phase 13: Documentation Improvements

**Goal**: Improve README, add API docs, tutorials, etc.

| ID | Action |
|----|--------|
| P13-1 | Improve README.md — project description, installation guide, quick start, architecture diagram, citations, license, paper link |
| P13-2 | Add API documentation (Sphinx/mkdocs) |
| P13-3 | Add tutorial notebooks |
| P13-4 | Add CONTRIBUTING.md |

---

## 📋 Implementation Order

```
Phase 1 (Audit) → Phase 2 (Environment Setup) → Phase 3 (Environment Verification)
                                                      │
Phase 4 (PyTorch Adaptation) → Phase 5 (Python Adaptation) → Phase 6 (Model Testing)
                                                              │
Phase 7 (Notebook Verification) → Phase 8 (Training Improvements) → Phase 9 (Code Quality)
                                                        │
Phase 10 (Model Architecture) → Phase 11 (Evaluation System) → Phase 12 (Deployment)
                                                    │
Phase 13 (Documentation)
```

### After Each Phase

1. ✅ Run all **[TEST]** tests for that phase
2. ✅ Complete **[VERIFY]** manual checks
3. ✅ Commit to Git (`git add` + `git commit`)
4. ✅ Confirm all pass before proceeding to next phase

---

## 🧪 Test Execution Guide

### Quick Tests (After Any Changes)

```bash
pytest tests/test_attention_softmax.py tests/test_model_same.py -v
```

### Phase-Specific Tests

```bash
pytest tests/test_*.py -k "phase_name" -v
```

### Full Test Suite

```bash
pytest tests/ -v --tb=short
```

### With Coverage

```bash
pytest tests/ --cov=src --cov-report=html -v
```

### CI Style (lint + type + test)

```bash
flake8 src/ && black --check src/ && mypy src/ && pytest tests/ -v
```

### Integration Tests Only

```bash
pytest tests/test_full_pipeline.py -v -s
```

---

## 📦 Test Data Strategy

### Automated Tests — Small Synthetic Dataset

> Avoid slow CI, use synthetic data

| Item | Details |
|------|---------|
| Data Size | 100 synthetic seismogram samples (3 channels, 60 time steps) |
| Storage | `tests/fixtures/synthetic_data.hdf5` |
| Training Epochs | Minimal training (1-3 epochs) |
| Batch Size | 4 (for speed) |

### Integration Tests — Real Data

| Item | Details |
|------|---------|
| Pre-trained Model | `Branch_Encoder_Decoder_LSTM/` |
| Sample Data | `notebooks/continuous_data/` |
| Marker | `@pytest.mark.integration` (fast CI can skip) |

---

## ⚠️ Important Notes

- **Runtime Environment**: All development, testing, and training run on the Host machine (not Docker container)
- **Virtual Environment**: Use `venv` (`.venv` directory), added to `.gitignore`
- **Python Version**: 3.12 (via `python3.12` command)
- **CUDA**: PyTorch uses CUDA version, training runs on Host GPU
- All changes committed to Git, commit after each Phase
- When loading pre-trained models (.pth files) with `torch.load`, ensure `map_location` is set correctly
- HDF5 data files are large, no need to regenerate, can use existing data directly
- **Note**: Use of `torch.float64` (double precision) is intentional (seismic data requires high precision), preserve precision during migration
- Tests for each phase must all pass before moving to the next phase
- Tests serve as regression protection: if future changes break functionality, the test suite will catch it

---

> **Document Version**: v1.3 | **Last Updated**: 2026-05-18 | **Environment**: Python 3.12 + venv + CUDA PyTorch 2.12.0+cu130 | **Maintainer**: Jiuxun Yin
