# Stable Diffusion 模型微调项目

[English](README_EN.md) | 中文

这是一个全面的 Stable Diffusion 和 SDXL 模型微调框架，包含多种微调方法、数据处理流程和推理脚本。

## 📋 项目概述

本项目提供了从基础微调到高级 LoRA 适配的完整解决方案，用于在自定义数据上训练文本到图像生成模型。支持 Stable Diffusion 1.5、SDXL 及其各种微调技术。

### 主要特性

- ✨ **多种微调方法**：包括基础微调、DreamBooth、LoRA 和部分层微调
- 🚀 **模型支持**：Stable Diffusion 1.5 和 SDXL 1.0
- 💾 **灵活的数据处理**：支持 Parquet、本地图像目录等多种格式
- 🎯 **内存优化**：梯度累积、混合精度训练、模型 CPU 卸载
- 📊 **完整的训练管道**：包含日志记录、检查点保存和可视化
- 🔧 **模块化设计**：易于扩展和定制

## 📁 项目结构

```
.
├── basic/                    # 基础 Stable Diffusion 1.5 示例
│   ├── run.py               # 文本到图像生成示例脚本
│   └── train.py             # SD 1.5 基础微调脚本
│
├── dataset/                 # 数据集处理示例
│   └── train.py             # 诗歌漫画数据集训练脚本
│
├── dreambooth/              # DreamBooth 微调方法
│   └── train.py             # SDXL DreamBooth + LoRA 训练脚本
│
├── SDXL/                    # SDXL 部分层微调
│   └── train.py             # SDXL 交叉注意力微调脚本
│
├── SDXL-lora/               # SDXL LoRA 微调完整实现
│   ├── train.py             # LoRA 训练脚本
│   └── test.py              # LoRA 推理和图像生成脚本
│
└── report.pdf               # 项目报告文档
```

## 🔧 环境配置

### 系统要求

- Python 3.8 或更高版本
- CUDA 11.8 或更高版本（用于 GPU 加速）
- 至少 16GB VRAM（推荐 24GB+）
- 充足的磁盘空间（模型文件约 10-50GB）

### 安装依赖

```bash
# 创建虚拟环境（推荐）
python -m venv venv
source venv/bin/activate  # Linux/Mac
# 或
venv\Scripts\activate  # Windows

# 安装依赖包
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install transformers diffusers accelerate peft datasets pandas pillow tqdm
pip install tensorboard  # 可选，用于训练可视化
```

### 配置 HuggingFace 镜像（中国用户）

```bash
# 在代码中或环境变量中设置
export HF_ENDPOINT="https://hf-mirror.com"
```

## 📚 使用指南

### 1. 基础使用 - Stable Diffusion 1.5

#### 简单推理示例

```bash
cd basic
python run.py
```

**示例代码**：
```python
from diffusers import StableDiffusionPipeline
import torch

model_id = "sd-legacy/stable-diffusion-v1-5"
pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
pipe = pipe.to("cuda")

prompt = "一只宇航员骑着马"
image = pipe(prompt).images[0]
image.save("output.png")
```

#### 微调训练

```bash
cd basic
python train.py
```

**关键参数**：
- `batch_size`: 16（根据 VRAM 调整）
- `num_train_epochs`: 100
- `learning_rate`: 1e-6

### 2. 数据集处理

#### 支持的数据格式

- **Parquet 格式**（推荐）：包含 `image` 和 `caption` 列
- **本地目录**：图像文件和对应的文本文件
- **Hugging Face Datasets**：任何兼容的数据集

#### Parquet 数据集结构

```python
# 创建 Parquet 数据集示例
import pandas as pd
from PIL import Image

df = pd.DataFrame({
    'image': [image_bytes_1, image_bytes_2, ...],
    'text': ['图像描述 1', '图像描述 2', ...]
})
df.to_parquet('dataset.parquet')
```

### 3. DreamBooth 微调

```bash
cd dreambooth
python train.py \
    --instance_data_dir="/path/to/images" \
    --instance_prompt="一个 V 的照片" \
    --output_dir="./dreambooth_model" \
    --num_train_epochs=100 \
    --learning_rate=1e-6
```

**DreamBooth 特点**：
- 使用少量图像（3-5 张）微调模型
- 保持所学特征的通用性
- 支持 LoRA 以减少显存占用

**关键参数**：
- `instance_prompt`: 特定概念的描述
- `class_prompt`: （可选）类别信息
- `lora_rank`: LoRA 秩（默认 8，越大效果越好但显存占用越多）

### 4. SDXL 部分层微调

```bash
cd SDXL
python train.py
```

**微调策略**：

```python
# 支持的策略：
# - "cross_attention_only": 仅训练交叉注意力（推荐，显存最优）
# - "attention_only": 训练所有注意力层
# - "output_blocks": 训练输出块
# - "custom": 自定义策略
```

配置在代码中：
```python
unet = freeze_partial_layers(unet, strategy="cross_attention_only")
```

**优势**：
- 显存占用减少 60-80%
- 训练速度快 2-3 倍
- 仍能获得良好的微调效果

### 5. SDXL LoRA 微调（推荐）

#### 训练

```bash
cd SDXL-lora
python train.py \
    --dataset_path="path/to/dataset.parquet" \
    --output_dir="./lora_weights" \
    --batch_size=1 \
    --num_epochs=10 \
    --lora_rank=16 \
    --learning_rate=1e-4
```

**训练参数详解**：
- `lora_rank`: LoRA 低秩（8-32，默认 16）
- `lora_alpha`: LoRA 缩放因子（默认 32）
- `lora_dropout`: LoRA dropout 率（默认 0.1）
- `batch_size`: 批次大小（根据 VRAM 调整）
- `mixed_precision`: 混合精度（"fp16" 或 "bf16"）

#### 推理/图像生成

```bash
cd SDXL-lora
python test.py \
    --lora_weights="./lora_weights" \
    --prompts "一只可爱的猫" "中式庭院" \
    --output_dir="./generated_images"
```

**生成参数**：
```python
generation_kwargs = {
    "height": 1024,           # 图像高度
    "width": 1024,            # 图像宽度
    "num_inference_steps": 20, # 推理步数（越多质量越好但速度越慢）
    "guidance_scale": 7.5,    # 提示词引导权重
    "num_images_per_prompt": 1 # 每个提示词生成的图像数
}
```

## 📊 训练配置对比

| 方法 | 显存占用 | 训练速度 | 效果 | 灵活性 | 推荐场景 |
|------|---------|---------|------|--------|---------|
| 全量微调 | 最多 | 慢 | 最好 | 低 | 大规模数据集 |
| DreamBooth | 中等 | 中 | 很好 | 中 | 少量特定样本 |
| 部分层微调 | 少 | 快 | 好 | 中 | 有限显存场景 |
| LoRA | 最少 | 最快 | 好 | 最高 | 多样化微调 |

## 🎯 最佳实践

### 1. 数据准备

```bash
# 检查数据集大小
- 少量数据（<100 张）→ DreamBooth
- 中量数据（100-10k 张）→ LoRA 或部分层微调
- 大量数据（>10k 张）→ 全量微调
```

### 2. 学习率设置

```python
# 微调通常使用较小的学习率
LoRA:        1e-4 ~ 5e-4  # 推荐：1e-4
DreamBooth:  1e-6 ~ 5e-6  # 推荐：1e-6
部分层微调:   1e-5 ~ 1e-4  # 推荐：1e-5
```

### 3. 显存优化技巧

```python
# 启用梯度检查点（降低显存 20-30%）
unet.enable_gradient_checkpointing()

# 混合精度训练（fp16）
mixed_precision = "fp16"

# 梯度累积
gradient_accumulation_steps = 4

# 模型 CPU 卸载
pipe.enable_model_cpu_offload()
```

### 4. 超参数调整

```python
# 根据数据集调整
num_train_epochs = {
    'small': 100,      # < 100 张图像
    'medium': 50,      # 100-5000 张
    'large': 10        # > 5000 张
}

# 检查点保存频率
save_steps = len(dataset) // batch_size  # 每个 epoch 保存一次
```

## 📈 监控训练

### TensorBoard 可视化

```bash
# 启用 TensorBoard 日志
tensorboard --logdir=./output

# 访问 http://localhost:6006
```

### 关键指标

- **Loss**: 应稳定下降，最终收敛
- **Learning Rate**: 余弦衰减，逐步降低
- **Batch Time**: 监控训练速度变化

## 🔄 常见问题解决

### 1. OOM 错误（显存不足）

```python
# 方案 1: 减小批次大小
batch_size = 1

# 方案 2: 启用梯度累积
gradient_accumulation_steps = 4

# 方案 3: 启用梯度检查点
unet.enable_gradient_checkpointing()

# 方案 4: 使用 LoRA 代替全量微调
```

### 2. 训练不稳定（Loss 激增）

```python
# 检查：
# 1. 学习率是否过大 → 降低 10 倍
# 2. 数据是否规范化 → 检查像素值范围 [-1, 1]
# 3. 梯度是否爆炸 → 启用梯度裁剪
max_grad_norm = 1.0

# 4. 数据质量 → 清理低质量样本
```

### 3. 生成质量差

```python
# 检查点：
# 1. 训练数据质量
# 2. 提示词是否匹配训练中使用的格式
# 3. 推理步数是否足够（20-30）
# 4. Guidance scale 是否合适（7.5）
```

### 4. HuggingFace 下载超时

```bash
# 配置镜像地址
export HF_ENDPOINT="https://hf-mirror.com"

# 或在代码中设置
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
```

## 🚀 高级用法

### 1. 组合微调方法

```python
# 先用 DreamBooth 学习特定概念
# 再用 LoRA 适配风格

# DreamBooth 训练
python dreambooth/train.py ...

# 在 LoRA 中加载 DreamBooth 模型
pipe = StableDiffusionXLPipeline.from_pretrained(
    "path/to/dreambooth/model",
    torch_dtype=torch.float16
)
pipe.load_lora_weights("path/to/lora/weights")
```

### 2. 多 GPU 训练

```bash
# 使用 accelerate 进行分布式训练
accelerate config  # 配置分布式设置
accelerate launch train.py ...
```

### 3. 自定义数据集

```python
from torch.utils.data import Dataset

class CustomDataset(Dataset):
    def __init__(self, image_paths, captions):
        self.image_paths = image_paths
        self.captions = captions
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert('RGB')
        image = image.resize((512, 512))
        image = torch.from_numpy(np.array(image)).float() / 127.5 - 1.0
        image = image.permute(2, 0, 1)
        
        return {
            "image": image,
            "caption": self.captions[idx]
        }
```

## 📖 主要脚本说明

### basic/run.py
- 功能：简单的文本到图像生成
- 无需训练，直接使用预训练模型
- 用于快速测试和演示

### basic/train.py
- 功能：Stable Diffusion 1.5 全量微调
- 支持 Parquet 数据集
- 包含完整的训练循环和检查点保存

### dreambooth/train.py
- 功能：SDXL DreamBooth 微调
- 支持 LoRA 以减少显存
- 完整的数据验证和错误处理
- 多次stability检查防止 NaN

### SDXL/train.py
- 功能：SDXL 部分层微调
- 提供多种冻结策略
- 优化的内存管理和性能监控

### SDXL-lora/train.py
- 功能：完整的 SDXL LoRA 微调实现
- 详细的日志和监控
- 支持多种混合精度模式
- 检查点保存和恢复

### SDXL-lora/test.py
- 功能：使用训练好的 LoRA 权重进行推理
- 支持批量图像生成
- 自动寻找最新的检查点
- 性能计时和统计

## 💡 技术亮点

1. **稳定的训练**：
   - 数值稳定性检查（NaN/Inf 检测）
   - 梯度裁剪
   - 适应性学习率调度

2. **内存优化**：
   - 梯度累积
   - 梯度检查点
   - 选择性参数冻结

3. **灵活的架构**：
   - 模块化设计
   - 易于扩展
   - 支持自定义数据和模型

4. **完整的工具链**：
   - TensorBoard 集成
   - 检查点管理
   - 分布式训练支持

## 📝 引用和致谢

- 感谢 Hugging Face 提供的 `diffusers` 库
- Stable Diffusion 由 Stability AI 开发
- LoRA 技术参考论文：[LoRA: Low-Rank Adaptation](https://arxiv.org/abs/2106.09685)
- DreamBooth 论文：[DreamBooth: Fine Tuning Text-to-Image Diffusion Models](https://arxiv.org/abs/2208.12242)

## 📄 许可证

MIT License - 详见 LICENSE 文件

## 🤝 贡献指南

欢迎提交 Issue 和 Pull Request！

1. Fork 本仓库
2. 创建功能分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 开启 Pull Request

## 📧 联系方式

如有问题或建议，请提交 Issue 或联系项目维护者。

---

**最后更新**：2024 年 4 月

**项目状态**：✅ 活跃开发中

**支持的平台**：Linux, Windows, macOS
