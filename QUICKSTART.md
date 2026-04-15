# 快速入门指南 (Quick Start)

本指南将帮助你快速开始使用本项目。

## ⚡ 5 分钟快速开始

### 步骤 1: 克隆并安装

```bash
# 克隆仓库
git clone https://github.com/yourusername/stable-diffusion-finetuning.git
cd stable-diffusion-finetuning

# 创建虚拟环境
python -m venv venv
source venv/bin/activate  # Linux/Mac
# 或
venv\Scripts\activate  # Windows

# 安装依赖
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install transformers diffusers accelerate peft datasets pandas pillow
```

### 步骤 2: 第一次运行

```bash
# 进入 basic 目录
cd basic

# 运行推理脚本（自动下载模型）
python run.py
```

✅ 完成！你的第一张生成图像应该已保存为 `astronaut_rides_horse.png`

---

## 🎯 不同场景的快速指南

### 场景 1: 我只想生成图像（不训练）

```bash
cd basic
python run.py
```

修改 `run.py` 中的提示词：
```python
prompt = "你的提示词"
```

---

### 场景 2: 我有少量照片想学习特定对象（< 10 张）

**推荐方法**: DreamBooth

```bash
cd dreambooth

# 准备图像
mkdir my_images
# 把你的 3-5 张图像放到 my_images 文件夹

# 开始训练
python train.py \
    --instance_data_dir="my_images" \
    --instance_prompt="a photo of sks person" \
    --output_dir="./my_model" \
    --num_train_epochs=50 \
    --train_batch_size=1 \
    --learning_rate=1e-6
```

**训练时间**: ~2 小时（单 GPU）

---

### 场景 3: 我有大量图像数据想微调模型

**推荐方法**: SDXL LoRA

#### 准备数据

```python
# 创建 dataset.parquet
import pandas as pd
from PIL import Image
import io

images = []
captions = []

# 读取你的图像和描述
for img_path, caption in your_data:
    img = Image.open(img_path)
    img_bytes = io.BytesIO()
    img.save(img_bytes, format='PNG')
    images.append({'bytes': img_bytes.getvalue()})
    captions.append(caption)

df = pd.DataFrame({
    'image': images,
    'text': captions
})
df.to_parquet('dataset.parquet')
```

#### 训练

```bash
cd SDXL-lora

python train.py \
    --dataset_path="../dataset.parquet" \
    --output_dir="./lora_weights" \
    --num_epochs=10 \
    --batch_size=1 \
    --lora_rank=16
```

**训练时间**: 数小时到数天（取决于数据量和 GPU）

#### 生成图像

```bash
python test.py \
    --lora_weights="./lora_weights" \
    --prompts "你的提示词 1" "你的提示词 2" \
    --output_dir="./generated"
```

---

### 场景 4: 我有 GPU 显存限制（< 8GB）

**推荐方法**: LoRA + 优化设置

```bash
cd SDXL-lora

# 编辑 train.py，设置这些参数：
batch_size = 1
gradient_accumulation_steps = 4
mixed_precision = "fp16"
lora_rank = 8

# 启用梯度检查点
unet.enable_gradient_checkpointing()

# 然后运行训练
python train.py
```

期望显存使用: 4-6GB

---

## 📊 快速参考表

### 根据你的需求选择方法

| 需求 | 推荐方法 | 数据量 | 显存 | 训练时间 |
|------|---------|-------|------|---------|
| 快速体验 | basic/run.py | - | 4GB | 2 分钟 |
| 学习特定对象 | DreamBooth | 3-5 张 | 16GB | 2 小时 |
| 学习风格 | LoRA | 100-1000 | 8GB | 2-8 小时 |
| 全面微调 | 全量微调 | 10k+ | 24GB | 1-7 天 |

### 学习率快速选择

```python
# 根据方法选择合适的学习率
learning_rate = {
    'basic': 1e-6,
    'dreambooth': 1e-6,
    'lora': 1e-4,
    'full_finetune': 1e-6
}
```

---

## 🔧 常见问题速解

### Q1: CUDA out of memory

```python
# 最快的解决方案（三选一）：

# 方案 A: 减小批大小
batch_size = 1

# 方案 B: 启用梯度累积
gradient_accumulation_steps = 4

# 方案 C: 使用 LoRA
# （自动更省显存）
```

### Q2: 模型下载慢

```bash
# 设置镜像加速（中国用户）
export HF_ENDPOINT="https://hf-mirror.com"
```

### Q3: 训练很慢

```python
# 改进建议（优先级）：
# 1. 检查是否用了 CPU (应该用 GPU)
# 2. 减小图像尺寸（但效果会下降）
# 3. 减少推理步数（inference_steps）
# 4. 关闭梯度检查点（如果显存足够）
```

### Q4: 生成质量差

```python
# 检查列表：
# 1. 数据质量 ✓
# 2. 训练长度 ✓（可能需要更多 epoch）
# 3. 提示词格式 ✓（要与训练数据匹配）
# 4. 采样步数 >=20 ✓
# 5. 引导权重 7.5 ✓
```

---

## 📁 文件说明

### 训练脚本

| 文件 | 用途 | 难度 | 推荐 |
|------|------|------|------|
| basic/run.py | 推理/生成 | ⭐ | 初学者 |
| basic/train.py | 基础训练 | ⭐⭐ | 入门 |
| dreambooth/train.py | DreamBooth | ⭐⭐⭐ | 中级 |
| SDXL/train.py | 部分微调 | ⭐⭐⭐⭐ | 高级 |
| SDXL-lora/train.py | LoRA 训练 | ⭐⭐⭐ | 中级 |
| SDXL-lora/test.py | LoRA 推理 | ⭐⭐ | 入门 |

---

## 🚀 进阶建议

### 1. 监控训练

```bash
# 实时查看训练指标
tensorboard --logdir=./output
```

### 2. 比较不同方法

```bash
# 用同样的数据分别训练三种方法
# 1. 基础微调 (basic/train.py)
# 2. DreamBooth (dreambooth/train.py)
# 3. LoRA (SDXL-lora/train.py)

# 对比：速度、显存、质量
```

### 3. 调参优化

```python
# 常见的调参方向

# 提升质量：
learning_rate *= 0.5  # 更稳定的训练
num_train_epochs *= 2  # 更充分的学习

# 加快训练：
batch_size *= 2  # 如果显存允许
learning_rate *= 2  # 更快的收敛

# 节省显存：
lora_rank //= 2  # LoRA 秩减半
gradient_accumulation_steps *= 2
```

---

## 📚 后续学习

### 推荐流程

1. **第一步**: 运行 `basic/run.py` (5 分钟)
2. **第二步**: 用自己的图像运行 DreamBooth (2 小时)
3. **第三步**: 用数据集进行 LoRA 训练 (4 小时)
4. **第四步**: 阅读完整 README.md 了解高级特性
5. **第五步**: 尝试组合不同方法

### 推荐阅读

- 详细指南: [README.md](README.md)或 [README_EN.md](README_EN.md)
- DreamBooth 论文: https://arxiv.org/abs/2208.12242
- LoRA 论文: https://arxiv.org/abs/2106.09685
- Stable Diffusion 文档: https://huggingface.co/docs/diffusers

---

## 💡 小技巧

### 技巧 1: 快速测试

```bash
# 用小数据快速测试（5 分钟）
python train.py \
    --num_train_epochs=1 \
    --num_training_samples=10

# 确认没问题后再用完整数据训练
```

### 技巧 2: 保存多个检查点

```python
# 在训练配置中：
save_steps = len(dataset) // batch_size  # 每个 epoch 保存一次

# 这样可以：
# 1. 比较不同 epoch 的质量
# 2. 如果显存溢出可以从中间恢复
```

### 技巧 3: 使用权重最佳实践

```bash
# 保存多个版本做对比
checkpoints/
├── epoch_10/          # 早期
├── epoch_50/          # 中期
├── epoch_100/         # 晚期
└── final/             # 最佳选择

# 然后用不同版本测试：
python test.py --lora_weights "./checkpoints/epoch_50"
```

---

## 📝 检查清单

运行你的第一个训练前，确保：

- [ ] Python 3.8+ 已安装
- [ ] CUDA 11.8+ 已安装
- [ ] 虚拟环境已创建
- [ ] 依赖已安装
- [ ] 数据已准备好
- [ ] 输出目录已创建

---

## 🆘 仍需帮助？

1. 查看 README.md 中的常见问题部分
2. 提交 Issue (包含错误日志)
3. 查看项目中的详细注释

**记住**: 大多数问题都由显存、学习率或数据质量引起！

---

**祝你训练顺利！** 🎉
