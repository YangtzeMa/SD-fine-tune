# Stable Diffusion Model Fine-tuning Project

[English](README.md) | [中文](README_cn.md)

A comprehensive framework for fine-tuning Stable Diffusion and SDXL models on custom datasets, featuring multiple fine-tuning methods, data processing pipelines, and inference scripts.

## 📋 Project Overview

This project provides a complete solution from basic fine-tuning to advanced LoRA adaptation for training text-to-image generation models. Supports Stable Diffusion 1.5, SDXL, and various fine-tuning techniques.

### Key Features

- ✨ **Multiple Fine-tuning Methods**: Basic fine-tuning, DreamBooth, LoRA, and partial layer fine-tuning
- 🚀 **Model Support**: Stable Diffusion 1.5 and SDXL 1.0
- 💾 **Flexible Data Handling**: Support for Parquet, local image directories, and more
- 🎯 **Memory Optimization**: Gradient accumulation, mixed precision training, model CPU offload
- 📊 **Complete Training Pipeline**: Logging, checkpoint saving, and visualization
- 🔧 **Modular Design**: Easy to extend and customize

## 📁 Project Structure

```
.
├── basic/                    # Basic Stable Diffusion 1.5 examples
│   ├── run.py               # Text-to-image generation example
│   └── train.py             # SD 1.5 basic fine-tuning
│
├── dataset/                 # Dataset processing examples
│   └── train.py             # Poetry-cartoon dataset training
│
├── dreambooth/              # DreamBooth fine-tuning method
│   └── train.py             # SDXL DreamBooth + LoRA training
│
├── SDXL/                    # SDXL partial layer fine-tuning
│   └── train.py             # SDXL cross-attention fine-tuning
│
├── SDXL-lora/               # SDXL LoRA fine-tuning complete implementation
│   ├── train.py             # LoRA training script
│   └── test.py              # LoRA inference and image generation
│
└── report.pdf               # Project report
```

## 🔧 Environment Setup

### System Requirements

- Python 3.8 or higher
- CUDA 11.8 or higher (for GPU acceleration)
- At least 16GB VRAM (24GB+ recommended)
- Sufficient disk space (model files ~10-50GB)

### Install Dependencies

```bash
# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate  # Windows

# Install dependencies
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install transformers diffusers accelerate peft datasets pandas pillow tqdm
pip install tensorboard  # Optional, for training visualization
```

### Configure HuggingFace Mirror (Optional)

```bash
# Set environment variable
export HF_ENDPOINT="https://hf-mirror.com"  # China users
```

## 📚 Usage Guide

### 1. Basic Usage - Stable Diffusion 1.5

#### Simple Inference Example

```bash
cd basic
python run.py
```

**Example Code**:
```python
from diffusers import StableDiffusionPipeline
import torch

model_id = "sd-legacy/stable-diffusion-v1-5"
pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
pipe = pipe.to("cuda")

prompt = "an astronaut riding a horse"
image = pipe(prompt).images[0]
image.save("output.png")
```

#### Fine-tuning Training

```bash
cd basic
python train.py
```

**Key Parameters**:
- `batch_size`: 16 (adjust based on VRAM)
- `num_train_epochs`: 100
- `learning_rate`: 1e-6

### 2. Dataset Processing

#### Supported Data Formats

- **Parquet Format** (recommended): Contains `image` and `caption` columns
- **Local Directory**: Image files and corresponding text files
- **Hugging Face Datasets**: Any compatible dataset

#### Parquet Dataset Structure

```python
# Create Parquet dataset example
import pandas as pd

df = pd.DataFrame({
    'image': [image_bytes_1, image_bytes_2, ...],
    'text': ['image description 1', 'image description 2', ...]
})
df.to_parquet('dataset.parquet')
```

### 3. DreamBooth Fine-tuning

```bash
cd dreambooth
python train.py \
    --instance_data_dir="/path/to/images" \
    --instance_prompt="a photo of V" \
    --output_dir="./dreambooth_model" \
    --num_train_epochs=100 \
    --learning_rate=1e-6
```

**DreamBooth Characteristics**:
- Fine-tune with few images (3-5)
- Preserve learned features generality
- Support LoRA to reduce VRAM usage

**Key Parameters**:
- `instance_prompt`: Description of specific concept
- `class_prompt`: (optional) Class information
- `lora_rank`: LoRA rank (default 8, higher = better but more VRAM)

### 4. SDXL Partial Layer Fine-tuning

```bash
cd SDXL
python train.py
```

**Fine-tuning Strategies**:

```python
# Available strategies:
# - "cross_attention_only": Only train cross-attention (recommended, best VRAM)
# - "attention_only": Train all attention layers
# - "output_blocks": Train output blocks
# - "custom": Custom strategy
```

Configure in code:
```python
unet = freeze_partial_layers(unet, strategy="cross_attention_only")
```

**Advantages**:
- 60-80% reduction in VRAM usage
- 2-3x faster training
- Good fine-tuning results

### 5. SDXL LoRA Fine-tuning (Recommended)

#### Training

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

**Parameter Explanation**:
- `lora_rank`: LoRA low rank (8-32, default 16)
- `lora_alpha`: LoRA scaling factor (default 32)
- `lora_dropout`: LoRA dropout rate (default 0.1)
- `batch_size`: Batch size (adjust based on VRAM)
- `mixed_precision`: Mixed precision ("fp16" or "bf16")

#### Inference/Image Generation

```bash
cd SDXL-lora
python test.py \
    --lora_weights="./lora_weights" \
    --prompts "a cute cat" "Chinese garden" \
    --output_dir="./generated_images"
```

**Generation Parameters**:
```python
generation_kwargs = {
    "height": 1024,            # Image height
    "width": 1024,             # Image width
    "num_inference_steps": 20, # Inference steps (more = better but slower)
    "guidance_scale": 7.5,     # Prompt guidance weight
    "num_images_per_prompt": 1 # Images per prompt
}
```

## 📊 Training Method Comparison

| Method | VRAM | Speed | Quality | Flexibility | Best For |
|--------|------|-------|---------|-------------|----------|
| Full Fine-tune | Most | Slow | Best | Low | Large datasets |
| DreamBooth | Medium | Medium | Very Good | Medium | Few specific samples |
| Partial Layer | Less | Fast | Good | Medium | Limited VRAM |
| LoRA | Least | Fastest | Good | Highest | Diverse fine-tuning |

## 🎯 Best Practices

### 1. Data Preparation

```bash
# Dataset size guidelines:
- Small (<100 images)     → DreamBooth
- Medium (100-10k)        → LoRA or partial layer fine-tuning
- Large (>10k)            → Full fine-tuning
```

### 2. Learning Rate Settings

```python
# Fine-tuning typically uses smaller learning rates
LoRA:           1e-4 ~ 5e-4  # Recommended: 1e-4
DreamBooth:     1e-6 ~ 5e-6  # Recommended: 1e-6
Partial layers: 1e-5 ~ 1e-4  # Recommended: 1e-5
```

### 3. Memory Optimization Tips

```python
# Enable gradient checkpointing (reduces VRAM by 20-30%)
unet.enable_gradient_checkpointing()

# Mixed precision training (fp16)
mixed_precision = "fp16"

# Gradient accumulation
gradient_accumulation_steps = 4

# Model CPU offload
pipe.enable_model_cpu_offload()
```

### 4. Hyperparameter Tuning

```python
# Adjust based on dataset size
num_train_epochs = {
    'small': 100,      # < 100 images
    'medium': 50,      # 100-5000
    'large': 10        # > 5000
}

# Checkpoint save frequency
save_steps = len(dataset) // batch_size  # Once per epoch
```

## 📈 Training Monitoring

### TensorBoard Visualization

```bash
# Enable TensorBoard logging
tensorboard --logdir=./output

# Visit http://localhost:6006
```

### Key Metrics

- **Loss**: Should decrease steadily and converge
- **Learning Rate**: Cosine decay, gradually decreasing
- **Batch Time**: Monitor training speed changes

## 🔄 Troubleshooting

### 1. OOM Error (Out of Memory)

```python
# Solution 1: Reduce batch size
batch_size = 1

# Solution 2: Enable gradient accumulation
gradient_accumulation_steps = 4

# Solution 3: Enable gradient checkpointing
unet.enable_gradient_checkpointing()

# Solution 4: Use LoRA instead of full fine-tuning
```

### 2. Unstable Training (Loss Spikes)

```python
# Check:
# 1. Learning rate too high → reduce 10x
# 2. Data normalization → check pixel range [-1, 1]
# 3. Gradient explosion → enable gradient clipping
max_grad_norm = 1.0

# 4. Data quality → clean low quality samples
```

### 3. Poor Generation Quality

```python
# Checklist:
# 1. Training data quality
# 2. Prompt format matches training data
# 3. Sufficient inference steps (20-30)
# 4. Proper guidance scale (7.5)
```

### 4. HuggingFace Download Timeout

```bash
# Configure mirror
export HF_ENDPOINT="https://hf-mirror.com"

# Or set in code
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
```

## 🚀 Advanced Usage

### 1. Combine Fine-tuning Methods

```python
# First: DreamBooth to learn specific concept
# Then: LoRA to adapt style

# DreamBooth training
python dreambooth/train.py ...

# Load DreamBooth in LoRA
pipe = StableDiffusionXLPipeline.from_pretrained(
    "path/to/dreambooth/model",
    torch_dtype=torch.float16
)
pipe.load_lora_weights("path/to/lora/weights")
```

### 2. Multi-GPU Training

```bash
# Configure accelerate for distributed training
accelerate config
accelerate launch train.py ...
```

### 3. Custom Datasets

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

## 📖 Main Script Descriptions

### basic/run.py
- Function: Simple text-to-image generation
- No training needed, uses pre-trained models
- For quick testing and demos

### basic/train.py
- Function: Stable Diffusion 1.5 full fine-tuning
- Supports Parquet datasets
- Complete training loop and checkpoint saving

### dreambooth/train.py
- Function: SDXL DreamBooth fine-tuning
- LoRA support to reduce VRAM
- Complete data validation and error handling
- Multiple stability checks to prevent NaN

### SDXL/train.py
- Function: SDXL partial layer fine-tuning
- Multiple freezing strategies
- Optimized memory management and performance monitoring

### SDXL-lora/train.py
- Function: Complete SDXL LoRA fine-tuning
- Detailed logging and monitoring
- Multiple mixed precision modes
- Checkpoint saving and recovery

### SDXL-lora/test.py
- Function: Inference using trained LoRA weights
- Batch image generation
- Auto-find latest checkpoint
- Performance timing and statistics

## 💡 Technical Highlights

1. **Stable Training**:
   - Numerical stability checks (NaN/Inf detection)
   - Gradient clipping
   - Adaptive learning rate scheduling

2. **Memory Optimization**:
   - Gradient accumulation
   - Gradient checkpointing
   - Selective parameter freezing

3. **Flexible Architecture**:
   - Modular design
   - Easy to extend
   - Custom data and model support

4. **Complete Toolchain**:
   - TensorBoard integration
   - Checkpoint management
   - Distributed training support

## 📝 References and Acknowledgments

- Thanks to Hugging Face for `diffusers` library
- Stable Diffusion developed by Stability AI
- LoRA paper: [LoRA: Low-Rank Adaptation](https://arxiv.org/abs/2106.09685)
- DreamBooth paper: [DreamBooth: Fine Tuning Text-to-Image Diffusion Models](https://arxiv.org/abs/2208.12242)

## 📄 License

MIT License - See LICENSE file for details

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📧 Contact

For questions or suggestions, please open an Issue or contact the project maintainers.

---

**Last Updated**: April 2024

**Project Status**: ✅ Active Development

**Supported Platforms**: Linux, Windows, macOS
