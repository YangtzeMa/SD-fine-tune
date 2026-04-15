#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DreamBooth训练脚本 - 从原始SDXL模型开始训练倪瓒风格
"""

import os
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import argparse
from tqdm import tqdm
import logging
from accelerate import Accelerator
import numpy as np
import math
from transformers import CLIPTextModel, CLIPTextModelWithProjection, CLIPTokenizer
from diffusers import (
    StableDiffusionXLPipeline, 
    UNet2DConditionModel, 
    AutoencoderKL, 
    DDPMScheduler,
    EulerDiscreteScheduler
)
from peft import LoraConfig, get_peft_model, TaskType
import json
from pathlib import Path
import random

# 设置huggingface镜像
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DreamBoothDataset(Dataset):
    """DreamBooth数据集类"""
    
    def __init__(self, image_dir, instance_prompt, class_prompt=None, size=1024, tokenizer_1=None, tokenizer_2=None, dtype=torch.float32):
        self.image_dir = Path(image_dir)
        self.instance_prompt = instance_prompt
        self.class_prompt = class_prompt
        self.size = size
        self.tokenizer_1 = tokenizer_1
        self.tokenizer_2 = tokenizer_2
        self.dtype = dtype
        
        # 支持的图像格式
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp']
        
        # 加载所有图像文件
        self.image_paths = []
        for ext in image_extensions:
            self.image_paths.extend(self.image_dir.glob(f'*{ext}'))
            self.image_paths.extend(self.image_dir.glob(f'*{ext.upper()}'))
        
        logger.info(f"Found {len(self.image_paths)} training images")
        
        if len(self.image_paths) == 0:
            raise ValueError(f"No image files found in {image_dir}")
        
        # 打印找到的图像
        for path in self.image_paths:
            logger.info(f"  - {path.name}")
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        
        try:
            # 加载和预处理图像
            image = Image.open(image_path).convert('RGB')
            original_size = image.size
            
            # 调整图像大小，保持宽高比
            image = image.resize((self.size, self.size), Image.Resampling.LANCZOS)
            
            # 转换为tensor，范围[-1, 1]
            image = torch.from_numpy(np.array(image)).to(dtype=torch.float32) / 127.5 - 1.0
            image = image.permute(2, 0, 1)  # HWC -> CHW
            
            # 确保图像数据类型与模型一致
            if self.dtype == torch.float16:
                image = image.half()
            elif self.dtype == torch.float32:
                image = image.float()
            
            # 使用实例提示词
            prompt = self.instance_prompt
            
            # 分词 - tokenizer_1
            inputs_1 = self.tokenizer_1(
                prompt,
                max_length=77,
                padding="max_length",
                truncation=True,
                return_tensors="pt"
            )
            
            # 分词 - tokenizer_2  
            inputs_2 = self.tokenizer_2(
                prompt,
                max_length=77,
                padding="max_length",
                truncation=True,
                return_tensors="pt"
            )
            
            return {
                "pixel_values": image,
                "input_ids_1": inputs_1.input_ids.squeeze(),
                "attention_mask_1": inputs_1.attention_mask.squeeze(),
                "input_ids_2": inputs_2.input_ids.squeeze(),
                "attention_mask_2": inputs_2.attention_mask.squeeze(),
                "prompt": prompt
            }
            
        except Exception as e:
            logger.error(f"Error processing image {image_path}: {e}")
            # 返回第一张图像作为fallback
            return self.__getitem__(0)

def collate_fn(batch):
    """数据整理函数"""
    pixel_values = torch.stack([item["pixel_values"] for item in batch])
    input_ids_1 = torch.stack([item["input_ids_1"] for item in batch])
    attention_mask_1 = torch.stack([item["attention_mask_1"] for item in batch])
    input_ids_2 = torch.stack([item["input_ids_2"] for item in batch])
    attention_mask_2 = torch.stack([item["attention_mask_2"] for item in batch])
    prompts = [item["prompt"] for item in batch]
    
    return {
        "pixel_values": pixel_values,
        "input_ids_1": input_ids_1,
        "attention_mask_1": attention_mask_1,
        "input_ids_2": input_ids_2,
        "attention_mask_2": attention_mask_2,
        "prompts": prompts
    }

def get_time_ids(height, width, device, dtype=torch.float32):
    """生成SDXL需要的time_ids"""
    target_size = (height, width)
    original_size = (height, width) 
    crop_coords = (0, 0)
    
    add_time_ids = list(original_size + crop_coords + target_size)
    add_time_ids = torch.tensor([add_time_ids], dtype=dtype, device=device)
    
    return add_time_ids

def main(args):
    # 强制设置环境变量以提高数值稳定性
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    
    # 初始化accelerator
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with="tensorboard" if args.use_tensorboard else None,
        project_dir=args.output_dir if args.use_tensorboard else None
    )
    
    # 设置随机种子
    if args.seed is not None:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        random.seed(args.seed)
    
    logger.info("Loading SDXL model components...")
    
    # 加载分词器
    tokenizer_1 = CLIPTokenizer.from_pretrained(
        args.model_path,
        subfolder="tokenizer",
        cache_dir=args.cache_dir
    )
    
    tokenizer_2 = CLIPTokenizer.from_pretrained(
        args.model_path,
        subfolder="tokenizer_2", 
        cache_dir=args.cache_dir
    )
    
    # 加载文本编码器
    text_encoder_1 = CLIPTextModel.from_pretrained(
        args.model_path,
        subfolder="text_encoder",
        cache_dir=args.cache_dir,
        torch_dtype=torch.float16 if args.mixed_precision == "fp16" else torch.float32
    )
    
    text_encoder_2 = CLIPTextModelWithProjection.from_pretrained(
        args.model_path,
        subfolder="text_encoder_2",
        cache_dir=args.cache_dir,
        torch_dtype=torch.float16 if args.mixed_precision == "fp16" else torch.float32
    )
    
    # 加载VAE
    vae = AutoencoderKL.from_pretrained(
        args.model_path,
        subfolder="vae",
        cache_dir=args.cache_dir,
        torch_dtype=torch.float16 if args.mixed_precision == "fp16" else torch.float32
    )
    
    # 加载UNet
    unet = UNet2DConditionModel.from_pretrained(
        args.model_path,
        subfolder="unet",
        cache_dir=args.cache_dir,
        torch_dtype=torch.float16 if args.mixed_precision == "fp16" else torch.float32
    )
    
    # 配置LoRA
    lora_config = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        target_modules=["to_k", "to_q", "to_v", "to_out.0", "ff.net.0.proj", "ff.net.2"],
        lora_dropout=args.lora_dropout,
        bias="none"
        # 注意：对于diffusion模型，我们不需要指定task_type
    )
    
    # 应用LoRA到UNet
    unet = get_peft_model(unet, lora_config)
    logger.info(f"Applied LoRA with rank={args.lora_rank}, alpha={args.lora_alpha}")
    
    # 强制初始化LoRA权重为更小的值
    with torch.no_grad():
        for name, param in unet.named_parameters():
            if "lora" in name and param.requires_grad:
                if "lora_A" in name:
                    torch.nn.init.normal_(param, mean=0.0, std=0.01)
                elif "lora_B" in name:
                    torch.nn.init.zeros_(param)
    
    # 冻结非LoRA参数
    text_encoder_1.requires_grad_(False)
    text_encoder_2.requires_grad_(False)
    vae.requires_grad_(False)
    
    # 确保只有LoRA参数需要梯度
    for name, param in unet.named_parameters():
        if "lora" not in name:
            param.requires_grad = False
    
    # 打印可训练参数数量
    trainable_params = sum(p.numel() for p in unet.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in unet.parameters())
    logger.info(f"Trainable parameters: {trainable_params:,} / {total_params:,} ({100 * trainable_params / total_params:.2f}%)")
    
    # 设置噪声调度器
    noise_scheduler = DDPMScheduler.from_pretrained(
        args.model_path,
        subfolder="scheduler",
        cache_dir=args.cache_dir
    )
    
    # 设置更稳定的训练配置
    noise_scheduler.set_timesteps(1000)  # 确保时间步数合理
    
    # 创建数据集
    train_dataset = DreamBoothDataset(
        image_dir=args.instance_data_dir,
        instance_prompt=args.instance_prompt,
        class_prompt=args.class_prompt,
        size=args.resolution,
        tokenizer_1=tokenizer_1,
        tokenizer_2=tokenizer_2,
        dtype=torch.float16 if args.mixed_precision == "fp16" else torch.float32
    )
    
    # 创建数据加载器
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.train_batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    # 配置优化器 - 使用更保守的设置
    optimizer = torch.optim.AdamW(
        unet.parameters(),
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
        amsgrad=True  # 使用AMSGrad变体以提高稳定性
    )
    
    # 学习率调度器
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.max_train_steps if args.max_train_steps else args.num_train_epochs * len(train_dataloader),
        eta_min=args.learning_rate * 0.1
    )
    
    # 准备训练组件
    unet, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        unet, optimizer, train_dataloader, lr_scheduler
    )
    
    # 将其他模型移到设备
    text_encoder_1.to(accelerator.device)
    text_encoder_2.to(accelerator.device)
    vae.to(accelerator.device)
    
    # 计算训练步数
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    
    logger.info("***** Running DreamBooth Training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size = {args.train_batch_size}")
    logger.info(f"  Gradient accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total train batch size = {args.train_batch_size * args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    logger.info(f"  Learning rate = {args.learning_rate}")
    
    # 训练循环
    global_step = 0
    progress_bar = tqdm(range(args.max_train_steps), disable=not accelerator.is_local_main_process)
    
    for epoch in range(args.num_train_epochs):
        unet.train()
        train_loss = 0.0
        
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(unet):
                # 编码图像到潜空间
                with torch.no_grad():
                    # 确保图像数据类型与VAE一致
                    pixel_values = batch["pixel_values"].to(vae.dtype).to(vae.device)
                    
                    # 检查输入数据
                    if torch.isnan(pixel_values).any() or torch.isinf(pixel_values).any():
                        logger.warning("NaN or Inf detected in input images, skipping this step")
                        continue
                    
                    # 限制像素值范围
                    pixel_values = torch.clamp(pixel_values, -1.0, 1.0)
                    
                    latents = vae.encode(pixel_values).latent_dist.sample()
                    latents = latents * vae.config.scaling_factor
                    
                    # 检查latents
                    if torch.isnan(latents).any() or torch.isinf(latents).any():
                        logger.warning("NaN or Inf detected in latents, skipping this step")
                        continue
                
                # 添加噪声
                noise = torch.randn_like(latents)
                bsz = latents.shape[0]
                timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device)
                timesteps = timesteps.long()
                
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
                
                # 编码文本
                with torch.no_grad():
                    # 第一个文本编码器
                    prompt_embeds_1 = text_encoder_1(
                        batch["input_ids_1"], 
                        attention_mask=batch["attention_mask_1"]
                    ).last_hidden_state
                    
                    # 第二个文本编码器 (带投影)
                    prompt_embeds_2 = text_encoder_2(
                        batch["input_ids_2"], 
                        attention_mask=batch["attention_mask_2"]
                    ).last_hidden_state
                    
                    # 获取池化输出 (text_embeds)
                    pooled_prompt_embeds = text_encoder_2(
                        batch["input_ids_2"], 
                        attention_mask=batch["attention_mask_2"]
                    ).text_embeds  # 池化的文本嵌入
                    
                    # 合并两个文本编码器的输出
                    encoder_hidden_states = torch.cat([prompt_embeds_1, prompt_embeds_2], dim=-1)
                    
                    # 生成time_ids和text_embeds
                    add_time_ids = get_time_ids(
                        args.resolution, 
                        args.resolution, 
                        latents.device,
                        dtype=encoder_hidden_states.dtype
                    )
                    add_time_ids = add_time_ids.repeat(bsz, 1)
                    
                    # 添加条件
                    added_cond_kwargs = {
                        "time_ids": add_time_ids,
                        "text_embeds": pooled_prompt_embeds
                    }
                
                # 预测噪声
                model_pred = unet(
                    noisy_latents,
                    timesteps,
                    encoder_hidden_states=encoder_hidden_states,
                    added_cond_kwargs=added_cond_kwargs
                ).sample
                
                # 检查模型输出的数值稳定性
                if torch.isnan(model_pred).any() or torch.isinf(model_pred).any():
                    logger.warning("NaN or Inf detected in model prediction, skipping this step")
                    continue
                
                # 计算损失
                if noise_scheduler.config.prediction_type == "epsilon":
                    target = noise
                elif noise_scheduler.config.prediction_type == "v_prediction":
                    target = noise_scheduler.get_velocity(latents, noise, timesteps)
                else:
                    raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")
                
                # 确保数据类型一致并检查数值稳定性
                model_pred = model_pred.float()
                target = target.float()
                
                # 检查是否有nan或inf值
                if torch.isnan(model_pred).any() or torch.isinf(model_pred).any():
                    logger.warning("NaN or Inf detected in model prediction, skipping this step")
                    continue
                if torch.isnan(target).any() or torch.isinf(target).any():
                    logger.warning("NaN or Inf detected in target, skipping this step")
                    continue
                
                loss = F.mse_loss(model_pred, target, reduction="mean")
                
                # 检查损失是否为nan
                if torch.isnan(loss) or torch.isinf(loss):
                    logger.warning(f"NaN or Inf loss detected: {loss.item()}, skipping this step")
                    continue
                
                # 反向传播
                accelerator.backward(loss)
                
                # 检查梯度数值稳定性
                if accelerator.sync_gradients:
                    # 检查梯度是否有nan或inf
                    has_nan_grad = False
                    for param in unet.parameters():
                        if param.grad is not None:
                            if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
                                has_nan_grad = True
                                break
                    
                    if has_nan_grad:
                        logger.warning("NaN or Inf detected in gradients, skipping optimizer step")
                        optimizer.zero_grad()
                        continue
                    
                    # 梯度裁剪
                    accelerator.clip_grad_norm_(unet.parameters(), args.max_grad_norm)
                
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
            
            # 更新进度
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1
                train_loss += loss.detach().item()
                
                # 记录日志
                if global_step % args.log_steps == 0:
                    avg_loss = train_loss / args.log_steps
                    current_lr = lr_scheduler.get_last_lr()[0]
                    logger.info(f"Step {global_step}, Loss: {avg_loss:.4f}, LR: {current_lr:.2e}")
                    
                    if accelerator.is_main_process and args.use_tensorboard:
                        accelerator.log({
                            "train_loss": avg_loss, 
                            "learning_rate": current_lr
                        }, step=global_step)
                    
                    train_loss = 0.0
                
                # 保存检查点
                if global_step % args.save_steps == 0 and accelerator.is_main_process:
                    save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                    accelerator.save_state(save_path)
                    
                    # 保存LoRA权重
                    unet_lora = accelerator.unwrap_model(unet)
                    lora_save_path = os.path.join(save_path, "lora_weights")
                    unet_lora.save_pretrained(lora_save_path)
                    logger.info(f"Saved checkpoint to {save_path}")
                
                if global_step >= args.max_train_steps:
                    break
    
    # 保存最终模型
    accelerator.end_training()
    if accelerator.is_main_process:
        unet_lora = accelerator.unwrap_model(unet)
        final_save_path = os.path.join(args.output_dir, "final_lora_weights")
        unet_lora.save_pretrained(final_save_path)
        logger.info(f"Training completed! Final model saved to {final_save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DreamBooth Training Script")
    
    # Model and data paths
    parser.add_argument("--model_path", type=str, default="stabilityai/stable-diffusion-xl-base-1.0",
                       help="Path to pretrained model")
    parser.add_argument("--instance_data_dir", type=str, required=True,
                       help="Directory containing instance images")
    parser.add_argument("--output_dir", type=str, required=True,
                       help="Output directory")
    parser.add_argument("--cache_dir", type=str, default=None,
                       help="Cache directory for models")
    
    # Training hyperparameters
    parser.add_argument("--instance_prompt", type=str, required=True,
                       help="Instance prompt")
    parser.add_argument("--class_prompt", type=str, default=None,
                       help="Class prompt")
    parser.add_argument("--resolution", type=int, default=1024,
                       help="Image resolution")
    parser.add_argument("--train_batch_size", type=int, default=1,
                       help="Training batch size")
    parser.add_argument("--num_train_epochs", type=int, default=100,
                       help="Number of training epochs")
    parser.add_argument("--max_train_steps", type=int, default=None,
                       help="Maximum training steps")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4,
                       help="Gradient accumulation steps")
    
    # LoRA parameters
    parser.add_argument("--lora_rank", type=int, default=16,
                       help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=32,
                       help="LoRA alpha")
    parser.add_argument("--lora_dropout", type=float, default=0.1,
                       help="LoRA dropout")
    
    # Optimizer parameters
    parser.add_argument("--learning_rate", type=float, default=1e-4,
                       help="Learning rate")
    parser.add_argument("--adam_beta1", type=float, default=0.9,
                       help="Adam beta1")
    parser.add_argument("--adam_beta2", type=float, default=0.999,
                       help="Adam beta2")
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2,
                       help="Adam weight decay")
    parser.add_argument("--adam_epsilon", type=float, default=1e-08,
                       help="Adam epsilon")
    parser.add_argument("--max_grad_norm", type=float, default=1.0,
                       help="Max gradient norm")
    
    # Other parameters
    parser.add_argument("--seed", type=int, default=None,
                       help="Random seed")
    parser.add_argument("--mixed_precision", type=str, default="fp16", 
                       choices=["no", "fp16", "bf16"],
                       help="Mixed precision")
    parser.add_argument("--save_steps", type=int, default=500,
                       help="Save steps")
    parser.add_argument("--log_steps", type=int, default=50,
                       help="Log steps")
    parser.add_argument("--use_tensorboard", action="store_true",
                       help="Use tensorboard")
    parser.add_argument("--num_workers", type=int, default=2,
                       help="Number of workers")
    
    args = parser.parse_args()
    
    # Validate arguments
    if not os.path.exists(args.instance_data_dir):
        raise FileNotFoundError(f"Instance data directory not found: {args.instance_data_dir}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Print configuration
    logger.info("DreamBooth Training Configuration:")
    for arg, value in vars(args).items():
        logger.info(f"  {arg}: {value}")
    
    main(args)
