import os
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from PIL import Image
import io
from diffusers import StableDiffusionXLPipeline, UNet2DConditionModel, AutoencoderKL, DDPMScheduler
from transformers import CLIPTextModel, CLIPTextModelWithProjection, CLIPTokenizer
from peft import LoraConfig, get_peft_model, TaskType
import argparse
from tqdm import tqdm
import logging
from accelerate import Accelerator
import numpy as np
import math

# 设置huggingface镜像
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ImageTextDataset(Dataset):
    def __init__(self, parquet_path, tokenizer_1, tokenizer_2, size=1024):
        self.df = pd.read_parquet(parquet_path)
        self.tokenizer_1 = tokenizer_1
        self.tokenizer_2 = tokenizer_2
        self.size = size
        
        logger.info(f"Loaded dataset with {len(self.df)} samples")
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        try:
            # 解析图像
            image_bytes = row['image']['bytes']
            image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
            
            # 调整图像大小
            image = image.resize((self.size, self.size), Image.LANCZOS)
            
            # 修复：使用float32保证数值稳定性，后续转换为VAE所需精度
            image = torch.from_numpy(np.array(image)).to(dtype=torch.float32) / 127.5 - 1.0
            image = image.permute(2, 0, 1)  # HWC -> CHW
            
            # 数值稳定性检查
            if torch.isnan(image).any() or torch.isinf(image).any():
                logger.warning(f"Invalid values in image {idx}, using default")
                image = torch.zeros(3, self.size, self.size, dtype=torch.float16)

            # 处理文本
            text = str(row['text']).strip()
            if not text:
                text = "一张图片"  # 默认文本
            
            # 为SDXL编码文本（需要两个text encoder）
            inputs_1 = self.tokenizer_1(
                text,
                max_length=77,
                padding="max_length",
                truncation=True,
                return_tensors="pt"
            )
            
            inputs_2 = self.tokenizer_2(
                text,
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
                "text": text
            }
        except Exception as e:
            logger.warning(f"Error processing sample {idx}: {e}")
            # 返回默认样本
            default_image = torch.zeros(3, self.size, self.size, dtype=torch.float16)
            default_text = "一张图片"
            
            inputs_1 = self.tokenizer_1(
                default_text,
                max_length=77,
                padding="max_length",
                truncation=True,
                return_tensors="pt"
            )
            
            inputs_2 = self.tokenizer_2(
                default_text,
                max_length=77,
                padding="max_length",
                truncation=True,
                return_tensors="pt"
            )
            
            return {
                "pixel_values": default_image,
                "input_ids_1": inputs_1.input_ids.squeeze(),
                "attention_mask_1": inputs_1.attention_mask.squeeze(),
                "input_ids_2": inputs_2.input_ids.squeeze(),
                "attention_mask_2": inputs_2.attention_mask.squeeze(),
                "text": default_text
            }

def collate_fn(batch):
    pixel_values = torch.stack([item["pixel_values"] for item in batch])
    input_ids_1 = torch.stack([item["input_ids_1"] for item in batch])
    attention_mask_1 = torch.stack([item["attention_mask_1"] for item in batch])
    input_ids_2 = torch.stack([item["input_ids_2"] for item in batch])
    attention_mask_2 = torch.stack([item["attention_mask_2"] for item in batch])
    texts = [item["text"] for item in batch]
    
    return {
        "pixel_values": pixel_values,
        "input_ids_1": input_ids_1,
        "attention_mask_1": attention_mask_1,
        "input_ids_2": input_ids_2,
        "attention_mask_2": attention_mask_2,
        "texts": texts
    }

def get_time_ids(height, width, device):
    """生成SDXL需要的time_ids"""
    target_size = (height, width)
    original_size = (height, width)
    crop_coords = (0, 0)
    
    add_time_ids = list(original_size + crop_coords + target_size)
    add_time_ids = torch.tensor([add_time_ids], dtype=torch.float32, device=device)
    
    return add_time_ids

def main(args):
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with="tensorboard" if args.use_tensorboard else None,
        project_dir=args.output_dir if args.use_tensorboard else None
    )
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 加载模型组件
    logger.info("Loading SDXL components...")
    
    # 加载tokenizers
    tokenizer_1 = CLIPTokenizer.from_pretrained(
        args.model_id, 
        subfolder="tokenizer"
    )
    tokenizer_2 = CLIPTokenizer.from_pretrained(
        args.model_id, 
        subfolder="tokenizer_2"
    )
    
    # 加载text encoders
    text_encoder_1 = CLIPTextModel.from_pretrained(
        args.model_id,
        subfolder="text_encoder",
        torch_dtype=torch.float16
    )
    text_encoder_2 = CLIPTextModelWithProjection.from_pretrained(
        args.model_id,
        subfolder="text_encoder_2", 
        torch_dtype=torch.float16
    )
    
    # 加载VAE - 使用float32或bfloat16避免NaN问题
    vae_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float32
    logger.info(f"Using VAE dtype: {vae_dtype}")
    vae = AutoencoderKL.from_pretrained(
        args.model_id,
        subfolder="vae",
        torch_dtype=vae_dtype
    )
    
    # 加载UNet
    unet = UNet2DConditionModel.from_pretrained(
        args.model_id,
        subfolder="unet",
        torch_dtype=torch.float16
    )
    
    # 加载噪声调度器 - 关键修复！
    noise_scheduler = DDPMScheduler.from_pretrained(
        args.model_id,
        subfolder="scheduler"
    )
    
    # 冻结不需要训练的模型
    text_encoder_1.requires_grad_(False)
    text_encoder_2.requires_grad_(False)
    vae.requires_grad_(False)
    unet.requires_grad_(False)  # 先冻结，等添加LoRA后再解冻相关参数
    
    # 配置LoRA
    lora_config = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        target_modules=[
            "to_k", "to_q", "to_v", "to_out.0",
            "ff.net.0.proj", "ff.net.2"
        ],
        lora_dropout=args.lora_dropout,
    )
    
    # 应用LoRA到UNet
    unet = get_peft_model(unet, lora_config)
    
    if accelerator.is_main_process:
        unet.print_trainable_parameters()
    
    # 创建数据集和数据加载器
    logger.info("Creating dataset...")
    dataset = ImageTextDataset(
        args.dataset_path,
        tokenizer_1,
        tokenizer_2,
        size=args.resolution
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    # 优化器和学习率调度器
    optimizer = torch.optim.AdamW(
        unet.parameters(),
        lr=args.learning_rate,
        betas=(0.9, 0.999),
        weight_decay=args.weight_decay,
        eps=1e-08
    )
    
    # 计算总步数
    num_update_steps_per_epoch = math.ceil(len(dataloader) / args.gradient_accumulation_steps)
    max_train_steps = args.num_epochs * num_update_steps_per_epoch
    
    # 学习率调度器
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max_train_steps, eta_min=args.learning_rate * 0.1
    )
    
    # 使用accelerator准备模型和数据加载器
    unet, optimizer, dataloader, lr_scheduler = accelerator.prepare(
        unet, optimizer, dataloader, lr_scheduler
    )
    
    # 将其他模型移到GPU
    text_encoder_1 = text_encoder_1.to(accelerator.device)
    text_encoder_2 = text_encoder_2.to(accelerator.device)
    vae = vae.to(accelerator.device)
    
    # 初始化tracker
    if args.use_tensorboard and accelerator.is_main_process:
        accelerator.init_trackers("lora_sdxl_training")
    
    # 训练循环
    logger.info("Starting training...")
    global_step = 0
    
    for epoch in range(args.num_epochs):
        unet.train()
        epoch_loss = 0
        
        progress_bar = tqdm(
            dataloader, 
            desc=f"Epoch {epoch+1}/{args.num_epochs}",
            disable=not accelerator.is_local_main_process
        )
        
        for step, batch in enumerate(progress_bar):
            with accelerator.accumulate(unet):
                # 编码图像到潜在空间
                with torch.no_grad():
                    # 确保输入数据类型与VAE匹配
                    pixel_values = batch["pixel_values"].to(device=accelerator.device, dtype=vae.dtype)
                    
                    # 数值检查
                    if torch.isnan(pixel_values).any() or torch.isinf(pixel_values).any():
                        logger.warning(f"Invalid pixel values at step {step}, skipping")
                        continue
                        
                    # VAE编码 - 现在使用稳定的精度
                    latents = vae.encode(pixel_values).latent_dist.sample()
                    latents = latents * vae.config.scaling_factor
                    
                    # 转换回训练精度（如果需要）
                    if args.mixed_precision == "fp16":
                        latents = latents.to(dtype=torch.float16)
                    
                    # 检查潜在向量
                    if torch.isnan(latents).any() or torch.isinf(latents).any():
                        logger.warning(f"Invalid latents at step {step}, skipping")
                        continue
                
                # 添加噪声
                noise = torch.randn_like(latents)
                bsz = latents.shape[0]
                
                # 随机时间步
                timesteps = torch.randint(
                    0, noise_scheduler.config.num_train_timesteps, 
                    (bsz,), 
                    device=latents.device
                ).long()
                
                # 使用正确的噪声调度 - 关键修复！
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
                
                # 检查噪声后的潜在向量
                if torch.isnan(noisy_latents).any() or torch.isinf(noisy_latents).any():
                    logger.warning(f"Invalid noisy latents at step {step}, skipping")
                    continue

                # # 前向扩散过程 (添加噪声)
                # sqrt_alpha_prod = torch.sqrt(1 - timesteps.float() / 1000).view(-1, 1, 1, 1)
                # sqrt_one_minus_alpha_prod = torch.sqrt(timesteps.float() / 1000).view(-1, 1, 1, 1)
                # noisy_latents = sqrt_alpha_prod * latents + sqrt_one_minus_alpha_prod * noise
                
                # 编码文本
                with torch.no_grad():
                    # Text encoder 1
                    text_embeddings_1 = text_encoder_1(
                        batch["input_ids_1"],
                        attention_mask=batch["attention_mask_1"]
                    ).last_hidden_state
                    
                    # Text encoder 2
                    text_encoder_2_output = text_encoder_2(
                        batch["input_ids_2"],
                        attention_mask=batch["attention_mask_2"]
                    )
                    pooled_text_embeddings = text_encoder_2_output.text_embeds
                    text_embeddings_2 = text_encoder_2_output.last_hidden_state
                    
                    # 合并text embeddings
                    text_embeddings = torch.cat([text_embeddings_1, text_embeddings_2], dim=-1)
                    
                    # 检查文本嵌入
                    if torch.isnan(text_embeddings).any() or torch.isinf(text_embeddings).any():
                        logger.warning(f"Invalid text embeddings at step {step}, skipping")
                        continue
                
                # 生成time_ids
                add_time_ids = get_time_ids(args.resolution, args.resolution, accelerator.device)
                add_time_ids = add_time_ids.repeat(bsz, 1)
                
                # 预测噪声
                added_cond_kwargs = {
                    "text_embeds": pooled_text_embeddings,
                    "time_ids": add_time_ids
                }
                
                noise_pred = unet(
                    noisy_latents,
                    timesteps,
                    encoder_hidden_states=text_embeddings,
                    added_cond_kwargs=added_cond_kwargs,
                    return_dict=False
                )[0]
                
                # 检查预测结果
                if torch.isnan(noise_pred).any() or torch.isinf(noise_pred).any():
                    logger.warning(f"Invalid noise prediction at step {step}, skipping")
                    continue
                
                # 计算损失 - 添加数值稳定性
                loss = F.mse_loss(noise_pred.float(), noise.float(), reduction="mean")
                
                # 检查损失
                if torch.isnan(loss) or torch.isinf(loss):
                    logger.warning(f"Invalid loss at step {step}: {loss.item()}, skipping")
                    continue
                
                # 损失缩放检查
                if loss.item() > 100.0:  # 异常大的损失
                    logger.warning(f"Abnormally large loss: {loss.item()}, clipping")
                    loss = torch.clamp(loss, 0, 10.0)
                
                # 反向传播
                accelerator.backward(loss)
                
                if accelerator.sync_gradients:
                    # 梯度检查
                    total_norm = accelerator.clip_grad_norm_(unet.parameters(), args.max_grad_norm)
                    if torch.isnan(total_norm) or torch.isinf(total_norm):
                        logger.warning(f"Invalid gradient norm: {total_norm}, skipping optimizer step")
                        optimizer.zero_grad()
                        continue
                
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                
                # 记录loss
                epoch_loss += loss.item()
                
                if accelerator.sync_gradients:
                    global_step += 1
                
                # 更新进度条
                progress_bar.set_postfix({
                    "loss": f"{loss.item():.4f}",
                    "lr": f"{lr_scheduler.get_last_lr()[0]:.2e}",
                    "step": global_step
                })
                
                # 记录到tensorboard
                if args.use_tensorboard and accelerator.is_main_process and global_step % args.log_steps == 0:
                    accelerator.log({
                        "train/loss": loss.item(),
                        "train/learning_rate": lr_scheduler.get_last_lr()[0],
                        "train/epoch": epoch
                    }, step=global_step)
                
                # 保存checkpoint
                if global_step % args.save_steps == 0:
                    save_path = f"{args.output_dir}/checkpoint-step-{global_step}"
                    accelerator.save_state(save_path)
                    
                    # 保存LoRA权重
                    if accelerator.is_main_process:
                        unet_lora = accelerator.unwrap_model(unet)
                        lora_save_path = f"{save_path}/lora_weights"
                        unet_lora.save_pretrained(lora_save_path)
                        logger.info(f"Checkpoint saved to {save_path}")
        
        # Epoch结束
        avg_loss = epoch_loss / len(dataloader)
        
        if accelerator.is_main_process:
            logger.info(f"Epoch {epoch+1} completed. Average loss: {avg_loss:.4f}")
            
            # 保存epoch checkpoint
            epoch_save_path = f"{args.output_dir}/checkpoint-epoch-{epoch+1}"
            accelerator.save_state(epoch_save_path)
            
            unet_lora = accelerator.unwrap_model(unet)
            lora_save_path = f"{epoch_save_path}/lora_weights"
            unet_lora.save_pretrained(lora_save_path)
            
            if args.use_tensorboard:
                accelerator.log({
                    "epoch/loss": avg_loss,
                    "epoch/number": epoch + 1
                }, step=global_step)
    
    # 保存最终模型
    if accelerator.is_main_process:
        final_save_path = f"{args.output_dir}/final_lora_weights"
        os.makedirs(final_save_path, exist_ok=True)
        unet_lora = accelerator.unwrap_model(unet)
        unet_lora.save_pretrained(final_save_path)
        logger.info(f"Final LoRA weights saved to {final_save_path}")
    
    # 结束tracker
    if args.use_tensorboard:
        accelerator.end_training()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train LoRA for SDXL")
    
    # 模型参数
    parser.add_argument("--model_id", type=str, default="stabilityai/stable-diffusion-xl-base-1.0")
    parser.add_argument("--dataset_path", type=str, default="../stable-diffusion-v1-5/dataset.parquet")
    parser.add_argument("--output_dir", type=str, default="./output")
    parser.add_argument("--resolution", type=int, default=1024)
    
    # 训练参数
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_epochs", type=int, default=10)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--mixed_precision", type=str, default="fp16", choices=["no", "fp16", "bf16"])
    
    # LoRA参数
    parser.add_argument("--lora_rank", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.1)
    
    # 保存和日志参数
    parser.add_argument("--save_steps", type=int, default=500)
    parser.add_argument("--log_steps", type=int, default=50)
    parser.add_argument("--use_tensorboard", action="store_true", help="Use tensorboard for logging")
    
    # 数据加载参数
    parser.add_argument("--num_workers", type=int, default=2)
    
    args = parser.parse_args()
    
    # 验证参数
    if not os.path.exists(args.dataset_path):
        raise FileNotFoundError(f"Dataset not found at {args.dataset_path}")
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 打印配置
    logger.info("Training configuration:")
    for arg, value in vars(args).items():
        logger.info(f"  {arg}: {value}")
    
    main(args)
