import torch
from diffusers import StableDiffusionPipeline, DDPMScheduler, UNet2DConditionModel
from diffusers.optimization import get_cosine_schedule_with_warmup
from accelerate import Accelerator
from datasets import load_dataset
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os
from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import AutoencoderKL
import io
import numpy as np
from tqdm.auto import tqdm

class ParquetDataset(Dataset):
    def __init__(self, parquet_file, image_column="image", caption_column="caption"):
        self.data = pd.read_parquet(parquet_file)
        self.image_column = image_column
        self.caption_column = caption_column
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data.iloc[idx]
        
        # 处理图片数据
        image_data = item[self.image_column]
        if isinstance(image_data, dict):
            # 如果是字典格式，尝试获取图片数据
            if 'bytes' in image_data:
                image = Image.open(io.BytesIO(image_data['bytes']))
            elif 'data' in image_data:
                image = Image.open(io.BytesIO(image_data['data']))
            else:
                raise ValueError(f"无法识别的图片数据格式: {image_data.keys()}")
        elif isinstance(image_data, bytes):
            # 如果是字节格式
            image = Image.open(io.BytesIO(image_data))
        elif isinstance(image_data, str):
            # 如果是文件路径
            image = Image.open(image_data)
        else:
            raise ValueError(f"不支持的图片数据格式: {type(image_data)}")
            
        # 确保图片是RGB格式
        if image.mode != "RGB":
            image = image.convert("RGB")
            
        # 调整图片大小为模型期望的尺寸
        image = image.resize((512, 512))
        
        # 转换为张量
        image = torch.from_numpy(np.array(image)).float() / 127.5 - 1.0
        image = image.permute(2, 0, 1)  # 转换为CHW格式
        
        caption = item[self.caption_column]
        return {"image": image, "caption": caption}

def main():
    # 初始化accelerator
    accelerator = Accelerator(
        gradient_accumulation_steps=1,
        mixed_precision="fp16"
    )
    
    # 加载模型组件
    model_id = "sd-legacy/stable-diffusion-v1-5"
    tokenizer = CLIPTokenizer.from_pretrained(model_id, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(model_id, subfolder="text_encoder")
    vae = AutoencoderKL.from_pretrained(model_id, subfolder="vae")
    unet = UNet2DConditionModel.from_pretrained(model_id, subfolder="unet")
    noise_scheduler = DDPMScheduler.from_pretrained(model_id, subfolder="scheduler")
    
    # 将模型移动到GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")    
    text_encoder = text_encoder.to(device)
    vae = vae.to(device)
    unet = unet.to(device)
    
    # 冻结VAE和文本编码器
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    
    # 加载数据集
    dataset = ParquetDataset("dataset.parquet", image_column="image", caption_column="text")
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)
    
    # 训练参数
    train_batch_size = 16
    num_train_epochs = 100
    learning_rate = 1e-6
    
    # 优化器
    optimizer = torch.optim.AdamW(
        unet.parameters(),
        lr=learning_rate,
    )
    
    # 学习率调度器
    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=500,
        num_training_steps=len(dataloader) * num_train_epochs,
    )
    
    # 准备训练
    unet, optimizer, lr_scheduler, dataloader = accelerator.prepare(
        unet, optimizer, lr_scheduler, dataloader
    )
    
    # 训练循环
    for epoch in range(num_train_epochs):
        unet.train()
        epoch_loss = 0.0
        
        # 添加进度条
        progress_bar = tqdm(
            total=len(dataloader), 
            disable=not accelerator.is_local_main_process,
            desc=f"Epoch {epoch}"
        )
        
        for batch in dataloader:
            # 将数据移动到正确的设备
            images = batch["image"].to(device)
            captions = batch["caption"]
            
            # 处理图像
            latents = vae.encode(images).latent_dist.sample()
            latents = latents * 0.18215
            
            # 处理文本
            text_inputs = tokenizer(
                captions,
                padding="max_length",
                max_length=tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt"
            ).to(device)
            text_embeddings = text_encoder(text_inputs.input_ids)[0]
            
            # 添加噪声
            noise = torch.randn_like(latents)
            timesteps = torch.randint(0, 1000, (latents.shape[0],), device=device)
            noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
            
            # 预测噪声
            noise_pred = unet(noisy_latents, timesteps, encoder_hidden_states=text_embeddings).sample
            
            # 计算损失
            loss = torch.nn.functional.mse_loss(noise_pred, noise)
            
            # 反向传播
            accelerator.backward(loss)
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            
            # 累积损失
            epoch_loss += loss.detach().item()
            
            # 更新进度条
            progress_bar.update(1)
        
        # 关闭进度条
        progress_bar.close()
        
        # 计算并打印平均loss
        average_loss = epoch_loss / len(dataloader)
        if accelerator.is_main_process:
            print(f"Epoch {epoch} 完成, 平均损失: {average_loss:.4f}")
            
        # 保存检查点
        if epoch % 10 == 0:
            accelerator.wait_for_everyone()
            unwrapped_unet = accelerator.unwrap_model(unet)
            unwrapped_unet.save_pretrained(f"checkpoint-{epoch}")

if __name__ == "__main__":
    main() 