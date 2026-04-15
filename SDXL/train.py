import torch
import os
import gc

# 在文件开头添加这些内存优化设置
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True,max_split_size_mb:64"
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

# 设置更激进的内存管理
torch.backends.cudnn.benchmark = False  # 改为False避免额外内存分配
torch.backends.cuda.matmul.allow_tf32 = True

# 清理所有缓存
torch.cuda.empty_cache()
gc.collect()


from diffusers import StableDiffusionXLPipeline, DDPMScheduler, UNet2DConditionModel
from diffusers.optimization import get_cosine_schedule_with_warmup
from accelerate import Accelerator
from datasets import load_dataset
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import time
from transformers import CLIPTextModel, CLIPTextModelWithProjection, CLIPTokenizer
from diffusers import AutoencoderKL
import io
import numpy as np
from tqdm.auto import tqdm
from args import default_args
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

def freeze_partial_layers(unet, strategy="cross_attention_only"):
    """
    部分层冻结策略
    strategy选项:
    - "attention_only": 只训练注意力层
    - "cross_attention_only": 只训练交叉注意力层(推荐)
    - "output_blocks": 只训练输出块
    - "last_layers": 只训练最后几层
    - "custom": 自定义策略
    """
    
    # 首先冻结所有参数
    for param in unet.parameters():
        param.requires_grad = False
    
    trainable_params = 0
    total_params = sum(p.numel() for p in unet.parameters())
    
    if strategy == "attention_only":
        # 只训练所有注意力层
        print("🔧 策略: 训练所有注意力层")
        for name, module in unet.named_modules():
            if any(attn_name in name for attn_name in ['attn1', 'attn2', 'attention']):
                for param in module.parameters():
                    param.requires_grad = True
                    trainable_params += param.numel()
                    
    elif strategy == "cross_attention_only":
        # 只训练交叉注意力层（与文本最相关）
        print("🔧 策略: 只训练交叉注意力层（推荐）")
        for name, module in unet.named_modules():
            if 'attn2' in name or 'cross_attention' in name:
                for param in module.parameters():
                    param.requires_grad = True
                    trainable_params += param.numel()
                    
    elif strategy == "output_blocks":
        # 只训练上采样/输出块
        print("🔧 策略: 训练输出块")
        for name, module in unet.named_modules():
            if any(block_name in name for block_name in ['up_blocks', 'conv_out', 'conv_norm_out']):
                for param in module.parameters():
                    param.requires_grad = True
                    trainable_params += param.numel()
                    
    elif strategy == "last_layers":
        # 只训练最后几层
        print("🔧 策略: 训练最后几层")
        target_layers = ['conv_out', 'conv_norm_out']
        for name, module in unet.named_modules():
            if any(layer_name in name for layer_name in target_layers):
                for param in module.parameters():
                    param.requires_grad = True
                    trainable_params += param.numel()
                    
    elif strategy == "custom":
        # 自定义策略：交叉注意力 + 部分上采样块
        print("🔧 策略: 自定义（交叉注意力 + 上采样块）")
        for name, module in unet.named_modules():
            # 交叉注意力层
            if 'attn2' in name or 'cross_attention' in name:
                for param in module.parameters():
                    param.requires_grad = True
                    trainable_params += param.numel()
            # 最后两个上采样块
            elif name.startswith('up_blocks.2') or name.startswith('up_blocks.3'):
                for param in module.parameters():
                    param.requires_grad = True
                    trainable_params += param.numel()
            # 输出层
            elif any(layer in name for layer in ['conv_out', 'conv_norm_out']):
                for param in module.parameters():
                    param.requires_grad = True
                    trainable_params += param.numel()
    
    print(f"📊 参数统计:")
    print(f"   可训练参数: {trainable_params:,}")
    print(f"   总参数: {total_params:,}")
    print(f"   训练比例: {trainable_params/total_params*100:.2f}%")
    print(f"   预计显存节省: {(1-trainable_params/total_params)*100:.1f}%")
    
    return unet

def print_trainable_layers(unet):
    """打印可训练的层信息"""
    print("\n🔍 可训练层详情:")
    trainable_layers = []
    for name, param in unet.named_parameters():
        if param.requires_grad:
            trainable_layers.append(name)
    
    if len(trainable_layers) > 20:
        print(f"   总共 {len(trainable_layers)} 个可训练层")
        print("   前10个:")
        for layer in trainable_layers[:10]:
            print(f"     - {layer}")
        print("   ...")
        print("   后10个:")
        for layer in trainable_layers[-10:]:
            print(f"     - {layer}")
    else:
        for layer in trainable_layers:
            print(f"   - {layer}")
    print()

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
            if 'bytes' in image_data:
                image = Image.open(io.BytesIO(image_data['bytes']))
            elif 'data' in image_data:
                image = Image.open(io.BytesIO(image_data['data']))
            else:
                raise ValueError(f"无法识别的图片数据格式: {image_data.keys()}")
        elif isinstance(image_data, bytes):
            image = Image.open(io.BytesIO(image_data))
        elif isinstance(image_data, str):
            image = Image.open(image_data)
        else:
            raise ValueError(f"不支持的图片数据格式: {type(image_data)}")
            
        if image.mode != "RGB":
            image = image.convert("RGB")
            
        # SDXL 使用 512x512 分辨率
        image = image.resize((512, 512))
        
        image = torch.from_numpy(np.array(image)).float() / 127.5 - 1.0
        image = image.permute(2, 0, 1)
        
        caption = item[self.caption_column]
        return {"image": image, "caption": caption}

def main():
    args = default_args
    
    # 添加训练时间统计
    training_start_time = time.time()
    print("🎨 开始 SDXL 中国画风格训练 (部分层微调版本)...")

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision="fp16"
    )
    
    # 使用 SDXL 模型
    model_id = "stabilityai/stable-diffusion-xl-base-1.0"
    
    # SDXL 使用双文本编码器
    tokenizer = CLIPTokenizer.from_pretrained(model_id, subfolder="tokenizer")
    tokenizer_2 = CLIPTokenizer.from_pretrained(model_id, subfolder="tokenizer_2")
    text_encoder = CLIPTextModel.from_pretrained(model_id, subfolder="text_encoder")
    text_encoder_2 = CLIPTextModelWithProjection.from_pretrained(model_id, subfolder="text_encoder_2")
    
    vae = AutoencoderKL.from_pretrained(model_id, subfolder="vae")
    unet = UNet2DConditionModel.from_pretrained(model_id, subfolder="unet")
    noise_scheduler = DDPMScheduler.from_pretrained(model_id, subfolder="scheduler")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")    
    
    # 应用部分层冻结策略
    print("\n" + "="*50)
    print("🔧 应用部分层微调策略")
    print("="*50)
    
    # 您可以在这里选择不同的策略：
    # "cross_attention_only" - 推荐，显存节省最多且效果好
    # "attention_only" - 训练所有注意力层
    # "output_blocks" - 训练输出块
    # "custom" - 自定义策略
    
    unet = freeze_partial_layers(unet, strategy="cross_attention_only")
    print_trainable_layers(unet)
    
    text_encoder = text_encoder.to(device)
    text_encoder_2 = text_encoder_2.to(device)
    vae = vae.to(device)
    unet = unet.to(device)
    
    if args.gradient_checkpointing:
        unet.enable_gradient_checkpointing()
    
    # 冻结编码器
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    text_encoder_2.requires_grad_(False)
    
    dataset = ParquetDataset("../stable-diffusion-v1-5/dataset.parquet", image_column="image", caption_column="text")
    
    # 因为显存节省了，可以适当增加batch size
    effective_batch_size = 2  # 从1增加到2
    args.gradient_accumulation_steps = 8  # 相应减少，保持总的有效batch size
    
    dataloader = DataLoader(dataset, batch_size=effective_batch_size, shuffle=True)
    
    # 只对可训练参数创建优化器
    trainable_params = [p for p in unet.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=args.learning_rate,
    )
    
    print(f"\n📈 训练配置:")
    print(f"   批次大小: {effective_batch_size}")
    print(f"   梯度累积步数: {args.gradient_accumulation_steps}")
    print(f"   有效批次大小: {effective_batch_size * args.gradient_accumulation_steps}")
    print(f"   优化器参数数量: {len(trainable_params)}")
    
    if args.lr_scheduler == "constant":
        lr_scheduler = get_cosine_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=args.lr_warmup_steps,
            num_training_steps=args.max_train_steps,
        )
    
    unet, optimizer, lr_scheduler, dataloader = accelerator.prepare(
        unet, optimizer, lr_scheduler, dataloader
    )
    
    # 创建主进度条
    main_progress = tqdm(
        total=args.max_train_steps,
        desc="🎨 SDXL部分微调进度",
        unit="step",
        disable=not accelerator.is_local_main_process
    )
    
    global_step = 0
    while global_step < args.max_train_steps:
        unet.train()
        epoch_loss = 0.0
        epoch_start_time = time.time()
        
        # 创建epoch进度条
        epoch_progress = tqdm(
            total=len(dataloader), 
            disable=not accelerator.is_local_main_process,
            desc=f"📊 Epoch进度 (Step {global_step}/{args.max_train_steps})"
        )
        
        for batch_idx, batch in enumerate(dataloader):
            # 每5步清理一次内存
            if batch_idx % 5 == 0:
                torch.cuda.empty_cache()
                gc.collect()
                
            batch_start_time = time.time()
            
            images = batch["image"].to(device)
            captions = batch["caption"]

            # 使用with torch.no_grad()包装不需要梯度的操作
            with torch.no_grad():
                # VAE编码
                latents = vae.encode(images).latent_dist.sample()
                latents = latents * vae.config.scaling_factor
                
                # 文本编码
                text_inputs = tokenizer(
                    captions,
                    padding="max_length",
                    max_length=tokenizer.model_max_length,
                    truncation=True,
                    return_tensors="pt"
                ).to(device)
                
                text_inputs_2 = tokenizer_2(
                    captions,
                    padding="max_length",
                    max_length=tokenizer_2.model_max_length,
                    truncation=True,
                    return_tensors="pt"
                ).to(device)
                
                text_encoder_output = text_encoder(text_inputs.input_ids)
                text_embeddings = text_encoder_output.last_hidden_state
                
                text_encoder_2_output = text_encoder_2(text_inputs_2.input_ids)
                text_embeddings_2 = text_encoder_2_output.last_hidden_state
                pooled_text_embeddings = text_encoder_2_output.text_embeds
                
                encoder_hidden_states = torch.cat([text_embeddings, text_embeddings_2], dim=-1)
                add_time_ids = torch.tensor([[512, 512, 0, 0, 512, 512]], device=device).repeat(images.shape[0], 1)
                
            noise = torch.randn_like(latents)
            timesteps = torch.randint(0, 1000, (latents.shape[0],), device=device)
            noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
            
            # 只有UNet前向传播需要梯度
            noise_pred = unet(
                noisy_latents, 
                timesteps, 
                encoder_hidden_states=encoder_hidden_states,
                added_cond_kwargs={"text_embeds": pooled_text_embeddings, "time_ids": add_time_ids}
            ).sample
            
            loss = torch.nn.functional.mse_loss(noise_pred, noise)
            loss = loss / args.gradient_accumulation_steps
            
            accelerator.backward(loss)
            
            # 删除中间变量释放内存
            del noise_pred, noise, noisy_latents, latents
            del text_embeddings, text_embeddings_2, encoder_hidden_states
            del text_inputs, text_inputs_2
            torch.cuda.empty_cache()
            
            if (global_step + 1) % args.gradient_accumulation_steps == 0:
                if args.max_grad_norm is not None:
                    accelerator.clip_grad_norm_(unet.parameters(), args.max_grad_norm)
                
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
            
            epoch_loss += loss.detach().item() * args.gradient_accumulation_steps
            batch_time = time.time() - batch_start_time
            
            # 更新进度条
            current_lr = optimizer.param_groups[0]['lr']
            epoch_progress.set_postfix({
                'Loss': f'{loss.item() * args.gradient_accumulation_steps:.4f}',
                'Avg Loss': f'{epoch_loss / (batch_idx + 1):.4f}',
                'LR': f'{current_lr:.2e}',
                'Time/batch': f'{batch_time:.1f}s'
            })
            epoch_progress.update(1)
            
            # 更新主进度条
            main_progress.update(1)
            steps_per_sec = 1.0 / batch_time if batch_time > 0 else 0
            total_time = time.time() - training_start_time
            eta_seconds = (args.max_train_steps - global_step) / steps_per_sec if steps_per_sec > 0 else 0
            eta_hours = eta_seconds / 3600
            
            main_progress.set_postfix({
                'Loss': f'{loss.item() * args.gradient_accumulation_steps:.4f}',
                'Step/s': f'{steps_per_sec:.2f}',
                'ETA': f'{eta_hours:.1f}h'
            })
            
            global_step += 1
            
            if global_step >= args.max_train_steps:
                break
        
        epoch_progress.close()
        
        # Epoch 统计
        epoch_time = time.time() - epoch_start_time
        average_loss = epoch_loss / len(dataloader)
        total_hours = (time.time() - training_start_time) / 3600
        
        if accelerator.is_main_process:
            print(f"\n📈 Epoch完成 - Step {global_step}/{args.max_train_steps}")
            print(f"   平均损失: {average_loss:.4f}")
            print(f"   Epoch用时: {epoch_time:.1f}秒")
            print(f"   总训练时间: {total_hours:.2f}小时\n")
            
        # 保存检查点
        if global_step % 1000 == 0:
            accelerator.wait_for_everyone()
            unwrapped_unet = accelerator.unwrap_model(unet)
            checkpoint_path = os.path.join(args.output_dir + "_partial", f"checkpoint-{global_step}")
            os.makedirs(checkpoint_path, exist_ok=True)
            unwrapped_unet.save_pretrained(checkpoint_path)
            
            if accelerator.is_main_process:
                print(f"✅ 部分微调检查点已保存: {checkpoint_path}")

    main_progress.close()
    
    # 训练完成统计
    total_training_time = time.time() - training_start_time
    avg_time_per_step = total_training_time / args.max_train_steps
    
    if accelerator.is_main_process:
        print(f"\n🎉 部分层微调训练完成!")
        print(f"   总训练时间: {total_training_time/3600:.2f}小时")
        print(f"   平均每步用时: {avg_time_per_step:.2f}秒")
        print(f"   最终平均损失: {average_loss:.4f}")
        print(f"   模型保存在: {args.output_dir}_partial")
        
        # 保存最终模型
        final_path = os.path.join(args.output_dir + "_partial", "final_model")
        os.makedirs(final_path, exist_ok=True)
        accelerator.unwrap_model(unet).save_pretrained(final_path)
        print(f"   最终模型保存在: {final_path}")

if __name__ == "__main__":
    main()
