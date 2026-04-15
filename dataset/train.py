from dataset import PoemCartoonDataset
import torch
from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import UNet2DConditionModel, DDPMScheduler, AutoencoderKL
from datasets import load_dataset
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from accelerate import Accelerator
import os

def main():
    # Initialize accelerator
    accelerator = Accelerator(
        gradient_accumulation_steps=1,
        mixed_precision="fp16"
    )
    
    # Load dataset
    dataset = PoemCartoonDataset(split='train')
    
    # Load model components
    model_id = "sd-legacy/stable-diffusion-v1-5"
    tokenizer = CLIPTokenizer.from_pretrained(model_id, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(model_id, subfolder="text_encoder")
    vae = AutoencoderKL.from_pretrained(model_id, subfolder="vae")
    unet = UNet2DConditionModel.from_pretrained(model_id, subfolder="unet")
    noise_scheduler = DDPMScheduler.from_pretrained(model_id, subfolder="scheduler")
    
    # Move models to GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    text_encoder = text_encoder.to(device)
    vae = vae.to(device)
    unet = unet.to(device)
    
    # Freeze VAE and text encoder
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    
    # Prepare DataLoader
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)
    
    # Training parameters
    num_train_epochs = 100
    learning_rate = 1e-6
    
    # Optimizer
    optimizer = torch.optim.AdamW(
        unet.parameters(),
        lr=learning_rate,
    )
    
    # Training loop
    for epoch in range(num_train_epochs):
        unet.train()
        epoch_loss = 0.0
        
        progress_bar = tqdm(total=len(dataloader), desc=f"Epoch {epoch}")
        
        for batch in dataloader:
            images = batch['image'].to(device)
            texts = batch['text']
            
            # Process images
            latents = vae.encode(images).latent_dist.sample()
            latents = latents * 0.18215
            
            # Process text
            text_inputs = tokenizer(
                texts,
                padding="max_length",
                max_length=tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt"
            ).to(device)
            text_embeddings = text_encoder(text_inputs.input_ids)[0]
            
            # Add noise
            noise = torch.randn_like(latents)
            timesteps = torch.randint(0, 1000, (latents.shape[0],), device=device)
            noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
            
            # Predict noise
            noise_pred = unet(noisy_latents, timesteps, encoder_hidden_states=text_embeddings).sample
            
            # Calculate loss
            loss = torch.nn.functional.mse_loss(noise_pred, noise)
            
            # Backpropagation
            accelerator.backward(loss)
            optimizer.step()
            optimizer.zero_grad()
            
            epoch_loss += loss.detach().item()
            progress_bar.update(1)
        
        progress_bar.close()
        average_loss = epoch_loss / len(dataloader)
        print(f"Epoch {epoch} completed, average loss: {average_loss:.4f}")

if __name__ == "__main__":
    main()