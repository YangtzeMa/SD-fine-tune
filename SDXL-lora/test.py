import os
import torch
from diffusers import StableDiffusionXLPipeline, DPMSolverMultistepScheduler
from PIL import Image
import argparse
import time

# 设置huggingface镜像
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

def load_pipeline_with_lora(base_model_id, lora_weights_path, device="cuda"):
    """加载SDXL管道并应用LoRA权重"""
    
    print(f"🔄 正在加载基础模型: {base_model_id}")
    
    # 加载基础SDXL管道
    pipe = StableDiffusionXLPipeline.from_pretrained(
        base_model_id,
        torch_dtype=torch.float16,
        use_safetensors=True,
        variant="fp16"
    ).to(device)
    
    # 使用更快的调度器
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    
    # 加载LoRA权重
    if lora_weights_path and os.path.exists(lora_weights_path):
        print(f"🔄 正在加载LoRA权重: {lora_weights_path}")
        pipe.load_lora_weights(lora_weights_path)
        print("✅ LoRA权重加载成功")
    else:
        print("⚠️  未找到LoRA权重，使用基础模型")
    
    # 启用内存优化
    pipe.enable_attention_slicing()
    pipe.enable_vae_slicing()
    
    # 如果显存足够，可以启用模型CPU卸载
    # pipe.enable_model_cpu_offload()
    
    print("✅ 管道加载完成")
    return pipe

def generate_images(pipe, prompts, output_dir="./generated_images", **generation_kwargs):
    """生成图像"""
    
    os.makedirs(output_dir, exist_ok=True)
    
    default_kwargs = {
        "height": 1024,
        "width": 1024,
        "num_inference_steps": 20,
        "guidance_scale": 7.5,
        "num_images_per_prompt": 1,
        "generator": torch.Generator(device=pipe.device).manual_seed(42)
    }
    
    # 合并用户参数
    default_kwargs.update(generation_kwargs)
    
    results = []
    
    for i, prompt in enumerate(prompts):
        print(f"\n🎨 正在生成第 {i+1}/{len(prompts)} 张图像...")
        print(f"📝 提示词: {prompt}")
        
        start_time = time.time()
        
        try:
            # 生成图像
            with torch.no_grad():
                images = pipe(prompt, **default_kwargs).images
            
            generation_time = time.time() - start_time
            print(f"⏱️  生成时间: {generation_time:.2f}秒")
            
            # 保存图像
            for j, image in enumerate(images):
                filename = f"generated_{i+1}_{j+1}.png"
                filepath = os.path.join(output_dir, filename)
                image.save(filepath)
                
                results.append({
                    "prompt": prompt,
                    "filepath": filepath,
                    "generation_time": generation_time
                })
                
                print(f"💾 图像已保存: {filepath}")
                
        except Exception as e:
            print(f"❌ 生成失败: {e}")
            continue
    
    return results

def main():
    parser = argparse.ArgumentParser(description="使用训练好的LoRA模型生成图像")
    
    # 模型参数
    parser.add_argument("--base_model", type=str, 
                       default="stabilityai/stable-diffusion-xl-base-1.0",
                       help="基础SDXL模型ID")
    parser.add_argument("--lora_weights", type=str,
                       default="./output/final_lora_weights",
                       help="LoRA权重路径")
    parser.add_argument("--output_dir", type=str,
                       default="./generated_images",
                       help="输出目录")
    
    # 生成参数
    parser.add_argument("--prompts", type=str, nargs="+",
                       default=["一只可爱的猫咪在花园里玩耍", "古典中式建筑在夕阳下的美景"],
                       help="生成提示词列表")
    parser.add_argument("--height", type=int, default=1024, help="图像高度")
    parser.add_argument("--width", type=int, default=1024, help="图像宽度")
    parser.add_argument("--steps", type=int, default=20, help="推理步数")
    parser.add_argument("--guidance_scale", type=float, default=7.5, help="引导权重")
    parser.add_argument("--num_images", type=int, default=1, help="每个提示词生成的图像数量")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    
    # 设备参数
    parser.add_argument("--device", type=str, default="cuda", help="设备")
    
    args = parser.parse_args()
    
    # 检查设备
    if args.device == "cuda" and not torch.cuda.is_available():
        print("⚠️  CUDA不可用，切换到CPU")
        args.device = "cpu"
    
    print("🚀 开始图像生成...")
    print(f"设备: {args.device}")
    print(f"基础模型: {args.base_model}")
    print(f"LoRA权重: {args.lora_weights}")
    print(f"输出目录: {args.output_dir}")
    
    # 加载管道
    pipe = load_pipeline_with_lora(
        args.base_model,
        args.lora_weights,
        args.device
    )
    
    # 生成参数
    generation_kwargs = {
        "height": args.height,
        "width": args.width,
        "num_inference_steps": args.steps,
        "guidance_scale": args.guidance_scale,
        "num_images_per_prompt": args.num_images,
        "generator": torch.Generator(device=args.device).manual_seed(args.seed)
    }
    
    # 生成图像
    results = generate_images(
        pipe,
        args.prompts,
        args.output_dir,
        **generation_kwargs
    )
    
    # 总结结果
    print(f"\n🎉 生成完成!")
    print(f"总共生成了 {len(results)} 张图像")
    print(f"图像保存在: {args.output_dir}")
    
    for result in results:
        print(f"  - {result['filepath']} ({result['generation_time']:.2f}s)")

if __name__ == "__main__":
    # 如果想直接运行测试，可以取消注释以下代码
    # main()
    
    # 或者直接运行简单测试
    print("🚀 开始快速测试...")
    
    # 检查可用的检查点
    output_dir = "./output"
    lora_paths = []
    
    # 查找可用的LoRA权重
    if os.path.exists(output_dir):
        for item in os.listdir(output_dir):
            if item.startswith("checkpoint-"):
                lora_path = os.path.join(output_dir, item, "lora_weights")
                if os.path.exists(lora_path):
                    lora_paths.append((item, lora_path))
        
        # 检查最终权重
        final_path = os.path.join(output_dir, "final_lora_weights")
        if os.path.exists(final_path):
            lora_paths.append(("final", final_path))
    
    if lora_paths:
        print("🔍 找到以下LoRA权重:")
        for i, (name, path) in enumerate(lora_paths):
            print(f"  {i+1}. {name}: {path}")
        
        # 使用最新的权重（通常是final或最大的epoch/step）
        best_lora = lora_paths[-1][1]
        print(f"\n使用权重: {best_lora}")
        
        # 快速测试
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        pipe = load_pipeline_with_lora(
            "stabilityai/stable-diffusion-xl-base-1.0",
            best_lora,
            device
        )
        
        # 测试提示词
        test_prompts = [
            "一只可爱的小猫在阳光下睡觉",
            "古色古香的中式庭院，小桥流水",
            "现代都市夜景，霓虹灯闪烁"
        ]
        
        results = generate_images(
            pipe, 
            test_prompts,
            "./test_generated",
            height=1024,
            width=1024,
            num_inference_steps=20,
            guidance_scale=7.5
        )
        
        print(f"\n🎉 测试完成! 生成了 {len(results)} 张图像")
        
    else:
        print("❌ 未找到训练好的LoRA权重")
        print("请确保训练已完成并且权重文件存在")