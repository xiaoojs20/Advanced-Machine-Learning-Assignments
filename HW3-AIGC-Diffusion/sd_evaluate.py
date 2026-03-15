from diffusers import StableDiffusionPipeline, EulerDiscreteScheduler
import torch
from PIL import Image
import numpy as np
from torchmetrics.functional.multimodal import clip_score
from functools import partial
import os


from diffusers import AutoPipelineForText2Image
import torch


model_baseline = "runwayml/stable-diffusion-v1-5"
model_sd21 = "stabilityai/stable-diffusion-2-1"
model_clip = "openai/clip-vit-base-patch16"
lora_path_base = "/flash2/aml/xiaojs24/AML_3/hw3-base/sddata/finetune/lora"
save_path_base = "./image"
torch.manual_seed(1337)

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["CUDA_VISIBLE_DEVICES"] = "6"
os.makedirs(save_path_base, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# baseline + sd21 共8个任务
task_list = ["pokemon_baseline", "simpsons_baseline", "cartoon_baseline", "rubberduck_baseline",\
            "pokemon_better", "simpsons_better", "cartoon_better", "rubberduck_better"]
task_prompts  = 2*["pokemon-type", "simpsons-type", "cartoon-type", "rubberduck-type"]
prompt_list = 2 * ["A fierce dragon-type Pokémon breathing fire over a medieval castle.",\
                    "A person eating a donut in the living room, The Simpsons.",\
                    "Two cartoon bears having a picnic under a giant mushroom in a magical forest.",\
                    "A bright yellow rubber duck floating on a calm lake."]

AML_prompts = [ \
    "A futuristic AI-powered classroom environment where an intelligent virtual assistant, represented as a holographic figure, engages interactively with diverse students seated at smart desks equipped with touchscreens.", \
    "A state-of-the-art robotics lab where students are actively interacting with advanced humanoid robots. The scene includes robots demonstrating various machine learning applications, such as natural language processing, computer vision, and autonomous navigation.", \
    "An intricate and highly detailed visualization of a deep neural network architecture, featuring multiple layers of interconnected neurons with varying activation functions.", \
    "A vibrant and imaginative depiction of a small town entirely powered by large language model (LLM) agents. The townscape includes smart buildings with integrated digital interfaces, autonomous electric vehicles navigating the streets, and interactive kiosks." \
    ]


clip_score_fn = partial(clip_score, model_name_or_path=model_clip)

def calculate_clip_score(images, prompts):
    
    # import pdb;pdb.set_trace()
    # images_int = (np.asarray(images[0]) * 255).astype("uint8")
    images_int = (np.asarray(images) * 255).astype("uint8")
    clip_score = clip_score_fn(torch.from_numpy(images_int).permute(0, 3, 1, 2), prompts).detach()
    return round(float(clip_score), 4)

def sd_evaluate():
    """未经微调模型的evaluate"""
    for idx, task in enumerate(task_list):
        print(f"evaluating {task}...")
        # 每个model的clip_score
        clip_score = []
        
        if "baseline" in task:
            model_name = model_baseline
        else:
            model_name = model_sd21
            
        prompt = prompt_list[idx]
        save_path = os.path.join(save_path_base, task)

        pipeline = AutoPipelineForText2Image.from_pretrained(model_name, torch_dtype=torch.float16).to(device)
        images = pipeline(prompt).images
        sd_clip_score = calculate_clip_score(images, prompt)
        
        print(f"{task} without lora avg clip_score: {sd_clip_score}")
            

def evaluate():
    """几个微调模型的evaluate"""
    for idx, task in enumerate(task_list):
        print(f"evaluating {task}...")
        # 每个model的clip_score
        clip_score = []
        
        if "baseline" in task:
            model_name = model_baseline
        else:
            model_name = model_sd21
        
        prompt = prompt_list[idx] 
        lora_path = os.path.join(lora_path_base, task)
        save_path = os.path.join(save_path_base, task)

        pipeline = AutoPipelineForText2Image.from_pretrained(model_name, torch_dtype=torch.float16).to(device)

        for epoch in range(500, 15001, 500):
            pipeline.load_lora_weights(lora_path.format(epoch=epoch), weight_name="pytorch_lora_weights.safetensors")
            images = pipeline(prompt).images
            sd_clip_score = calculate_clip_score(images, prompt)
            clip_score.append(sd_clip_score)
            # save fig
            os.makedirs(save_path, exist_ok=True)
            images[0].save(os.path.join(save_path, f"image_{epoch}.png"))
        # save clip_score
        np.save(os.path.join(save_path, f"clip_score_{task}.npy"), clip_score)
        # avg clip_score
        print(f"{task} avg clip_score: {np.mean(clip_score)}")


def find_npy_max(npy_path):
    if not os.path.exists(npy_path):
        raise FileNotFoundError(f"文件 {npy_path} 不存在。")
    data = np.load(npy_path, allow_pickle=True)
    
    if isinstance(data, list):
        data = np.array(data)
    
    if not np.issubdtype(data.dtype, np.number):
        raise ValueError(".npy文件中的数据不是数字类型。")
    
    max_value = np.max(data)
    max_index = np.argmax(data)
    return max_value, max_index

def best_clip_score():
    """几个微调模型的best clip_score"""
    for idx, task in enumerate(task_list):
        print(f"evaluating {task}...")
        save_path = os.path.join(save_path_base, task)
        npy_path = os.path.join(save_path, f"clip_score_{task}.npy")
        
        max_value, max_index = find_npy_max(npy_path)
        print(f"{task} best clip_score: {max_value} at epoch {max_index*500+500}")
        
best_epochs = [13500, 5000, 14000, 4500, 13500, 2500, 10500, 13000]
def AML_assistant():
    """AML助教相关的推理"""
    # 风格 - 主题
    for idx, task in enumerate(task_list):
        print(f"Doing {task}...")
        if "baseline" in task:
            model_name = model_baseline
        else:
            model_name = model_sd21
        
        for jdx, prompt in enumerate(AML_prompts):
            lora_path = os.path.join(lora_path_base, task)
            save_path = os.path.join(save_path_base, "AML")
            os.makedirs(save_path, exist_ok=True)
            
            # pipeline = AutoPipelineForText2Image.from_pretrained(model_name, torch_dtype=torch.float16).to(device)
            
            prompt = task_prompts[idx] + prompt

        
            pipeline = AutoPipelineForText2Image.from_pretrained(model_name, torch_dtype=torch.float16).to(device)
            pipeline.load_lora_weights(lora_path.format(epoch=best_epochs[idx]), weight_name="pytorch_lora_weights.safetensors")
            images = pipeline(prompt).images
            sd_clip_score = calculate_clip_score(images, prompt)
            print(f"{task} best clip_score: {sd_clip_score}")
        
            images[0].save(os.path.join(save_path, f"image_{task}_{jdx+1}.png"))
        
        

if __name__ == "__main__":
    # sd_evaluate()
    # evaluate()
    # best_clip_score()
    AML_assistant()
    



# pokemon_baseline without lora avg clip_score: 34.1421
# simpsons_baseline without lora avg clip_score: 21.0196
# cartoon_baseline without lora avg clip_score: 37.4851
# evaluating rubberduck_baseline...
# rubberduck_baseline without lora avg clip_score: 30.4532

# pokemon_better without lora avg clip_score: 33.3011
# simpsons_better without lora avg clip_score: 29.1853
# cartoon_better without lora avg clip_score: 31.4626
# rubberduck_better without lora avg clip_score: 28.5196

# pokemon_baseline avg clip_score: 29.763863333333333
# simpsons_baseline avg clip_score: 27.257029999999997
# cartoon_baseline avg clip_score: 34.20535
# rubberduck_baseline avg clip_score: 25.791363333333333

# pokemon_better avg clip_score: 33.425073333333344
# simpsons_better avg clip_score: 32.07601666666667
# cartoon_better avg clip_score: 33.815540000000006
# rubberduck_better avg clip_score: 28.827123333333336

# evaluating pokemon_baseline...
# pokemon_baseline best clip_score: 36.2479 at epoch 13500
# evaluating simpsons_baseline...
# simpsons_baseline best clip_score: 33.7736 at epoch 5000
# evaluating cartoon_baseline...
# cartoon_baseline best clip_score: 37.8002 at epoch 14000
# evaluating rubberduck_baseline...
# rubberduck_baseline best clip_score: 29.7436 at epoch 4500
# evaluating pokemon_better...
# pokemon_better best clip_score: 36.7394 at epoch 13500
# evaluating simpsons_better...
# simpsons_better best clip_score: 37.2311 at epoch 2500
# evaluating cartoon_better...
# cartoon_better best clip_score: 37.0746 at epoch 10500
# evaluating rubberduck_better...
# rubberduck_better best clip_score: 32.3995 at epoch 13000