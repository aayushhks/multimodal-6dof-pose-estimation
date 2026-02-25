import sys
import os
import json
import torch
from PIL import Image

# 1. SETUP PATHS
PROJECT_ROOT = "/projectnb/cs598/aayushks/EgoOrientBench/model_train/InternVL_4B/internvl_chat"
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

JSON_PATH = "/projectnb/cs598/data_storage/msi_datasets/EgoOrientBench/all_data/EgocentricDataset/benchmark_data/benchmark.json"
IMAGE_DIR = "/projectnb/cs598/data_storage/msi_datasets/EgoOrientBench/all_data/EgocentricDataset/images/imagenet_after/"

from internvl.model.internvl_chat import InternVLChatModel
from transformers import AutoTokenizer
from internvl.train.dataset import build_transform

def run_experiment():
    # FORCE SINGLE GPU (cuda:0)
    device = torch.device("cuda:0")
    print(f" Loading model onto {device} (Single-GPU Mode)...")
    
    tokenizer = AutoTokenizer.from_pretrained("OpenGVLab/InternVL2-4B", trust_remote_code=True, use_fast=False)
    
    # We load specifically to cuda:0 to avoid multi-device fragmentation
    model = InternVLChatModel.from_pretrained(
        "OpenGVLab/InternVL2-4B", 
        low_cpu_mem_usage=True, 
        torch_dtype=torch.bfloat16
    ).to(device).eval()
    
    model.config.use_cache = False

    with open(JSON_PATH, "r") as f:
        data = json.load(f)
    
    print(f" Setup verified. Processing samples...")
    correct, total = 0, 0

    for i, item in enumerate(data[:50]):
        filename = os.path.basename(item['image'])
        img_path = os.path.join(IMAGE_DIR, filename)

        if not os.path.exists(img_path):
            continue

        try:
            # 1. Transform Image
            transform = build_transform(is_train=False, input_size=448)
            pixel_values = transform(Image.open(img_path).convert('RGB')).unsqueeze(0)
            
            # 2. MOVE PIXELS TO SAME DEVICE AS MODEL
            pixel_values = pixel_values.to(torch.bfloat16).to(device)
            
            question = f"<image>\n{item['question']}\nAnswer with only the letter."
            
            # 3. Inference
            response = model.chat(
                tokenizer=tokenizer,
                pixel_values=pixel_values,
                question=question,
                generation_config={"max_new_tokens": 10, "do_sample": False}
            )
            
            ans = response.strip().upper()
            label = item['label'].strip().upper()
            
            is_correct = (len(ans) > 0 and (ans[0] == label or label in ans))
            if is_correct:
                correct += 1
            total += 1
            
            print(f"[{total}/50] Response: {ans} | Goal: {label} | Match: {is_correct}")

        except Exception as e:
            print(f" Error on {filename}: {e}")
            continue

    if total > 0:
        print("\n" + "="*40)
        print(f"📊 FINAL ACCURACY: {(correct/total)*100:.2f}% ({correct}/{total})")
        print("="*40)

if __name__ == "__main__":
    run_experiment()
