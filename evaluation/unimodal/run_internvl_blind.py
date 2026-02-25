import sys
import os
import json
import torch
import torch.nn as nn
import csv
from PIL import Image
from tqdm import tqdm

# 1. SETUP
PROJECT_ROOT = "/projectnb/cs598/aayushks/EgoOrientBench/model_train/InternVL_4B/internvl_chat"
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

JSON_PATH = "/projectnb/cs598/data_storage/msi_datasets/EgoOrientBench/all_data/EgocentricDataset/benchmark_data/benchmark.json"
IMAGE_DIR = "/projectnb/cs598/data_storage/msi_datasets/EgoOrientBench/all_data/EgocentricDataset/images/imagenet_after/"
OUTPUT_CSV = "results_blind.csv"

# 2. IMPORTS
try:
    from internvl.model.internvl_chat import InternVLChatModel
    from internvl.model.internvl_chat.modeling_intern_vit import InternVisionEncoder, InternVisionEncoderLayer
    from transformers import AutoTokenizer
    from internvl.train.dataset import build_transform
except ImportError as e:
    print(f"Import Error: {e}")
    sys.exit(1)

# 3. PATCHES
def fixed_vision_encoder_init(self, config):
    nn.Module.__init__(self) 
    self.config = config
    dpr = [x.item() for x in torch.linspace(0, config.drop_path_rate, config.num_hidden_layers, device='cpu')]
    self.layers = nn.ModuleList([InternVisionEncoderLayer(config, dpr[idx]) for idx in range(config.num_hidden_layers)])
    self.gradient_checkpointing = True

InternVisionEncoder.__init__ = fixed_vision_encoder_init
InternVLChatModel.all_tied_weights_keys = property(lambda self: {})

# 4. SMART LABELS (Ordered longest to shortest to prevent partial matches)
VALID_LABELS = [
    "FACING RIGHT WHILE FACING THE CAMERA",
    "FACING LEFT WHILE FACING THE CAMERA",
    "FACING THE CAMERA",
    "FACING LEFT",
    "FACING RIGHT",
    "FRONT RIGHT",
    "FRONT LEFT",
    "BACK RIGHT",
    "BACK LEFT",
    "FRONT",
    "BACK",
    "LEFT",
    "RIGHT",
    "YES",
    "NO"
]

# 5. EXECUTION
def run_experiment():
    device = torch.device("cuda:0")
    print(f"Loading model onto {device} (Smart Parser Mode)...")
    
    tokenizer = AutoTokenizer.from_pretrained("OpenGVLab/InternVL2-4B", trust_remote_code=True, use_fast=False)
    
    model = InternVLChatModel.from_pretrained(
        "OpenGVLab/InternVL2-4B", 
        low_cpu_mem_usage=True, 
        torch_dtype=torch.bfloat16
    ).to(device).eval()
    
    model.config.use_cache = False

    with open(JSON_PATH, "r") as f:
        data = json.load(f)
    
    print(f"Processing {len(data)} images with Split Prompting...")
    
    fieldnames = ['image', 'question', 'raw_response', 'prediction', 'ground_truth', 'correct']
    
    with open(OUTPUT_CSV, mode='w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        correct = 0
        total = 0
     
        for i, item in enumerate(tqdm(data[:600])):
            filename = os.path.basename(item['image'])
            img_path = os.path.join(IMAGE_DIR, filename)

            if not os.path.exists(img_path):
                if not os.path.exists(os.path.join(IMAGE_DIR, os.path.basename(filename))):
                    continue

            try:
                transform = build_transform(is_train=False, input_size=448)
                pixel_values = transform(Image.open(img_path).convert('RGB')).unsqueeze(0).to(torch.bfloat16).to(device)
                
                # --- SPLIT PROMPTING LOGIC ---
                orig_q = item['question']
                is_binary = "yes' or 'no'" in orig_q.lower() or "yes/no" in orig_q.lower()
                
                if is_binary:
                    # Force a strict Yes/No answer
                    question = f"<image>\n{orig_q}\nAnswer strictly with YES or NO. Do not add any other words."
                else:
                    # Let the model use the A-H choices already in the question
                    question = f"<image>\n{orig_q}\nAnswer strictly with the exact category name from the choices provided. Do not use full sentences."
                
                response = model.chat(
                    tokenizer=tokenizer,
                    pixel_values=pixel_values,
                    question=question,
                    generation_config={"max_new_tokens": 20, "do_sample": False}
                )
                
                # Cleaning
                raw_res = response.strip().upper()
                gt = str(item['label']).strip().upper()
                
                # --- SMART EXTRACTION ---
                pred = raw_res
                for label in VALID_LABELS:
                    if label in raw_res:
                        pred = label
                        break # Stops at the longest match!
                
                is_correct = (pred == gt)
                
                if is_correct: correct += 1
                total += 1
                
                writer.writerow({
                    'image': filename,
                    'question': orig_q,
                    'raw_response': raw_res,
                    'prediction': pred,
                    'ground_truth': gt,
                    'correct': str(is_correct)
                })

            except Exception as e:
                continue

    print(f"\nFINAL RESULTS SAVED TO {OUTPUT_CSV}")
    if total > 0:
        print(f"Accuracy: {(correct/total)*100:.2f}% ({correct}/{total})")

if __name__ == "__main__":
    run_experiment()
