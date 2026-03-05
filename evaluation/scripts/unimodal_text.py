import sys
import os
import json
import torch
import torch.nn as nn
import csv
from tqdm import tqdm
from transformers import AutoTokenizer

# 1. PATH SETUP (Must be first to avoid ImportErrors)
PROJECT_ROOT = "/projectnb/cs598/aayushks/EgoOrientBench/model_train/InternVL_4B/internvl_chat"
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# 2. IMPORTS
try:
    from internvl.model.internvl_chat import InternVLChatModel
    from internvl.model.internvl_chat.modeling_intern_vit import InternVisionEncoder, InternVisionEncoderLayer
except ImportError as e:
    print(f"Import Error: {e}. Check if PROJECT_ROOT is correct.")
    sys.exit(1)

# 3. PATH DATA
JSON_PATH = "/projectnb/cs598/data_storage/msi_datasets/EgoOrientBench/all_data/EgocentricDataset/benchmark_data/benchmark.json"
OUTPUT_CSV = "unimodel-text.csv"

# 4. PATCHES
def fixed_vision_encoder_init(self, config):
    nn.Module.__init__(self) 
    self.config = config
    dpr = [x.item() for x in torch.linspace(0, config.drop_path_rate, config.num_hidden_layers, device='cpu')]
    self.layers = nn.ModuleList([InternVisionEncoderLayer(config, dpr[idx]) for idx in range(config.num_hidden_layers)])
    self.gradient_checkpointing = True

InternVisionEncoder.__init__ = fixed_vision_encoder_init
InternVLChatModel.all_tied_weights_keys = property(lambda self: {})

VALID_LABELS = [
    "FACING RIGHT WHILE FACING THE CAMERA", "FACING LEFT WHILE FACING THE CAMERA", 
    "FACING THE CAMERA", "FACING LEFT", "FACING RIGHT", "FRONT RIGHT", "FRONT LEFT", 
    "BACK RIGHT", "BACK LEFT", "FRONT", "BACK", "LEFT", "RIGHT", "YES", "NO"
]

# 5. EXECUTION
def run_experiment():
    device = torch.device("cuda:0")
    print(f"Loading TEXT-ONLY mode onto {device}...")
    
    tokenizer = AutoTokenizer.from_pretrained("OpenGVLab/InternVL2-4B", trust_remote_code=True, use_fast=False)
    model = InternVLChatModel.from_pretrained(
        "OpenGVLab/InternVL2-4B", 
        low_cpu_mem_usage=True, 
        torch_dtype=torch.bfloat16
    ).to(device).eval()
    
    model.config.use_cache = False

    if not os.path.exists(JSON_PATH):
        print(f"Error: JSON file not found at {JSON_PATH}")
        return

    with open(JSON_PATH, "r") as f:
        data = json.load(f)
    
    with open(OUTPUT_CSV, mode='w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['image', 'question', 'raw_response', 'prediction', 'ground_truth', 'correct'])
        writer.writeheader()
        correct, total = 0, 0
     
        for item in tqdm(data):
            orig_q = item['question']
            is_binary = "yes' or 'no'" in orig_q.lower() or "yes/no" in orig_q.lower()
            
            # Pure text prompt - NO <image> tag
            if is_binary:
                question = f"{orig_q}\nAnswer strictly with YES or NO. Do not add any other words."
            else:
                question = f"{orig_q}\nAnswer strictly with the exact category name from the choices provided. Do not use full sentences."
            
            try:
                # Setting pixel_values=None forces the model to ignore vision
                response = model.chat(
                    tokenizer=tokenizer,
                    pixel_values=None, 
                    question=question,
                    generation_config={"max_new_tokens": 20, "do_sample": False}
                )
                
                raw_res = response.strip().upper()
                gt = str(item['label']).strip().upper()
                
                pred = raw_res
                for label in VALID_LABELS:
                    if label in raw_res:
                        pred = label
                        break
                
                is_correct = (pred == gt)
                if is_correct: correct += 1
                total += 1
                
                writer.writerow({
                    'image': os.path.basename(item['image']), 
                    'question': orig_q,
                    'raw_response': raw_res, 
                    'prediction': pred,
                    'ground_truth': gt, 
                    'correct': str(is_correct)
                })

            except Exception as e:
                continue

    if total > 0:
        print(f"\nText-Only Accuracy: {(correct/total)*100:.2f}% ({correct}/{total})")

if __name__ == "__main__":
    run_experiment()