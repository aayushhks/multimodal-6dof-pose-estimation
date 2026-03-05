import sys, os, json, torch, csv, re
from PIL import Image
from tqdm import tqdm
from transformers import AutoTokenizer

# 1. PATH SETUP

sys.path.insert(0, "/projectnb/cs598/aayushks/EgoOrientBench/model_train/InternVL_4B/internvl_chat")

try:
    from internvl.model.internvl_chat import InternVLChatModel
    from internvl.model.internvl_chat.modeling_intern_vit import InternVisionEncoder, InternVisionEncoderLayer
    from internvl.train.dataset import build_transform

except ImportError as e:
    print(f"Import Error: {e}")
    sys.exit(1)

# 2. PATCHES

def f_init(self, cfg):
    torch.nn.Module.__init__(self) 
    self.config = cfg
    dpr = [x.item() for x in torch.linspace(0, cfg.drop_path_rate, cfg.num_hidden_layers, device='cpu')]
    self.layers = torch.nn.ModuleList([InternVisionEncoderLayer(cfg, dpr[i]) for i in range(cfg.num_hidden_layers)])
    self.gradient_checkpointing = True

InternVisionEncoder.__init__ = f_init
InternVLChatModel.all_tied_weights_keys = property(lambda self: {})


# 3. GLOBAL LABELS (Sorted longest to shortest so "FRONT RIGHT" matches before "RIGHT")

L = ['TOWARD RIGHT WHILE FACING AWAY THE CAMERA', 'TOWARD LEFT WHILE FACING AWAY THE CAMERA', 
     'FACING RIGHT WHILE FACING THE CAMERA', 'FACING LEFT WHILE FACING THE CAMERA', 
     'FACING AWAY THE CAMERA', 'FACING THE CAMERA', 'FACING RIGHT', 'FACING LEFT', 
     'FRONT RIGHT', 'BACK RIGHT', 'FRONT LEFT', 'BACK LEFT', 'RIGHT', 'FRONT', 'LEFT', 'BACK', 'YES', 'NO']

L.sort(key=len, reverse=True)

# 4. BULLETPROOF PARSER

def parse_prediction(question, response):
    response = response.upper()
    question = question.upper()

    # Map the choices from the question, e.g., (A) FRONT (B) LEFT -> {'A': 'FRONT', 'B': 'LEFT'}

    matches = re.findall(r'\(([A-Z])\)\s*(.*?)(?=\s*\([A-Z]\)|$)', question)
    choice_map = {m[0]: m[1].strip() for m in matches}

    # Isolate the final answer if the model followed instructions

    ans_part = response.split("FINAL ANSWER:")[-1].strip() if "FINAL ANSWER:" in response else response

    # Safety 1: Did it just output a single mapped letter? (e.g., "A" or "C")

    clean_ans = re.sub(r'[^A-Z]', '', ans_part)
    if len(clean_ans) == 1 and clean_ans in choice_map:
        return choice_map[clean_ans]

    # Safety 2: Search for mapped choice texts in the answer block

    mapped_texts = sorted(list(choice_map.values()), key=len, reverse=True)
    for text in mapped_texts:
        if text in ans_part: return text

    # Safety 3: Search for global labels in the answer block

    for label in L:
        if label in ans_part: return label

    # Safety 4: Desperation search through its entire descriptive paragraph

    for text in mapped_texts:
        if text in response: return text

    for label in L:
        if label in response: return label

    return ans_part # Absolute fallback

# 5. MAIN RUN

def run():
    device = torch.device("cuda:0")
    tok = AutoTokenizer.from_pretrained("OpenGVLab/InternVL2-4B", trust_remote_code=True, use_fast=False)
    model = InternVLChatModel.from_pretrained("OpenGVLab/InternVL2-4B", low_cpu_mem_usage=True, torch_dtype=torch.bfloat16).to(device).eval()

    with open("/projectnb/cs598/data_storage/msi_datasets/EgoOrientBench/all_data/EgocentricDataset/benchmark_data/benchmark.json") as f:
        data = json.load(f)[:12000]

    root = "/projectnb/cs598/data_storage/msi_datasets/EgoOrientBench/all_data/EgocentricDataset/images/"
    f_map = {f: os.path.join(r, f) for r, _, fs in os.walk(root) for f in fs}

    with open("multimodal-text-image.csv", 'w', newline='') as f:

        w = csv.DictWriter(f, fieldnames=['image', 'prediction', 'ground_truth', 'correct'])
        w.writeheader()
        c, t = 0, 0

        for i in tqdm(data):
            nm = os.path.basename(i['image'])
            p = f_map.get(nm)
            if not p: continue
            try:
                px = build_transform(is_train=False, input_size=448)(Image.open(p).convert('RGB')).unsqueeze(0).to(torch.bfloat16).to(device)

                # The Golden Prompt: Let it describe, then force a final answer
                q = f"<image>\nQuestion: {i['question']}\nAnalyze the image descriptively to determine the orientation. End your response with 'FINAL ANSWER: ' followed by the exact option."

                # Upped tokens to 80 so it has room to be descriptive!
                res = model.chat(tokenizer=tok, pixel_values=px, question=q, generation_config={"max_new_tokens": 80, "do_sample": False}).strip()

                gt = str(i['label']).strip().upper()
                pr = parse_prediction(i['question'], res)
                is_correct = (pr == gt)
                w.writerow({'image': nm, 'prediction': pr, 'ground_truth': gt, 'correct': str(is_correct)})

                if is_correct: c += 1
                t += 1

            except Exception as e: 
                continue

    print(f"\nFinal Extracted Accuracy: {c/t*100:.2f}%")

if __name__ == "__main__": run()