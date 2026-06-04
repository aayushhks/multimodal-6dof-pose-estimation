"""
LoRA v6 — Key fixes over v5:
1. Dual-format training: original CoT conversations (teach reasoning) + clean labels (teach output format)
2. Paper's hyperparams: LR=4e-5, r=16, alpha=32, warmup=0.3, weight_decay=0.05
3. Use ALL data with oversample to median (not hard cap at 400)
4. Open question uses constrained choice list in BOTH train and eval
5. Object name extracted from JSONL and used in prompts (matches benchmark format)
"""
import sys, os, json, torch, random, re, math
from PIL import Image, ImageOps
from tqdm import tqdm
from transformers import AutoTokenizer
from peft import LoraConfig, get_peft_model
from collections import defaultdict

sys.path.insert(0, "/projectnb/cs598/aayushks/EgoOrientBench/model_train/InternVL_4B/internvl_chat")
try:
    from internvl.model.internvl_chat import InternVLChatModel
    from internvl.model.internvl_chat.modeling_intern_vit import InternVisionEncoder, InternVisionEncoderLayer
    from internvl.train.dataset import build_transform
except ImportError as e:
    print(f"Import Error: {e}"); sys.exit(1)

def f_init(self, cfg):
    torch.nn.Module.__init__(self)
    self.config = cfg
    dpr = [x.item() for x in torch.linspace(0, cfg.drop_path_rate, cfg.num_hidden_layers, device='cpu')]
    self.layers = torch.nn.ModuleList([InternVisionEncoderLayer(cfg, dpr[i]) for i in range(cfg.num_hidden_layers)])
    self.gradient_checkpointing = True
InternVisionEncoder.__init__ = f_init
InternVLChatModel.all_tied_weights_keys = property(lambda self: {})

# ─── LABEL MAPPINGS ───
MCQ_CHOICES = "A.front B.front right C.right D.back right E.back F.back left G.left H.front left"
COMPASS_TO_LETTER = {
    'front':'A','front right':'B','right':'C','back right':'D',
    'back':'E','back left':'F','left':'G','front left':'H',
}
COMPASS_TO_VERBOSE = {
    'front':'facing the camera',
    'back':'facing away the camera',
    'right':'facing right',
    'left':'facing left',
    'front right':'facing right while facing the camera',
    'front left':'facing left while facing the camera',
    'back right':'toward right while facing away the camera',
    'back left':'toward left while facing away the camera',
}
COMPASS_FLIP = {
    'front':'front','back':'back',
    'right':'left','left':'right',
    'front right':'front left','front left':'front right',
    'back right':'back left','back left':'back right',
}
ALL_COMPASS = list(COMPASS_TO_LETTER.keys())

OPEN_CHOICE_LIST = (
    "Choose the single best answer from: facing the camera, facing right, facing left, "
    "facing away the camera, facing right while facing the camera, facing left while facing the camera, "
    "toward right while facing away the camera, toward left while facing away the camera.\n"
    "Answer with the exact phrase only."
)

# ─── LABEL EXTRACTOR ───
def extract_compass(answer_text):
    text = answer_text.upper()
    if '(ANSWER THE ORIENTATION):' in text:
        text = text.split('(ANSWER THE ORIENTATION):')[1]
    first_clause = re.split(r'\b(?:SO (?:IT|YOU|THE)|THEREFORE|THUS)\b', text)[0]
    patterns = [
        (r'FACING THE CAMERA(?:/OBSERVER)?','front'),
        (r'FACING (?:DIRECTLY )?(?:AT )?THE CAMERA','front'),
        (r'FACING (?:DIRECTLY )?(?:TOWARDS? )?THE OBSERVER','front'),
        (r'FACING FORWARD','front'),(r'\bFRONTAL\b','front'),
        (r'FACING AWAY(?: FROM THE CAMERA)?','back'),
        (r'AWAY FROM THE CAMERA','back'),
        (r'FACING FRONT[- ]?RIGHT','front right'),(r'\bFRONT[- ]?RIGHT\b','front right'),
        (r'SLIGHTLY RIGHT WHILE FACING','front right'),
        (r'GAZING TOWARDS? (?:THE )?FRONT RIGHT','front right'),
        (r'FACING FRONT[- ]?LEFT','front left'),(r'\bFRONT[- ]?LEFT\b','front left'),
        (r'SLIGHTLY LEFT WHILE FACING','front left'),
        (r'FACING BACK[- ]?RIGHT','back right'),(r'\bBACK[- ]?RIGHT\b','back right'),
        (r'FACING BACK[- ]?LEFT','back left'),(r'\bBACK[- ]?LEFT\b','back left'),
        (r'FACING (?:SLIGHTLY )?(?:TO )?(?:THE )?RIGHT','right'),
        (r'TURNED TO THE RIGHT','right'),(r'GAZING TOWARDS? (?:THE )?RIGHT','right'),
        (r'FACING (?:SLIGHTLY )?(?:TO )?(?:THE )?LEFT','left'),
        (r'TURNED TO THE LEFT','left'),(r'GAZING TOWARDS? (?:THE )?LEFT','left'),
    ]
    for pat, compass in patterns:
        if re.search(pat, first_clause): return compass
    for pat, compass in patterns:
        if re.search(pat, text): return compass
    return None

# ─── OBJECT NAME EXTRACTION ───
def extract_object_name(conversation_text):
    """Extract object name from the JSONL conversation question."""
    text = conversation_text.lower()
    # Try to extract from "the {object}" or "is the {object} facing"
    patterns = [
        r'is the (.+?) facing',
        r'orientation (?:of|is) the (.+?)[\?\.]',
        r'the (.+?) (?:is facing|have to|in the)',
        r'describe.*?the (.+?)[\.\?]',
        r'features.*?(?:of|indicate).*?the (.+?)[\.\?]',
    ]
    for pat in patterns:
        m = re.search(pat, text)
        if m:
            name = m.group(1).strip()
            # Clean up common prefixes
            name = re.sub(r'^(common |a |an |the )', '', name)
            if 2 <= len(name) <= 40:
                return name
    return "object"

# ─── TRAINING EXAMPLE BUILDERS ───
def make_clean_label_examples(image_placeholder, compass, object_name, rng):
    """Generate clean-label training examples matching benchmark format."""
    verbose = COMPASS_TO_VERBOSE[compass]
    letter  = COMPASS_TO_LETTER[compass]
    examples = []

    # MCQ — matches benchmark Choose task format exactly
    examples.append((
        f"{image_placeholder}\n"
        f"From the perspective of the camera, which orientation is the {object_name} in the photo facing? "
        f"{MCQ_CHOICES}. "
        f"Answer with the option's letter and word from the given choices directly.",
        f"{letter} {compass}"
    ))

    # Open — matches benchmark Freeform task + constrained choices
    examples.append((
        f"{image_placeholder}\n"
        f"From the perspective of the camera, Answer what orientation the {object_name} in the picture is facing.\n"
        f"{OPEN_CHOICE_LIST}",
        verbose
    ))

    # YES/NO positive — matches benchmark Verify task format
    examples.append((
        f"{image_placeholder}\n"
        f"Is the {object_name} facing \"{compass}\" from the camera's perspective? "
        f"Answer with \"yes\" or \"no\" only.",
        "yes"
    ))

    # YES/NO negative
    wrong_compass = rng.choice([c for c in ALL_COMPASS if c != compass])
    examples.append((
        f"{image_placeholder}\n"
        f"Is the {object_name} facing \"{wrong_compass}\" from the camera's perspective? "
        f"Answer with \"yes\" or \"no\" only.",
        "no"
    ))

    return examples

def make_cot_example(image_placeholder, conversation):
    """Use the original CoT conversation from JSONL — teaches reasoning."""
    question = conversation[0]['value']
    answer = conversation[1]['value']
    # Replace any image tags in the question with our placeholder
    question = re.sub(r'<image>|<img>.*?</img>', '', question).strip()
    prompt = f"{image_placeholder}\n{question}"
    return prompt, answer

# ─── MODEL SETUP ───
def setup_model(method='lora'):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained("OpenGVLab/InternVL2-4B", trust_remote_code=True, use_fast=False)
    model = InternVLChatModel.from_pretrained(
        "OpenGVLab/InternVL2-4B", low_cpu_mem_usage=True, torch_dtype=torch.bfloat16
    ).to(device)

    img_ctx_id = tokenizer.convert_tokens_to_ids('<IMG_CONTEXT>')
    if img_ctx_id == tokenizer.unk_token_id:
        tokenizer.add_special_tokens({'additional_special_tokens': ['<IMG_CONTEXT>']})
        model.resize_token_embeddings(len(tokenizer))
        img_ctx_id = tokenizer.convert_tokens_to_ids('<IMG_CONTEXT>')
    model.img_context_token_id = img_ctx_id
    print(f"img_context_token_id: {img_ctx_id}")
    assert len(tokenizer.encode('<IMG_CONTEXT>', add_special_tokens=False)) == 1
    num_image_token = model.num_image_token
    print(f"num_image_token: {num_image_token}")

    model.vision_model.requires_grad_(False)

    if method == 'lora':
        # Paper's config for InternVL2: r=16, alpha=32
        config = LoraConfig(
            r=16, lora_alpha=32,
            target_modules=["q_proj","k_proj","v_proj","o_proj","down_proj","up_proj","gate_proj"],
            lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
        )
    elif method == 'dora':
        config = LoraConfig(
            r=16, lora_alpha=32,
            target_modules=["q_proj","k_proj","v_proj","o_proj","down_proj","up_proj","gate_proj"],
            lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
            use_dora=True,
        )
    else:
        raise ValueError(f"Unknown method: {method}")

    model.language_model = get_peft_model(model.language_model, config)
    model.language_model.print_trainable_parameters()
    return model, tokenizer, device, num_image_token

# ─── MAIN ───
def run_training():
    # ── CONFIG ──
    METHOD = 'lora'          # Change to 'dora' for DoRA
    EPOCHS = 3
    LR = 4e-5                # Paper's LR for InternVL2
    WEIGHT_DECAY = 0.05      # Paper's weight decay
    WARMUP_RATIO = 0.3       # Paper's warmup ratio
    SAVE_DIR = f"./egoorient_{METHOD}_weights_v6"

    rng = random.Random(42)
    model, tokenizer, device, num_image_token = setup_model(METHOD)

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LR, weight_decay=WEIGHT_DECAY
    )

    json_path = "/projectnb/cs598/data_storage/msi_datasets/EgoOrientBench/all_data/EgocentricDataset/train_data/imagenet_train_internvl.jsonl"
    root_img_dir = "/projectnb/cs598/data_storage/msi_datasets/EgoOrientBench/all_data/EgocentricDataset/images/"
    f_map = {f: os.path.join(r, f) for r, _, fs in os.walk(root_img_dir) for f in fs}

    # ── Load and parse training data ──
    print("Loading and parsing training data...")
    raw_data = []
    with open(json_path) as f:
        for line in f:
            if line.strip(): raw_data.append(json.loads(line))

    labeled = []  # (img_path, compass, is_flipped, object_name, conversations)
    skipped = 0
    for item in raw_data:
        nm = os.path.basename(item['image'])
        p = f_map.get(nm)
        if not p: skipped += 1; continue
        compass = extract_compass(item['conversations'][1]['value'])
        if compass is None: skipped += 1; continue

        # Extract object name from the conversation
        obj_name = extract_object_name(item['conversations'][0]['value'])

        # Original orientation
        labeled.append((p, compass, False, obj_name, item['conversations']))
        # Flipped orientation (swap left/right)
        flipped_compass = COMPASS_FLIP[compass]
        labeled.append((p, flipped_compass, True, obj_name, item['conversations']))

    dist = defaultdict(int)
    for _,c,_,_,_ in labeled: dist[c] += 1
    print(f"Parsed {len(labeled)} samples ({skipped} skipped)")
    print("Distribution:", dict(sorted(dist.items())))

    # ── Smart oversampling: bring all classes to MEDIAN count ──
    by_dir = defaultdict(list)
    for item in labeled: by_dir[item[1]].append(item)

    counts = sorted(len(v) for v in by_dir.values())
    target = counts[len(counts)//2]  # median
    target = max(target, 400)        # at least 400
    print(f"Oversampling target per direction: {target}")

    balanced = []
    for compass, items in by_dir.items():
        rng.shuffle(items)
        if len(items) >= target:
            balanced.extend(items[:target])
        else:
            oversampled = []
            while len(oversampled) < target:
                tmp = items.copy(); rng.shuffle(tmp); oversampled.extend(tmp)
            balanced.extend(oversampled[:target])

    dist2 = defaultdict(int)
    for _,c,_,_,_ in balanced: dist2[c] += 1
    print(f"Balanced: {len(balanced)} samples")
    print("Balanced dist:", dict(sorted(dist2.items())))

    # ── Build training examples ──
    image_placeholder = '<img>' + '<IMG_CONTEXT>' * num_image_token + '</img>'
    all_examples = []  # (img_path, is_flipped, prompt, answer)

    for img_path, compass, is_flipped, obj_name, conversations in balanced:
        # 1. CLEAN LABEL EXAMPLES (teaches output format) — 4 examples
        for prompt, answer in make_clean_label_examples(image_placeholder, compass, obj_name, rng):
            all_examples.append((img_path, is_flipped, prompt, answer))

        # 2. ORIGINAL CoT EXAMPLE (teaches reasoning) — 1 example
        # Only for non-flipped (CoT reasoning references specific visual features
        # that would be wrong if image is flipped)
        if not is_flipped:
            try:
                cot_prompt, cot_answer = make_cot_example(image_placeholder, conversations)
                all_examples.append((img_path, False, cot_prompt, cot_answer))
            except Exception:
                pass

    total_steps = len(all_examples) * EPOCHS
    warmup_steps = int(total_steps * WARMUP_RATIO)
    print(f"\nExamples/epoch: {len(all_examples)} | Total steps: {total_steps}")
    print(f"Warmup steps: {warmup_steps} | Method: {METHOD}")

    # ── Scheduler: linear warmup + cosine decay (paper's setup) ──
    from torch.optim.lr_scheduler import LambdaLR

    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = LambdaLR(optimizer, lr_lambda)

    # ── Training loop ──
    model.train()
    image_flags = torch.tensor([1], dtype=torch.long).to(device)
    global_step = 0
    running_loss = 0.0
    transform_train = build_transform(is_train=True, input_size=448)

    for epoch in range(EPOCHS):
        epoch_ex = all_examples.copy()
        rng.shuffle(epoch_ex)
        print(f"\n--- Epoch {epoch+1}/{EPOCHS} ---")

        for img_path, is_flipped, prompt, answer in tqdm(epoch_ex):
            try:
                img = Image.open(img_path).convert('RGB')
                if is_flipped:
                    img = ImageOps.mirror(img)
                px = transform_train(img).unsqueeze(0).to(torch.bfloat16).to(device)

                full_text = prompt + answer + tokenizer.eos_token
                tokenized = tokenizer(full_text, return_tensors="pt")
                input_ids = tokenized.input_ids.to(device)
                attention_mask = tokenized.attention_mask.to(device)
                labels = input_ids.clone()
                prompt_len = len(tokenizer(prompt, return_tensors="pt").input_ids[0])
                labels[0, :prompt_len] = -100

                optimizer.zero_grad()
                outputs = model(
                    pixel_values=px, input_ids=input_ids,
                    attention_mask=attention_mask, image_flags=image_flags, labels=labels
                )
                outputs.loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                scheduler.step()

                running_loss += outputs.loss.item()
                global_step += 1
                if global_step % 500 == 0:
                    cur_lr = scheduler.get_last_lr()[0]
                    print(f"Step {global_step}/{total_steps} | Avg Loss: {running_loss/500:.4f} | LR: {cur_lr:.2e}")
                    running_loss = 0.0

            except Exception as e:
                print(f"\n[!] Step {global_step}: {e}"); continue

    print(f"\nSaving {METHOD.upper()} weights to {SAVE_DIR}...")
    model.language_model.save_pretrained(SAVE_DIR, save_embedding_layers=False)
    print(f"Done. Saved to {SAVE_DIR}")

if __name__ == "__main__":
    run_training()