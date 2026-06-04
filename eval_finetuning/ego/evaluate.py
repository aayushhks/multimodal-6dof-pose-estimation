
"""
Unified evaluation script for LoRA v6 / DoRA v2 / SFT v2.
Usage:
  python evaluate_unified.py --method lora --weights ./egoorient_lora_weights_v6
  python evaluate_unified.py --method dora --weights ./egoorient_dora_weights_v6
  python evaluate_unified.py --method sft  --weights ./egoorient_sft_weights_v2
"""
import sys, os, json, torch, csv, re, argparse
from PIL import Image
from tqdm import tqdm
from transformers import AutoTokenizer

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

# ─── LABELS ───
VERBOSE_LABELS = [
    'TOWARD RIGHT WHILE FACING AWAY THE CAMERA','TOWARD LEFT WHILE FACING AWAY THE CAMERA',
    'FACING RIGHT WHILE FACING THE CAMERA','FACING LEFT WHILE FACING THE CAMERA',
    'FACING AWAY THE CAMERA','FACING THE CAMERA','FACING RIGHT','FACING LEFT',
]
VERBOSE_LABELS.sort(key=len, reverse=True)
COMPASS_LABELS = ['FRONT RIGHT','BACK RIGHT','FRONT LEFT','BACK LEFT','FRONT','RIGHT','LEFT','BACK']
COMPASS_LABELS.sort(key=len, reverse=True)
LETTER_TO_COMPASS = {
    'A':'FRONT','B':'FRONT RIGHT','C':'RIGHT','D':'BACK RIGHT',
    'E':'BACK','F':'BACK LEFT','G':'LEFT','H':'FRONT LEFT',
}
FORWARD_SYNS = re.compile(r'\bFACING FORWARD\b|\bFACING FRONT\b|\bFRONTAL\b|\bFACING DIRECTLY\b|\bFACING STRAIGHT\b')
AWAY_SYNS    = re.compile(r'\bFACING AWAY\b|\bAWAY FROM THE CAMERA\b|\bFACING BACKWARD\b|\bFACING BACKWARDS\b')

def get_qtype(question):
    q = question.lower()
    if re.search(r'\b[a-h]\.', q): return 'mcq'
    if "yes or no" in q or "answer with 'yes' or 'no'" in q or \
       'answer with "yes" or "no"' in q or re.match(r'^is ', q.strip()): return 'yesno'
    return 'open'

def parse_mcq_choices(question):
    matches = re.findall(r'\b([A-Ha-h])\.([a-zA-Z ]+?)(?=\s+[A-Ha-h]\.|$|\n)', question, re.IGNORECASE)
    return {m[0].upper(): m[1].strip().upper() for m in matches}

def parse_mcq(question, response):
    resp_up = response.upper().strip()
    choice_map = parse_mcq_choices(question)
    ans = resp_up.split("FINAL ANSWER:")[-1].strip() if "FINAL ANSWER:" in resp_up else resp_up

    # "G left" format
    lw = re.match(r'^([A-H])\s+([A-Z ]+)', re.sub(r'[^A-Z ]',' ',ans).strip())
    if lw:
        L = lw.group(1)
        if L in choice_map: return choice_map[L]
        if L in LETTER_TO_COMPASS: return LETTER_TO_COMPASS[L]

    # Bare letter
    bL = re.match(r'^([A-H])\b', re.sub(r'[^A-Z ]',' ',ans).strip())
    if bL:
        L = bL.group(1)
        if L in choice_map: return choice_map[L]
        if L in LETTER_TO_COMPASS: return LETTER_TO_COMPASS[L]

    # Choice text verbatim
    for ltr,text in sorted(choice_map.items(), key=lambda x:-len(x[1])):
        if re.search(r'\b'+re.escape(text)+r'\b', ans): return text

    # Compass label
    for label in COMPASS_LABELS:
        if re.search(r'\b'+label+r'\b', ans): return label

    # Full response scan
    for L in 'ABCDEFGH':
        if re.search(r'\b'+L+r'\b', resp_up):
            if L in choice_map: return choice_map[L]
            if L in LETTER_TO_COMPASS: return LETTER_TO_COMPASS[L]
    for label in COMPASS_LABELS:
        if re.search(r'\b'+label+r'\b', resp_up): return label
    return ans[:50]


def parse_open(response):
    resp_up = response.upper().strip()
    ans = resp_up.split("FINAL ANSWER:")[-1].strip() if "FINAL ANSWER:" in resp_up else resp_up
    has_while = 'WHILE' in ans

    for label in VERBOSE_LABELS:
        if label.startswith('TOWARD') and label in ans: return label

    for label in VERBOSE_LABELS:
        if label in ans:
            if 'WHILE' in label and not has_while: continue
            return label

    if FORWARD_SYNS.search(ans): return 'FACING THE CAMERA'
    if AWAY_SYNS.search(ans):    return 'FACING AWAY THE CAMERA'

    for label in VERBOSE_LABELS:
        if 'WHILE' not in label and label in ans: return label

    sentence_patterns = [
        (r'TOWARD(?:S)? (?:THE )?RIGHT WHILE FACING AWAY', 'TOWARD RIGHT WHILE FACING AWAY THE CAMERA'),
        (r'TOWARD(?:S)? (?:THE )?LEFT WHILE FACING AWAY',  'TOWARD LEFT WHILE FACING AWAY THE CAMERA'),
        (r'FACING (?:TO (?:THE )?)?RIGHT WHILE FACING',    'FACING RIGHT WHILE FACING THE CAMERA'),
        (r'FACING (?:TO (?:THE )?)?LEFT WHILE FACING',     'FACING LEFT WHILE FACING THE CAMERA'),
        (r'FACING (?:DIRECTLY )?(?:THE )?CAMERA',          'FACING THE CAMERA'),
        (r'FACING AWAY',                                    'FACING AWAY THE CAMERA'),
        (r'FACING (?:TO (?:THE )?)?RIGHT',                 'FACING RIGHT'),
        (r'FACING (?:TO (?:THE )?)?LEFT',                  'FACING LEFT'),
        (r'\bTO THE RIGHT\b',                              'FACING RIGHT'),
        (r'\bTO THE LEFT\b',                               'FACING LEFT'),
        (r'\bFACING LEFT SIDE\b',                          'FACING LEFT'),
        (r'\bFACING RIGHT SIDE\b',                         'FACING RIGHT'),
    ]
    for pat, label in sentence_patterns:
        if re.search(pat, ans): return label

    if FORWARD_SYNS.search(resp_up): return 'FACING THE CAMERA'
    if AWAY_SYNS.search(resp_up):    return 'FACING AWAY THE CAMERA'
    for label in VERBOSE_LABELS:
        if label in resp_up:
            if 'WHILE' in label and 'WHILE' not in resp_up: continue
            return label
    for pat, label in sentence_patterns:
        if re.search(pat, resp_up): return label
    return ans[:50]


def parse_yesno(response):
    resp_up = response.upper().strip()
    ans = resp_up.split("FINAL ANSWER:")[-1].strip() if "FINAL ANSWER:" in resp_up else resp_up
    if ans.startswith('YES'): return 'YES'
    if ans.startswith('NO'):  return 'NO'
    if 'YES' in ans: return 'YES'
    if 'NO'  in ans: return 'NO'
    return ans[:10]


# ─── PROMPT ───
OPEN_CHOICE_LIST = (
    "Choose the single best answer from: facing the camera, facing right, facing left, "
    "facing away the camera, facing right while facing the camera, facing left while facing the camera, "
    "toward right while facing away the camera, toward left while facing away the camera.\n"
    "Answer with the exact phrase only."
)

def build_prompt(question, qtype):
    if qtype == 'open':
        return f"<image>\n{question}\n{OPEN_CHOICE_LIST}"
    return f"<image>\n{question}"


# ─── MAIN ───
def run(args):
    device = torch.device("cuda:0")
    tok = AutoTokenizer.from_pretrained("OpenGVLab/InternVL2-4B", trust_remote_code=True, use_fast=False)

    print("Loading Base Model...")
    base_model = InternVLChatModel.from_pretrained(
        "OpenGVLab/InternVL2-4B", low_cpu_mem_usage=True, torch_dtype=torch.bfloat16
    ).to(device)
    img_ctx_id = tok.convert_tokens_to_ids('<IMG_CONTEXT>')
    base_model.img_context_token_id = img_ctx_id
    print(f"img_context_token_id: {img_ctx_id}")

    if args.method in ('lora', 'dora', 'adalora'):
        from peft import PeftModel
        print(f"Loading {args.method.upper()} weights from {args.weights}...")
        base_model.language_model = PeftModel.from_pretrained(
            base_model.language_model, args.weights
        )
    elif args.method == 'sft':
        print(f"Loading SFT weights from {args.weights}...")
        sft_weights = torch.load(
            os.path.join(args.weights, "language_model.pt"), map_location=device
        )
        base_model.language_model.load_state_dict(sft_weights)
        del sft_weights
        torch.cuda.empty_cache()
    else:
        print("Running zero-shot (no fine-tuning weights loaded)")

    model = base_model
    model.eval()
    print("Model ready.")

    with open("/projectnb/cs598/data_storage/msi_datasets/EgoOrientBench/all_data/EgocentricDataset/benchmark_data/benchmark.json") as f:
        data = json.load(f)

    root = "/projectnb/cs598/data_storage/msi_datasets/EgoOrientBench/all_data/EgocentricDataset/images/"
    f_map = {f: os.path.join(r, f) for r, _, fs in os.walk(root) for f in fs}

    output_csv = f"{args.method}-v6-finetuned-33k.csv"
    type_counts = {'mcq':[0,0],'yesno':[0,0],'open':[0,0]}

    with open(output_csv,'w',newline='') as f:
        w = csv.DictWriter(f, fieldnames=['image','qtype','raw_response','prediction','ground_truth','correct'])
        w.writeheader()
        c, t = 0, 0

        for item in tqdm(data):
            nm = os.path.basename(item['image'])
            p = f_map.get(nm)
            if not p: continue
            try:
                px = (build_transform(is_train=False, input_size=448)(Image.open(p).convert('RGB'))
                      .unsqueeze(0).to(torch.bfloat16).to(device))
                qtype = get_qtype(item['question'])
                q = build_prompt(item['question'], qtype)

                if qtype == 'mcq':
                    gen_cfg = {"max_new_tokens":10, "do_sample":False}
                elif qtype == 'yesno':
                    gen_cfg = {"max_new_tokens":5,  "do_sample":False}
                else:
                    gen_cfg = {"max_new_tokens":25, "do_sample":False}

                res = model.chat(tokenizer=tok, pixel_values=px, question=q,
                                 generation_config=gen_cfg).strip()
                gt = str(item['label']).strip().upper()
                if qtype=='mcq':     pr = parse_mcq(item['question'], res)
                elif qtype=='yesno': pr = parse_yesno(res)
                else:                pr = parse_open(res)

                is_correct = (pr == gt)
                w.writerow({'image':nm,'qtype':qtype,'raw_response':res[:120],
                            'prediction':pr,'ground_truth':gt,'correct':str(is_correct)})
                if is_correct:
                    c += 1; type_counts[qtype][0] += 1
                type_counts[qtype][1] += 1
                t += 1

            except Exception as e:
                print(f"\n[!] {nm}: {e}"); continue

    if t == 0:
        print("\n[ERROR] No samples evaluated! All failed.")
        return

    print(f"\nFinal Accuracy: {c/t*100:.2f}% ({c}/{t})")
    for qt,(cr,tot) in type_counts.items():
        if tot: print(f"  {qt}: {cr/tot*100:.1f}% ({cr}/{tot})")
    print(f"Saved to {output_csv}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--method', choices=['lora','dora','adalora','sft','zeroshot'], default='lora')
    parser.add_argument('--weights', type=str, default='./egoorient_lora_weights_v6')
    args = parser.parse_args()
    run(args)