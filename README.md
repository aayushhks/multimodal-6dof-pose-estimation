# Multimodal 6DoF Orientation Reasoning with PEFT

This repository studies orientation reasoning in multimodal vision-language models using parameter-efficient fine-tuning. The focus is EgoOrientBench, a benchmark for evaluating whether a model can reason about object orientation from the camera's perspective.

The main comparison is between LoRA, DoRA, and AdaLoRA on multimodal orientation tasks. The goal is to measure not only overall accuracy, but also whether the model handles different question formats and orientation classes reliably.

## Summary

Parameter-efficient fine-tuning improves orientation reasoning over zero-shot prompting. Among the PEFT methods tested, AdaLoRA gives the strongest result on EgoOrientBench, reaching 49.5% overall accuracy.

The main finding is that adaptive rank allocation can improve orientation reasoning, but the benefit depends on the dataset and the pruning schedule. AdaLoRA performs best on DORI and EgoOrientBench, while LoRA and DoRA remain stronger on Spatial457.

## Results

| Method | DORI | EgoOrientBench | Spatial457 |
|---|---:|---:|---:|
| Random | 27.2 | 22.9 | 26.0 |
| Zero-shot text | 34.7 | 29.7 | 21.9 |
| Zero-shot multimodal | 37.4 | 35.1 | 32.7 |
| LoRA | 53.7 | 47.6 | 50.8 |
| DoRA | 52.4 | 47.2 | 50.8 |
| AdaLoRA | 55.1 | 49.5 | 42.7 |

AdaLoRA is the best method on DORI and EgoOrientBench. On EgoOrientBench, it improves over LoRA by 1.9 points and over DoRA by 2.3 points.

## EgoOrientBench Breakdown

| Method | Overall | Choose | Verify | Freeform |
|---|---:|---:|---:|---:|
| LoRA | 47.6 | 32.0 | 63.1 | 31.9 |
| DoRA | 47.2 | 31.8 | 62.8 | 31.2 |
| AdaLoRA | 49.5 | 34.3 | 66.6 | 30.4 |

AdaLoRA gives the best overall performance on EgoOrientBench. It is strongest on the Choose and Verify splits, which test multiple-choice orientation selection and yes/no orientation verification.

## Method

The fine-tuning setup uses InternVL2-4B as the base multimodal model. The vision encoder is frozen, and PEFT adapters are applied to the language model.

The training pipeline uses a mixture of:

- clean-label examples for direct answer formatting
- original chain-of-thought examples for preserving reasoning behavior
- median-based class oversampling
- horizontal flip augmentation with corrected left/right orientation labels
- deterministic evaluation parsing for reproducible scoring

The LoRA and DoRA runs use rank 16 and alpha 32. AdaLoRA starts with a higher rank budget and dynamically reallocates rank during training.

## Finding 1: PEFT Improves Orientation Reasoning

Zero-shot multimodal prompting reaches 35.1% on EgoOrientBench. Fine-tuning with PEFT improves this substantially:

| Method | EgoOrientBench |
|---|---:|
| Zero-shot multimodal | 35.1 |
| LoRA | 47.6 |
| DoRA | 47.2 |
| AdaLoRA | 49.5 |

This shows that orientation reasoning benefits from task-specific adaptation, even when only a small number of trainable parameters are introduced.

## Finding 2: AdaLoRA Helps on EgoOrientBench

AdaLoRA performs best on EgoOrientBench. Its improvement is strongest on structured question formats:

| Split | LoRA | AdaLoRA | Gain |
|---|---:|---:|---:|
| Choose | 32.0 | 34.3 | +2.3 |
| Verify | 63.1 | 66.6 | +3.5 |
| Overall | 47.6 | 49.5 | +1.9 |

This suggests that adaptive rank allocation helps the model better represent the orientation decision boundaries needed for camera-perspective reasoning.

## Finding 3: Average Accuracy Hides Class Collapse

The LoRA model reaches 47.6% overall on EgoOrientBench, but its multiple-choice predictions are highly concentrated in a small subset of orientation classes.

This means the model can achieve a reasonable average score while still failing difficult orientation categories such as left-facing and diagonal directions. For orientation reasoning, class-level analysis is necessary because overall accuracy alone can hide systematic spatial failures.

## Finding 4: Adapter Behavior Differs Across Datasets

Layer-wise adapter analysis shows that LoRA, DoRA, and AdaLoRA use model capacity differently.

LoRA and DoRA keep a fixed rank across targeted layers. AdaLoRA dynamically reallocates rank based on layer importance. This helps on DORI and EgoOrientBench, but hurts on Spatial457, where AdaLoRA drops below LoRA and DoRA.

This suggests that adaptive rank allocation is useful, but only when the pruning schedule matches the dataset and training budget.

## Reproducibility

Train LoRA:

```bash
python eval_finetuning/ego/train_lora.py
```

Train DoRA:

```bash
python eval_finetuning/ego/train_dora.py
```

Train AdaLoRA:

```bash
python eval_finetuning/ego/train_adalora.py
```

Evaluate LoRA:

```bash
python eval_finetuning/ego/evaluate.py --method lora --weights ./egoorient_lora_weights_v6
```

Evaluate DoRA:

```bash
python eval_finetuning/ego/evaluate.py --method dora --weights ./egoorient_dora_weights_v6
```

Evaluate AdaLoRA:

```bash
python eval_finetuning/ego/evaluate.py --method adalora --weights ./egoorient_adalora_weights_v1
```

Verify dataset files:

```bash
python scripts/verify_dataset.py
```

Verify model setup:

```bash
python scripts/verify_model.py
```

## Takeaway

PEFT is effective for multimodal orientation reasoning, but the choice of adapter matters. AdaLoRA gives the best EgoOrientBench result in this setup, while the class-level analysis shows that orientation benchmarks require more than overall accuracy: models must be evaluated by question type, class distribution, and layer-level adaptation behavior.
