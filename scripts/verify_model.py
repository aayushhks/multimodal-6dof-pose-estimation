import os
import importlib
import torch
import numpy as np
from PIL import Image
import sys
import argparse
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.mllm_base import get_registered_models


def import_models_from_folder(folder_path="models"):
    """Dynamically imports all python files in the folder to trigger registration."""
    for file in os.listdir(folder_path):
        if file.endswith(".py") and file != "mllm_base.py" and not file.startswith("__"):
            # Ensure we use the correct module path relative to the script
            module_name = f"models.{file[:-3]}"
            try:
                importlib.import_module(module_name)
                # Only print loading info if not just listing
            except Exception as e:
                print(f"❌ Failed to load module {module_name}: {e}")


def run_verification(target_model=None, list_only=False):
    # 1. Discover models
    import_models_from_folder()
    all_models = get_registered_models()

    if not all_models:
        print("No models registered. Check your decorators and imports.")
        return

    # 2. Handle --list flag
    if list_only:
        print("\n📋 Available Registered Models:")
        for name in all_models.keys():
            print(f"  - {name}")
        return

    # 3. Filter models if flag is provided
    models_to_test = {}
    if target_model:
        if target_model in all_models:
            models_to_test = {target_model: all_models[target_model]}
        else:
            print(f"❌ Model '{target_model}' not found in registry.")
            print(f"Available models: {list(all_models.keys())}")
            return
    else:
        models_to_test = all_models

    print(f"\n--- Starting Verification for {len(models_to_test)} model(s) ---\n")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 4. Loop through selected models
    for name, model_class in models_to_test.items():
        print(f"Testing Model: [{name}]")
        try:
            print(f"⏳ Initializing...")
            instance = model_class(model_name=name, device=device)
            print(f"✅ Load Success")

            print(f"⏳ Running verify()...")
            instance.verify()

            print(f"  - - - - - - - - - - - - - - - - - - -")
        except Exception as e:
            print(f"  ❌ Model [{name}] failed: {e}")
            import traceback

            traceback.print_exc()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Verify registered MLLM models.")

    parser.add_argument(
        "--model",
        type=str,
        help="Specific model name to verify. If omitted, all models are verified.",
        default=None,
    )

    parser.add_argument("--list", action="store_true", help="List all registered model names and exit.")

    args = parser.parse_args()

    run_verification(target_model=args.model, list_only=args.list)
