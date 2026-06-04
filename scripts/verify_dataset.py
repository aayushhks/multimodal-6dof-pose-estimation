#!/usr/bin/env python3
"""
Dataset Verification Script

This script validates that student-implemented datasets conform to the MSI dataset standard.
It checks:
1. Proper use of StandardDataset class
2. Presence of required fields based on modality and task type
3. Data type consistency across samples
4. Proper registration in the dataset registry
5. Sample accessibility and validity

Usage:
    python scripts/verify_dataset.py --dataset sat
    python scripts/verify_dataset.py --dataset sat --verbose
    python scripts/verify_dataset.py --all  # Check all registered datasets
"""

import argparse
import sys
from typing import Dict, Any, List, Tuple, Optional
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from msi_datasets.utils import load_msi_dataset, list_registered_datasets, StandardDataset


class DatasetValidator:
    """Validates datasets against the MSI standard."""

    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self.errors = []
        self.warnings = []
        self.info = []

    def log_error(self, msg: str):
        """Log an error message."""
        self.errors.append(msg)
        if self.verbose:
            print(f"❌ ERROR: {msg}")

    def log_warning(self, msg: str):
        """Log a warning message."""
        self.warnings.append(msg)
        if self.verbose:
            print(f"⚠️  WARNING: {msg}")

    def log_info(self, msg: str):
        """Log an info message."""
        self.info.append(msg)
        if self.verbose:
            print(f"ℹ️  INFO: {msg}")

    def validate_dataset(self, dataset_name: str, config: Optional[Dict[str, Any]] = None) -> Tuple[bool, Dict[str, Any]]:
        """
        Validate a dataset comprehensively.

        Args:
            dataset_name: Name of the dataset to validate
            config: Optional configuration for the dataset

        Returns:
            Tuple of (is_valid, report_dict)
        """
        self.errors = []
        self.warnings = []
        self.info = []

        report = {
            "dataset": dataset_name,
            "valid": False,
            "errors": [],
            "warnings": [],
            "info": {},
        }

        # Step 1: Check registration
        if not self._check_registration(dataset_name):
            self.log_error(f"Dataset '{dataset_name}' is not registered!")
            report["errors"] = self.errors
            return False, report

        # Step 2: Load dataset
        try:
            config = config or {}
            dataset = load_msi_dataset(dataset_name, config)
            self.log_info(f"Successfully loaded dataset '{dataset_name}'")
        except Exception as e:
            self.log_error(f"Failed to load dataset: {str(e)}")
            report["errors"] = self.errors
            return False, report

        # Step 3: Check StandardDataset inheritance
        if not self._check_inheritance(dataset):
            report["errors"] = self.errors
            return False, report

        # Step 4: Check dataset structure
        self._check_dataset_structure(dataset)

        # Step 5: Check sample validity
        self._check_samples(dataset)

        # Step 6: Check modality-specific requirements
        self._check_modality_requirements(dataset)

        # Step 7: print out 1 data point so students can see what is inside
        try:
            sample_raw = dataset.get_one_sample_structure(return_raw_data=True)
            sample_structure = dataset.get_one_sample_structure(return_raw_data=False)
            ## pretty print the structure
            import json

            # Convert type names to strings for pretty printing
            readable_structure = {k: (v if isinstance(v, dict) else str(v)) for k, v in sample_structure.items()}

            print(f"\n---------- One sample from {dataset.name} ({dataset.split} split) ----------")
            print(f"Structure of the sample:")
            print(json.dumps(readable_structure, indent=4))
            print("-----------------------")
            ## print raw data
            print(f"Raw data of the sample:")
            print(json.dumps(sample_raw, indent=4, default=str))
            print("-" * 50)

        except Exception as e:
            self.log_warning(f"Could not retrieve one sample structure: {str(e)}")

        # Compile report
        report["errors"] = self.errors
        report["warnings"] = self.warnings
        report["info"] = dataset.info()

        report["valid"] = len(self.errors) == 0

        return report["valid"], report

    def _check_registration(self, dataset_name: str) -> bool:
        """Check if dataset is properly registered."""
        registered = list_registered_datasets()
        if dataset_name not in registered:
            self.log_error(f"Dataset not found in registry. Available: {list(registered.keys())}")
            return False
        self.log_info(f"Dataset is properly registered")
        return True

    def _check_inheritance(self, dataset: Any) -> bool:
        """Check if dataset inherits from StandardDataset."""
        if not isinstance(dataset, StandardDataset):
            self.log_error(f"Dataset must inherit from StandardDataset. Got: {type(dataset).__name__}")
            return False
        self.log_info("Dataset properly inherits from StandardDataset")
        return True

    def _check_dataset_structure(self, dataset: StandardDataset):
        """Check required dataset attributes."""
        # Check required attributes
        required_attrs = ["name", "modalities", "task_type", "split"]
        for attr in required_attrs:
            if not hasattr(dataset, attr):
                self.log_error(f"Missing required attribute: {attr}")
            else:
                self.log_info(f"✓ Has attribute '{attr}': {getattr(dataset, attr)}")

        # Check types
        if not isinstance(dataset.modalities, (list, tuple, set)):
            self.log_error(f"'modalities' must be a list. Got: {type(dataset.modalities)}")

        if not isinstance(dataset.task_type, str):
            self.log_error(f"'task_type' must be a string. Got: {type(dataset.task_type)}")

        if dataset.split not in ["train", "val", "test", "validation", "validation_train"]:
            self.log_warning(f"Unexpected split value: {dataset.split}")

    def _check_samples(self, dataset: StandardDataset):
        """Check individual samples for validity."""
        if len(dataset) == 0:
            self.log_error("Dataset is empty!")
            return

        self.log_info(f"Checking {min(5, len(dataset))} sample(s) for validity...")

        # Check first 5 samples
        for idx in range(min(5, len(dataset))):
            try:
                sample = dataset[idx]
                if not isinstance(sample, dict):
                    self.log_error(f"Sample {idx}: must be a dict. Got: {type(sample)}")
                    continue

                # Check required fields
                required_fields = ["question", "ground_truth"]
                for field in required_fields:
                    if field not in sample:
                        self.log_warning(f"Sample {idx}: missing '{field}' field")

                self.log_info(f"Sample {idx}: ✓ Valid structure with fields: {list(sample.keys())}")

            except Exception as e:
                self.log_error(f"Sample {idx}: {str(e)}")

    def _check_modality_requirements(self, dataset: StandardDataset):
        """Check that required fields exist for each modality."""
        modality_fields = {
            "image": ["image"],
            "video": ["video"],
            "text": ["text"],
            "point_cloud": ["point_cloud"],
            "audio": ["audio"],
        }

        self.log_info(f"Checking modality requirements for: {dataset.modalities}")

        # Sample a few items to check modality fields
        for idx in range(min(3, len(dataset))):
            sample = dataset[idx]
            for modality in dataset.modalities:
                if modality in modality_fields:
                    required = modality_fields[modality]
                    for field in required:
                        if field not in sample:
                            self.log_warning(f"Sample {idx}: modality '{modality}' requires " f"field '{field}' but it's missing")
                        else:
                            self.log_info(f"✓ Modality '{modality}' field present: '{field}'")


def print_report(report: Dict[str, Any], verbose: bool = False):
    """Print validation report in a user-friendly format."""
    dataset_name = report["dataset"]
    is_valid = report["valid"]

    print("\n" + "=" * 70)
    print(f"Dataset Validation Report: {dataset_name}")
    print("=" * 70)

    status = "✅ VALID" if is_valid else "❌ INVALID"
    print(f"\nStatus: {status}")

    # Print dataset info
    if "info" in report and report["info"]:
        print("\n📊 Dataset Information:")
        for key, value in report["info"].items():
            print(f"  • {key}: {value}")

    # Print errors
    if report["errors"]:
        print("\n❌ Errors:")
        for error in report["errors"]:
            print(f"  • {error}")

    # Print warnings
    if report["warnings"] and verbose:
        print("\n⚠️  Warnings:")
        for warning in report["warnings"]:
            print(f"  • {warning}")
    elif report["warnings"]:
        print(f"\n⚠️  {len(report['warnings'])} warning(s) found (use --verbose to see details)")

    print("\n" + "=" * 70)


def main():
    parser = argparse.ArgumentParser(description="Validate MSI dataset implementations against the standard format.")
    parser.add_argument("--dataset", type=str, help="Name of the dataset to validate (e.g., 'sat')")
    parser.add_argument("--all", action="store_true", help="Validate all registered datasets")
    parser.add_argument("--verbose", action="store_true", help="Print detailed validation information")
    parser.add_argument("--list", action="store_true", help="List all registered datasets")

    args = parser.parse_args()

    validator = DatasetValidator(verbose=args.verbose)

    # List datasets
    if args.list:
        registered = list_registered_datasets()
        print("\nRegistered Datasets:")
        for name, info in registered.items():
            print(f"  • {name}")
            if info.get("docstring"):
                first_line = info["docstring"].split("\n")[0].strip()
                print(f"    {first_line}")
        return 0

    # Validate all datasets
    if args.all:
        registered = list_registered_datasets()
        if not registered:
            print("No datasets registered!")
            return 1

        all_valid = True
        for dataset_name in registered:
            is_valid, report = validator.validate_dataset(dataset_name)
            print_report(report, args.verbose)
            if not is_valid:
                all_valid = False

        return 0 if all_valid else 1

    # Validate single dataset
    if args.dataset:
        is_valid, report = validator.validate_dataset(args.dataset)
        print_report(report, args.verbose)
        return 0 if is_valid else 1

    # No action specified
    parser.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())
