import json

json_path = "/projectnb/cs598/data_storage/msi_datasets/EgoOrientBench/all_data/EgocentricDataset/benchmark_data/benchmark.json"

with open(json_path, 'r') as f:
    data = json.load(f)

# Collect every unique label found in the entire dataset
unique_labels = set()
for entry in data:
    label = str(entry.get('label', '')).strip().upper()
    if label:
        unique_labels.add(label)

# Sort them by length (descending) for the best parsing results
sorted_labels = sorted(list(unique_labels), key=len, reverse=True)

print("--- COPY AND PASTE THIS INTO YOUR MAIN SCRIPT ---")
print(f"L = {sorted_labels}")
print("-------------------------------------------------")
print(f"Total unique labels: {len(sorted_labels)}")