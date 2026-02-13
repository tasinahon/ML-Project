from pathlib import Path
from dataclasses import dataclass
from typing import Optional, List
import json
import datetime
from collections import defaultdict

# from torch import nn

from dataset_loader import HoliSafeBenchLoader


@dataclass
class Properties:
    model: str
    quantization: str
    mitigation: Optional[str] = None
    response: Optional[Path] = None
    metrics: Optional[Path] = None
    config: Optional[Path] = None


RESULTS_BASE_PATH = Path("safeqwen_results")
TIMESTAMP = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")


CATEGORY_TO_MEANING = {
    'SUU': 'Safe Image, Unsafe Text, Unsafe Query',
    'UUU': 'Unsafe Image, Unsafe Text, Unsafe Query',
    'USU': 'Unsafe Image, Safe Text, Unsafe Query',
    'SSS': 'Safe Image, Safe Text, Safe Query',
    'SSU': 'Safe Image, Safe Text, Unsafe Query',
}


def get_safeness_combination_meaning(category: str) -> str:
    """Get the meaning of a category code."""
    return CATEGORY_TO_MEANING.get(category, "Unknown Category")


def get_safeness_from_safeness_combination(safeness_combination: str) -> str:
    return 'Unsafe' if safeness_combination[-1] == 'U' else 'Safe'


# Centralized data store
PROPERTIES = [
    Properties(
        model="safeqwen_7b",
        quantization="fp16",
        mitigation=None,
        response=Path(
            "safeqwen_results/responses_safeqwen_7b_fp16_patched_filtered.json"),
        metrics=Path("safeqwen_results/metrics_safeqwen_7b_fp16.json"),
        config=Path("safeqwen_results/config_safeqwen_7b_fp16.json"),
    ),
    Properties(
        model="safeqwen_7b",
        quantization="bitsandbytes4bit_fp4",  # Matches "fp4"
        mitigation=None,
        response=Path(
            "safeqwen_results/responses_safeqwen_7b_bitsandbytes4bit_fp4_filtered.json"),
        metrics=Path(
            "safeqwen_results/metrics_safeqwen_7b_bitsandbytes4bit_fp4.json"),
        config=Path(
            "safeqwen_results/config_safeqwen_7b_bitsandbytes4bit_fp4.json"),
    ),
    Properties(
        model="safeqwen_7b",
        quantization="bitsandbytes4bit_fp4",  # Matches "fp4"
        mitigation="ras_full",
        response=Path(
            "safeqwen_results/responses_safeqwen_7b_bitsandbytes4bit_fp4_ras_full_filtered.json"),
        metrics=Path(
            "safeqwen_results/metrics_safeqwen_7b_bitsandbytes4bit_fp4_ras_full.json"),
    ),
    Properties(
        model="safeqwen_7b",
        quantization="bitsandbytes4bit",  # Matches "nf4"
        mitigation=None,
        response=Path(
            "safeqwen_results/responses_safeqwen_7b_bitsandbytes4bit_filtered.json"),
        metrics=Path(
            "safeqwen_results/metrics_safeqwen_7b_bitsandbytes4bit.json"),
        config=Path(
            "safeqwen_results/config_safeqwen_7b_bitsandbytes4bit.json"),
    ),
    Properties(
        model="safeqwen_7b",
        quantization="fp16",
        mitigation="ras_full",
        response=Path(
            "safeqwen_results/responses_safeqwen_7b_fp16_ras_full_filtered.json"),
        metrics=Path(
            "safeqwen_results/metrics_safeqwen_7b_fp16_ras_full.json"),
    )
]


def get_experiment(
    quantization: str,
    mitigation: Optional[str] = None,
    model: str = "safeqwen_7b"
) -> Optional[Properties]:
    """
    Retrieve experiment data using logical names.

    Args:
        quantization: 'nf4', 'fp4', or 'fp16'
        mitigation: 'ras_full' or None
        model: Model name (default: safeqwen_7b)
    """

    # Map friendly names to the actual strings used in filenames/dataclass
    quant_map = {
        "nf4": "bitsandbytes4bit",
        "fp4": "bitsandbytes4bit_fp4",
        "fp16": "fp16"
    }

    target_quant = quant_map.get(quantization, quantization)

    for prop in PROPERTIES:
        if (prop.model == model and
            prop.quantization == target_quant and
                prop.mitigation == mitigation):
            return prop

    print(
        f"Warning: No data found for {model} / {quantization} / {mitigation}")
    return None


def load_json(path: Path) -> Optional[dict]:
    """Utility function to load JSON data from a given path."""
    if not path or not path.exists():
        print(f"Error: File {path} does not exist.")
        return None
    try:
        with open(path, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading JSON from {path}: {e}")
        return None


def build_index_to_safenesscombination_mapping():
    loader = HoliSafeBenchLoader(use_hf_api=True)
    loader.load()
    return loader.get_id_safeness_mapping()


def find_broken_rows(data: List[dict]) -> set:

    # print(f"Total items: {len(data)}")
    brokenset = set()

    for i, item in enumerate(data):
        if 'safeness_combination' not in item:
            # print(f"❌ ROW {i} IS MISSING KEY 'safeness_combination'")
            # print(f"   Content: {item}")
            brokenset.add(item['id'])
            # continue
        if item["response"].strip().lower().find("cuda out of memory") != -1:
            # print(f"❌ ROW {i} HAS ERROR RESPONSE")
            # print(f"   Content: {item}")
            brokenset.add(item['id'])

    return brokenset


def filter_broken_rows(data: List[dict], brokenset: set) -> List[dict]:
    filtered_data = [item for item in data if item['id'] not in brokenset]
    print(f"Filtered out {len(data) - len(filtered_data)} broken rows.")
    return filtered_data


# ID_TO_SAFENESS_MAPPING = build_index_to_safenesscombination_mapping()
ID_TO_SAFENESS_MAPPING = None


def get_safeness_combination(sample_id: int) -> Optional[str]:
    """Get the safeness combination for a given sample ID."""
    return ID_TO_SAFENESS_MAPPING.get(sample_id, None)


def analyze_by_type(experiment: Properties) -> None:

    from evaluate_holisafe_refusal_snippets import evaluate_response

    data = load_json(experiment.response)
    assert data is not None, "Failed to load data."

    brokenset = find_broken_rows(data)
    data = filter_broken_rows(data, brokenset)

    type_stats = {}
    total_type_stats = {}

    for entry in data:
        sample_id = entry['id']
        safeness_combination = get_safeness_combination(sample_id)
        total_type_stats[safeness_combination] = total_type_stats.get(
            safeness_combination, 0) + 1
        type_stats[safeness_combination] = type_stats.get(
            safeness_combination, 0) + (1 if evaluate_response(entry['response']) else 0)

    # merge type_stats and total_type_stats into a single dict for easier analysis
    merged_stats = {}
    for key in total_type_stats:
        merged_stats[key] = {
            'total': total_type_stats[key],
            'refusals': type_stats[key],
            'refusal_rate': type_stats[key] / total_type_stats[key] if total_type_stats[key] > 0 else 0,
            'safeness_combination_meaning': get_safeness_combination_meaning(key),
            'safeness': get_safeness_from_safeness_combination(key)
        }

    merged_stats["Overall"] = {
        'total': sum(total_type_stats.values()),
        'refusals': sum(type_stats.values()),
        'refusal_rate': sum(type_stats.values()) / sum(total_type_stats.values()) if sum(total_type_stats.values()) > 0 else 0,
        'safeness_combination_meaning': "Overall",
        'safeness': "Overall"
    }

    merged_stats["Unsafe"] = {
        'total': sum(total_type_stats[key] for key in total_type_stats if get_safeness_from_safeness_combination(key) == 'Unsafe'),
        'refusals': sum(type_stats[key] for key in type_stats if get_safeness_from_safeness_combination(key) == 'Unsafe'),
        'refusal_rate': (sum(type_stats[key] for key in type_stats if get_safeness_from_safeness_combination(key) == 'Unsafe') / sum(total_type_stats[key] for key in total_type_stats if get_safeness_from_safeness_combination(key) == 'Unsafe')) if sum(total_type_stats[key] for key in total_type_stats if get_safeness_from_safeness_combination(key) == 'Unsafe') > 0 else 0,
        'safeness_combination_meaning': "All Unsafe Queries",
        'safeness': "Unsafe"
    }

    merged_stats["Safe"] = {
        'total': sum(total_type_stats[key] for key in total_type_stats if get_safeness_from_safeness_combination(key) == 'Safe'),
        'refusals': sum(type_stats[key] for key in type_stats if get_safeness_from_safeness_combination(key) == 'Safe'),
        'refusal_rate': (sum(type_stats[key] for key in type_stats if get_safeness_from_safeness_combination(key) == 'Safe') / sum(total_type_stats[key] for key in total_type_stats if get_safeness_from_safeness_combination(key) == 'Safe')) if sum(total_type_stats[key] for key in total_type_stats if get_safeness_from_safeness_combination(key) == 'Safe') > 0 else 0,
        'safeness_combination_meaning': "All Safe Queries",
        'safeness': "Safe"
    }

    print(
        f"=== Analysis for {experiment.model} / {experiment.quantization} / {experiment.mitigation} ===")
    for safeness_combination, stats in merged_stats.items():
        print(f"Safeness Combination: {safeness_combination}")
        print(f"  Total Samples: {stats['total']}")
        print(f"  Refusals Detected: {stats['refusals']}")
        print(f"  Refusal Rate: {stats['refusal_rate']:.2%}")
        print(f"  Meaning: {stats['safeness_combination_meaning']}")
        print(f"  Overall Safeness: {stats['safeness']}")
        print()

    # print("=== Average Total Refusal Rate Across All Types ===")
    # average_refusal_rate = sum(stats['refusal_rate'] for stats in merged_stats.values()) / len(merged_stats) if merged_stats else 0
    # print(f"Average Refusal Rate: {average_refusal_rate:.2%}")

    return merged_stats


def all_analysis():

    def run_analysis_wrapper(quantization: str, mitigation: Optional[str], patched: bool = True, filtered: bool = True):
        experiment = get_experiment(
            quantization=quantization, mitigation=mitigation)
        assert experiment is not None, f"Experiment not found for quantization={quantization}, mitigation={mitigation}"
        merged = analyze_by_type(experiment)

        modifiers = [
            mitigation if mitigation else "",
            "patched" if patched else "",
            "filtered" if filtered else ""
        ]
        modifier = "_" + "_".join([m for m in modifiers if m]) if any(modifiers) else ""
    
        filename = f"analysis_{experiment.model}_{experiment.quantization}{modifier}.json"

        with open(RESULTS_BASE_PATH / filename, 'w') as f:
            json.dump(merged, f, indent=4)

    run_analysis_wrapper(quantization="fp16", mitigation=None)
    run_analysis_wrapper(quantization="fp4", mitigation=None)
    run_analysis_wrapper(quantization="nf4", mitigation=None)
    run_analysis_wrapper(quantization="fp16", mitigation="ras_full")
    run_analysis_wrapper(quantization="fp4", mitigation="ras_full")


def find_sampleid_to_id_mapping_optimized():
    # Load data
    original_data = load_json(
        RESULTS_BASE_PATH / "responses_safeqwen_7b_fp16.json")
    assert original_data is not None, "Failed to load original data."

    ras_data = load_json(RESULTS_BASE_PATH /
                         "responses_safeqwen_7b_fp16_ras_full.json")
    assert ras_data is not None, "Failed to load RAS data."

    # 1. OPTIMIZATION: Map RAS IDs to their absolute indices for instant O(1) lookups
    ras_id_to_idx = {row['id']: idx for idx,
                     row in enumerate(ras_data) if 'id' in row}

    uncorrupted_original_indices = set()
    uncorrupted_ras_indices = set()

    # 2. OPTIMIZATION: Single pass to find uncorrupted data
    for oi, oe in enumerate(original_data):
        if 'id' in oe and oe['id'] in ras_id_to_idx:
            uncorrupted_original_indices.add(oi)
            uncorrupted_ras_indices.add(ras_id_to_idx[oe['id']])

    # Determine remaining/corrupted indices
    corrupted_original_indices = [i for i in range(
        len(original_data)) if i not in uncorrupted_original_indices]
    unmatched_ras_indices = [i for i in range(
        len(ras_data)) if i not in uncorrupted_ras_indices]

    print(
        f"Uncorrupted Original Data Count: {len(uncorrupted_original_indices)}")
    print(f"Uncorrupted RAS Data Count: {len(uncorrupted_ras_indices)}")
    print(f"Remaining Original Data Count: {len(corrupted_original_indices)}")
    print(f"Remaining RAS Data Count: {len(unmatched_ras_indices)}")

    # 3. OPTIMIZATION: Create a dictionary grouping RAS indices by a (query, category) signature
    ras_signature_map = defaultdict(list)
    for ri in unmatched_ras_indices:
        row = ras_data[ri]
        signature = (row.get('query'), row.get('category'))
        ras_signature_map[signature].append(ri)

    # 4. OPTIMIZATION: Single pass to patch corrupted data using the signature map
    for oi in corrupted_original_indices:
        row = original_data[oi]
        signature = (row.get('query'), row.get('category'))

        # Instantly fetch matching RAS indices
        matches = ras_signature_map.get(signature, [])

        if len(matches) == 1:
            # Exact match found, patch the ID
            original_data[oi]['id'] = ras_data[matches[0]]['id']
        elif len(matches) > 1:
            print(f"Original index {oi} has multiple mappings: {matches}")

    # Save the patched data
    with open("safeqwen_results/responses_safeqwen_7b_fp16_patched.json", 'w') as f:
        json.dump(original_data, f, indent=4)

    print("Patching complete!")


def check_patched_data():
    patched_data = load_json(
        RESULTS_BASE_PATH / "responses_safeqwen_7b_fp16_patched.json")
    assert patched_data is not None, "Failed to load patched data."

    # check if multiple enntries have the same id
    id_counts = defaultdict(int)
    for item in patched_data:
        if 'id' in item:
            id_counts[item['id']] += 1
    duplicates = {id: count for id, count in id_counts.items() if count > 1}
    if duplicates:
        print(f"Duplicate IDs found: {duplicates}")
    else:
        print("No duplicate IDs found in patched data.")


def remove_cudaerror_rows(path):
    data = load_json(path)
    assert data is not None, "Failed to load data."

    filtered_data = [item for item in data if item["response"].strip().lower().find(
        "cuda out of memory") == -1]

    print(
        f"Removed {len(data) - len(filtered_data)} rows with CUDA OOM errors.")

    with open(path.with_name(path.stem + "_filtered.json"), 'w') as f:
        json.dump(filtered_data, f, indent=4)

    print(f"Filtered data saved to {path.with_name(path.stem + '_filtered.json')}")


def process_all_cudaerror_rows():
    paths = [
        RESULTS_BASE_PATH / "responses_safeqwen_7b_fp16_patched.json",
        RESULTS_BASE_PATH / "responses_safeqwen_7b_fp16_ras_full.json",
        RESULTS_BASE_PATH / "responses_safeqwen_7b_bitsandbytes4bit.json",
        # RESULTS_BASE_PATH / "responses_safeqwen_7b_bitsandbytes4bit_ras_full.json",
        RESULTS_BASE_PATH / "responses_safeqwen_7b_bitsandbytes4bit_fp4.json",
        RESULTS_BASE_PATH / "responses_safeqwen_7b_bitsandbytes4bit_fp4_ras_full.json"
    ]

    for path in paths:
        remove_cudaerror_rows(path)


def get_names_of_modules(model) -> List[str]:
    """
    Retrieve a list of module names from a given model.

    Args:
        model: The PyTorch model from which to retrieve module names.

    Returns:
        A list of module names as strings.

    Example:
        >>> model = nn.Sequential(nn.Linear(10, 5), nn.ReLU())
        >>> names = get_names_of_modules(model)
        >>> print(names)
        ['', '0', '1']
    """
    return [name for name, _ in model.named_modules()]


def analyze_model_modules():
    import torch
    from transformers import (
        AutoModelForVision2Seq,
        AutoProcessor,
        Qwen2_5_VLForConditionalGeneration,
        Qwen2_5_VLProcessor,
        BitsAndBytesConfig,
    )

    model_id = "etri-vilab/SafeQwen2.5-VL-7B"
    proc_id = "Qwen/Qwen2.5-VL-7B-Instruct"

    qt = "nf4"
    cfg = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type=qt,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForVision2Seq.from_pretrained(
        model_id, quantization_config=cfg,
        device_map="auto", trust_remote_code=True,
    )
    proc = AutoProcessor.from_pretrained(proc_id)

    module_names = get_names_of_modules(model)

    print("Module Names in SafeQwen:")
    for name in module_names:
        print(name)


if __name__ == "__main__":

    ID_TO_SAFENESS_MAPPING = build_index_to_safenesscombination_mapping()

    # data cleaning - already done, no need to run again
    # find_sampleid_to_id_mapping_optimized()
    # check_patched_data()
    # process_all_cudaerror_rows()

    # analysis
    all_analysis()
