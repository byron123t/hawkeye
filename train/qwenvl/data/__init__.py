"""Dataset registry for qwen-vl-finetune (Hawkeye release).

`--dataset_use` takes registry names or paths, comma-separated, each optionally suffixed with
%N to subsample N percent. The Hawkeye entries point at the LLaVA-style JSON files written by
train/prepare_data.py; override their locations with environment variables.

    bstrb_train_video   $BSTRB_TRAIN_JSON  (default data/bstrb_train_video.json)
    bstrb_val           $BSTRB_VAL_JSON    (default data/bstrb_val.json)

Image paths inside the JSON files are absolute, so data_path is "/".
"""
import os
import re

data_dict = {
    "bstrb_train_video": {"annotation_path": os.environ.get("BSTRB_TRAIN_JSON", "data/bstrb_train_video.json"),
                          "data_path": "/"},
    "bstrb_val": {"annotation_path": os.environ.get("BSTRB_VAL_JSON", "data/bstrb_val.json"),
                  "data_path": "/"},
}


def parse_sampling_rate(dataset_name):
    match = re.search(r"%(\d+)$", dataset_name)
    if match:
        return int(match.group(1)) / 100.0
    return 1.0


def data_list(dataset_names):
    config_list = []
    for dataset_name in dataset_names:
        sampling_rate = parse_sampling_rate(dataset_name)
        dataset_name = re.sub(r"%(\d+)$", "", dataset_name)
        if dataset_name in data_dict:
            config = data_dict[dataset_name].copy()
        elif dataset_name.endswith((".json", ".jsonl")) and os.path.exists(dataset_name):
            config = {"annotation_path": dataset_name, "data_path": "/"}
        else:
            raise ValueError(f"unknown dataset {dataset_name!r} (not registered and not a file)")
        config["sampling_rate"] = sampling_rate
        config_list.append(config)
    return config_list
