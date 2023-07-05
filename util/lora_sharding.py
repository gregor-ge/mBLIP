import os

import torch
import json
from tqdm import tqdm

def prepare_shards():
    index = json.load(open("pytorch_model.bin.index.json"))
    shards = set(index["weight_map"].values())
    print(shards)
    target_modules = [".q.", ".v."]

    for i, shard in enumerate(shards):
        target_modules_shard = dict()
        shard = torch.load(shard)
        keys = list(shard.keys())
        for key in tqdm(keys):
            if any(m in key for m in target_modules):
                print(key)
                target_modules_shard[key] = shard.pop(key)
                index["weight_map"][key] = f"pytorch_model-0000{2*i+2}-of-0000{2*len(shards)}.bin"
            else:
                index["weight_map"][key] = f"pytorch_model-0000{2 * i + 1}-of-0000{2 * len(shards)}.bin"
        torch.save(shard, f"original/pytorch_model-0000{2 * i + 1}-of-0000{2 * len(shards)}.bin")
        torch.save(target_modules_shard, f"original/pytorch_model-0000{2*i+2}-of-0000{2*len(shards)}.bin")
    json.dump(index, open("original/pytorch_model.bin.index.json", "w"))


def merge_lora(name, adapter_folder):
    index = json.load(open("original/pytorch_model.bin.index.json"))
    config = json.load(open("original/config.json"))
    shards = sorted(list(set(index["weight_map"].values())))

    for k, v in index["weight_map"].items():
        for i in range(0, len(shards), 2):
            shard = shards[i]
            index["weight_map"][k] = index["weight_map"][k].replace(shard, f"../original/{shard}")

    os.makedirs(name, exist_ok=True)

    json.dump(index, open(f"{name}/pytorch_model.bin.index.json", "w"))
    json.dump(config, open(f"{name}/config.json", "w"))

    adapter_config = json.load(open(os.path.join(adapter_folder, "adapter_config.json")))
    adapter_state_dict = torch.load(os.path.join(adapter_folder, "adapter_model.bin"))

    alpha = adapter_config["lora_alpha"]
    r = adapter_config["r"]
    scaling = alpha/r
    for i in range(1, len(shards) + 1, 2):
        shard_name = shards[i]
        print(shard_name)
        shard = torch.load(f"original/{shard_name}")
        for key in shard:
            lora_a = adapter_state_dict[f"base_model.model.{key[:-len('weight')]}lora_A.weight"].cpu()
            lora_b = adapter_state_dict[f"base_model.model.{key[:-len('weight')]}lora_B.weight"].cpu()
            w = shard[key] + scaling * lora_b @ lora_a
            shard[key] = w
        torch.save(shard, f"{name}/{shard_name}")

def merge_lora_all(name, adapter_folder):
    index = json.load(open("original/pytorch_model.bin.index.json"))
    config = json.load(open("original/config.json"))
    shards = sorted(list(set(index["weight_map"].values())))

    # for k, v in index["weight_map"].items():
    #     for i in range(0, len(shards), 2):
    #         shard = shards[i]
    #         index["weight_map"][k] = index["weight_map"][k].replace(shard, f"../original/{shard}")

    os.makedirs(name, exist_ok=True)

    json.dump(index, open(f"{name}/pytorch_model.bin.index.json", "w"))
    json.dump(config, open(f"{name}/config.json", "w"))

    adapter_config = json.load(open(os.path.join(adapter_folder, "adapter_config.json")))
    adapter_state_dict = torch.load(os.path.join(adapter_folder, "adapter_model.bin"))

    alpha = adapter_config["lora_alpha"]
    r = adapter_config["r"]
    scaling = alpha/r
    for i in range(len(shards)):
        shard_name = shards[i]
        print(shard_name)
        shard = torch.load(f"original/{shard_name}")
        for key in shard:
            if f"base_model.model.{key[:-len('weight')]}lora_A.weight" in adapter_state_dict:
                lora_a = adapter_state_dict[f"base_model.model.{key[:-len('weight')]}lora_A.weight"].cpu()
                lora_b = adapter_state_dict[f"base_model.model.{key[:-len('weight')]}lora_B.weight"].cpu()
                w = shard[key] + scaling * lora_b @ lora_a
                shard[key] = w
            else:
                print("Skip ", key)
        torch.save(shard, f"{name}/{shard_name}")

if __name__ == "__main__":
    # prepare_shards()
    # merge_lora("05_30_2023_08_32_17-2-61785", "../05_30_2023_08_32_17/2-61785")

    output_directory = "05_30_2023_08_32_17-2-61785"
    lora_directory = "../05_30_2023_08_32_17/2-61785"
    merge_lora_all(output_directory, lora_directory)