## LoRA merging
After instruction training, we need to merge the LoRA weights into the LLM if we want to use
LoRA again to finetune for a downstream task.

### Option 1: Simple but needs enough RAM
If you can load the entire LLM without 8-bit quantization, then you can simply load it, 
use the [peft utilities to merge weights into the model](https://github.com/huggingface/peft/blob/main/src/peft/tuners/lora.py#L391)
and then dump the model to disk (`.save_pretrained(...)`).

### Option 2: 'Manual' merging (RAM efficient)
This is the option used by us.
1. Download the model checkpoint shards, config, and shard index from the model hub (e.g. [mt0-xl](https://huggingface.co/bigscience/mt0-xl/tree/main)) and place them in a
folder (we use `original`.)
2. Edit in `lora_sharding.py` the output and LoRA directory to use your LoRA weights and run it. This will
merge the LoRA weights into the model, going shard by shard for memory efficiency.

Extra:
If you only use 'standard' LoRA (using only query and value matrices), then you can use
the `prepare_shards()` and `merge_lora()` functions for space efficiency:
This combination first re-shards the model to put the query and value matrices in separate shards and then only
saves those shards after merging LoRA into them while re-using the shards from the original model.

## Creating $blip_checkpoint
[This script](creating_blip_checkpoint.py) shows how to create prepare the $blip_checkpoint by removing all language model parameters
from the checkpoint downloaded from https://huggingface.co/Salesforce/blip2-flan-t5-xl/tree/main.