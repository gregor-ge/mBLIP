import torch

# download from https://huggingface.co/Salesforce/blip2-flan-t5-xl/tree/main
state_dict = torch.load("pytorch_model-00001-of-00002.bin")

state_dict = {k:v for k,v in state_dict.items() if "language_model" not in k}

torch.save(state_dict, "blip2-flant5xl-nolm.bin")