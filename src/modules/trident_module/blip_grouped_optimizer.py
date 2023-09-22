import hydra

from trident import TridentModule
import torch
from typing import Union, Optional, cast
from transformers import BatchEncoding

class GroupedOptimizerTridentModule(TridentModule):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def configure_optimizers(self):
        """Prepares optimizer and scheduler."""
        if weight_decay := getattr(self.hparams.optimizer, "weight_decay", None):
            blip_lr = self.hparams.optimizer.blip_lr
            llm_lora_lr = self.hparams.optimizer.llm_lora_lr
            delattr(self.hparams.optimizer, "blip_lr")
            delattr(self.hparams.optimizer, "llm_lora_lr")
            print(blip_lr, llm_lora_lr)
            def is_llm(param):
                return "language_model" in param

            param_optimizer = list(self.named_parameters())
            no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
            parameters = [
                {
                    "params": [
                        p
                        for n, p in param_optimizer
                        if not any(nd in n for nd in no_decay) and is_llm(n)
                    ],
                    "weight_decay": weight_decay,
                    "lr": llm_lora_lr
                },
                {
                    "params": [
                        p for n, p in param_optimizer if any(nd in n for nd in no_decay) and is_llm(n)
                    ],
                    "weight_decay": 0.0,
                    "lr": llm_lora_lr
                },
                {
                    "params": [
                        p
                        for n, p in param_optimizer
                        if not any(nd in n for nd in no_decay) and not is_llm(n)
                    ],
                    "weight_decay": weight_decay,
                    "lr": blip_lr
                },
                {
                    "params": [
                        p for n, p in param_optimizer if any(nd in n for nd in no_decay) and not is_llm(n)
                    ],
                    "weight_decay": 0.0,
                    "lr": blip_lr
                },
            ]
        else:
            parameters = self.parameters()

        optimizer = hydra.utils.instantiate(self.hparams.optimizer, parameters)
        if scheduler_cfg := getattr(self.hparams, "scheduler"):
            scheduler = self.configure_scheduler(optimizer, scheduler_cfg)
            return [optimizer], [scheduler]
        return [optimizer]

