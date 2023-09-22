import os
import time
from typing import Dict, Any
import torch
from lightning.pytorch.callbacks import ModelCheckpoint, Callback
from weakref import proxy

class mBLIPModelCheckpoint(ModelCheckpoint):
    def __init__(self,
                 freeze_vit=True,
                 freeze_qformer=True,
                 freeze_lm=True,
                 freeze_projection=False,
                 lora=False,
                 lora_vit=False,
                 **kwargs):
        super().__init__(**kwargs)
        self.freeze_vit = freeze_vit
        self.freeze_qformer = freeze_qformer
        self.freeze_lm = freeze_lm

        self.filter_keys = set()
        if freeze_vit:
            self.filter_keys.add("vision_model")
        if freeze_qformer:
            self.filter_keys.add("query_tokens")
            self.filter_keys.add("qformer")
        if freeze_lm:
            self.filter_keys.add("language_model")
        if freeze_projection:
            self.filter_keys.add("language_projection")
        self.lora = lora
        self.lora_vit = lora_vit
        self.deepspeed = False

    def _save_checkpoint(self, trainer: "pl.Trainer", filepath: str) -> None:
        torch.cuda.empty_cache()
        trainer.save_checkpoint(filepath, self.save_weights_only)
        self._last_global_step_saved = trainer.global_step
        # notify loggers
        if trainer.is_global_zero:
            print("saving checkpoint to ", filepath)
            if self.lora:
                path, file = os.path.split(filepath)
                idx = file.split(".")[0]
                if self.deepspeed:
                    model = trainer.model.module.lightning_module.model.model
                else:
                    try:
                        model = trainer.model.model.model
                    except AttributeError:
                        model = trainer.model.module.lightning_module.model.model
                model.language_model.save_pretrained(os.path.join(path, idx))

            torch.cuda.empty_cache()
            for logger in trainer.loggers:
                logger.after_save_checkpoint(proxy(self))



    def on_save_checkpoint(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", checkpoint: Dict[str, Any]
    ) -> None:
        checkpoint['state_dict'] = {k:v for k,v in checkpoint['state_dict'].items() if not any(f in k for f in self.filter_keys)}
        pass

# During teardown after test and fit, the model is moved to CPU, which with LoRA somehow (cannot reproduce) moves the entire LLM to CPU which gives me OOM death.
# We wait a bit to give wandb and checkpoints time to save before the process will die.
# Yes, this is a bandaid solution at best but it works enough.
class TeardownCallback(Callback):
    def on_test_end(self, trainer, pl_module):
        print("Waiting before tearing down after test")
        time.sleep(120)
    def on_fit_end(self, trainer, pl_module):
        print("Waiting before tearing down after fit")
        time.sleep(120)