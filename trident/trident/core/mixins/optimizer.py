from typing import Union

import hydra
from lightning import LightningModule
from lightning.pytorch.utilities.parsing import AttributeDict
from omegaconf.dictconfig import DictConfig

from trident.utils.logging import get_logger

log = get_logger(__name__)

from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR


class OptimizerMixin(LightningModule):
    """Mixin for base model to define configuration of optimizer and scheduler.

    The OptimizerMixin provides functionality to:
        * compute the number of training steps (:obj:`OptimizerMixin.num_training_steps`)
        * configure the optimizer(s) (:obj:`OptimizerMixin.configure_optimizers`)
        * configure the scheduler (:obj:`OptimizerMixin.configure_scheduler`)

    Examples:
        * Optimizer: :repo:`AdamW <configs/optimizer/adamw.yaml>`
        * Scheduler: :repo:`Linear Warm-Up <configs/scheduler/linear_warm_up.yaml>`

    """

    hparams: AttributeDict

    @property
    def num_training_steps(self):
        """Computes the number of training steps per device, accounting for gradient accumulation."""
        if self.trainer.max_steps != -1:
            return self.trainer.max_steps
        accumulate_grad_batches = getattr(self.trainer, "accumulate_grad_batches", 1)
        if datamodule := getattr(self.trainer, "datamodule"):
            dataloader = datamodule.train_dataloader()
            if isinstance(dataloader, dict):
                num_training_batches = max([len(dl) for dl in dataloader.values()])
            else:
                num_training_batches = len(dataloader)
        else:
            num_training_batches = len(self.train_dataloader())
        return (
            num_training_batches
            * self.trainer.max_epochs
            // max(1, self.trainer.num_devices)
            // accumulate_grad_batches
        )

    def configure_scheduler(
        self, optimizer: Optimizer, scheduler_cfg: DictConfig
    ) -> dict[str, Union[str, int, LambdaLR]]:
        """Configures the LR scheduler for the optimizer.

        The instantiation of the scheduler takes the optimizer as the first positional argument.

        .. code-block:: python

            # hparams.scheduler: passed config
            scheduler: LambdaLR = hydra.utils.instantiate(self.hparams.scheduler, optimizer,)


        Note that the below values are hard-coded for the time being:
            * interval: step
            * frequency: 1

        Args:
            optimizer: pytorch optimizer

        Returns:
            dict[str, Union[str, int, LambdaLR]: scheduler in pytorch-lightning format
        """
        if hasattr(scheduler_cfg, "num_warmup_steps") and isinstance(
            scheduler_cfg.num_warmup_steps, float
        ):
            scheduler_cfg.num_warmup_steps *= self.num_training_steps
        scheduler_cfg.num_training_steps = self.num_training_steps
        log.info(
            f"Warm up for {scheduler_cfg.num_warmup_steps} of {self.num_training_steps}"
        )
        scheduler = hydra.utils.instantiate(
            scheduler_cfg,
            optimizer,
        )
        # TODO(fdschmidt93): more flexible LR schedules?
        scheduler = {"scheduler": scheduler, "interval": "step", "frequency": 1}
        return scheduler

    def configure_optimizers(self):
        """Prepares optimizer and scheduler."""
        if weight_decay := getattr(self.hparams.optimizer, "weight_decay", None):
            param_optimizer = list(self.named_parameters())
            no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
            parameters = [
                {
                    "params": [
                        p
                        for n, p in param_optimizer
                        if not any(nd in n for nd in no_decay)
                    ],
                    "weight_decay": weight_decay,
                },
                {
                    "params": [
                        p for n, p in param_optimizer if any(nd in n for nd in no_decay)
                    ],
                    "weight_decay": 0.0,
                },
            ]
        else:
            parameters = self.parameters()

        optimizer = hydra.utils.instantiate(self.hparams.optimizer, parameters)
        if scheduler_cfg := getattr(self.hparams, "scheduler"):
            scheduler = self.configure_scheduler(optimizer, scheduler_cfg)
            return [optimizer], [scheduler]
        return [optimizer]
