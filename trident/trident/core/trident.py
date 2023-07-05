from types import MethodType
from typing import Any, Callable, Optional, Union

import hydra
import torch
from lightning import LightningModule
from lightning.pytorch.utilities.parsing import AttributeDict
from omegaconf import DictConfig
from torch import nn

from trident.core.mixins.evaluation import EvalMixin
from trident.core.mixins.optimizer import OptimizerMixin
from trident.utils import deepgetitem
from trident.utils.logging import get_logger

log = get_logger(__name__)


# TODO(fdschmidt93): function signatures fn(self, ...)
# TODO(fdschmidt93): clean API for instantiation to depend on datamodule (token classification: label2id, id2label
class TridentModule(OptimizerMixin, EvalMixin, LightningModule):
    """Base module of Trident that wraps model, optimizer, scheduler, evaluation.

    Args:
        model (:obj:`omegaconf.dictconfig.DictConfig`):
            Needs to instantiate a :obj:`torch.nn.Module` that

            * Takes the batch unpacked
            * Returns a container with "loss" and other required attrs

            .. seealso:: :py:meth:`src.modules.base.TridentModule.forward`, :py:meth:`src.modules.base.TridentModule.training_step`, :repo:`tiny bert example <configs/module/tiny_bert.yaml>`

        optimizer (:obj:`omegaconf.dictconfig.DictConfig`):
            Configuration for the optimizer of your :py:class:`src.modules.base.TridentModule`.

            .. seealso:: :py:class:`src.modules.mixin.optimizer.OptimizerMixin`, :repo:`AdamW config <configs/optimizer/adamw.yaml>`

        scheduler (:obj:`omegaconf.dictconfig.DictConfig`):
            Configuration for the scheduler of the optimizer of your :py:class:`src.modules.base.TridentModule`.

            .. seealso:: :py:class:`src.modules.mixin.optimizer.OptimizerMixin`, :repo:`Linear Warm-Up config <configs/scheduler/linear_warm_up.yaml>`

        evaluation (:obj:`omegaconf.dictconfig.DictConfig`):

            Please refer to :ref:`evaluation`

            .. seealso:: :py:class:`src.modules.mixin.evaluation.EvalMixin`, :repo:`Classification Evaluation config <configs/evaluation/sequence_classification.yaml>`

        overrides (:obj:`omegaconf.dictconfig.DictConfig`):
            Allows you to override existing functions of the :py:class:`src.modules.base.TridentModule`.

            Example:

            `src/my_functions/model_overrides.py`

            .. code-block:: python

                def keep_batch(self, batch: dict) -> ModelOutput:
                    return self.model(batch)

            `configs/overrides/my_model_overrides.yaml`

            .. code-block:: yaml

                forward:
                  _target_: src.utils.hydra.partial
                  _partial_: src.my_functions.model_overrides.keep_batch

            At last, you'd link to the overrides config as follows:

            `configs/module/my_module.yaml`

            .. code-block:: yaml

                defaults:
                    - /overrides: my_model_overrides


            For further details, please refer to :ref:`evaluation`

        mixins (:obj:`omegaconf.dictconfig.DictConfig`):

            **Preemptive warning: do not call super().__init__() in your mixin!**

            Mixins for the :py:class:`src.modules.base.TridentModule` essentially groups of methods.
            You can overrwrite any functionality of the module with your mixins.

            Should you require any attributes from :py:meth:`src.modules.base.TridentModule.__init__` you can access them via `self.hparams`.

            .. seealso:: :repo:`Evaluation <src/modules/mixin/evaluation/sequence_classification.yaml>`
    """

    hparams: AttributeDict

    # TODO: think about `_cfg` suffix
    def __init__(
        self,
        model: DictConfig,
        optimizer: DictConfig,
        scheduler: Optional[DictConfig],
        evaluation: Optional[DictConfig] = None,
        overrides: Optional[DictConfig] = None,
        mixins: Optional[DictConfig] = None,
        module_cfg: Optional[DictConfig] = DictConfig({}),
        *args: Any,
        **kwargs: Any,
    ):
        self.save_hyperparameters()
        # TODO(fdschmidt93): verify ordering
        LightningModule.__init__(self)
        super().__init__()
        # Lightning 2.0 requires manual management of evaluation step outputs
        self._eval_outputs = []

        # TODO(fschmidt93): consider removing override syntax, as typically TridentModule is sub-classed
        # forcefully override trident methods
        self.overrides = hydra.utils.instantiate(self.hparams.overrides)
        if self.overrides is not None:
            for key, value in self.overrides.items():
                setattr(self, key, MethodType(value, self))

    # TODO
    def setup(self, stage: str):
        """Sets up the TridentModule.

        Setup the model if it does not exist yet. This enables inter-operability between your datamodule and model if you pass define a setup function in `module_cfg`, as the datamodule will be set up _before_ the model.

        In case you pass : :obj:`setup` to :obj:`module_cfg` the function should follow the below schema:

            .. code-block:: python

                def setup(module: TridentModule, stage: str):
                    # the module has to be setup
                    module.model = hydra.utils.instantiate(module.hparams.model)

        Since the :obj:`module` exposes :obj:`module.trainer.datamodule`, you can use your custom setup function to enable inter-operability between the module and datamodule.

        """
        # TODO: maybe we can simplify and integrate this even better
        setup_cfg = getattr(self.hparams.module_cfg, "setup", None)
        if setup_cfg is None:
            if not hasattr(self, "model"):
                self.model = hydra.utils.instantiate(self.hparams.model)
        else:
            hydra.utils.instantiate(setup_cfg, module=self, stage=stage)
            assert isinstance(
                self.model, nn.Module
            ), "Please set up the model appriopriately"
        if ckpt := getattr(self.hparams.module_cfg, "weights_from_checkpoint", None):
            self.weights_from_checkpoint(**ckpt)

    def weights_from_checkpoint(
        self,
        ckpt_path: str,
        map_location: str = "cpu",
        strict: bool = True,
        processing_fn: Optional[Callable] = None,
        map_weights: Optional[Union[str, dict]] = None,
    ):
        """Loads weights from existing Pytorch-Lightning checkpoint.


        Args:
            ckpt_path (:obj:`str`):
                path to lightning checkpoint
            map_location (:obj:`str`):
                device to map to
            strict (:obj:`bool`):
                force agreement (:obj:`True`) or accept non-matching keys (:obj:`False`)
            map_weights (:obj:`Union[None, str, dict[str, str]]`, `optional`):
                * If :obj:`None`: assumes 1-to-1 alignment between checkpoint and :obj:`TridentModule`
                * If :obj:`str`: maps the weight of the `str` to :obj:`self.model`
                * If :obj:`dict`: keys of the checkpoint's :obj:`state_dict` are mapped by values to :obj:`self.model`
                    - Values should be in absolute dot notation e.g. :code:`encoder.embeddings: transformer.embeddings`
            map_location (:obj:`Optional[Callable]`):
                function that takes the state dict of the loaded checkpoint, process and outputs the state dict
        """
        ckpt = torch.load(ckpt_path, map_location)["state_dict"]
        if isinstance(processing_fn, Callable):
            processing_fn = hydra.utils.instantiate(processing_fn)
            if isinstance(processing_fn, Callable):
                ckpt = processing_fn(ckpt)
        if map_weights is None:
            self.load_state_dict(ckpt, strict=strict)
        elif isinstance(map_weights, str):
            ckpt = ckpt[map_weights]
            self.load_state_dict(ckpt, strict=strict)
        elif isinstance(map_weights, dict):
            # for from, to in weights
            for src_weights, trg_weights in map_weights.items():
                src_module = deepgetitem(ckpt, src_weights)
                assert (
                    src_module is not None
                ), f"{src_module} is not a valid attribute of state_dict"
                trg_module: Union[nn.Module, None] = deepgetitem(self, trg_weights)
                assert (
                    trg_module is not None
                ), f"{trg_module} is not a valid attribute of self.model"
                trg_module.load_state_dict(src_module, strict=strict)

        log.info(f"Successfully reloaded weights from {ckpt_path}")

    def forward(self, batch: dict) -> dict:
        """Plain forward pass of your model for which the batch is unpacked.


        **Implementation:**

            .. code-block:: python

                return self.model(**batch)


        Args:
            batch: input to your model

        Returns:
            ModelOutput: container with attributes required for evaluation
        """
        return self.model(**batch)

    def training_step(self, batch: dict, batch_idx: int) -> dict[str, Any]:
        """Comprises training step of your model which takes a forward pass.

        **Notes:**
            If you want to extend `training_step`, add a `on_train_batch_end` method via overrides.
            See: Pytorch-Lightning's `on_train_batch_end <https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html#on-train-batch-end>`_

        **Implementation:**

            .. code-block:: python

                def training_step(
                    self, batch: BatchEncoding, batch_idx: int
                ) -> Union[dict[str, Any], ModelOutput]:
                    outputs = self(batch)
                    self.log("train/loss", outputs.loss)
                    return outputs

        Args:
            batch: typically comprising input_ids, attention_mask, and position_ids
            batch_idx: variable used internally by pytorch-lightning

        Returns:
            Union[dict[str, Any], ModelOutput]: model output that must have 'loss' as attr or key
        """
        outputs = self(batch)
        self.log("train/loss", outputs["loss"])
        return outputs
