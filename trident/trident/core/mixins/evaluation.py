from typing import Any, Callable, NamedTuple, Optional, Union

import hydra
import torch
from lightning import LightningModule
from lightning.pytorch.utilities.parsing import AttributeDict
from omegaconf import OmegaConf
from omegaconf.base import DictKeyType
from omegaconf.dictconfig import DictConfig
from omegaconf.listconfig import ListConfig
from torch.utils.data import Dataset

from trident.utils import deepgetitem
from trident.utils.logging import get_logger
from trident.utils.transform import flatten_dict

log = get_logger(__name__)


class EvalMixin(LightningModule):
    r"""Mixin for base model to define evaluation loop largely via hydra.

    The evaluation mixin enables writing evaluation via yaml files, here is an
    example for sequence classification, borrowed from configs/evaluation/classification.yaml.

    .. code-block:: yaml

        # apply transformation function
        prepare_cfg:
          batch: null # on each step
          outputs:    # on each step
            _target_: src.utils.hydra.partial
            _partial_: src.evaluation.classification.get_preds
            .. code-block: python

                # we link evaluation.apply.outputs against get_preds
                def get_preds(outputs):
                    outputs.preds = outputs.logits.argmax(dim=-1)
                    return outputs

          step_outputs: null  # on flattened outputs of what's collected from steps

        # Which keys/attributes are supposed to be collected from `outputs` and `batch`
        step_outputs:
          outputs: "preds" # can be a str
          batch: # or a list[str]
            - labels

        # either metrics or val_metrics and test_metrics
        # where the latter
        metrics_cfg:
          # name of the metric used eg for logging
          accuracy:
            # instructions to instantiate metric, preferrably torchmetrics.Metric
            metric:
              _target_: torchmetrics.Accuracy
            # either on_step: true or on_epoch: true
            on_step: true
            compute:
              preds: "outputs:preds"
              target: "batch:labels"
          f1:
            metric:
              _target_: torchmetrics.F1
            on_step: true
            compute:
              preds: "outputs:preds"
              target: "batch:labels"

    """
    hparams: AttributeDict
    log: Callable

    def __init__(self) -> None:
        # hparams used to fast-forward required attributes
        self.evaluation = hydra.utils.instantiate(self.hparams.evaluation)

    # TODO better message
    def _validate_tensors_epoch_end(
        self,
        outputs: dict,
        num_samples: int,
        stage: str,
        dataset: Optional[str] = None,
    ) -> None:
        keys = []
        prefix = f"{stage}: " if dataset is None else f"{stage} - {dataset}: "
        for k, v in outputs.items():
            if isinstance(v, torch.Tensor):
                if v.shape[0] != num_samples:
                    message = prefix + f"{k} has {v.shape[0]}/{num_samples} rows"
                    log.warn(message)
                    keys.append(k)

        for k in keys:
            outputs[k] = outputs[k][:num_samples]
            message = (
                prefix
                + f"Truncating {k} to {outputs[k].shape[0]} (#samples={num_samples}) rows"
            )
            log.warn(message)

    def on_eval_start(self, stage: str) -> None:
        metrics_cfg = self.evaluation.metrics_cfg
        if metrics_cfg is None:
            return
        stage_metrics_cfg: Union[None, DictConfig] = self.evaluation.metrics_cfg.get(
            stage, None
        )
        dataset: Union[Dataset, dict[str, Dataset]] = getattr(self.trainer.datamodule, f"dataset_{stage}")  # type: ignore - datamodule not appropriately embedded

        if stage_metrics_cfg is not None:
            configs = (
                stage_metrics_cfg["_datasets_"]
                if "_datasets_" in stage_metrics_cfg
                else {"val": stage_metrics_cfg}
            )
            # torchmetrics must be moved to GPU
            for cfg in configs.values():
                for metric_cfg in cfg.values():
                    metric = metric_cfg["metric"]
                    if (
                        hasattr(metric, "to")
                        and hasattr(metric, "device")
                        and metric.device != self.device
                    ):
                        metric.to(self.device)

            # deepcopy original metrics cfg for each dataset
            # TODO(fdschmidt93): probably should be part of config_callbacks but torchmetrics OOP api requires later merging
            if isinstance(dataset, dict) and not "_datasets_" in stage_metrics_cfg:
                self.evaluation.metrics_cfg[stage] = {}
                self.evaluation.metrics_cfg[stage]["_datasets_"] = {}
                for name in dataset:
                    if self.hparams.evaluation is not None and hasattr(
                        self.hparams.evaluation, "metrics_cfg"
                    ):
                        self.evaluation.metrics_cfg[stage]["_datasets_"][
                            name
                        ] = hydra.utils.instantiate(
                            self.hparams.evaluation.metrics_cfg.get(stage)
                        )
                        metrics: DictConfig = self.evaluation.metrics_cfg[stage][
                            "_datasets_"
                        ][name]
                        for cfg in metrics.values():
                            metric = cfg["metric"]
                            if (
                                hasattr(metric, "to")
                                and hasattr(metric, "device")
                                and metric.device != self.device
                            ):
                                metric.to(self.device)

    def on_validation_start(self) -> None:
        self.on_eval_start(stage="val")

    def on_test_start(self) -> None:
        self.on_eval_start(stage="test")

    def logging(
        self,
        stage: str,
        metric_key: Union[str, DictKeyType],
        input: Union[int, float, dict],
        log_kwargs: Optional[dict[str, Any]] = None,
        dataset_name: Optional[str] = None,
    ):
        # TODO(fdschmidt93): document logging function
        fn: Union[None, Callable] = OmegaConf.select(
            self.evaluation, f"metrics_cfg.{stage}.{metric_key}.logging"
        )
        if isinstance(fn, Callable):
            input = fn(input)
        log_kwargs = log_kwargs if log_kwargs is not None else {}
        log_kwargs["prog_bar"] = True
        if isinstance(input, dict):
            prefix = dataset_name + "/" + stage if dataset_name is not None else stage
            log_kwargs["dictionary"] = {f"{prefix}/{k}": v for k, v in input.items()}
            self.log_dict(**log_kwargs)
        else:
            log_kwargs["name"] = f"{stage}/{metric_key}"
            if dataset_name is not None:
                log_kwargs["name"] = f"{dataset_name}/{log_kwargs['name']}"
            log_kwargs["value"] = input
            self.log(**log_kwargs)

    # TODO(fdschmidt93): can we reduce overhead?
    def prepare_batch(
        self, stage: str, batch: dict, dataset: Optional[str] = None
    ) -> dict:
        fn: Union[None, Callable, DictConfig] = OmegaConf.select(
            self.evaluation, f"prepare_cfg.{stage}.batch"
        )
        if fn and isinstance(fn, DictConfig) and dataset is not None:
            fn = fn._datasets_.get(dataset)
        if isinstance(fn, Callable):
            return fn(self, batch=batch, stage=stage)
        return batch

    def prepare_outputs(
        self, stage: str, outputs: dict, batch: dict, dataset: Optional[str] = None
    ) -> dict:
        fn: Union[None, Callable, DictConfig] = OmegaConf.select(
            self.evaluation, f"prepare_cfg.{stage}.outputs"
        )
        if fn and isinstance(fn, DictConfig) and dataset is not None:
            fn = fn._datasets_.get(dataset)
        if isinstance(fn, Callable):
            return fn(self, outputs=outputs, batch=batch, stage=stage)
        return outputs

    def prepare_step_outputs(
        self, stage: str, step_outputs: dict, dataset: Optional[str] = None
    ) -> dict:
        fn: Union[None, Callable] = OmegaConf.select(
            self.evaluation, f"prepare_cfg.{stage}.step_outputs"
        )
        if fn and isinstance(fn, DictConfig) and dataset is not None:
            fn = fn._datasets_.get(dataset)
        if isinstance(fn, Callable):
            return fn(self, outputs=step_outputs, stage=stage, dataset=dataset)
        return step_outputs

    # TODO(fdschmidt93): switch from `locals` to kwargs?
    def _prepare_metric_input(
        self,
        cfg: Union[dict, DictConfig],
        outputs: Union[dict, NamedTuple],
        batch: Optional[Union[dict, NamedTuple]] = None,
    ) -> dict:
        """Collects user-defined attributes of outputs & batch to compute metric.


        Args:
            outputs:
            batch: [TODO:description]
            cfg: [TODO:description]

        Returns:
            dict: [TODO:description]

        Raises:
            AssertionError: [TODO:description]
        """
        # TODO(fdschmidt93): allow self
        ret = {}
        local_vars = locals()
        for k, v in cfg.items():
            var, key = v.split(":")
            # TODO(fdschmidt93): rfc
            input_: dict = local_vars.get(var, {})
            if var == "self":
                val = deepgetitem(input_, key)
            else:
                val = (
                    input_.get(key, None)
                    if isinstance(input_, dict)
                    else getattr(input_, key, None)
                )
            if val is not None:
                ret[k] = val
            else:
                raise AssertionError(f"{k} not found in {var}")
        return ret

    @staticmethod
    def _collect_step_output(
        outputs: dict, batch: dict, stage_dico: Optional[Union[dict, DictConfig]] = None
    ) -> dict:
        """Collects user-defined attributes of outputs & batch at end of eval_step in dict."""
        # TODO(fdschmidt93): validate uniqueness
        # TODO(fdschmidt93): enable putting to other device
        # TODO(fdschmidt93): define clear behaviour if no step_outputs is defined
        # TODO(fdschmidt93): restricting step_output arguments to function arguments via inspect library
        if stage_dico is not None:
            ret = {}
            local_vars = locals()

            def set_val(dico, key, val):
                ret_val = local_vars.get(key, {}).get(val, None)
                if ret_val is not None:
                    # TODO: refactor, fusion of step outputs over datasets instead of batch results with some keys missing
                    # raise AttributeError(f"{val} not in {key}")
                    dico[val] = ret_val

            for key, vals in stage_dico.items():
                if isinstance(vals, (ListConfig, list)):
                    for val in vals:
                        set_val(ret, key, val)
                elif isinstance(vals, str):
                    set_val(ret, key, vals)
                else:
                    raise TypeError(
                        f"Should be either str or list[str], not {type(vals)}"
                    )
            return ret
        return {"outputs": outputs, "batch": batch}

    def eval_step(
        self, stage: str, batch: dict, dataloader_idx: Optional[int] = None
    ) -> None:
        """Performs model forward & user batch transformation in an eval step."""
        # TODO(fdschmidt93): can we maybe make accessing faster?
        # TODO(fdschmidt93): implement pattern get("stage", default=base_config)? - so on_eval_start discussion
        metrics_cfg = self.evaluation.metrics_cfg
        if metrics_cfg is None:
            return
        metrics_cfg: DictConfig = self.evaluation.metrics_cfg.get(stage, metrics_cfg)
        if metrics_cfg is not None and "_datasets_" in metrics_cfg:
            idx2dataset: dict[int, str] = getattr(self.trainer.datamodule, f"idx2dataset_{stage}")  # type: ignore - datamodule not appropriately embedded
            # satisfy linter
            assert idx2dataset is not None and dataloader_idx is not None
            # dataloader_idx must come from list so datasets with varying length are appropriately iterated
            # since Pytorch Lightning's CombinedLoader either reduces to shortest or extends to longest dataset otherwise
            metrics_cfg = metrics_cfg["_datasets_"][idx2dataset[dataloader_idx]]
        # `step_collection_dico` maps what to collect from `outputs` and `batch`
        # eg {"outputs": "logits", "batch": ["input_ids", "attention_mask"]

        dataset = None
        if dataloader_idx is not None:
            idx2dataset = getattr(self.trainer.datamodule, f"idx2dataset_{stage}")
            dataset = idx2dataset[dataloader_idx]
        step_collection_dico: Union[None, DictConfig] = OmegaConf.select(
            self.evaluation, f"step_outputs.{stage}"
        )
        if (
            dataset is not None
            and step_collection_dico is not None
            and "_datasets_" in step_collection_dico
        ):
            step_collection_dico = step_collection_dico._datasets_.get(dataset)
        # if multiple datasets val or test dataloaders
        batch = self.prepare_batch(stage=stage, batch=batch, dataset=dataset)
        outputs = self.prepare_outputs(stage, self(batch), batch, dataset=dataset)
        if metrics_cfg is not None:
            for v in metrics_cfg.values():
                if getattr(v, "compute_on", False) == "eval_step":
                    kwargs = self._prepare_metric_input(v.kwargs, outputs, batch)
                    v["metric"](**kwargs)

        # lightning 2.0 now requires user to maintain step outputs themselves
        # dataloader_idx == None is considered the equivalent of
        # the first dataset in a multi dataset evaluation
        # to harmonize handling on epoch end
        if dataloader_idx is None:
            dataloader_idx = 0
        # TODO(fdschmidt93): better validation against datasets (eg include dataset name)
        try:
            eval_outputs = self._eval_outputs[dataloader_idx]
        except IndexError as _:
            if len(self._eval_outputs) == dataloader_idx:
                eval_outputs = []
                self._eval_outputs.append(eval_outputs)
            else:
                raise IndexError(
                    f"dataloader_idx {dataloader_idx} is not subsequent index in evaluation, check warranted!"
                )
        except Exception:
            raise Exception("This must not happen.")
        eval_outputs.append(
            self._collect_step_output(outputs, batch, step_collection_dico)
        )

    def eval_epoch_end_dataset(
        self,
        stage: str,
        step_outputs: list[dict],
        metrics_cfg: DictConfig,
        dataset_name: Optional[str] = None,
    ) -> None:
        """Runs evaluation configuration on step_outputs for passed dataset.

        Args:
            stage: either "val", "test", or "predict"
            step_outputs: end_of_epoch `step_outputs` for corresponding dataset
            metrics_cfg: evaluation configuration of corresponding datasets
            dataset_name: name of dataset as denoted in datamodule config
        """
        flattened_step_outputs = flatten_dict(step_outputs)
        flattened_step_outputs = self.prepare_step_outputs(
            stage, flattened_step_outputs, dataset_name
        )
        for metric, metric_cfg in metrics_cfg.items():
            if getattr(metric_cfg, "compute_on", False) == "eval_step":
                # TODO(fdschmidt93): do not rely on having to call `compute` here
                self.logging(
                    stage=stage,
                    metric_key=metric,
                    input=metric_cfg["metric"],
                )
            if getattr(metric_cfg, "compute_on", False) == "epoch_end":
                kwargs: dict = self._prepare_metric_input(
                    metric_cfg.kwargs, flattened_step_outputs, None
                )
                self.logging(
                    stage=stage,
                    metric_key=metric,
                    input=metric_cfg["metric"](**kwargs),
                    dataset_name=dataset_name,
                )

    def on_eval_epoch_end(self, stage: str) -> None:
        """Computes evaluation metric at epoch end for respective `stage` for dataset(s).

        Notes:
            - Loops over datasets only once all
            - dataset(s) may potentially have individual evaluation configuration.


        Args:
            stage: typically either 'val' or 'test', affects logging
            step_outputs: outputs of eval steps & flattened at start of `eval_epoch_end`

        Returns:
            dict: flattened outputs from evaluation steps
        """
        metrics_cfg = self.evaluation.metrics_cfg
        if metrics_cfg is not None:
            metrics_cfg = self.evaluation.metrics_cfg.get(
                stage, self.evaluation.metrics_cfg
            )
        if metrics_cfg is None:
            return

        step_outputs: list[list[dict]] = self._eval_outputs
        idx2dataset: Union[None, dict[int, str]] = getattr(self.trainer.datamodule, f"idx2dataset_{stage}")  # type: ignore - datamodule not appropriately embedded
        # multiple datasets in `stage` dataloaders
        if idx2dataset is not None:
            for idx, dataset_name in idx2dataset.items():
                dataset_metrics = metrics_cfg._datasets_[dataset_name]
                dataset_outputs = step_outputs[idx]
                assert isinstance(dataset_outputs, list)  # avoid linting error
                self.eval_epoch_end_dataset(
                    stage=stage,
                    step_outputs=dataset_outputs,
                    dataset_name=dataset_name,
                    metrics_cfg=dataset_metrics,
                )
        else:
            assert (
                len(step_outputs) == 1
            ), "Something' off: `step_outputs` does not refer to single dataset!"
            self.eval_epoch_end_dataset(
                stage=stage,
                # TODO(fdschmidt93): resolve linting error
                step_outputs=step_outputs[0],
                metrics_cfg=metrics_cfg,
            )
        # clean up cached dataset(s) outputs
        self._eval_outputs.clear()

    def validation_step(
        self, batch: dict, batch_idx: int, dataloader_idx: Optional[int] = None
    ) -> None:
        return self.eval_step("val", batch, dataloader_idx)

    def on_validation_epoch_end(self):
        return self.on_eval_epoch_end("val")

    def test_step(
        self, batch: dict, batch_idx: int, dataloader_idx: Optional[int] = None
    ) -> None:
        return self.eval_step("test", batch, dataloader_idx)

    def on_test_epoch_end(self):
        return self.on_eval_epoch_end("test")
