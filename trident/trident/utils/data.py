from typing import Optional, Tuple, Union

import hydra
from omegaconf import OmegaConf
from omegaconf.dictconfig import DictConfig
from torch.utils.data import Dataset

from trident.core import TridentDataModule
from trident.utils.hydra import instantiate_and_apply


def load_dataset(
    cfg: Union[None, DictConfig], keep_raw_dataset: bool
) -> Tuple[Dataset, Union[None, Dataset]]:
    dataset_raw = None
    if cfg is not None and keep_raw_dataset:
        raw_cfg = OmegaConf.masked_copy(
            cfg,
            [
                str(key) for key in cfg if key not in ["_method_", "_apply_"]
            ],  # str(key) -> satisfy linter
        )
        dataset_raw = hydra.utils.instantiate(raw_cfg)
    dataset = instantiate_and_apply(cfg)
    return (dataset, dataset_raw)


def setup_dataset(
    self: TridentDataModule, cfg: Union[None, DictConfig], split: str
) -> None:
    keep_raw_dataset = self.datamodule_cfg.get("keep_raw_dataset", False)
    self._datamodule_hook("on_before_dataset_setup", split=split)
    if cfg and "_datasets_" in cfg:
        dataset = {}
        dataset_raw = {}
        for dataset_name, dataset_cfg in cfg["_datasets_"].items():
            dataset[dataset_name], dataset_raw[dataset_name] = load_dataset(
                dataset_cfg, keep_raw_dataset
            )
    else:
        dataset, dataset_raw = load_dataset(cfg, keep_raw_dataset)
    setattr(self, f"dataset_{split}", dataset)
    setattr(self, f"dataset_{split}_raw", dataset_raw)
    self._datamodule_hook("on_after_dataset_setup", split=split)


def setup(
    self: TridentDataModule,
    stage: Optional[str],
    config: DictConfig,
) -> None:
    # instantiate_and_apply extends `hydra.utils.instantiate` with[str, bool]
    # - _method_: call methods onto the instantiated object
    # - _apply_: call any function onto the instantiated object

    if stage in (None, "fit"):
        setup_dataset(self, config.get("train", None), "train")
    if stage in (None, "fit", "validate"):
        setup_dataset(self, config.get("val", None), "val")
    if stage in (None, "test"):
        setup_dataset(self, config.get("test", None), "test")
    if stage in (None, "predict"):
        setup_dataset(self, config.get("predict", None), "predict")
