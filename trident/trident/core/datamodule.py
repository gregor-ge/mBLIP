from functools import cached_property
from types import MethodType
from typing import Optional, Union

import hydra
from lightning import LightningDataModule
from lightning.pytorch.utilities import CombinedLoader
from omegaconf.dictconfig import DictConfig
from omegaconf.omegaconf import OmegaConf
from torch.utils.data import DataLoader, Dataset, IterableDataset

from trident.utils.logging import get_logger

log = get_logger(__name__)

# TODO(fdschmidt93): link dataloader against global cfg
# TODO(fdschmidt93): add docs for {train, val, test} options
# TODO(fdschmidt93): handle predict in more principled fashion
# TODO(fdschmidt93): update documentation once preprocessing routines are set
# TODO(fdschmidt93): document partial trident functions


class TridentDataModule(LightningDataModule):
    """
    The base class for all datamodules.

    The :obj:`TridentDataModule` facilitates writing a :obj:`LightningDataModule` with little to no boilerplate via Hydra configuration. It splits into

    - :obj:`dataset_cfg`:
    - :obj:`dataloader_cfg`:

    Args:
        dataset_cfg (:obj:`omegaconf.dictconfig.DictConfig`):

            A hierarchical :obj:`DictConfig` that instantiates or returns the dataset for :obj:`self.dataset_{train, val, test}`, respectively.

            Typical configurations follow the below pattern:

    .. seealso:: :py:func:`src.utils.hydra.instantiate_and_apply`, :py:func:`src.utils.hydra.expand`
        dataloader_cfg (:obj:`omegaconf.dictconfig.DictConfig`):


            .. seealso:: :py:func:`src.utils.hydra.expand`

        seed (:obj:`int`, `optional`):
            Linked against `config.seed` by default for convenience and maybe used
            for functionality that is not yet set by :obj:`lightning.seed_everything`,
            which sets the seed for `pytorch`, `numpy` and `python.random`.

    Notes:
        - The `train`, `val`, and `test` keys of :obj:`dataset_cfg` and :obj:`dataloader_cfg` join remaining configurations with priority to existing config
        - :obj:`dataloader_cfg` automatically generates `train`, `val`, and `test` keys for convenience as the config is evaluated lazily (i.e. when a :obj:`DataLoader` is requested)

    Example:

        .. code-block:: yaml

            _target_: src.datamodules.base.TridentDataModule
            _recursive_: false

            dataset_cfg:
              _target_: datasets.load.load_dataset
              # access methods of the instantiated object
              _method_:
                map: # dataset.map for e.g. tokenization
                  # kwargs for dataset.map
                  function:
                    _target_: src.utils.hydra.partial
                    _partial_: src.datamodules.preprocessing.{...}
                  num_proc: 12

              path: glue
              name: mnli
              train:
                split: "train"
              val:
                # inherits `path`, `name`, etc.
                split: "validation_mismatched+validation_matched"
              test:
                # set `path`, `name`, `lang` specifically, remainder inherited
                path: xtreme
                name: xnli
                lang: de
                split: "test"

            dataloader_cfg:
              _target_: torch.utils.data.dataloader.DataLoader
              batch_size: 8
              num_workers: 0
              pin_memory: true
              # linked against global cfg
              seed: ${seed}
    """

    def __init__(
        self,
        datamodule_cfg: DictConfig,
        dataset_cfg: DictConfig,
        dataloader_cfg: DictConfig,
        seed: int = 42,
        overrides: Optional[DictConfig] = None,
    ):
        super().__init__()
        self.seed = seed

        # variables
        self.datamodule_cfg = datamodule_cfg
        self.dataset_cfg = dataset_cfg
        self.dataloader_cfg = dataloader_cfg

        # reserved attributes
        self.dataset_train: Optional[Union[Dataset, dict[str, Dataset]]] = None
        self.dataset_val: Optional[Union[Dataset, dict[str, Dataset]]] = None
        self.dataset_test: Optional[Union[Dataset, dict[str, Dataset]]] = None
        self.dataset_predict: Optional[Union[Dataset, dict[str, Dataset]]] = None

        self.dataset_train_raw: Optional[Union[Dataset, dict[str, Dataset]]] = None
        self.dataset_val_raw: Optional[Union[Dataset, dict[str, Dataset]]] = None
        self.dataset_test_raw: Optional[Union[Dataset, dict[str, Dataset]]] = None
        self.dataset_predict_raw: Optional[Union[Dataset, dict[str, Dataset]]] = None

        self.overrides = hydra.utils.instantiate(overrides)
        if self.overrides is not None:
            for key, value in self.overrides.items():
                setattr(self, key, MethodType(value, self))

    @cached_property
    def idx2dataset_train(self) -> Optional[dict[int, str]]:
        """Returns dict aligning dataloader_idx to dataset name for multi-train datasets."""
        if isinstance(self.dataset_train, dict):
            return {i: k for i, k in enumerate(self.dataset_train.keys())}

    @cached_property
    def idx2dataset_val(self) -> Optional[dict[int, str]]:
        """Returns dict aligning dataloader_idx to dataset name for multi-val datasets."""
        if isinstance(self.dataset_val, dict):
            return {i: k for i, k in enumerate(self.dataset_val.keys())}

    @cached_property
    def idx2dataset_test(self) -> Optional[dict[int, str]]:
        """Returns dict aligning dataloader_idx to dataset name for multi-test datasets."""
        if isinstance(self.dataset_test, dict):
            return {i: k for i, k in enumerate(self.dataset_test.keys())}

    @cached_property
    def idx2dataset_predict(self) -> Optional[dict]:
        """Returns dict aligning dataloader_idx to dataset name for multi-predict datasets."""
        if isinstance(self.dataset_predict, dict):
            return {i: k for i, k in enumerate(self.dataset_predict.keys())}

    @cached_property
    def _signature_columns(self) -> Optional[list[str]]:
        import inspect

        trainer = self.trainer
        if module := getattr(trainer, "model"):
            assert hasattr(
                module, "model"
            ), f"{type(module)} does not have a `model` attribute!"
            # Inspect model forward signature to keep only the arguments it accepts.
            signature = inspect.signature(module.model.forward)
            _signature_columns = list(signature.parameters.keys())
            # Labels may be named label or label_ids, the default data collator handles that.
            _signature_columns += ["label", "label_ids"]
            return _signature_columns

    def __len__(self) -> int:
        """Returns the number of instances in :obj:`dataset_train`."""
        if self.dataset_train is None:
            return 0
        elif isinstance(self.dataset_train, dict):
            return max([len(dataset) for dataset in self.dataset_train.values()])  # type: ignore
        else:
            if not isinstance(self.dataset_train, IterableDataset):
                return len(self.dataset_train)  # type: ignore
            else:
                return self.trainer.global_step

    def prepare_data(self) -> None:
        """
        Notice: untested.

        .. seealso:: `LightningDataModule.prepare_data <https://pytorch-lightning.readthedocs.io/en/latest/extensions/datamodules.html#prepare-data>`_
        """
        hydra.utils.call(self.datamodule_cfg.get("prepare_data", None), self)

    # TODO(fdschmidt93): enable overrides
    # def transfer_batch_to_device(self, batch, dataloader_idx: int) -> None:
    #     """
    #     .. seealso:: `LightningDataModule.transfer_batch_to_device <https://pytorch-lightning.readthedocs.io/en/latest/extensions/datamodules.html#transfer-batch-to-device>`_
    #     """
    #     self.datamodule_cfg.transfer_batch_to_device(self, batch, dataloader_idx)

    # def on_before_batch_transfer(self, batch, dataloader_idx: int) -> None:
    #     """
    #     .. seealso:: `LightningDataModule.on_before_batch_transfer <https://lightning.readthedocs.io/en/latest/extensions/datamodules.html#on-before-batch-transfer>`_
    #     """
    #     self.datamodule_cfg.on_before_batch_transfer(self, batch, dataloader_idx)

    # def on_after_batch_transfer(self, batch, dataloader_idx: int) -> None:
    #     """
    #     .. seealso:: `LightningDataModule.on_after_batch_transfer <https://pytorch-lightning.readthedocs.io/en/latest/extensions/datamodules.html#on-after-batch-transfer>`_
    #     """
    #     self.datamodule_cfg.on_after_batch_transfer(self, batch, dataloader_idx)

    def _datamodule_hook(self, kind: str, *args, **kwargs) -> None:
        # `call` returns None if cfg is None
        hydra.utils.call(
            getattr(self.datamodule_cfg, kind, None), datamodule=self, *args, **kwargs
        )

    def setup(self, stage: Optional[str] = None) -> None:
        """
        Sets up `self.dataset_{train, val, test}` datasets that are fed to the corresponding :obj:`DataLoader` instances.

        :obj:`dataset_cfg` must point to a function with the following specification. The below example shows a naive use case:

        .. code-block:: python

            def setup(cfg: DictConfig, datamodule: TridentDataModule, stage: Optional[str]) -> None:
                # load and process the dataset as defined in dataset_cfg.my_dataset configuration
                dataset = hydra.utils.instantiate(cfg.my_dataset)
                datamodule.dataset_train = dataset["train"]
                datamodule.dataset_val = dataset["val"]
                datamodule.dataset_test = dataset["test"]

        The corresponding :obj:`datamodule` configuration may then look as follows:


        .. code-block:: yaml

            dataset_cfg:
                _target_: src.custom.my_setup_func
                _partial_: true

                my_dataset:
                    _target_: src.custom.dataset.MyDataset
                    ...

        Args:
            self: datamodule
            stage: one of "fit", "validate", "test"

        .. seealso:: `LightningDataModule.setup <https://pytorch-lightning.readthedocs.io/en/latest/extensions/datamodules.html#setup>`_, :py:func:`src.datamodules.utils.load_dataset`, :py:func:`src.utils.hydra.instantiate_and_apply`
        """
        self._datamodule_hook(kind="on_before_setup")
        hydra.utils.call(
            self.datamodule_cfg.setup, self, stage, getattr(self, "dataset_cfg", None)
        )
        self._datamodule_hook(kind="on_after_setup")

    def _init_dataloader(
        self, dataset: Dataset, dataloader_cfg: DictConfig
    ) -> DataLoader:
        """Internal method to instantiate `DataLoader` with `dataset` and `dataloader_cfg`."""
        if OmegaConf.select(self.datamodule_cfg, "remove_unused_columns"):
            dataset = self._remove_unused_columns(dataset)
        return hydra.utils.call(dataloader_cfg, dataset=dataset)

    # TODO(fdschmidt93): document (or refactor) _remove_unused_columns
    def _get_dataloader(self, split: str) -> Union[DataLoader, CombinedLoader]:
        """Checks existence of dataset for :obj:`split` and returns :obj:`DataLoader` with cfg.

        The return type of this function typically depends on the scenario:
            * :obj:`DataLoader`: simple, single datasets
            * :obj:`CombinedLoader`: for modes, see CombinedLoader documentation
                - mode = "sequential" common in zero-shot cross-lingual transfer, evaluating on many varying datasets
                - mode = "max_size_cycle" common in zero-shot cross-lingual transfer, evaluating on many varying datasets

            .. seealso:: :py:meth:`trident.core.datamodule.TridentDataModule._get_dataloader`

        Args:
            split: one of :obj:`train`, :obj:`val`, :obj:`test`, or :obj:`predict`

        Returns:
            Union[DataLoader, list[DataLoader], dict[str, DataLoader]]: [TODO:description]
        """
        dataset: Union[None, Dataset, dict[str, Dataset]] = getattr(
            self, f"dataset_{split}"
        )
        assert dataset is not None, f"Dataset for {split} missing!"

        # single dataset
        if not isinstance(dataset, dict):
            return self._init_dataloader(
                dataset, getattr(self, "dataloader_cfg")[split]
            )
        else:
            dataloader_dict: dict[str, DataLoader] = {}
            dataloader_cfg = getattr(self, "dataloader_cfg")[split]

            # dataloader configuration shared for all datasets
            if not "_datasets_" in dataloader_cfg:
                log.info(
                    f"Sharing {split}-dataloader configuration for {list(dataset.keys())}"
                )
                dl_cfg_iter: dict[str, DictConfig] = {
                    k: dataloader_cfg.copy() for k in dataset.keys()
                }
            else:
                dl_cfg_iter: dict[str, DictConfig] = dataloader_cfg._datasets_
            assert set(dataset.keys()) == set(
                dl_cfg_iter.keys()
            ), "Keys between datasets and dataloaders configuration does not align!"

            for name, cfg in dl_cfg_iter.items():
                dataloader_dict[name] = self._init_dataloader(dataset[name], cfg)
            # training split automatically wrapped
            if split == "train":
                return CombinedLoader(
                    dataloader_dict,
                    mode=getattr(
                        self.datamodule_cfg,
                        "train_multi_dataloader_mode",
                        "max_size_cycle",
                    ),
                )
            else:
                dloaders: list[DataLoader] = list(dataloader_dict.values())  # type: ignore
                # verify order
                idx2dataset: dict[str, str] = getattr(self, f"idx2dataset_{split}")
                for idx, dataset_name_ in idx2dataset.items():
                    # `is` checks equality of memory addresses
                    assert dloaders[int(idx)] is dataloader_dict[dataset_name_]
                # lightning 2.0 now supports sequential CombinedLoader
                return CombinedLoader(dloaders, mode="sequential")

    def train_dataloader(
        self,
    ) -> Union[DataLoader, CombinedLoader]:
        return self._get_dataloader("train")

    def val_dataloader(
        self,
    ) -> Union[DataLoader, CombinedLoader]:
        return self._get_dataloader("val")

    def test_dataloader(
        self,
    ) -> Union[DataLoader, CombinedLoader]:
        return self._get_dataloader("test")

    def predict_dataloader(
        self,
    ) -> Union[DataLoader, CombinedLoader]:
        return self._get_dataloader("predict")

    # TODO(fdschmidt93): maybe move out-of trident-core and into trident-xtreme
    def _remove_unused_columns(
        self,
        dataset: Dataset,
    ) -> Dataset:
        signature_columns = (
            self._signature_columns if self._signature_columns is not None else []
        )
        column_names: list[str] = getattr(dataset, "column_names")
        ignored_columns = list(set(column_names) - set(signature_columns))
        if len(ignored_columns) > 0:
            log.info(
                f"The following columns don't have a corresponding argument in "
                f"`{self.trainer.model.__class__.__name__}.forward` and have been ignored: {', '.join(ignored_columns)}."  # type: ignore
            )
        # ignoring as new_fingerprint typically not passed
        return dataset.remove_columns(ignored_columns)  # type: ignore
