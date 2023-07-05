import functools
from typing import Any, Union

import hydra
from hydra.utils import get_method
from omegaconf import DictConfig, OmegaConf

_ExtraKeys = ["_method_", "_apply_"]

from trident.utils.logging import get_logger

log = get_logger(__name__)


# TODO(fdschmidt93): test when wrapped up in partial
def get_local(var):
    import inspect

    frame = inspect.currentframe()
    locals: Union[None, dict] = getattr(getattr(frame, "f_back"), "f_locals")
    if locals is None:
        return None
    args = var.split(".")
    objects = [locals.get(args[0], None)]
    for i, arg in enumerate(args[1:]):
        val = getattr(objects[i], arg, None)
        objects.append(val)
    return objects[-1]


def partial(*args, **kwargs):
    """Implements functools.partial for Hydra.

    :obj:`partial` is very handy for repeated function calls as :obj:`hydra.utils.call` incurs a high latency.

    Example:

        Code:

        .. code-block:: python

            # module: src.custom.functional
            my_transform(batch: BatchEncoding) -> BatchEncoding:
                batch["sequence_length"] = batch["attention_mask"].sum(dim=-1)
                return batch

        Config:

        .. code-block:: yaml

            my_key:
                _target_: src.utils.hydra.partial
                _partial_: src.custom.functional.my_transform

        :py:func:`src.utils.hydra.partial` can also be leveraged to partially define class methods. The configuration follows the below pattern:

        .. code-block:: python
            # Hydra configuration translated to function/classmethod signature
            # <--- _partial_ ---><-self> <args, kwargs>
            cfg: DictConfig # your configuration
            my_cls_obj = hydra.utils.instantiate(cfg.self) # `self` in classmethod
            MyClass.__function__(my_cls_obj, *args, **kwargs)

        Your :obj:`_target_` points to the classmethod to be partially defined, while the :obj:`self` key lays out how the corresponding class instance is instantiated. The below example configuration illustrates how to declaratively partially define a Huggingface Tokenizer.

        .. code-block:: yaml

            tokenizer:
                # classmethod to call
                _target_: src.utils.hydra.partial
                _partial_: transformers.tokenization_utils_base.PreTrainedTokenizerBase.__call__
                # `hydra.utils.instantiate` for self in classmethod
                self:
                    _target_: transformers.AutoTokenizer.from_pretrained
                    pretrained_model_name_or_path: roberta-base
                # other kwargs for classmethod
                padding: true
                truncation: true
                max_length: 512

        You might think that the above example is rather convoluted. However, the pattern allows you to flexibly combine functions and class methods alike with a transparent syntax that avoids indirection via additional wrappers that would otherwise be required to define a class and its associated method call.
    """
    _partial_ = kwargs.pop("_partial_")
    method = get_method(_partial_)
    if "self" in kwargs:
        obj = kwargs.pop("self")
        args = (obj, *args)
    return functools.partial(method, *args, **kwargs)


def expand(
    cfg: DictConfig,
    merge_keys: Union[str, list[str]] = ["train", "val", "test"],
    gen_keys: bool = False,
) -> DictConfig:
    """Expands partial configuration of `keys` in `cfg` with the residual configuration.

    Most useful when configuring modules that have a substantial shared component.

    Applied by default on :obj:`dataset_cfg` (with :code:`create_keys=False`) and :obj:`dataloader_cfg` (with :code:`create_keys=True`) of :obj:`DataModule` config.

    Notes:
        - Shared config reflects all configuration excluding set :obj:`keys`.

    Args:
        keys (:obj:`Union[str, list[str])`):
            Keys that comprise dedicated configuration for which shared config will be merged.

        gen_keys (:obj:`bool`):
            Whether (:code:`True`) or not (:code:`False) to create :code:`keys` in :code:`cfg: with shared configuration if :code:`keys` do not exist yet.

    Example:
        :code:`expand(cfg, keys=["train", "val", "test"], create_keys=True)` with the following config

        .. code-block:: yaml

            dataloader_cfg:
                batch_size: 4
                num_workers: 8
                train:
                    batch_size: 8
                    shuffle: True
                test:
                    shuffle: False

        resolves to

        .. code-block:: yaml

            dataloader_cfg:
                train:
                    shuffle: True
                    batch_size: 8
                    num_workers: 8
                val:
                    batch_size: 4
                    num_workers: 8
                test:
                    shuffle: False
                    batch_size: 4
                    num_workers: 8

        while only the original config is the one being logged.
    """
    special_keys = ["_datasets_"]
    if cfg is not None:
        if isinstance(merge_keys, str):
            merge_keys = [merge_keys]
        shared_keys = [key for key in cfg.keys() if key not in merge_keys]
        cfg_excl_keys = OmegaConf.masked_copy(cfg, shared_keys)
        for key in merge_keys:  # train, val, test
            if key_cfg := cfg.get(key, None):  # cfg
                # if special key for multiple datasets/dataloaders
                if any(
                    # TODO(fdschmidt93): declare more "globally"
                    # TODO(fdschmidt93): recursion to make this nicer?
                    c in key_cfg
                    for c in special_keys
                ):  # if _datasets_ in cfg
                    for sub_key in special_keys:  # for each special in cfg
                        if sub_key_cfg := key_cfg.get(sub_key, None):  # "_datasets_"
                            for name in sub_key_cfg:  # source: ... target: ...
                                # merge True by default
                                cfg[key][sub_key][name] = OmegaConf.merge(
                                    cfg_excl_keys, cfg[key][sub_key][name]
                                )
                else:
                    cfg[key] = OmegaConf.merge(cfg_excl_keys, cfg[key])
            else:
                if gen_keys:
                    cfg[key] = cfg_excl_keys
        for key in shared_keys:
            cfg.pop(key)
        return cfg


# TODO(fdschmidt93): update documentation once preprocessing routines are set
# TODO(fdschmidt93): add _keep_ to docs
def instantiate_and_apply(cfg: Union[None, DictConfig]) -> Any:
    r"""Adds :obj:`_method_` and :obj:`_apply_` keywords for :code:`hydra.utils.instantiate`.

    :obj:`_method_` and :obj:`_apply_` describe methods and custom functions to be applied on the instantiated object in order of the configuration. Most commonly, you want to make use of :obj:`dataset` `processing methods <https://huggingface.co/docs/datasets/process.html>`_\. For convenience

    Args:
        cfg (:obj:`omegaconf.dictconfig.DictConfig`):
            The :obj:`dataset_cfg` for your :obj:`TridentDataModule`

    Notes:
        - If :obj:`_method_` and :obj:`_apply_` are not set, :obj:`instantiate_and_apply` essentially reduces to :obj:`hydra.utils.instantiate`
        - :obj:`instantiate_and_apply` is only applied for the :obj:`dataset_cfg` of the TridentDataModule
        - The function arguments must be dictionaries
        - Any transformation is applied sequentially -- **order matters**!
            * If you want to intertwine :code:`_method_` and :code:`_apply_` use the former via the latter as per the final example.

    Example:
        The below example levers `datasets <https://huggingface.co/docs/datasets/>`_ to instantiate MNLI seamlessly with the below Python code and YAML configuration.

        .. code-block:: python

            dataset = instantiate_and_apply(cfg=dataset_cfg)

        .. code-block:: yaml

            dataset_cfg:
                _target_: datasets.load.load_dataset
                # required key!
                _recursive_: false
                # apply methods of _target_ object
                _method_:
                  map: # uses method of instantiated object, here dataset.map
                    # kwargs for dataset.map
                    function:
                      _target_: src.datamodules.preprocessing.{...}
                    batched: true
                  set_transform: # use dataset.set_transform
                    # kwargs for dataset.set_transform
                    _target_: src.custom.my_transform
                _apply_:
                    my_transform:
                        _target_: src.hydra.utils.partial
                        _partial_: src.custom.utils.my_transform

        The order of transformation is (1) :code:`map`, (2) :code:`set_transform`, and (3) :code:`my_transform`.

        \(1) :code:`map`, (2) :code:`my_transform`, and (3) :code:`set_transform` would be possible as follows.

        .. code-block:: yaml

            # ...
            _apply_:
                map:
                    _target_: dataset.arrow_dataset.Dataset.map
                    # ...
                my_transform:
                    _target_: src.hydra.utils.partial
                    _partial_: src.custom.utils.my_transform
                    # ...
                set_transform:
                    _target_: dataset.arrow_dataset.Dataset.set_transform
                    # ...

    Returns:
        Any: your instantiated object processed with _method_ & _apply_ functions
    """
    if cfg is None:
        return None

    # instantiate top-level cfg
    cfg_keys = list(cfg.keys())  # avoid changing dictionary size in loop
    extra_kwds = {key: cfg.pop(key) for key in cfg_keys if key in _ExtraKeys}
    obj = hydra.utils.instantiate(cfg)

    if not extra_kwds:
        return obj
    extra_kwds = hydra.utils.instantiate(OmegaConf.create(extra_kwds))
    # kwd: {_method_, _apply_}
    # kwd_config: their respective collections of functions
    # key: name of user method or function
    # kwd_config: their respective config
    # TODO(fdschmidt93): handle ListConfig?
    for kwd, kwd_cfg in extra_kwds.items():
        for key, key_cfg in kwd_cfg.items():
            # _method_ is for convenience
            # construct partial wrapper, instantiate with cfg, and apply to ret
            if kwd == "_method_":
                key_cfg[
                    "_target_"
                ] = f"{obj.__class__.__module__}.{obj.__class__.__name__}.{key}"
                key_cfg["_partial_"] = True
                fn = hydra.utils.instantiate(key_cfg)
                val = fn(obj)
                # `fn` might mutate ret in-place
                if val is not None:
                    obj = val
            else:
                obj = key_cfg(obj)
    return obj


def config_callbacks(cfg: DictConfig, cb_cfg: DictConfig) -> DictConfig:
    """Amends configuration with user callback by configuration key.

    Hydra excels at depth-first, bottom-up config resolution. However,
    such a paradigm does not always allow you to elegantly express scenarios
    that are very relevant in experimentation. One instance, where :obj:`trident`
    levers :obj:`config_callback`s is the `Huggingface datasets <https://huggingface.co/docs/datasets/>`_ integration.

    An example configuration may look like follows:

    .. code-block:: yaml

        config: # global config
          datamodule:
            dataset_cfg:
              # ${SHARED}
              _target_: datasets.load.load_dataset
              #     trident-integration into huggingface datasets
              #     to lever dataset methods within yaml configuration
              _method_:
                function:
                  _target_: src.utils.hydra.partial
                  _partial_: src.datamodules.utils.preprocessing.text_classification
                  tokenizer:
                    _target_: src.datamodules.utils.tokenization.HydraTokenizer
                    pretrained_model_name_or_path: roberta-base
                    max_length: 53
                batched: false
                num_proc: 12

              path: glue
              name: mnli

              # ${INDIVIDUAL}
              train:
                split: "train"
                # ${SHARED} will be merged into {train, val test} with priority for existing config
              val:
                split: "validation_mismatched+validation_matched"
              test:
                path: xtreme # overrides shared glue
                name: xnli # overrides shared mnli
                lang: de
                split: "test"


    Args:
        cfg:
        cb_cfg:

    Returns:
        DictConfig:

    .. seealso:: :py:func:`src.utils.hydra.expand`, :py:func:`src.utils.hydra.instantiate_and_apply`, :py:func:`src.datamodule.utils.load_dataset`
    """
    for key in cb_cfg:
        if to_process_cfg := OmegaConf.select(cfg, key):
            processed_cfg = hydra.utils.call(cb_cfg.get(key), to_process_cfg)
            OmegaConf.update(cfg, key, processed_cfg)
        else:
            log.info(f"Attempted to mutate non-existing {key} configuration.")
    return cfg
